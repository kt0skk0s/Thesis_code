import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Linear, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 0. Choose task
# ============================================================

# Βάλε:
# TASK = "prepost" για PRE/POST -> Valence/Arousal
# TASK = "game"    για EASY/HARD/OPTIMAL -> Positive/Negative Affect

TASK = "prepost"
#TASK = "game"


# ============================================================
# 1. Paths and task setup
# ============================================================

RESULTS_DIR = r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\gnn_results"

if TASK == "prepost":
    SPLIT_DIR = os.path.join(RESULTS_DIR, "subject_level_splits_prepost_va_original")
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "weighted_gcn_prepost_va_no_posweight")

    TASK_TITLE = "PRE/POST Valence-Arousal"
    TARGET_NAMES = ["valence", "arousal"]

elif TASK == "game":
    SPLIT_DIR = os.path.join(RESULTS_DIR, "subject_level_splits_game_geq_pa_na")
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "weighted_gcn_game_geq_pa_na_no_posweight")

    TASK_TITLE = "GAME GEQ Positive-Negative Affect"
    TARGET_NAMES = ["positive_affect", "negative_affect"]

else:
    raise ValueError("TASK must be either 'prepost' or 'game'.")


TRAIN_PATH = os.path.join(SPLIT_DIR, "train_graphs_subject.pt")
VAL_PATH = os.path.join(SPLIT_DIR, "val_graphs_subject.pt")
TEST_PATH = os.path.join(SPLIT_DIR, "test_graphs_subject.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_weighted_gcn_posweight.pt")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "test_summary.txt")


# ============================================================
# 2. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Task:", TASK_TITLE)
print("Split dir:", SPLIT_DIR)
print("Output dir:", OUTPUT_DIR)


# ============================================================
# 4. Load splits
# ============================================================

def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


print("\nLoading graph splits...")

train_graphs = torch_load(TRAIN_PATH)
val_graphs = torch_load(VAL_PATH)
test_graphs = torch_load(TEST_PATH)

print("Train graphs:", len(train_graphs))
print("Val graphs:", len(val_graphs))
print("Test graphs:", len(test_graphs))

print("\nExample train graph:")
print(train_graphs[0])
print("x shape:", train_graphs[0].x.shape)
print("edge_index shape:", train_graphs[0].edge_index.shape)
print("edge_attr shape:", train_graphs[0].edge_attr.shape)
print("y:", train_graphs[0].y)
print("subject_id:", train_graphs[0].subject_id)
print("condition:", train_graphs[0].condition)

if hasattr(train_graphs[0], "target_names"):
    print("target_names:", train_graphs[0].target_names)

if hasattr(train_graphs[0], "task_name"):
    print("task_name:", train_graphs[0].task_name)


# ============================================================
# 5. Label distribution and pos_weight
# ============================================================

def get_y_matrix(graphs):
    ys = []

    for g in graphs:
        y = g.y.view(-1)

        if y.numel() != 2:
            raise ValueError(f"Expected y with 2 labels, got shape {g.y.shape}")

        ys.append(y.view(1, 2))

    return torch.cat(ys, dim=0).float()


def print_label_distribution(name, graphs):
    y = get_y_matrix(graphs)

    first_0 = int(torch.sum(y[:, 0] == 0).item())
    first_1 = int(torch.sum(y[:, 0] == 1).item())

    second_0 = int(torch.sum(y[:, 1] == 0).item())
    second_1 = int(torch.sum(y[:, 1] == 1).item())

    joint_00 = int(torch.sum((y[:, 0] == 0) & (y[:, 1] == 0)).item())
    joint_01 = int(torch.sum((y[:, 0] == 0) & (y[:, 1] == 1)).item())
    joint_10 = int(torch.sum((y[:, 0] == 1) & (y[:, 1] == 0)).item())
    joint_11 = int(torch.sum((y[:, 0] == 1) & (y[:, 1] == 1)).item())

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("Graphs:", len(graphs))
    print(f"{TARGET_NAMES[0]} 0:", first_0)
    print(f"{TARGET_NAMES[0]} 1:", first_1)
    print(f"{TARGET_NAMES[1]} 0:", second_0)
    print(f"{TARGET_NAMES[1]} 1:", second_1)

    print("\nJoint labels:")
    print("[0,0]:", joint_00)
    print("[0,1]:", joint_01)
    print("[1,0]:", joint_10)
    print("[1,1]:", joint_11)


print_label_distribution("TRAIN label distribution", train_graphs)
print_label_distribution("VALIDATION label distribution", val_graphs)
print_label_distribution("TEST label distribution", test_graphs)


y_train = get_y_matrix(train_graphs)

first_pos = torch.sum(y_train[:, 0] == 1).item()
first_neg = torch.sum(y_train[:, 0] == 0).item()

second_pos = torch.sum(y_train[:, 1] == 1).item()
second_neg = torch.sum(y_train[:, 1] == 0).item()

pos_weight_first = first_neg / max(first_pos, 1)
pos_weight_second = second_neg / max(second_pos, 1)

pos_weight = torch.tensor(
    [pos_weight_first, pos_weight_second],
    dtype=torch.float32
).to(device)

print("\npos_weight:")
print(f"{TARGET_NAMES[0]}:", pos_weight_first)
print(f"{TARGET_NAMES[1]}:", pos_weight_second)


# ============================================================
# 6. DataLoaders
# ============================================================

BATCH_SIZE = 64

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 7. Weighted GCN model
# ============================================================

class WeightedGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels=50,
        hidden_channels=64,
        out_channels=2,
        dropout=0.4
    ):
        super().__init__()

        self.dropout_layer = Dropout(dropout)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # edge_attr = Pearson correlation.
        # Χρησιμοποιούμε absolute correlation strength ως edge weight.
        edge_weight = data.edge_attr.view(-1).float().abs()

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout_layer(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Node embeddings -> graph embedding
        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)

        out = self.lin2(x)

        return out


model = WeightedGCN(
    in_channels=50,
    hidden_channels=64,
    out_channels=2,
    dropout=0.4
).to(device)

print("\nModel:")
print(model)


# ============================================================
# 8. Loss and optimizer
# ============================================================

#criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# ============================================================
# 9. Evaluation function
# ============================================================

def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    all_true = []
    all_probs = []
    all_preds = []

    total_loss = 0.0
    total_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            logits = model(batch)

            # PyG μπορεί να κάνει flatten το y.
            y = batch.y.float().view(-1, 2)

            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            all_true.append(y.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

    y_true = torch.cat(all_true, dim=0).numpy()
    y_probs = torch.cat(all_probs, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()

    avg_loss = total_loss / total_graphs

    results = {}

    for i, name in enumerate(TARGET_NAMES):
        results[f"{name}_accuracy"] = accuracy_score(y_true[:, i], y_pred[:, i])

        results[f"{name}_precision"] = precision_score(
            y_true[:, i],
            y_pred[:, i],
            zero_division=0
        )

        results[f"{name}_recall"] = recall_score(
            y_true[:, i],
            y_pred[:, i],
            zero_division=0
        )

        results[f"{name}_f1"] = f1_score(
            y_true[:, i],
            y_pred[:, i],
            zero_division=0
        )

        results[f"{name}_macro_f1"] = f1_score(
            y_true[:, i],
            y_pred[:, i],
            average="macro",
            zero_division=0
        )

    results["mean_task_f1"] = (
        results[f"{TARGET_NAMES[0]}_f1"] +
        results[f"{TARGET_NAMES[1]}_f1"]
    ) / 2.0

    results["mean_binary_macro_f1"] = (
        results[f"{TARGET_NAMES[0]}_macro_f1"] +
        results[f"{TARGET_NAMES[1]}_macro_f1"]
    ) / 2.0

    # Για συμβατότητα με τα παλιά scripts
    results["macro_f1"] = results["mean_task_f1"]

    results["exact_match_accuracy"] = np.all(y_true == y_pred, axis=1).mean()

    results[f"pred_{TARGET_NAMES[0]}_0"] = int(np.sum(y_pred[:, 0] == 0))
    results[f"pred_{TARGET_NAMES[0]}_1"] = int(np.sum(y_pred[:, 0] == 1))
    results[f"pred_{TARGET_NAMES[1]}_0"] = int(np.sum(y_pred[:, 1] == 0))
    results[f"pred_{TARGET_NAMES[1]}_1"] = int(np.sum(y_pred[:, 1] == 1))

    return avg_loss, results, y_true, y_probs, y_pred


# ============================================================
# 10. Training loop with early stopping
# ============================================================

EPOCHS = 100
PATIENCE = 15

best_val_macro_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

print(f"\nStarting Weighted GCN training on {TASK_TITLE}...\n")

for_epoch_range = range(1, EPOCHS + 1)

for epoch in for_epoch_range:
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits = model(batch)
        y = batch.y.float().view(-1, 2)

        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    train_loss = total_loss / total_graphs

    val_loss, val_results, _, _, _ = evaluate(
        model,
        val_loader,
        criterion,
        device,
        threshold=0.5
    )

    val_macro_f1 = val_results["macro_f1"]

    improved = val_macro_f1 > best_val_macro_f1

    if improved:
        best_val_macro_f1 = val_macro_f1
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"{TARGET_NAMES[0]} F1: {val_results[f'{TARGET_NAMES[0]}_f1']:.4f} | "
        f"{TARGET_NAMES[1]} F1: {val_results[f'{TARGET_NAMES[1]}_f1']:.4f} | "
        f"Mean Task F1: {val_results['mean_task_f1']:.4f} | "
        f"Exact Match: {val_results['exact_match_accuracy']:.4f}"
    )

    print(
        f"Predictions | "
        f"{TARGET_NAMES[0]} 0/1: "
        f"{val_results[f'pred_{TARGET_NAMES[0]}_0']}/"
        f"{val_results[f'pred_{TARGET_NAMES[0]}_1']} | "
        f"{TARGET_NAMES[1]} 0/1: "
        f"{val_results[f'pred_{TARGET_NAMES[1]}_0']}/"
        f"{val_results[f'pred_{TARGET_NAMES[1]}_1']}"
    )

    print("-" * 100)

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation mean task F1: {best_val_macro_f1:.4f}")
        break


# ============================================================
# 11. Load best model and evaluate on test set
# ============================================================

print("\nLoading best model...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

test_loss, test_results, y_true, y_probs, y_pred = evaluate(
    model,
    test_loader,
    criterion,
    device,
    threshold=0.5
)

print("\nFinal TEST results")
print("Test loss:", test_loss)

print("\nMetrics:")
for k, v in test_results.items():
    print(f"{k}: {v}")


# ============================================================
# 12. Confusion matrices and reports
# ============================================================

cm_first = confusion_matrix(y_true[:, 0], y_pred[:, 0], labels=[0, 1])
cm_second = confusion_matrix(y_true[:, 1], y_pred[:, 1], labels=[0, 1])

print(f"\nConfusion matrix - {TARGET_NAMES[0]}")
print("Rows = true labels, columns = predicted labels")
print("          Pred 0   Pred 1")
print(f"True 0    {cm_first[0,0]:6d}   {cm_first[0,1]:6d}")
print(f"True 1    {cm_first[1,0]:6d}   {cm_first[1,1]:6d}")

print(f"\nConfusion matrix - {TARGET_NAMES[1]}")
print("Rows = true labels, columns = predicted labels")
print("          Pred 0   Pred 1")
print(f"True 0    {cm_second[0,0]:6d}   {cm_second[0,1]:6d}")
print(f"True 1    {cm_second[1,0]:6d}   {cm_second[1,1]:6d}")

print(f"\nClassification report - {TARGET_NAMES[0]}")
report_first = classification_report(
    y_true[:, 0],
    y_pred[:, 0],
    labels=[0, 1],
    target_names=[f"{TARGET_NAMES[0]}_0", f"{TARGET_NAMES[0]}_1"],
    zero_division=0
)
print(report_first)

print(f"\nClassification report - {TARGET_NAMES[1]}")
report_second = classification_report(
    y_true[:, 1],
    y_pred[:, 1],
    labels=[0, 1],
    target_names=[f"{TARGET_NAMES[1]}_0", f"{TARGET_NAMES[1]}_1"],
    zero_division=0
)
print(report_second)


# ============================================================
# 13. Save outputs
# ============================================================

np.save(os.path.join(OUTPUT_DIR, "y_true.npy"), y_true)
np.save(os.path.join(OUTPUT_DIR, "y_probs.npy"), y_probs)
np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"), y_pred)
np.save(os.path.join(OUTPUT_DIR, f"confusion_matrix_{TARGET_NAMES[0]}.npy"), cm_first)
np.save(os.path.join(OUTPUT_DIR, f"confusion_matrix_{TARGET_NAMES[1]}.npy"), cm_second)

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write(f"Weighted GCN on {TASK_TITLE}\n")
    f.write("=" * 80 + "\n\n")

    f.write("Dataset split:\n")
    f.write(f"SPLIT_DIR: {SPLIT_DIR}\n")
    f.write("Subject-level split, no subject overlap between train/validation/test\n\n")

    f.write("Targets:\n")
    f.write(f"Output 1: {TARGET_NAMES[0]}\n")
    f.write(f"Output 2: {TARGET_NAMES[1]}\n\n")

    f.write("Model:\n")
    f.write("Weighted GCN\n")
    f.write("GCNConv(50 -> 64)\n")
    f.write("GCNConv(64 -> 64)\n")
    f.write("Global mean pooling\n")
    f.write("Linear(64 -> 64)\n")
    f.write("Linear(64 -> 2)\n\n")

    f.write("Training setup:\n")
    f.write("Batch size: 64\n")
    f.write("Max epochs: 100\n")
    f.write("Optimizer: Adam\n")
    f.write("Learning rate: 0.001\n")
    f.write("Weight decay: 1e-4\n")
    f.write("Dropout: 0.4\n")
    f.write("Loss: BCEWithLogitsLoss with pos_weight\n")
    f.write("Threshold fixed at 0.5\n")
    f.write("Edge weight: absolute Pearson correlation |edge_attr|\n\n")

    f.write(f"pos_weight {TARGET_NAMES[0]}: {pos_weight_first}\n")
    f.write(f"pos_weight {TARGET_NAMES[1]}: {pos_weight_second}\n\n")

    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Best validation mean task F1: {best_val_macro_f1}\n\n")

    f.write(f"Test loss: {test_loss}\n\n")

    f.write("Test metrics:\n")
    for k, v in test_results.items():
        f.write(f"{k}: {v}\n")

    f.write(f"\nConfusion matrix - {TARGET_NAMES[0]}\n")
    f.write(str(cm_first))
    f.write("\n\n")

    f.write(f"Confusion matrix - {TARGET_NAMES[1]}\n")
    f.write(str(cm_second))
    f.write("\n\n")

    f.write(f"Classification report - {TARGET_NAMES[0]}\n")
    f.write(report_first)
    f.write("\n\n")

    f.write(f"Classification report - {TARGET_NAMES[1]}\n")
    f.write(report_second)

print("\nSaved outputs to:")
print(OUTPUT_DIR)
print("\nSummary saved to:")
print(SUMMARY_PATH)