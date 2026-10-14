import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Linear, Dropout
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 0. CHOOSE TARGET
# ============================================================

# Επιλογές:
# "valence"
# "arousal"
# "positive_affect"
# "negative_affect"

TARGET = "negative_affect"


# ============================================================
# 1. PATHS / TARGET SETUP
# ============================================================

RESULTS_DIR = r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\gnn_results"

if TARGET in ["valence", "arousal"]:

    SPLIT_DIR = os.path.join(
        RESULTS_DIR,
        "subject_level_splits_prepost_va_original"
    )

    TARGET_INDEX = 0 if TARGET == "valence" else 1

elif TARGET in ["positive_affect", "negative_affect"]:

    SPLIT_DIR = os.path.join(
        RESULTS_DIR,
        "subject_level_splits_game_geq_pa_na"
    )

    TARGET_INDEX = 0 if TARGET == "positive_affect" else 1

else:
    raise ValueError(
        "TARGET must be valence, arousal, "
        "positive_affect or negative_affect"
    )


OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    f"single_target_gat_{TARGET}"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TRAIN_PATH = os.path.join(
    SPLIT_DIR,
    "train_graphs_subject.pt"
)

VAL_PATH = os.path.join(
    SPLIT_DIR,
    "val_graphs_subject.pt"
)

TEST_PATH = os.path.join(
    SPLIT_DIR,
    "test_graphs_subject.pt"
)


BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    f"best_gat_{TARGET}.pt"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    f"results_{TARGET}.txt"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)
print("Model: GAT")
print("Target:", TARGET)
print("Split:", SPLIT_DIR)


# ============================================================
# 4. LOAD DATA
# ============================================================

def torch_load(path):

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


train_graphs = torch_load(
    TRAIN_PATH
)

val_graphs = torch_load(
    VAL_PATH
)

test_graphs = torch_load(
    TEST_PATH
)


print("\nGraphs:")
print("Train:", len(train_graphs))
print("Val:", len(val_graphs))
print("Test:", len(test_graphs))


# ============================================================
# 5. LABEL DISTRIBUTION
# ============================================================

def get_target_labels(graphs):

    labels = []

    for graph in graphs:

        y = graph.y.view(-1)

        label = int(
            y[TARGET_INDEX].item()
        )

        labels.append(label)

    return np.array(labels)


y_train_np = get_target_labels(
    train_graphs
)

y_val_np = get_target_labels(
    val_graphs
)

y_test_np = get_target_labels(
    test_graphs
)


def print_distribution(
    name,
    y
):

    print(f"\n{name}")

    print(
        "Class 0:",
        np.sum(y == 0)
    )

    print(
        "Class 1:",
        np.sum(y == 1)
    )


print_distribution(
    "TRAIN",
    y_train_np
)

print_distribution(
    "VALIDATION",
    y_val_np
)

print_distribution(
    "TEST",
    y_test_np
)


# ============================================================
# 6. DATALOADERS
# ============================================================

BATCH_SIZE = 64


train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 7. SINGLE-TARGET GAT MODEL
# ============================================================

class SingleTargetGAT(
    torch.nn.Module
):

    def __init__(
        self,
        in_channels=50,
        hidden_channels=64,
        heads=4,
        dropout=0.4
    ):

        super().__init__()

        self.dropout_layer = Dropout(
            dropout
        )

        # ----------------------------------------------
        # First GAT layer
        #
        # 4 attention heads
        # κάθε head δίνει 16 features
        #
        # 4 × 16 = 64 συνολικά
        # ----------------------------------------------

        head_channels = (
            hidden_channels // heads
        )

        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=head_channels,
            heads=heads,
            concat=True,

            # Δεν βάζουμε dropout
            # μέσα στο attention
            dropout=0.0
        )


        # ----------------------------------------------
        # Second GAT layer
        #
        # Input = 64
        # Output = 64
        #
        # Εδώ χρησιμοποιούμε 1 head
        # ----------------------------------------------

        self.gat2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=0.0
        )


        # ----------------------------------------------
        # Classification head
        # ----------------------------------------------

        self.lin1 = Linear(
            hidden_channels,
            hidden_channels
        )

        self.lin2 = Linear(
            hidden_channels,
            1
        )


    def forward(
        self,
        data
    ):

        # Node features
        x = data.x

        # Graph connections
        edge_index = data.edge_index

        # Ποιος node ανήκει
        # σε ποιο graph
        batch = data.batch


        # ====================================================
        # IMPORTANT
        # ====================================================
        #
        # ΔΕΝ περνάμε edge_attr μέσα στο GAT.
        #
        # Η Pearson correlation έχει ήδη χρησιμοποιηθεί
        # για να αποφασίσουμε ποιες ακμές υπάρχουν:
        #
        # |r| > 0.30
        #
        # Το GAT τώρα μαθαίνει μόνο του attention weights
        # πάνω στις υπάρχουσες ακμές.
        # ====================================================


        # --------------------------
        # First GAT layer
        # [nodes, 50]
        # ->
        # [nodes, 64]
        # --------------------------

        x = self.gat1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.dropout_layer(x)


        # --------------------------
        # Second GAT layer
        # [nodes, 64]
        # ->
        # [nodes, 64]
        # --------------------------

        x = self.gat2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.dropout_layer(x)


        # --------------------------
        # Graph pooling
        #
        # 8 nodes
        # ->
        # 1 graph representation
        # --------------------------

        x = global_mean_pool(
            x,
            batch
        )


        # --------------------------
        # Fully connected layer
        # --------------------------

        x = self.lin1(x)

        x = F.relu(x)

        x = self.dropout_layer(x)


        # --------------------------
        # Final binary output
        # --------------------------

        out = self.lin2(x)


        return out.view(-1)


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = SingleTargetGAT(
    in_channels=50,
    hidden_channels=64,
    heads=4,
    dropout=0.4
).to(device)


print("\nModel:")
print(model)


# ============================================================
# 9. LOSS + OPTIMIZER
# ============================================================

# Χωρίς pos_weight
# Χωρίς sampler

criterion = (
    torch.nn.BCEWithLogitsLoss()
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    all_true = []
    all_probs = []
    all_preds = []

    total_loss = 0.0
    total_graphs = 0


    with torch.no_grad():

        for batch in loader:

            batch = batch.to(
                device
            )


            # -------------------------
            # Forward pass
            # -------------------------

            logits = model(batch)


            # -------------------------
            # Get correct target
            # -------------------------

            all_y = (
                batch.y
                .float()
                .view(-1, 2)
            )

            y = all_y[
                :,
                TARGET_INDEX
            ]


            # -------------------------
            # Loss
            # -------------------------

            loss = criterion(
                logits,
                y
            )


            # -------------------------
            # Logits -> probabilities
            # -------------------------

            probs = torch.sigmoid(
                logits
            )


            # -------------------------
            # Probability -> class
            #
            # >= 0.5 -> 1
            # < 0.5  -> 0
            # -------------------------

            preds = (
                probs >= 0.5
            ).long()


            all_true.append(
                y.cpu()
            )

            all_probs.append(
                probs.cpu()
            )

            all_preds.append(
                preds.cpu()
            )


            total_loss += (
                loss.item()
                * batch.num_graphs
            )

            total_graphs += (
                batch.num_graphs
            )


    # ========================================================
    # Combine all batches
    # ========================================================

    y_true = torch.cat(
        all_true
    ).numpy()

    y_probs = torch.cat(
        all_probs
    ).numpy()

    y_pred = torch.cat(
        all_preds
    ).numpy()


    avg_loss = (
        total_loss
        / total_graphs
    )


    # ========================================================
    # Metrics
    # ========================================================

    results = {}


    results["accuracy"] = (
        accuracy_score(
            y_true,
            y_pred
        )
    )


    results["precision"] = (
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    results["recall"] = (
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    results["f1"] = (
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    )


    results["macro_f1"] = (
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )


    results["pred_0"] = int(
        np.sum(
            y_pred == 0
        )
    )


    results["pred_1"] = int(
        np.sum(
            y_pred == 1
        )
    )


    # ========================================================
    # Probability diagnostics
    # ========================================================

    results["prob_mean"] = float(
        np.mean(y_probs)
    )

    results["prob_min"] = float(
        np.min(y_probs)
    )

    results["prob_max"] = float(
        np.max(y_probs)
    )


    return (
        avg_loss,
        results,
        y_true,
        y_probs,
        y_pred
    )


# ============================================================
# 11. TRAINING SETTINGS
# ============================================================

EPOCHS = 100

PATIENCE = 20

# Δεν επιτρέπουμε early stopping
# πριν από το epoch 30

MIN_EPOCHS = 30


best_val_macro_f1 = -1.0

best_epoch = 0

epochs_without_improvement = 0


print(
    "\nStarting GAT training...\n"
)


# ============================================================
# 12. TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # ========================================================
    # TRAIN MODE
    # ========================================================

    model.train()

    total_loss = 0.0
    total_graphs = 0


    # ========================================================
    # Go through training batches
    # ========================================================

    for batch in train_loader:

        batch = batch.to(
            device
        )


        # --------------------------------------------
        # Clear previous gradients
        # --------------------------------------------

        optimizer.zero_grad()


        # --------------------------------------------
        # Forward pass
        # --------------------------------------------

        logits = model(batch)


        # --------------------------------------------
        # Get target labels
        # --------------------------------------------

        all_y = (
            batch.y
            .float()
            .view(-1, 2)
        )


        y = all_y[
            :,
            TARGET_INDEX
        ]


        # --------------------------------------------
        # Calculate loss
        # --------------------------------------------

        loss = criterion(
            logits,
            y
        )


        # --------------------------------------------
        # Backpropagation
        # --------------------------------------------

        loss.backward()


        # --------------------------------------------
        # Update model parameters
        # --------------------------------------------

        optimizer.step()


        total_loss += (
            loss.item()
            * batch.num_graphs
        )

        total_graphs += (
            batch.num_graphs
        )


    # ========================================================
    # TRAIN LOSS
    # ========================================================

    train_loss = (
        total_loss
        / total_graphs
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_loss,
        val_results,
        _,
        _,
        _
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device
    )


    val_macro_f1 = (
        val_results[
            "macro_f1"
        ]
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Macro-F1: {val_macro_f1:.4f} | "
        f"Pred 0/1: "
        f"{val_results['pred_0']}/"
        f"{val_results['pred_1']} | "
        f"Prob mean/min/max: "
        f"{val_results['prob_mean']:.3f}/"
        f"{val_results['prob_min']:.3f}/"
        f"{val_results['prob_max']:.3f}"
    )


    # ========================================================
    # CHECK IF MODEL IMPROVED
    # ========================================================

    if (
        val_macro_f1
        >
        best_val_macro_f1
    ):

        best_val_macro_f1 = (
            val_macro_f1
        )

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epoch >= MIN_EPOCHS
        and
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# 13. BEST VALIDATION RESULT
# ============================================================

print(
    "\nBest epoch:",
    best_epoch
)

print(
    "Best validation Macro-F1:",
    round(
        best_val_macro_f1,
        4
    )
)


# ============================================================
# 14. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)


# ============================================================
# 15. FINAL TEST
# ============================================================

(
    test_loss,
    test_results,
    y_true,
    y_probs,
    y_pred
) = evaluate(
    model,
    test_loader,
    criterion,
    device
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


# ============================================================
# 17. PRINT FINAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=" * 80
)


print(
    "Model: GAT"
)

print(
    "Target:",
    TARGET
)


print(
    f"Accuracy: "
    f"{test_results['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{test_results['precision']:.4f}"
)

print(
    f"Recall: "
    f"{test_results['recall']:.4f}"
)

print(
    f"Positive-class F1: "
    f"{test_results['f1']:.4f}"
)

print(
    f"Macro-F1: "
    f"{test_results['macro_f1']:.4f}"
)

print(
    "Predictions 0:",
    test_results["pred_0"]
)

print(
    "Predictions 1:",
    test_results["pred_1"]
)


print(
    "\nProbability statistics:"
)

print(
    "Mean:",
    round(
        test_results["prob_mean"],
        4
    )
)

print(
    "Min:",
    round(
        test_results["prob_min"],
        4
    )
)

print(
    "Max:",
    round(
        test_results["prob_max"],
        4
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    "          Pred 0   Pred 1"
)

print(
    f"True 0    "
    f"{cm[0,0]:6d}   "
    f"{cm[0,1]:6d}"
)

print(
    f"True 1    "
    f"{cm[1,0]:6d}   "
    f"{cm[1,1]:6d}"
)


# ============================================================
# 18. SAVE ARRAYS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_true.npy"
    ),
    y_true
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_probs.npy"
    ),
    y_probs
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_pred.npy"
    ),
    y_pred
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.npy"
    ),
    cm
)


# ============================================================
# 19. SAVE TEXT RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Single-target GAT\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )


    f.write(
        f"Target: {TARGET}\n\n"
    )


    f.write(
        "Architecture:\n"
    )

    f.write(
        "GATConv(50 -> 64, 4 heads)\n"
    )

    f.write(
        "GATConv(64 -> 64, 1 head)\n"
    )

    f.write(
        "Global Mean Pooling\n"
    )

    f.write(
        "Linear(64 -> 64)\n"
    )

    f.write(
        "Linear(64 -> 1)\n\n"
    )


    f.write(
        "Training:\n"
    )

    f.write(
        "Batch size: 64\n"
    )

    f.write(
        "Learning rate: 0.001\n"
    )

    f.write(
        "Weight decay: 1e-4\n"
    )

    f.write(
        "Dropout: 0.4\n"
    )

    f.write(
        "Loss: BCEWithLogitsLoss\n"
    )

    f.write(
        "Sampler: No\n"
    )

    f.write(
        "pos_weight: No\n"
    )

    f.write(
        "Graph threshold: 0.30\n"
    )

    f.write(
        "Pearson used for topology only\n\n"
    )


    f.write(
        f"Best epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        "Best validation Macro-F1: "
        f"{best_val_macro_f1:.4f}\n\n"
    )


    f.write(
        "FINAL TEST RESULTS\n"
    )

    f.write(
        "=" * 80
        + "\n"
    )


    f.write(
        f"Accuracy: "
        f"{test_results['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{test_results['precision']:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{test_results['recall']:.4f}\n"
    )

    f.write(
        f"Positive-class F1: "
        f"{test_results['f1']:.4f}\n"
    )

    f.write(
        f"Macro-F1: "
        f"{test_results['macro_f1']:.4f}\n"
    )

    f.write(
        f"Predictions 0: "
        f"{test_results['pred_0']}\n"
    )

    f.write(
        f"Predictions 1: "
        f"{test_results['pred_1']}\n"
    )


    f.write(
        "\nProbability statistics:\n"
    )

    f.write(
        f"Mean: "
        f"{test_results['prob_mean']:.4f}\n"
    )

    f.write(
        f"Min: "
        f"{test_results['prob_min']:.4f}\n"
    )

    f.write(
        f"Max: "
        f"{test_results['prob_max']:.4f}\n"
    )


    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


print(
    "\nResults saved to:"
)

print(
    RESULTS_PATH
)