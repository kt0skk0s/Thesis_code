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
    confusion_matrix
)


# ============================================================
# 1. ΕΠΙΛΟΓΗ TARGET
# ============================================================
#
# Αλλάζεις ΜΟΝΟ αυτή τη γραμμή.
#
# Επιλογές:
# "valence"
# "arousal"
# "positive_affect"
# "negative_affect"
#

TARGET = "negative_affect"


# ============================================================
# 2. ΒΑΣΙΚΑ PATHS
# ============================================================

RESULTS_DIR = r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\gnn_results"


# ============================================================
# 3. ΡΥΘΜΙΣΕΙΣ ΑΝΑΛΟΓΑ ΜΕ ΤΟ TARGET
# ============================================================
#
# Στα PRE/POST:
# y = [valence, arousal]
#
# Άρα:
# valence -> θέση 0
# arousal -> θέση 1
#
# Στο GAME:
# y = [positive_affect, negative_affect]
#
# Άρα:
# positive_affect -> θέση 0
# negative_affect -> θέση 1
#

if TARGET == "valence":

    SPLIT_DIR = os.path.join(
        RESULTS_DIR,
        "subject_level_splits_prepost_va_original"
    )

    TARGET_INDEX = 0


elif TARGET == "arousal":

    SPLIT_DIR = os.path.join(
        RESULTS_DIR,
        "subject_level_splits_prepost_va_original"
    )

    TARGET_INDEX = 1


elif TARGET == "positive_affect":

    SPLIT_DIR = os.path.join(
        RESULTS_DIR,
        "subject_level_splits_game_geq_pa_na"
    )

    TARGET_INDEX = 0


elif TARGET == "negative_affect":

    SPLIT_DIR = os.path.join(
        RESULTS_DIR,
        "subject_level_splits_game_geq_pa_na"
    )

    TARGET_INDEX = 1


else:
    raise ValueError("Λάθος TARGET.")


# Φάκελος αποτελεσμάτων για το συγκεκριμένο target
OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    f"single_target_gcn_{TARGET}"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    f"best_gcn_{TARGET}.pt"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    f"results_{TARGET}.txt"
)


# ============================================================
# 4. SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Target:", TARGET)
print("Split directory:", SPLIT_DIR)


# ============================================================
# 6. ΦΟΡΤΩΣΗ DATA
# ============================================================

def load_pt(path):

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


train_graphs = load_pt(TRAIN_PATH)
val_graphs = load_pt(VAL_PATH)
test_graphs = load_pt(TEST_PATH)


print("\nGraphs:")
print("Train:", len(train_graphs))
print("Validation:", len(val_graphs))
print("Test:", len(test_graphs))


# ============================================================
# 7. ΕΛΕΓΧΟΣ LABEL DISTRIBUTION
# ============================================================

def get_labels(graphs):

    labels = []

    for graph in graphs:

        # Το y κάθε graph έχει δύο labels.
        #
        # PRE/POST:
        # [valence, arousal]
        #
        # GAME:
        # [positive_affect, negative_affect]

        y = graph.y.view(-1)

        # Κρατάμε μόνο το target που θέλουμε.
        label = int(y[TARGET_INDEX].item())

        labels.append(label)

    return np.array(labels)


train_labels = get_labels(train_graphs)
val_labels = get_labels(val_graphs)
test_labels = get_labels(test_graphs)


def print_distribution(name, labels):

    n0 = np.sum(labels == 0)
    n1 = np.sum(labels == 1)

    print(f"\n{name}")
    print("Class 0:", n0)
    print("Class 1:", n1)


print_distribution("TRAIN", train_labels)
print_distribution("VALIDATION", val_labels)
print_distribution("TEST", test_labels)


# ============================================================
# 8. DATALOADERS
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
# 9. GCN MODEL
# ============================================================
#
# ΠΑΛΙΑ:
#
# GCN
# ↓
# 2 outputs
# [Valence, Arousal]
#
#
# ΤΩΡΑ:
#
# GCN
# ↓
# 1 output
# π.χ. μόνο Valence
#

class SingleTargetGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels=50,
        hidden_channels=64,
        dropout=0.4
    ):

        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.lin1 = Linear(
            hidden_channels,
            hidden_channels
        )

        # ΕΝΑ output
        self.lin2 = Linear(
            hidden_channels,
            1
        )

        self.dropout = Dropout(dropout)


    def forward(self, data):

        x = data.x

        edge_index = data.edge_index

        batch = data.batch

        # Το edge_attr περιέχει Pearson correlation.
        #
        # Όπως και πριν,
        # χρησιμοποιούμε την απόλυτη τιμή.
        edge_weight = (
            data.edge_attr
            .view(-1)
            .float()
            .abs()
        )


        # --------------------------------
        # GCN layer 1
        # --------------------------------

        x = self.conv1(
            x,
            edge_index,
            edge_weight=edge_weight
        )

        x = F.relu(x)

        x = self.dropout(x)


        # --------------------------------
        # GCN layer 2
        # --------------------------------

        x = self.conv2(
            x,
            edge_index,
            edge_weight=edge_weight
        )

        x = F.relu(x)

        x = self.dropout(x)


        # --------------------------------
        # Όλοι οι nodes -> ένα graph vector
        # --------------------------------

        x = global_mean_pool(
            x,
            batch
        )


        # --------------------------------
        # Fully connected layer
        # --------------------------------

        x = self.lin1(x)

        x = F.relu(x)

        x = self.dropout(x)


        # --------------------------------
        # Ένα τελικό output
        # --------------------------------

        x = self.lin2(x)


        # shape:
        # [batch_size, 1]
        #
        # το κάνουμε:
        # [batch_size]

        return x.view(-1)


model = SingleTargetGCN().to(device)

print("\nModel:")
print(model)


# ============================================================
# 10. LOSS + OPTIMIZER
# ============================================================
#
# Προς το παρόν:
#
# ΟΧΙ pos_weight
# ΟΧΙ sampler
#
# Θέλουμε πρώτα ένα καθαρό baseline.
#

criterion = torch.nn.BCEWithLogitsLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_true = []
    all_probs = []
    all_pred = []

    total_loss = 0
    total_graphs = 0


    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)


            # ----------------------------
            # Prediction
            # ----------------------------

            logits = model(batch)


            # ----------------------------
            # Παίρνουμε ΜΟΝΟ το target
            # που μας ενδιαφέρει
            # ----------------------------

            y_all = batch.y.float().view(-1, 2)

            y = y_all[:, TARGET_INDEX]


            # ----------------------------
            # Loss
            # ----------------------------

            loss = criterion(
                logits,
                y
            )


            # logits -> probabilities

            probs = torch.sigmoid(logits)


            # Binary prediction
            #
            # probability >= 0.5 -> 1
            # probability < 0.5 -> 0

            pred = (
                probs >= 0.5
            ).long()


            all_true.append(
                y.cpu()
            )

            all_probs.append(
                probs.cpu()
            )

            all_pred.append(
                pred.cpu()
            )


            total_loss += (
                loss.item()
                * batch.num_graphs
            )

            total_graphs += (
                batch.num_graphs
            )


    y_true = torch.cat(
        all_true
    ).numpy()

    y_probs = torch.cat(
        all_probs
    ).numpy()

    y_pred = torch.cat(
        all_pred
    ).numpy()


    average_loss = (
        total_loss
        / total_graphs
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    # F1 της class 1
    positive_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    # Αυτό είναι το βασικό μας metric.
    #
    # Υπολογίζει F1 και για class 0
    # και για class 1 και παίρνει mean.

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "positive_f1": positive_f1,
        "macro_f1": macro_f1,
        "pred_0": int(np.sum(y_pred == 0)),
        "pred_1": int(np.sum(y_pred == 1)),
        "y_true": y_true,
        "y_probs": y_probs,
        "y_pred": y_pred
    }


# ============================================================
# 12. TRAINING
# ============================================================

EPOCHS = 100

PATIENCE = 15


best_val_macro_f1 = -1

best_epoch = 0

epochs_without_improvement = 0


print("\nStarting training...")
print("=" * 80)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_train_loss = 0

    total_graphs = 0


    for batch in train_loader:

        batch = batch.to(device)


        # Μηδενίζουμε gradients
        optimizer.zero_grad()


        # Prediction
        logits = model(batch)


        # Παίρνουμε μόνο το target
        y_all = batch.y.float().view(-1, 2)

        y = y_all[:, TARGET_INDEX]


        # Loss
        loss = criterion(
            logits,
            y
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        total_train_loss += (
            loss.item()
            * batch.num_graphs
        )

        total_graphs += (
            batch.num_graphs
        )


    train_loss = (
        total_train_loss
        / total_graphs
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    val_results = evaluate(
        model,
        val_loader
    )


    val_macro_f1 = (
        val_results["macro_f1"]
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================
    #
    # Κρατάμε το model που έχει το καλύτερο
    # validation Macro-F1.
    #

    if val_macro_f1 > best_val_macro_f1:

        best_val_macro_f1 = val_macro_f1

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_results['loss']:.4f} | "
        f"Val Macro-F1: {val_results['macro_f1']:.4f} | "
        f"Pred 0/1: "
        f"{val_results['pred_0']}/"
        f"{val_results['pred_1']}"
    )


    # ========================================================
    # STOP
    # ========================================================

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping.")

        break


# ============================================================
# 13. LOAD BEST MODEL
# ============================================================

print("\nBest epoch:", best_epoch)

print(
    "Best validation Macro-F1:",
    round(best_val_macro_f1, 4)
)


model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)


# ============================================================
# 14. FINAL TEST
# ============================================================

test_results = evaluate(
    model,
    test_loader
)


print("\n" + "=" * 80)

print("FINAL TEST RESULTS")

print("=" * 80)

print("Target:", TARGET)

print(
    "Accuracy:",
    round(test_results["accuracy"], 4)
)

print(
    "Precision:",
    round(test_results["precision"], 4)
)

print(
    "Recall:",
    round(test_results["recall"], 4)
)

print(
    "Positive-class F1:",
    round(test_results["positive_f1"], 4)
)

print(
    "Macro-F1:",
    round(test_results["macro_f1"], 4)
)

print(
    "Predictions 0:",
    test_results["pred_0"]
)

print(
    "Predictions 1:",
    test_results["pred_1"]
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_results["y_true"],
    test_results["y_pred"],
    labels=[0, 1]
)

print("\nConfusion Matrix:")

print(
    "          Pred 0   Pred 1"
)

print(
    f"True 0    "
    f"{cm[0, 0]:6d}   "
    f"{cm[0, 1]:6d}"
)

print(
    f"True 1    "
    f"{cm[1, 0]:6d}   "
    f"{cm[1, 1]:6d}"
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Single-target GCN - {TARGET}\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Seed: {SEED}\n"
    )

    f.write(
        f"Best epoch: {best_epoch}\n"
    )

    f.write(
        f"Best validation Macro-F1: "
        f"{best_val_macro_f1:.6f}\n\n"
    )

    f.write("TEST RESULTS\n")

    f.write(
        f"Accuracy: "
        f"{test_results['accuracy']:.6f}\n"
    )

    f.write(
        f"Precision: "
        f"{test_results['precision']:.6f}\n"
    )

    f.write(
        f"Recall: "
        f"{test_results['recall']:.6f}\n"
    )

    f.write(
        f"Positive F1: "
        f"{test_results['positive_f1']:.6f}\n"
    )

    f.write(
        f"Macro-F1: "
        f"{test_results['macro_f1']:.6f}\n"
    )

    f.write(
        f"Pred 0: "
        f"{test_results['pred_0']}\n"
    )

    f.write(
        f"Pred 1: "
        f"{test_results['pred_1']}\n"
    )

    f.write("\nConfusion Matrix:\n")

    f.write(str(cm))


print("\nResults saved to:")

print(SUMMARY_PATH)