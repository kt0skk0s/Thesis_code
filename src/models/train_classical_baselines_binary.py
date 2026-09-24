import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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


#TASK = "prepost"
TASK = "game"


# Αν δεν έχεις πολύ χρόνο, κράτα το False.
# Το RBF SVM μπορεί να αργήσει αρκετά.
RUN_RBF_SVM = True


# ============================================================
# 1. Paths
# ============================================================

RESULTS_DIR = r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\gnn_results"

if TASK == "prepost":
    SPLIT_DIR = os.path.join(
        RESULTS_DIR,
        "subject_level_splits_prepost_va_original"
    )
    OUTPUT_DIR = os.path.join(
        RESULTS_DIR,
        "classical_baselines_prepost_va_binary"
    )

    TASK_TITLE = "PRE/POST Valence-Arousal binary"
    TARGET_NAMES = ["valence", "arousal"]

elif TASK == "game":
    SPLIT_DIR = os.path.join(
        RESULTS_DIR,
        "subject_level_splits_game_geq_pa_na"
    )
    OUTPUT_DIR = os.path.join(
        RESULTS_DIR,
        "classical_baselines_game_geq_pa_na_binary"
    )

    TASK_TITLE = "GAME GEQ Positive-Negative Affect binary"
    TARGET_NAMES = ["positive_affect", "negative_affect"]

else:
    raise ValueError("TASK must be either 'prepost' or 'game'.")


os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "baseline_results_summary.csv")
SUMMARY_TXT_PATH = os.path.join(OUTPUT_DIR, "baseline_results_summary.txt")


# ============================================================
# 2. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. Load graphs
# ============================================================

def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_split(split_name):
    candidates = [
        os.path.join(SPLIT_DIR, f"{split_name}_graphs_subject.pt"),
        os.path.join(SPLIT_DIR, f"{split_name}_graphs.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"Loading {split_name}: {path}")
            return torch_load(path)

    raise FileNotFoundError(
        f"Could not find {split_name} split in {SPLIT_DIR}"
    )


print("=" * 90)
print("Classical ML baselines")
print("=" * 90)
print("Task:", TASK_TITLE)
print("Split dir:", SPLIT_DIR)
print("Output dir:", OUTPUT_DIR)

train_graphs = load_split("train")
val_graphs = load_split("val")
test_graphs = load_split("test")

print("\nGraphs:")
print("Train:", len(train_graphs))
print("Val:", len(val_graphs))
print("Test:", len(test_graphs))


# ============================================================
# 4. Safety checks
# ============================================================

def check_subject_overlap(train_graphs, val_graphs, test_graphs):
    train_subjects = set(str(g.subject_id) for g in train_graphs)
    val_subjects = set(str(g.subject_id) for g in val_graphs)
    test_subjects = set(str(g.subject_id) for g in test_graphs)

    print("\nSubject overlap check:")
    print("train ∩ val:", len(train_subjects & val_subjects))
    print("train ∩ test:", len(train_subjects & test_subjects))
    print("val ∩ test:", len(val_subjects & test_subjects))

    if train_subjects & val_subjects:
        raise ValueError("Subject overlap found between train and val.")
    if train_subjects & test_subjects:
        raise ValueError("Subject overlap found between train and test.")
    if val_subjects & test_subjects:
        raise ValueError("Subject overlap found between val and test.")


check_subject_overlap(train_graphs, val_graphs, test_graphs)


# ============================================================
# 5. Convert graph data to flat features
# ============================================================

def graphs_to_flat_features_and_labels(graphs):
    X = []
    Y = []

    for g in graphs:
        x = g.x.detach().cpu().float().numpy()

        # Expected shape: [8, 50]
        x_flat = x.reshape(-1)

        y = g.y.detach().cpu().view(-1).numpy()

        if len(y) != 2:
            raise ValueError(f"Expected 2 labels, got y={y}")

        y = y.astype(int)

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError(f"Binary labels must be 0/1. Got y={y}")

        X.append(x_flat)
        Y.append(y)

    X = np.vstack(X).astype(np.float32)
    Y = np.vstack(Y).astype(int)

    # Σε περίπτωση που υπάρχει κάποιο NaN/inf στα features.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, Y


X_train, y_train = graphs_to_flat_features_and_labels(train_graphs)
X_val, y_val = graphs_to_flat_features_and_labels(val_graphs)
X_test, y_test = graphs_to_flat_features_and_labels(test_graphs)

print("\nFeature shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nLabel shapes:")
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)


def print_label_distribution(split_name, y):
    print("\n" + "=" * 80)
    print(f"{split_name} label distribution")
    print("=" * 80)

    for i, target_name in enumerate(TARGET_NAMES):
        counts = Counter(y[:, i].tolist())
        print(f"{target_name}: {dict(sorted(counts.items()))}")

    joint = Counter((y[:, 0] * 2 + y[:, 1]).tolist())
    print("Joint labels:")
    print("[0,0]->0, [0,1]->1, [1,0]->2, [1,1]->3")
    print(dict(sorted(joint.items())))


print_label_distribution("TRAIN", y_train)
print_label_distribution("VAL", y_val)
print_label_distribution("TEST", y_test)


# ============================================================
# 6. Define baseline models
# ============================================================

models = {}

models["Logistic Regression"] = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=SEED
    )
)

models["Linear SVM"] = make_pipeline(
    StandardScaler(),
    LinearSVC(
        class_weight="balanced",
        max_iter=20000,
        random_state=SEED
    )
)

if RUN_RBF_SVM:
    models["RBF SVM"] = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=SEED
        )
    )

models["Random Forest"] = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

models["MLP"] = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=SEED
    )
)


# ============================================================
# 7. Evaluation
# ============================================================

def evaluate_binary_two_outputs(y_true, y_pred):
    results = {}

    for i, target_name in enumerate(TARGET_NAMES):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]

        results[f"{target_name}_accuracy"] = accuracy_score(true_i, pred_i)

        results[f"{target_name}_precision"] = precision_score(
            true_i,
            pred_i,
            pos_label=1,
            zero_division=0
        )

        results[f"{target_name}_recall"] = recall_score(
            true_i,
            pred_i,
            pos_label=1,
            zero_division=0
        )

        results[f"{target_name}_f1"] = f1_score(
            true_i,
            pred_i,
            pos_label=1,
            zero_division=0
        )

        results[f"{target_name}_macro_f1"] = f1_score(
            true_i,
            pred_i,
            average="macro",
            zero_division=0
        )

        results[f"true_{target_name}_0"] = int(np.sum(true_i == 0))
        results[f"true_{target_name}_1"] = int(np.sum(true_i == 1))
        results[f"pred_{target_name}_0"] = int(np.sum(pred_i == 0))
        results[f"pred_{target_name}_1"] = int(np.sum(pred_i == 1))

    results["mean_task_f1"] = (
        results[f"{TARGET_NAMES[0]}_f1"] +
        results[f"{TARGET_NAMES[1]}_f1"]
    ) / 2.0

    results["mean_binary_macro_f1"] = (
        results[f"{TARGET_NAMES[0]}_macro_f1"] +
        results[f"{TARGET_NAMES[1]}_macro_f1"]
    ) / 2.0

    results["exact_match_accuracy"] = np.all(y_true == y_pred, axis=1).mean()

    return results


def train_two_independent_classifiers(model_name, base_model):
    print("\n" + "=" * 90)
    print(f"Training: {model_name}")
    print("=" * 90)

    val_pred = np.zeros_like(y_val)
    test_pred = np.zeros_like(y_test)

    fitted_models = {}

    for output_index, target_name in enumerate(TARGET_NAMES):
        print(f"\nTraining {model_name} for target: {target_name}")

        clf = clone(base_model)

        clf.fit(X_train, y_train[:, output_index])

        val_pred[:, output_index] = clf.predict(X_val)
        test_pred[:, output_index] = clf.predict(X_test)

        fitted_models[target_name] = clf

    val_results = evaluate_binary_two_outputs(y_val, val_pred)
    test_results = evaluate_binary_two_outputs(y_test, test_pred)

    print("\nValidation summary:")
    print(
        f"Mean task F1: {val_results['mean_task_f1']:.4f} | "
        f"Mean binary macro F1: {val_results['mean_binary_macro_f1']:.4f} | "
        f"Exact match: {val_results['exact_match_accuracy']:.4f}"
    )

    print("\nTest summary:")
    print(
        f"Mean task F1: {test_results['mean_task_f1']:.4f} | "
        f"Mean binary macro F1: {test_results['mean_binary_macro_f1']:.4f} | "
        f"Exact match: {test_results['exact_match_accuracy']:.4f}"
    )

    print("\nDetailed TEST metrics:")
    for k, v in test_results.items():
        print(f"{k}: {v}")

    # Confusion matrices and reports
    detailed_text = []
    detailed_text.append("=" * 90)
    detailed_text.append(f"MODEL: {model_name}")
    detailed_text.append("=" * 90)
    detailed_text.append("\nVALIDATION RESULTS:")
    for k, v in val_results.items():
        detailed_text.append(f"{k}: {v}")

    detailed_text.append("\nTEST RESULTS:")
    for k, v in test_results.items():
        detailed_text.append(f"{k}: {v}")

    for output_index, target_name in enumerate(TARGET_NAMES):
        cm = confusion_matrix(
            y_test[:, output_index],
            test_pred[:, output_index],
            labels=[0, 1]
        )

        report = classification_report(
            y_test[:, output_index],
            test_pred[:, output_index],
            labels=[0, 1],
            target_names=[
                f"{target_name}_0",
                f"{target_name}_1"
            ],
            zero_division=0
        )

        print(f"\nConfusion matrix - {target_name}")
        print("Rows = true labels, columns = predicted labels")
        print(cm)

        print(f"\nClassification report - {target_name}")
        print(report)

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"confusion_matrix_{model_name.replace(' ', '_')}_{target_name}.npy"
            ),
            cm
        )

        detailed_text.append(f"\nConfusion matrix - {target_name}")
        detailed_text.append(str(cm))
        detailed_text.append(f"\nClassification report - {target_name}")
        detailed_text.append(report)

    # Save predictions
    safe_name = model_name.replace(" ", "_").replace("/", "_")

    np.save(
        os.path.join(OUTPUT_DIR, f"y_pred_{safe_name}.npy"),
        test_pred
    )

    # Return row for summary table
    row = {
        "model": model_name,
        "task": TASK,
        "val_mean_task_f1": val_results["mean_task_f1"],
        "val_mean_binary_macro_f1": val_results["mean_binary_macro_f1"],
        "val_exact_match_accuracy": val_results["exact_match_accuracy"],
        "test_mean_task_f1": test_results["mean_task_f1"],
        "test_mean_binary_macro_f1": test_results["mean_binary_macro_f1"],
        "test_exact_match_accuracy": test_results["exact_match_accuracy"],
    }

    for target_name in TARGET_NAMES:
        row[f"test_{target_name}_f1"] = test_results[f"{target_name}_f1"]
        row[f"test_{target_name}_macro_f1"] = test_results[f"{target_name}_macro_f1"]
        row[f"test_{target_name}_accuracy"] = test_results[f"{target_name}_accuracy"]
        row[f"pred_{target_name}_0"] = test_results[f"pred_{target_name}_0"]
        row[f"pred_{target_name}_1"] = test_results[f"pred_{target_name}_1"]

    return row, "\n".join(detailed_text)


# ============================================================
# 8. Run all baselines
# ============================================================

all_rows = []
all_details = []

for model_name, base_model in models.items():
    row, details = train_two_independent_classifiers(model_name, base_model)
    all_rows.append(row)
    all_details.append(details)


# ============================================================
# 9. Save and print comparison table
# ============================================================

summary_df = pd.DataFrame(all_rows)

summary_df = summary_df.sort_values(
    by="test_mean_binary_macro_f1",
    ascending=False
)

summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Classical ML baselines for {TASK_TITLE}\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"SPLIT_DIR: {SPLIT_DIR}\n")
    f.write("Features: graph.x flattened from [8, 50] to [400]\n")
    f.write("Graph structure was NOT used in these baselines.\n")
    f.write("Two independent classifiers were trained, one for each output.\n\n")

    f.write("SUMMARY TABLE\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("\n\nDETAILED RESULTS\n\n")
    f.write("\n\n".join(all_details))

print("\n" + "=" * 90)
print("FINAL TEST COMPARISON")
print("=" * 90)
print(summary_df.to_string(index=False))

print("\nSaved summary CSV to:")
print(SUMMARY_CSV_PATH)

print("\nSaved summary TXT to:")
print(SUMMARY_TXT_PATH)

print("\nDone.")