import os
import json
import time
import random
from collections import Counter, defaultdict

import numpy as np
import torch


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL"

GRAPH_DATASET_PATH = os.path.join(
    BASE_DIR,
    "graph_dataset_hr",
    "graph_dataset_hr.pt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "gnn_results",
    "subject_level_splits"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_OUT = os.path.join(OUTPUT_DIR, "train_graphs_subject.pt")
VAL_OUT = os.path.join(OUTPUT_DIR, "val_graphs_subject.pt")
TEST_OUT = os.path.join(OUTPUT_DIR, "test_graphs_subject.pt")

REPORT_OUT = os.path.join(OUTPUT_DIR, "subject_split_report.txt")
SUBJECTS_OUT = os.path.join(OUTPUT_DIR, "subject_split_subjects.json")


# ============================================================
# 2. Settings
# ============================================================

SEED = 42

# Για αρχή 1000 είναι αρκετό.
# Αν τελειώσει γρήγορα και θες πιο καλό split, μπορείς μετά να βάλεις 5000.
N_TRIALS = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. Helper functions
# ============================================================

def get_subject_id(graph):
    """
    Στο δικό σου dataset το subject ID είναι αποθηκευμένο ως graph.subject_id.
    Παράδειγμα: 'SUB103'
    """
    return str(graph.subject_id)


def get_label_tuple(graph):
    """
    Παίρνει το y κάθε graph και το γυρίζει ως tuple.
    Παράδειγμα:
    tensor([0., 1.]) -> (0, 1)
    """
    y = graph.y.view(-1).cpu().numpy().astype(int)
    return tuple(y.tolist())


def empty_stats():
    return {
        "n_graphs": 0,
        "valence_0": 0,
        "valence_1": 0,
        "arousal_0": 0,
        "arousal_1": 0,
        "joint_00": 0,
        "joint_01": 0,
        "joint_10": 0,
        "joint_11": 0,
    }


def add_stats(a, b):
    """
    Προσθέτει δύο dictionaries με counts.
    """
    out = empty_stats()
    for key in out.keys():
        out[key] = a.get(key, 0) + b.get(key, 0)
    return out


def count_graph_labels(graphs):
    """
    Μετράει valence, arousal και joint labels για μία λίστα από graphs.
    """
    stats = empty_stats()
    stats["n_graphs"] = len(graphs)

    for g in graphs:
        label = get_label_tuple(g)

        valence = label[0]
        arousal = label[1]

        if valence == 0:
            stats["valence_0"] += 1
        else:
            stats["valence_1"] += 1

        if arousal == 0:
            stats["arousal_0"] += 1
        else:
            stats["arousal_1"] += 1

        if label == (0, 0):
            stats["joint_00"] += 1
        elif label == (0, 1):
            stats["joint_01"] += 1
        elif label == (1, 0):
            stats["joint_10"] += 1
        elif label == (1, 1):
            stats["joint_11"] += 1

    return stats


def stats_for_subject_set(subject_set, subject_stats):
    """
    Υπολογίζει γρήγορα τα συνολικά stats ενός split
    προσθέτοντας τα ήδη υπολογισμένα stats των subjects.
    """
    total = empty_stats()

    for subject_id in subject_set:
        total = add_stats(total, subject_stats[subject_id])

    return total


def label_rates_from_stats(stats):
    """
    Μετατρέπει counts σε rates για να συγκρίνουμε distributions.
    """
    n = max(stats["n_graphs"], 1)

    return {
        "valence_1_rate": stats["valence_1"] / n,
        "arousal_1_rate": stats["arousal_1"] / n,
        "joint": np.array([
            stats["joint_00"] / n,
            stats["joint_01"] / n,
            stats["joint_10"] / n,
            stats["joint_11"] / n,
        ])
    }


def split_score(train_stats, val_stats, test_stats, full_rates, total_graphs):
    """
    Όσο μικρότερο score, τόσο καλύτερο split.

    Θέλουμε:
    - τα graph ratios να είναι κοντά σε 56% train, 24% val, 20% test
    - οι label distributions να μοιάζουν με του full dataset
    - να μη λείπουν βασικές binary classes από κάποιο split
    """

    target_ratios = {
        "train": 0.56,
        "val": 0.24,
        "test": 0.20
    }

    split_stats = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats
    }

    score = 0.0

    for split_name, stats in split_stats.items():
        rates = label_rates_from_stats(stats)

        graph_ratio = stats["n_graphs"] / total_graphs

        # Ποινή αν το ποσοστό των graphs απέχει από τον στόχο
        score += 2.0 * abs(graph_ratio - target_ratios[split_name])

        # Ποινή αν το valence/arousal distribution απέχει από το full dataset
        score += 2.0 * abs(rates["valence_1_rate"] - full_rates["valence_1_rate"])
        score += 2.0 * abs(rates["arousal_1_rate"] - full_rates["arousal_1_rate"])

        # Ποινή για διαφορά στα joint labels [0,0], [0,1], [1,0], [1,1]
        score += 5.0 * np.sum(np.abs(rates["joint"] - full_rates["joint"]))

        # Μεγάλη ποινή αν λείπει τελείως κάποια binary class
        if stats["valence_0"] == 0 or stats["valence_1"] == 0:
            score += 100.0

        if stats["arousal_0"] == 0 or stats["arousal_1"] == 0:
            score += 100.0

    return score


def build_graphs_for_subjects(subject_set, graphs_by_subject):
    """
    Φτιάχνει τη λίστα των graphs για ένα split.
    """
    selected_graphs = []

    for subject_id in subject_set:
        selected_graphs.extend(graphs_by_subject[subject_id])

    return selected_graphs


def print_stats(name, subjects, stats):
    """
    Τυπώνει τα τελικά statistics ενός split.
    """
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print("Subjects:", len(subjects))
    print("Graphs:", stats["n_graphs"])

    print("\nValence:")
    print("Valence 0:", stats["valence_0"])
    print("Valence 1:", stats["valence_1"])

    print("\nArousal:")
    print("Arousal 0:", stats["arousal_0"])
    print("Arousal 1:", stats["arousal_1"])

    print("\nJoint labels:")
    print("[0,0]:", stats["joint_00"])
    print("[0,1]:", stats["joint_01"])
    print("[1,0]:", stats["joint_10"])
    print("[1,1]:", stats["joint_11"])


# ============================================================
# 4. Load full graph dataset
# ============================================================

print("Loading full HR graph dataset...")
print("Path:", GRAPH_DATASET_PATH)

graphs = torch.load(GRAPH_DATASET_PATH, weights_only=False)

print("Total graphs:", len(graphs))

subjects = [get_subject_id(g) for g in graphs]
unique_subjects = sorted(list(set(subjects)))

print("Total unique subjects:", len(unique_subjects))


# ============================================================
# 5. Group graphs by subject
# ============================================================

graphs_by_subject = defaultdict(list)

for graph in graphs:
    subject_id = get_subject_id(graph)
    graphs_by_subject[subject_id].append(graph)


# ============================================================
# 6. Precompute stats per subject
# ============================================================

print("\nComputing per-subject label statistics...")

subject_stats = {}

for subject_id, subject_graphs in graphs_by_subject.items():
    subject_stats[subject_id] = count_graph_labels(subject_graphs)


# ============================================================
# 7. Full dataset label distribution
# ============================================================

full_stats = count_graph_labels(graphs)
full_rates = label_rates_from_stats(full_stats)

print("\nFull dataset label distribution:")
print(full_stats)


# ============================================================
# 8. Decide subject counts
# ============================================================

n_subjects = len(unique_subjects)

n_test = int(round(0.20 * n_subjects))
n_train_full = n_subjects - n_test
n_val = int(round(0.30 * n_train_full))
n_train = n_train_full - n_val

print("\nTarget subject counts:")
print("Train subjects:", n_train)
print("Validation subjects:", n_val)
print("Test subjects:", n_test)


# ============================================================
# 9. Search for a good subject-level split
# ============================================================

print(f"\nSearching {N_TRIALS} possible subject-level splits...")

rng = np.random.default_rng(SEED)

best_score = float("inf")
best_split = None

subjects_array = np.array(unique_subjects)

start_time = time.time()

for trial in range(N_TRIALS):
    shuffled = subjects_array.copy()
    rng.shuffle(shuffled)

    train_subjects = set(shuffled[:n_train])
    val_subjects = set(shuffled[n_train:n_train + n_val])
    test_subjects = set(shuffled[n_train + n_val:])

    train_stats = stats_for_subject_set(train_subjects, subject_stats)
    val_stats = stats_for_subject_set(val_subjects, subject_stats)
    test_stats = stats_for_subject_set(test_subjects, subject_stats)

    score = split_score(
        train_stats,
        val_stats,
        test_stats,
        full_rates,
        total_graphs=len(graphs)
    )

    if score < best_score:
        best_score = score
        best_split = {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "test_stats": test_stats,
        }

    if (trial + 1) % 100 == 0:
        elapsed = time.time() - start_time
        time_per_trial = elapsed / (trial + 1)
        remaining_trials = N_TRIALS - (trial + 1)
        eta_seconds = time_per_trial * remaining_trials

        print(
            f"Trial {trial + 1}/{N_TRIALS} | "
            f"Best score: {best_score:.4f} | "
            f"Elapsed: {elapsed/60:.2f} min | "
            f"ETA: {eta_seconds/60:.2f} min",
            flush=True
        )


print("\nBest split score:", best_score)


# ============================================================
# 10. Extract best split
# ============================================================

train_subjects = best_split["train_subjects"]
val_subjects = best_split["val_subjects"]
test_subjects = best_split["test_subjects"]

train_stats = best_split["train_stats"]
val_stats = best_split["val_stats"]
test_stats = best_split["test_stats"]


# ============================================================
# 11. Overlap checks
# ============================================================

print("\nSubject overlap checks:")
print("Train-Val overlap:", len(train_subjects.intersection(val_subjects)))
print("Train-Test overlap:", len(train_subjects.intersection(test_subjects)))
print("Val-Test overlap:", len(val_subjects.intersection(test_subjects)))

assert len(train_subjects.intersection(val_subjects)) == 0
assert len(train_subjects.intersection(test_subjects)) == 0
assert len(val_subjects.intersection(test_subjects)) == 0


# ============================================================
# 12. Build final graph splits
# ============================================================

print("\nBuilding final graph lists...")

train_graphs = build_graphs_for_subjects(train_subjects, graphs_by_subject)
val_graphs = build_graphs_for_subjects(val_subjects, graphs_by_subject)
test_graphs = build_graphs_for_subjects(test_subjects, graphs_by_subject)


# ============================================================
# 13. Print final split statistics
# ============================================================

print_stats("TRAIN SPLIT", train_subjects, train_stats)
print_stats("VALIDATION SPLIT", val_subjects, val_stats)
print_stats("TEST SPLIT", test_subjects, test_stats)


# ============================================================
# 14. Save splits
# ============================================================

print("\nSaving subject-level splits...")

torch.save(train_graphs, TRAIN_OUT)
torch.save(val_graphs, VAL_OUT)
torch.save(test_graphs, TEST_OUT)

subjects_dict = {
    "train_subjects": sorted(list(train_subjects)),
    "val_subjects": sorted(list(val_subjects)),
    "test_subjects": sorted(list(test_subjects)),
}

with open(SUBJECTS_OUT, "w", encoding="utf-8") as f:
    json.dump(subjects_dict, f, indent=4)


# ============================================================
# 15. Save report
# ============================================================

with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write("Subject-level split report\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Graph dataset path: {GRAPH_DATASET_PATH}\n")
    f.write(f"Total graphs: {len(graphs)}\n")
    f.write(f"Total subjects: {len(unique_subjects)}\n")
    f.write(f"Best split score: {best_score}\n\n")

    f.write("Subject counts:\n")
    f.write(f"Train subjects: {len(train_subjects)}\n")
    f.write(f"Validation subjects: {len(val_subjects)}\n")
    f.write(f"Test subjects: {len(test_subjects)}\n\n")

    f.write("Graph counts:\n")
    f.write(f"Train graphs: {len(train_graphs)}\n")
    f.write(f"Validation graphs: {len(val_graphs)}\n")
    f.write(f"Test graphs: {len(test_graphs)}\n\n")

    f.write("Overlap checks:\n")
    f.write(f"Train-Val overlap: {len(train_subjects.intersection(val_subjects))}\n")
    f.write(f"Train-Test overlap: {len(train_subjects.intersection(test_subjects))}\n")
    f.write(f"Val-Test overlap: {len(val_subjects.intersection(test_subjects))}\n\n")

    f.write("Full dataset stats:\n")
    f.write(str(full_stats) + "\n\n")

    f.write("Train stats:\n")
    f.write(str(train_stats) + "\n\n")

    f.write("Validation stats:\n")
    f.write(str(val_stats) + "\n\n")

    f.write("Test stats:\n")
    f.write(str(test_stats) + "\n\n")

    f.write("Train subjects:\n")
    f.write(str(sorted(list(train_subjects))) + "\n\n")

    f.write("Validation subjects:\n")
    f.write(str(sorted(list(val_subjects))) + "\n\n")

    f.write("Test subjects:\n")
    f.write(str(sorted(list(test_subjects))) + "\n")


print("\nDone.")
print("\nSaved subject-level splits:")
print("Train:", TRAIN_OUT)
print("Validation:", VAL_OUT)
print("Test:", TEST_OUT)

print("\nSaved subject lists:")
print(SUBJECTS_OUT)

print("\nSaved report:")
print(REPORT_OUT)