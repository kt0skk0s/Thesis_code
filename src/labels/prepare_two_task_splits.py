import os
import copy
import json
from collections import Counter

import pandas as pd
import torch


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL"
RESULTS_DIR = os.path.join(PROCESSED_DIR, "gnn_results")

# ΠΡΟΣΟΧΗ:
# Εδώ θέλουμε το ORIGINAL split, όχι το prepost_constant.
ORIGINAL_SPLIT_DIR = os.path.join(RESULTS_DIR, "subject_level_splits")

# Εδώ είναι τα GEQ files σου.
GEQ_DIR = r"C:\Users\User\OneDrive\Desktop\Thesis\BERAFFE2_DATASET\GEQ_LEVELS"

# Output folders
PREPOST_OUT_DIR = os.path.join(
    RESULTS_DIR,
    "subject_level_splits_prepost_va_original"
)

GAME_OUT_DIR = os.path.join(
    RESULTS_DIR,
    "subject_level_splits_game_geq_pa_na"
)


# ============================================================
# GEQ SETTINGS
# ============================================================

# Mapping game condition -> GEQ file
RAW_GEQ_FILES = {
    "EASY": os.path.join(GEQ_DIR, "BIRAFFE2-metadata-RAW-GEQ-Level01.csv"),
    "HARD": os.path.join(GEQ_DIR, "BIRAFFE2-metadata-RAW-GEQ-Level02.csv"),
    "OPTIMAL": os.path.join(GEQ_DIR, "BIRAFFE2-metadata-RAW-GEQ-Level03.csv"),
}

# GEQ Core Module items
# Positive Affect = mean(items 1, 4, 6, 14, 20)
# Negative Affect = mean(items 7, 8, 9, 16)
PA_ITEMS = ["1", "4", "6", "14", "20"]
NA_ITEMS = ["7", "8", "9", "16"]

# Μετά τη μετατροπή 1-5 -> 0-4, το threshold > 2 σημαίνει high.
GEQ_THRESHOLD = 2.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_csv_auto(path):
    return pd.read_csv(path, sep=None, engine="python")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def get_condition(graph):
    if hasattr(graph, "condition"):
        return str(graph.condition).upper()
    if hasattr(graph, "part"):
        return str(graph.part).upper()
    raise ValueError("Graph has no condition or part attribute.")


def subject_to_int(subject_id):
    s = str(subject_id).upper().replace("SUB", "").strip()
    return int(s)


def y_to_joint(y):
    y = y.view(-1).long()
    return int(y[0].item()) * 2 + int(y[1].item())


def joint_counts(graphs):
    counts = Counter()
    for g in graphs:
        counts[y_to_joint(g.y)] += 1
    return dict(sorted(counts.items()))


def label_counts_two_outputs(graphs):
    first = Counter()
    second = Counter()

    for g in graphs:
        y = g.y.view(-1).long()
        first[int(y[0].item())] += 1
        second[int(y[1].item())] += 1

    return dict(first), dict(second)


def load_split(split_name):
    candidates = [
        os.path.join(ORIGINAL_SPLIT_DIR, f"{split_name}_graphs_subject.pt"),
        os.path.join(ORIGINAL_SPLIT_DIR, f"{split_name}_graphs.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"Loading {split_name}: {path}")
            return torch_load(path)

    raise FileNotFoundError(
        f"Could not find {split_name} split in {ORIGINAL_SPLIT_DIR}"
    )


def save_split(graphs, out_dir, split_name):
    os.makedirs(out_dir, exist_ok=True)

    # Save both names για να δουλεύει με όποιο naming χρησιμοποιεί το training script.
    torch.save(graphs, os.path.join(out_dir, f"{split_name}_graphs_subject.pt"))
    torch.save(graphs, os.path.join(out_dir, f"{split_name}_graphs.pt"))


def save_subject_lists(split_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    subject_lists = {}
    for split_name, graphs in split_dict.items():
        subject_lists[split_name] = sorted(set(str(g.subject_id) for g in graphs))

    out_path = os.path.join(out_dir, "subject_split_subjects.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subject_lists, f, indent=2)

    print(f"Saved subject lists to: {out_path}")


def print_split_info(title, split_dict, target_names):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    for split_name, graphs in split_dict.items():
        subjects = sorted(set(str(g.subject_id) for g in graphs))
        conditions = Counter(get_condition(g) for g in graphs)
        joint = joint_counts(graphs)
        first_counts, second_counts = label_counts_two_outputs(graphs)

        print(f"\n{split_name.upper()}")
        print(f"Graphs: {len(graphs)}")
        print(f"Subjects: {len(subjects)}")
        print(f"Conditions: {dict(conditions)}")
        print(f"{target_names[0]} counts: {first_counts}")
        print(f"{target_names[1]} counts: {second_counts}")
        print(f"Joint counts: {joint}")


# ============================================================
# COMPUTE GEQ PA/NA LABELS
# ============================================================

def compute_geq_labels_from_raw():
    """
    Δημιουργεί labels για τα game conditions.

    Για κάθε condition:
    EASY    -> Level01 GEQ
    HARD    -> Level02 GEQ
    OPTIMAL -> Level03 GEQ

    Για κάθε subject:
    positive_affect_score = mean(items 1,4,6,14,20)
    negative_affect_score = mean(items 7,8,9,16)

    Σημαντικό:
    Στα raw GEQ files:
    0 = αναπάντητη τιμή / missing
    1-5 = πραγματικές απαντήσεις

    Άρα:
    0 γίνεται NaN και δεν συμμετέχει στον μέσο όρο.
    1-5 μετατρέπεται σε 0-4 με -1.
    """

    labels = {}

    for condition, path in RAW_GEQ_FILES.items():
        print("\n" + "-" * 90)
        print(f"Reading GEQ for {condition}")
        print(f"Path: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing RAW GEQ file for {condition}: {path}"
            )

        df = clean_columns(read_csv_auto(path))

        if "ID" not in df.columns:
            raise ValueError(
                f"ID column not found in {path}. Columns: {list(df.columns)}"
            )

        needed_cols = PA_ITEMS + NA_ITEMS
        missing_cols = [c for c in needed_cols if c not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing GEQ item columns in {path}: {missing_cols}"
            )

        item_cols = [str(i) for i in range(1, 34) if str(i) in df.columns]

        for col in item_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        min_val = df[item_cols].min().min()
        max_val = df[item_cols].max().max()

        zero_count = int((df[item_cols] == 0).sum().sum())
        five_count = int((df[item_cols] == 5).sum().sum())
        total_values = int(df[item_cols].notna().sum().sum())

        print(f"{condition} raw GEQ scale: min={min_val}, max={max_val}")
        print(f"{condition}: zero_count={zero_count}")
        print(f"{condition}: five_count={five_count}")
        print(f"{condition}: total_values={total_values}")

        # Σωστή λογική για τα δικά σου raw GEQ:
        # 0 = unanswered/missing
        # 1-5 = answered scale
        bad_mask = (df[item_cols] > 5) | (df[item_cols] < 0)
        bad_count = int(bad_mask.sum().sum())

        if bad_count > 0:
            print("\nWARNING: Found unexpected GEQ values outside 0-5.")
            print(f"File: {path}")
            print(f"Found min={min_val}, max={max_val}")
            print(f"Bad value count: {bad_count}")

            bad_positions = bad_mask.stack()
            bad_positions = bad_positions[bad_positions]

            print("\nUnexpected values:")
            for (row_idx, col_name), _ in bad_positions.items():
                subject_id = df.loc[row_idx, "ID"]
                value = df.loc[row_idx, col_name]
                print(f"ID={subject_id}, item={col_name}, value={value}")

            print("\nTreating unexpected values as missing.")
            df[item_cols] = df[item_cols].mask(bad_mask, float("nan"))

        print(f"{condition}: Treating 0 as missing/unanswered.")
        print(f"{condition}: Converting answered values from 1-5 to 0-4.")

        # 0 -> missing
        df[item_cols] = df[item_cols].replace(0, float("nan"))

        # 1-5 -> 0-4
        df[item_cols] = df[item_cols] - 1

        # Υπολογισμός component scores
        df["positive_affect_score"] = df[PA_ITEMS].mean(axis=1, skipna=True)
        df["negative_affect_score"] = df[NA_ITEMS].mean(axis=1, skipna=True)

        # Binary labels
        df["positive_affect_label"] = (
            df["positive_affect_score"] > GEQ_THRESHOLD
        ).astype(int)

        df["negative_affect_label"] = (
            df["negative_affect_score"] > GEQ_THRESHOLD
        ).astype(int)

        # Diagnostics
        print(f"\n{condition} Positive Affect score summary:")
        print(df["positive_affect_score"].describe())

        print(f"\n{condition} Negative Affect score summary:")
        print(df["negative_affect_score"].describe())

        print(f"\n{condition} Positive Affect label counts:")
        print(df["positive_affect_label"].value_counts(dropna=False).sort_index())

        print(f"\n{condition} Negative Affect label counts:")
        print(df["negative_affect_label"].value_counts(dropna=False).sort_index())

        df["joint_pa_na"] = (
            df["positive_affect_label"] * 2 + df["negative_affect_label"]
        )

        print(f"\n{condition} joint PA/NA counts:")
        print(df["joint_pa_na"].value_counts(dropna=False).sort_index())

        # Save labels in dictionary
        skipped = 0

        for _, row in df.iterrows():
            sid = int(row["ID"])

            if pd.isna(row["positive_affect_score"]) or pd.isna(row["negative_affect_score"]):
                print(
                    f"WARNING: Missing PA/NA score for ID={sid}, "
                    f"condition={condition}. Skipping."
                )
                skipped += 1
                continue

            labels[(sid, condition)] = {
                "positive_affect_score": float(row["positive_affect_score"]),
                "negative_affect_score": float(row["negative_affect_score"]),
                "positive_affect_label": int(row["positive_affect_label"]),
                "negative_affect_label": int(row["negative_affect_label"]),
            }

        print(f"{condition}: saved labels for {len(df) - skipped} subjects")
        print(f"{condition}: skipped subjects = {skipped}")

    return labels


def save_geq_label_table(labels):
    os.makedirs(GAME_OUT_DIR, exist_ok=True)

    rows = []

    for (sid, condition), info in sorted(labels.items()):
        pa_label = info["positive_affect_label"]
        na_label = info["negative_affect_label"]

        rows.append({
            "ID": sid,
            "condition": condition,
            "positive_affect_score": info["positive_affect_score"],
            "negative_affect_score": info["negative_affect_score"],
            "positive_affect_label": pa_label,
            "negative_affect_label": na_label,
            "joint_pa_na": pa_label * 2 + na_label,
        })

    df = pd.DataFrame(rows)

    out_path = os.path.join(GAME_OUT_DIR, "game_geq_pa_na_labels.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 90)
    print("Saved GEQ PA/NA label table")
    print("=" * 90)
    print(out_path)

    print("\nGEQ label distribution by condition:")
    print(
        df.groupby("condition")[
            ["positive_affect_label", "negative_affect_label"]
        ].mean()
    )

    print("\nJoint PA/NA counts by condition:")
    print(df.groupby(["condition", "joint_pa_na"]).size())


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nLoading original subject-level splits...")
    print(f"ORIGINAL_SPLIT_DIR: {ORIGINAL_SPLIT_DIR}")
    print(f"GEQ_DIR: {GEQ_DIR}")

    train_graphs = load_split("train")
    val_graphs = load_split("val")
    test_graphs = load_split("test")

    original_splits = {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }

    # ========================================================
    # TASK A: PRE/POST Valence-Arousal
    # ========================================================

    prepost_splits = {}

    for split_name, graphs in original_splits.items():
        selected = []

        for g in graphs:
            cond = get_condition(g)

            if cond in {"PRE", "POST"}:
                ng = copy.deepcopy(g)

                # Κρατάμε τα ORIGINAL y όπως ήταν.
                # Δεν αλλάζουμε labels στα PRE/POST.
                ng.task_name = "prepost_valence_arousal"
                ng.target_names = ["valence", "arousal"]
                ng.label_source = "Procedure.csv ANS-VALENCE / ANS-AROUSAL"

                selected.append(ng)

        prepost_splits[split_name] = selected
        save_split(selected, PREPOST_OUT_DIR, split_name)

    save_subject_lists(prepost_splits, PREPOST_OUT_DIR)

    print_split_info(
        title="TASK A: PRE/POST Valence-Arousal",
        split_dict=prepost_splits,
        target_names=["valence", "arousal"],
    )

    # ========================================================
    # TASK B: GAME GEQ Positive/Negative Affect
    # ========================================================

    geq_labels = compute_geq_labels_from_raw()
    save_geq_label_table(geq_labels)

    game_splits = {}
    missing_labels = []

    for split_name, graphs in original_splits.items():
        selected = []

        for g in graphs:
            cond = get_condition(g)

            if cond in {"EASY", "HARD", "OPTIMAL"}:
                sid = subject_to_int(g.subject_id)
                key = (sid, cond)

                if key not in geq_labels:
                    missing_labels.append((split_name, str(g.subject_id), cond))
                    continue

                info = geq_labels[key]

                ng = copy.deepcopy(g)

                # Εδώ αλλάζουμε το y:
                # Πλέον δεν είναι [valence, arousal].
                # Είναι [positive_affect, negative_affect].
                ng.y = torch.tensor(
                    [
                        info["positive_affect_label"],
                        info["negative_affect_label"],
                    ],
                    dtype=torch.long,
                )

                ng.positive_affect_score = info["positive_affect_score"]
                ng.negative_affect_score = info["negative_affect_score"]

                ng.task_name = "game_geq_positive_negative_affect"
                ng.target_names = ["positive_affect", "negative_affect"]
                ng.label_source = "GEQ Core Module Positive Affect / Negative Affect"

                selected.append(ng)

        game_splits[split_name] = selected
        save_split(selected, GAME_OUT_DIR, split_name)

    save_subject_lists(game_splits, GAME_OUT_DIR)

    if missing_labels:
        print("\n" + "=" * 90)
        print("WARNING: Missing GEQ labels for some graphs")
        print("=" * 90)
        print(f"Total missing graph labels: {len(missing_labels)}")
        print("First 30 missing:")
        for item in missing_labels[:30]:
            print(item)
    else:
        print("\nNo missing GEQ labels for game graphs.")

    print_split_info(
        title="TASK B: GAME GEQ Positive/Negative Affect",
        split_dict=game_splits,
        target_names=["positive_affect", "negative_affect"],
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)

    print(f"\nPRE/POST split saved to:")
    print(PREPOST_OUT_DIR)

    print(f"\nGAME GEQ split saved to:")
    print(GAME_OUT_DIR)

    print("\nNext:")
    print("1. Train Weighted GCN on subject_level_splits_prepost_va_original")
    print("2. Train Weighted GCN on subject_level_splits_game_geq_pa_na")


if __name__ == "__main__":
    main()