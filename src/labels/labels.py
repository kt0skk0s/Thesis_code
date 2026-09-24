from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ======================================================
# PATHS
# ======================================================

WINDOW_METADATA_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\signalwise_pca"
)

PROCEDURE_ROOT = Path(
    r"C:\Users\User\OneDrive\Desktop\Thesis\BERAFFE2_DATASET\metadata\BIRAFFE2-procedure"
)

GAME_PREPROCESSED_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\games_preprocced"
)

OUTPUT_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\labels_prepost_constant"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================
# SETTINGS
# ======================================================

REFERENCE_SIGNAL = "ECG"

CONDITIONS = ["PRE", "POST", "EASY", "HARD", "OPTIMAL"]

GAME_FOLDER_MAP = {
    "EASY": "easy",
    "HARD": "hard",
    "OPTIMAL": "optimal",
}


# ======================================================
# HELPERS
# ======================================================

def normalize_timestamp(t):
    """
    Αν το timestamp είναι σε milliseconds, το κάνει seconds.
    Αν είναι ήδη σε seconds, το αφήνει όπως είναι.
    """
    t = float(t)

    if t > 1e11:
        return t / 1000.0

    return t


def load_procedure_file(subject_id):
    proc_path = PROCEDURE_ROOT / f"{subject_id}-Procedure.csv"

    if not proc_path.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε procedure file: {proc_path}")

    proc = pd.read_csv(proc_path, sep=";", low_memory=False)

    if len(proc.columns) == 1:
        proc = pd.read_csv(proc_path, sep=",", low_memory=False)

    proc.columns = proc.columns.str.strip().str.upper()

    required_cols = ["TIMESTAMP", "ANS-VALENCE", "ANS-AROUSAL"]

    for col in required_cols:
        if col not in proc.columns:
            raise ValueError(
                f"Λείπει η στήλη {col} από {proc_path}. "
                f"Βρέθηκαν στήλες: {list(proc.columns)}"
            )

    proc["TIMESTAMP"] = pd.to_numeric(proc["TIMESTAMP"], errors="coerce")
    proc["ANS-VALENCE"] = pd.to_numeric(proc["ANS-VALENCE"], errors="coerce")
    proc["ANS-AROUSAL"] = pd.to_numeric(proc["ANS-AROUSAL"], errors="coerce")

    proc = proc.dropna(
        subset=["TIMESTAMP", "ANS-VALENCE", "ANS-AROUSAL"]
    ).copy()

    if len(proc) == 0:
        raise ValueError(f"Δεν υπάρχουν valid label rows στο {proc_path}")

    proc["TIMESTAMP_SEC"] = proc["TIMESTAMP"].apply(normalize_timestamp)

    return proc


def raw_to_binary(valence_raw, arousal_raw):
    valence_label = 1 if float(valence_raw) > 5 else 0
    arousal_label = 1 if float(arousal_raw) > 5 else 0

    return valence_label, arousal_label


def get_nearest_label(proc, target_time_sec):
    timestamps = proc["TIMESTAMP_SEC"].values
    diffs = np.abs(timestamps - target_time_sec)

    best_idx = int(np.argmin(diffs))
    row = proc.iloc[best_idx]

    val_raw = float(row["ANS-VALENCE"])
    aro_raw = float(row["ANS-AROUSAL"])

    val_bin, aro_bin = raw_to_binary(val_raw, aro_raw)

    return {
        "valence_label": val_bin,
        "arousal_label": aro_bin,
        "ANS_VALENCE_raw": val_raw,
        "ANS_AROUSAL_raw": aro_raw,
        "matched_procedure_timestamp": float(row["TIMESTAMP_SEC"]),
        "time_diff_sec": abs(target_time_sec - float(row["TIMESTAMP_SEC"])),
    }


def get_label_after_or_near(proc, target_time_sec):
    """
    Για game conditions:
    προτιμάμε το πρώτο label μετά το τέλος του game interval.
    Αν δεν υπάρχει, παίρνουμε το κοντινότερο.
    """
    after = proc[proc["TIMESTAMP_SEC"] >= target_time_sec].copy()

    if len(after) > 0:
        after["diff"] = after["TIMESTAMP_SEC"] - target_time_sec
        row = after.sort_values("diff").iloc[0]
    else:
        timestamps = proc["TIMESTAMP_SEC"].values
        diffs = np.abs(timestamps - target_time_sec)
        best_idx = int(np.argmin(diffs))
        row = proc.iloc[best_idx]

    val_raw = float(row["ANS-VALENCE"])
    aro_raw = float(row["ANS-AROUSAL"])

    val_bin, aro_bin = raw_to_binary(val_raw, aro_raw)

    return {
        "valence_label": val_bin,
        "arousal_label": aro_bin,
        "ANS_VALENCE_raw": val_raw,
        "ANS_AROUSAL_raw": aro_raw,
        "matched_procedure_timestamp": float(row["TIMESTAMP_SEC"]),
        "time_diff_sec": abs(target_time_sec - float(row["TIMESTAMP_SEC"])),
    }


def load_condition_metadata(condition):
    metadata_path = (
        WINDOW_METADATA_ROOT
        / condition
        / f"{REFERENCE_SIGNAL}_metadata.csv"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε metadata: {metadata_path}")

    meta = pd.read_csv(metadata_path)

    required_cols = [
        "subject_id",
        "part",
        "window_id",
        "t_start",
        "t_end",
        "t_center",
    ]

    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(
                f"Λείπει η στήλη {col} από {metadata_path}. "
                f"Βρέθηκαν: {list(meta.columns)}"
            )

    meta["t_center_sec"] = meta["t_center"].apply(normalize_timestamp)

    return meta


def get_game_interval(condition, subject_id):
    """
    Παίρνει start/end από το condition-specific preprocessed .mat.
    Π.χ. EASY -> games_preprocced/easy/SUB120_preprocessed.mat
    """
    folder_name = GAME_FOLDER_MAP[condition]

    mat_path = (
        GAME_PREPROCESSED_ROOT
        / folder_name
        / f"{subject_id}_preprocessed.mat"
    )

    if not mat_path.exists():
        return None

    data = loadmat(mat_path)

    if "preprocessed" not in data:
        raise ValueError(f"Δεν βρέθηκε struct 'preprocessed' στο {mat_path}")

    pp = data["preprocessed"][0, 0]

    if "TIMESTAMP" not in pp.dtype.names:
        raise ValueError(f"Δεν βρέθηκε TIMESTAMP στο {mat_path}")

    timestamps = pp["TIMESTAMP"].squeeze()
    timestamps = np.array([normalize_timestamp(t) for t in timestamps])

    return {
        "game_start_sec": float(np.min(timestamps)),
        "game_end_sec": float(np.max(timestamps)),
        "n_timestamps": len(timestamps),
    }


# ======================================================
# LABEL CREATION
# ======================================================

def make_pre_post_labels(condition, metadata, procedure_cache):
    """
    Για PRE/POST:
    Δεν παίρνουμε διαφορετικό nearest label για κάθε window.
    Παίρνουμε ένα σταθερό label ανά subject_id + condition.

    Αυτό διορθώνει το πρόβλημα όπου τα PRE/POST labels
    άλλαζαν σχεδόν σε κάθε window.
    """

    labels = []
    rows = []

    prepost_label_cache = {}

    # Υπολογίζουμε πρώτα ένα target time ανά subject
    # για αυτό το condition.
    subject_target_times = {}

    for subject_id, subject_meta in metadata.groupby("subject_id"):
        subject_id = str(subject_id).strip()

        subject_meta = subject_meta.copy()
        subject_meta["t_center_sec"] = subject_meta["t_center"].apply(normalize_timestamp)

        # Επιλογή: παίρνουμε το τέλος του PRE/POST segment.
        # Μπορεί να αλλαχθεί σε mean/median αν χρειαστεί.
        target_time_sec = float(subject_meta["t_center_sec"].max())

        subject_target_times[subject_id] = target_time_sec

    for idx, row in metadata.iterrows():
        subject_id = str(row["subject_id"]).strip()

        if subject_id not in procedure_cache:
            procedure_cache[subject_id] = load_procedure_file(subject_id)

        proc = procedure_cache[subject_id]

        cache_key = (subject_id, condition)

        if cache_key not in prepost_label_cache:
            target_time_sec = subject_target_times[subject_id]

            label_info = get_nearest_label(
                proc=proc,
                target_time_sec=target_time_sec
            )

            prepost_label_cache[cache_key] = {
                "target_time_sec": target_time_sec,
                "label_info": label_info,
            }

        cached = prepost_label_cache[cache_key]
        target_time_sec = cached["target_time_sec"]
        label_info = cached["label_info"]

        labels.append([
            label_info["valence_label"],
            label_info["arousal_label"],
        ])

        rows.append({
            "row_id": idx,
            "subject_id": subject_id,
            "part": row["part"],
            "window_id": row["window_id"],
            "t_start": row["t_start"],
            "t_end": row["t_end"],
            "t_center": row["t_center"],
            "method": "constant_prepost_label_per_subject_condition",
            "prepost_target_time_sec": target_time_sec,
            "game_start_sec": np.nan,
            "game_end_sec": np.nan,
            "inside_game_interval": "",
            **label_info,
        })

    return np.array(labels, dtype=np.int64), pd.DataFrame(rows)

'''
def make_pre_post_labels(condition, metadata, procedure_cache):
    labels = []
    rows = []

    for idx, row in metadata.iterrows():
        subject_id = str(row["subject_id"]).strip()

        if subject_id not in procedure_cache:
            procedure_cache[subject_id] = load_procedure_file(subject_id)

        proc = procedure_cache[subject_id]

        label_info = get_nearest_label(
            proc=proc,
            target_time_sec=float(row["t_center_sec"])
        )

        labels.append([
            label_info["valence_label"],
            label_info["arousal_label"],
        ])

        rows.append({
            "row_id": idx,
            "subject_id": subject_id,
            "part": row["part"],
            "window_id": row["window_id"],
            "t_start": row["t_start"],
            "t_end": row["t_end"],
            "t_center": row["t_center"],
            "method": "nearest_window_timestamp",
            "game_start_sec": np.nan,
            "game_end_sec": np.nan,
            "inside_game_interval": "",
            **label_info,
        })

    return np.array(labels, dtype=np.int64), pd.DataFrame(rows)
'''

def make_game_labels(condition, metadata, procedure_cache):
    labels = []
    rows = []

    game_label_cache = {}

    for idx, row in metadata.iterrows():
        subject_id = str(row["subject_id"]).strip()

        if subject_id not in procedure_cache:
            procedure_cache[subject_id] = load_procedure_file(subject_id)

        proc = procedure_cache[subject_id]

        cache_key = (subject_id, condition)

        if cache_key not in game_label_cache:
            interval = get_game_interval(condition, subject_id)

            if interval is None:
                raise FileNotFoundError(
                    f"Δεν βρέθηκε game preprocessed interval για "
                    f"{subject_id}, condition={condition}"
                )

            label_info = get_label_after_or_near(
                proc=proc,
                target_time_sec=interval["game_end_sec"]
            )

            game_label_cache[cache_key] = {
                "interval": interval,
                "label_info": label_info,
            }

        cached = game_label_cache[cache_key]
        interval = cached["interval"]
        label_info = cached["label_info"]

        inside = (
            float(row["t_center_sec"]) >= interval["game_start_sec"]
            and float(row["t_center_sec"]) <= interval["game_end_sec"]
        )

        labels.append([
            label_info["valence_label"],
            label_info["arousal_label"],
        ])

        rows.append({
            "row_id": idx,
            "subject_id": subject_id,
            "part": row["part"],
            "window_id": row["window_id"],
            "t_start": row["t_start"],
            "t_end": row["t_end"],
            "t_center": row["t_center"],
            "method": "game_preprocessed_interval_end_label",
            "game_start_sec": interval["game_start_sec"],
            "game_end_sec": interval["game_end_sec"],
            "inside_game_interval": inside,
            **label_info,
        })

    return np.array(labels, dtype=np.int64), pd.DataFrame(rows)


# ======================================================
# MAIN
# ======================================================

def main():
    print("Ξεκινάω δημιουργία τελικών σωστών labels...")
    print(f"Window metadata root: {WINDOW_METADATA_ROOT}")
    print(f"Procedure root: {PROCEDURE_ROOT}")
    print(f"Game preprocessed root: {GAME_PREPROCESSED_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}\n")

    procedure_cache = {}

    for condition in CONDITIONS:
        print("=" * 70)
        print(f"Condition: {condition}")
        print("=" * 70)

        metadata = load_condition_metadata(condition)

        if condition in ["PRE", "POST"]:
            labels, labels_df = make_pre_post_labels(
                condition=condition,
                metadata=metadata,
                procedure_cache=procedure_cache
            )
        else:
            labels, labels_df = make_game_labels(
                condition=condition,
                metadata=metadata,
                procedure_cache=procedure_cache
            )

        out_folder = OUTPUT_ROOT / condition
        out_folder.mkdir(parents=True, exist_ok=True)

        labels_path = out_folder / f"{condition}_labels.npy"
        labels_meta_path = out_folder / f"{condition}_labels_metadata.csv"

        np.save(labels_path, labels)
        labels_df.to_csv(labels_meta_path, index=False)

        print(f"Metadata rows: {len(metadata)}")
        print(f"Labels shape: {labels.shape}")
        print(f"Saved labels: {labels_path}")
        print(f"Saved labels metadata: {labels_meta_path}")

        print("\nLabel distribution:")
        print(
            labels_df[["valence_label", "arousal_label"]]
            .value_counts()
            .sort_index()
        )

        print("\nTime difference summary:")
        print(labels_df["time_diff_sec"].describe())

        if condition in ["EASY", "HARD", "OPTIMAL"]:
            print("\nInside game interval:")
            print(labels_df["inside_game_interval"].value_counts(dropna=False))

        print()

    print("Ολοκληρώθηκε η δημιουργία τελικών σωστών labels.")
    

if __name__ == "__main__":
    main()