from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ======================================================
# PATHS
# ======================================================

# PCA features για EDA, GYR, ACC
PCA_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\pca50_features"
)

# PCA features για HR
HR_PCA_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\hr_pca50_features"
)

# Raw windows για EDA, GYR, ACC
RAW_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\signalwise_pca"
)

# Raw/interpolated HR windows
HR_RAW_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\hr_from_ecg"
)

# Labels
LABELS_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\labels_prepost_constant"
)

# Output
OUTPUT_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\graph_dataset_hr_prepost_constant"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================
# SETTINGS
# ======================================================

CONDITIONS = ["PRE", "POST", "EASY", "HARD", "OPTIMAL"]

# Πλέον ο πρώτος κόμβος είναι HR, όχι ECG
SIGNALS = [
    "HR",
    "EDA",
    "GYR-X",
    "GYR-Y",
    "GYR-Z",
    "ACC-X",
    "ACC-Y",
    "ACC-Z",
]

REFERENCE_SIGNAL = "ECG"   # κρατάμε ECG metadata γιατί αυτό έχει σωστά windows/timestamps

CORR_THRESHOLD = 0.3


# ======================================================
# EDGE CONSTRUCTION
# ======================================================

def build_edges_from_raw_window(raw_window_matrix, threshold=0.3):
    """
    raw_window_matrix shape = (8, 900)

    Υπολογίζει Pearson correlation μεταξύ των 8 signals.
    Αν |corr| > threshold, δημιουργεί edge.
    """

    edge_list = []
    edge_attr = []

    num_nodes = raw_window_matrix.shape[0]

    corr_matrix = np.corrcoef(raw_window_matrix)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue

            corr = corr_matrix[i, j]

            if np.isnan(corr):
                continue

            if abs(corr) > threshold:
                edge_list.append([i, j])
                edge_attr.append([corr])

    # Αν δεν βρεθεί κανένα edge, βάζουμε self-loops για να μη σπάσει το GNN
    if len(edge_list) == 0:
        for i in range(num_nodes):
            edge_list.append([i, i])
            edge_attr.append([1.0])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr


# ======================================================
# LOAD CONDITION DATA
# ======================================================

def load_condition_data(condition):
    print(f"\nLoading condition: {condition}")

    pca_data = {}
    raw_data = {}

    for signal in SIGNALS:

        # --------------------------------------------------
        # HR έχει δικά του PCA και raw αρχεία
        # --------------------------------------------------
        if signal == "HR":
            pca_path = HR_PCA_ROOT / condition / "HR_PCA50.npy"
            raw_path = HR_RAW_ROOT / condition / "HR_X.npy"

        # --------------------------------------------------
        # Τα υπόλοιπα signals μένουν όπως πριν
        # --------------------------------------------------
        else:
            pca_path = PCA_ROOT / condition / f"{signal}_PCA50.npy"
            raw_path = RAW_ROOT / condition / f"{signal}_X.npy"

        if not pca_path.exists():
            raise FileNotFoundError(f"Missing PCA file: {pca_path}")

        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw file: {raw_path}")

        pca_data[signal] = np.load(pca_path)
        raw_data[signal] = np.load(raw_path, mmap_mode="r")

        print(
            f"{signal:6s} PCA: {pca_data[signal].shape} | "
            f"RAW: {raw_data[signal].shape}"
        )

    labels_path = LABELS_ROOT / condition / f"{condition}_labels.npy"
    metadata_path = RAW_ROOT / condition / f"{REFERENCE_SIGNAL}_metadata.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    labels = np.load(labels_path)
    metadata = pd.read_csv(metadata_path)

    print(f"Labels: {labels.shape}")
    print(f"Metadata: {metadata.shape}")

    return pca_data, raw_data, labels, metadata


# ======================================================
# CHECK SHAPES
# ======================================================

def check_shapes(condition, pca_data, raw_data, labels, metadata):
    expected_n = None

    for signal in SIGNALS:
        pca_shape = pca_data[signal].shape
        raw_shape = raw_data[signal].shape

        if pca_shape[1] != 50:
            raise ValueError(
                f"{condition} {signal}: PCA features are not 50: {pca_shape}"
            )

        if raw_shape[1] != 900:
            raise ValueError(
                f"{condition} {signal}: raw samples are not 900: {raw_shape}"
            )

        if expected_n is None:
            expected_n = pca_shape[0]

        if pca_shape[0] != expected_n:
            raise ValueError(
                f"{condition} {signal}: PCA N mismatch "
                f"{pca_shape[0]} != {expected_n}"
            )

        if raw_shape[0] != expected_n:
            raise ValueError(
                f"{condition} {signal}: raw N mismatch "
                f"{raw_shape[0]} != {expected_n}"
            )

    if labels.shape[0] != expected_n:
        raise ValueError(
            f"{condition}: labels N mismatch: {labels.shape[0]} != {expected_n}"
        )

    if len(metadata) != expected_n:
        raise ValueError(
            f"{condition}: metadata N mismatch: {len(metadata)} != {expected_n}"
        )

    print(f"[OK] {condition}: all shapes match. N = {expected_n}")

    return expected_n


# ======================================================
# BUILD GRAPHS FOR ONE CONDITION
# ======================================================

def build_graphs_for_condition(condition):
    pca_data, raw_data, labels, metadata = load_condition_data(condition)

    n_windows = check_shapes(
        condition=condition,
        pca_data=pca_data,
        raw_data=raw_data,
        labels=labels,
        metadata=metadata,
    )

    graphs = []

    for i in range(n_windows):

        # -----------------------------
        # Node features x: (8, 50)
        # -----------------------------
        x_np = np.stack(
            [pca_data[signal][i] for signal in SIGNALS],
            axis=0
        )

        x = torch.tensor(x_np, dtype=torch.float)

        # -----------------------------
        # Raw window matrix: (8, 900)
        # Για correlation edges
        # -----------------------------
        raw_window_matrix = np.stack(
            [raw_data[signal][i] for signal in SIGNALS],
            axis=0
        )

        edge_index, edge_attr = build_edges_from_raw_window(
            raw_window_matrix=raw_window_matrix,
            threshold=CORR_THRESHOLD
        )

        # -----------------------------
        # Label y: (2,)
        # -----------------------------
        y = torch.tensor(labels[i], dtype=torch.float)

        # -----------------------------
        # Metadata
        # -----------------------------
        meta_row = metadata.iloc[i]

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )

        graph.subject_id = str(meta_row["subject_id"])
        graph.condition = condition
        graph.part = str(meta_row["part"])
        graph.window_id = int(meta_row["window_id"])
        graph.t_start = float(meta_row["t_start"])
        graph.t_end = float(meta_row["t_end"])
        graph.t_center = float(meta_row["t_center"])

        # για να θυμόμαστε ότι αυτό είναι το HR graph dataset
        graph.signal_names = SIGNALS

        graphs.append(graph)

        if (i + 1) % 1000 == 0:
            print(f"{condition}: built {i + 1}/{n_windows} graphs")

    print(f"{condition}: completed {len(graphs)} graphs")

    return graphs


# ======================================================
# MAIN
# ======================================================

def main():
    print("Ξεκινάω δημιουργία graph dataset με HR αντί για ECG...")
    print(f"Correlation threshold: |corr| > {CORR_THRESHOLD}")
    print(f"Signals: {SIGNALS}")

    all_graphs = []

    for condition in CONDITIONS:
        condition_graphs = build_graphs_for_condition(condition)
        all_graphs.extend(condition_graphs)

    output_path = OUTPUT_ROOT / "graph_dataset_hr.pt"
    torch.save(all_graphs, output_path)

    print("\n====================================")
    print("HR Graph dataset completed.")
    print(f"Total graphs: {len(all_graphs)}")
    print(f"Saved to: {output_path}")
    print("====================================")

    # Quick check
    g = all_graphs[0]

    print("\nExample graph:")
    print(g)
    print("x shape:", g.x.shape)
    print("edge_index shape:", g.edge_index.shape)
    print("edge_attr shape:", g.edge_attr.shape)
    print("y:", g.y)
    print("subject:", g.subject_id)
    print("condition:", g.condition)
    print("signals:", g.signal_names)


if __name__ == "__main__":
    main()