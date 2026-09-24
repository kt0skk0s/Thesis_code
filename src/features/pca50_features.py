from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib


# ======================================================
# ΡΥΘΜΙΣΕΙΣ
# ======================================================

INPUT_ROOT = Path(r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\signalwise_pca")
OUTPUT_ROOT = Path(r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\pca50_features")

N_COMPONENTS = 50

CONDITIONS = ["PRE", "POST", "EASY", "HARD", "OPTIMAL"]

SIGNALS = [
    "ECG",
    "EDA",
    "GYR-X",
    "GYR-Y",
    "GYR-Z",
    "ACC-X",
    "ACC-Y",
    "ACC-Z",
]


# ======================================================
# ΣΥΝΑΡΤΗΣΗ PCA ΓΙΑ ΕΝΑ SIGNAL
# ======================================================

def run_pca_for_signal(condition: str, signal_name: str):
    input_folder = INPUT_ROOT / condition
    output_folder = OUTPUT_ROOT / condition
    output_folder.mkdir(parents=True, exist_ok=True)

    x_path = input_folder / f"{signal_name}_X.npy"

    if not x_path.exists():
        print(f"[SKIP] Δεν βρέθηκε: {x_path}")
        return

    print(f"\nCondition: {condition} | Signal: {signal_name}")

    # Φόρτωση raw windows
    X = np.load(x_path)

    print(f"Input shape: {X.shape}")

    if X.ndim != 2:
        raise ValueError(f"Λάθος shape για {x_path}: {X.shape}")

    if X.shape[1] != 900:
        print(f"[WARNING] Περίμενα 900 samples, αλλά βρήκα {X.shape[1]}")

    if X.shape[1] < N_COMPONENTS:
        raise ValueError(
            f"Δεν μπορώ να κρατήσω {N_COMPONENTS} components, "
            f"γιατί το signal έχει μόνο {X.shape[1]} features."
        )

    # PCA
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X)

    X_pca = X_pca.astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()

    print(f"Output shape: {X_pca.shape}")
    print(f"Cumulative explained variance @ {N_COMPONENTS}: {explained:.4f}")

    # Paths εξόδου
    pca_features_path = output_folder / f"{signal_name}_PCA50.npy"
    pca_model_path = output_folder / f"{signal_name}_PCA50_model.pkl"
    variance_path = output_folder / f"{signal_name}_PCA50_variance.csv"

    # Αποθήκευση PCA features
    np.save(pca_features_path, X_pca)

    # Αποθήκευση PCA model
    joblib.dump(pca, pca_model_path)

    # Αποθήκευση variance
    variance_df = pd.DataFrame({
        "component": np.arange(1, N_COMPONENTS + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)
    })

    variance_df.to_csv(variance_path, index=False)

    print(f"Saved: {pca_features_path}")
    print(f"Saved: {pca_model_path}")
    print(f"Saved: {variance_path}")


# ======================================================
# MAIN
# ======================================================

def main():
    print("Ξεκινάω PCA50 feature extraction...")
    print(f"Input folder: {INPUT_ROOT}")
    print(f"Output folder: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:
        condition_folder = INPUT_ROOT / condition

        if not condition_folder.exists():
            print(f"\n[SKIP] Δεν υπάρχει condition folder: {condition_folder}")
            continue

        print(f"\n==============================")
        print(f"Processing condition: {condition}")
        print(f"==============================")

        for signal_name in SIGNALS:
            run_pca_for_signal(condition, signal_name)

    print("\nΟλοκληρώθηκε η δημιουργία PCA50 features.")


if __name__ == "__main__":
    main()