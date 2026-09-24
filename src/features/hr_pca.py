from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ======================================================
# PATHS
# ======================================================

HR_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\hr_from_ecg"
)

OUTPUT_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\hr_pca50_features"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================
# SETTINGS
# ======================================================

CONDITIONS = ["PRE", "POST", "EASY", "HARD", "OPTIMAL"]

N_COMPONENTS = 50


# ======================================================
# MAIN
# ======================================================

def main():
    print("Ξεκινάω PCA50 για HR signals...\n")

    for condition in CONDITIONS:
        print("=" * 70)
        print(f"Condition: {condition}")
        print("=" * 70)

        hr_path = HR_ROOT / condition / "HR_X.npy"

        if not hr_path.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε HR file: {hr_path}")

        X = np.load(hr_path)

        print(f"Input HR shape: {X.shape}")

        if X.shape[1] != 900:
            raise ValueError(f"Περίμενα 900 samples, αλλά βρήκα {X.shape[1]}")

        # --------------------------------------------------
        # Standardization πριν το PCA
        # --------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --------------------------------------------------
        # PCA
        # --------------------------------------------------
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        print(f"Output HR PCA shape: {X_pca.shape}")
        print(
            f"Cumulative explained variance @ {N_COMPONENTS}: "
            f"{pca.explained_variance_ratio_.sum():.4f}"
        )

        # --------------------------------------------------
        # Save outputs
        # --------------------------------------------------
        out_condition = OUTPUT_ROOT / condition
        out_condition.mkdir(parents=True, exist_ok=True)

        pca_path = out_condition / "HR_PCA50.npy"
        model_path = out_condition / "HR_PCA50_model.pkl"
        scaler_path = out_condition / "HR_PCA50_scaler.pkl"
        variance_path = out_condition / "HR_PCA50_variance.csv"

        np.save(pca_path, X_pca)
        joblib.dump(pca, model_path)
        joblib.dump(scaler, scaler_path)

        variance_df = pd.DataFrame({
            "component": np.arange(1, N_COMPONENTS + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            )
        })

        variance_df.to_csv(variance_path, index=False)

        print(f"Saved: {pca_path}")
        print(f"Saved: {model_path}")
        print(f"Saved: {scaler_path}")
        print(f"Saved: {variance_path}")
        print()

    print("Ολοκληρώθηκε το PCA50 για HR.")


if __name__ == "__main__":
    main()