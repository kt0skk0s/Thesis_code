from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


# ======================================================
# PATHS
# ======================================================

INPUT_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\signalwise_pca"
)

OUTPUT_ROOT = Path(
    r"D:\BIRAFFE2_DATASET\PROCESSED_FINAL\hr_from_ecg"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================
# SETTINGS
# ======================================================

CONDITIONS = ["PRE", "POST", "EASY", "HARD", "OPTIMAL"]

FS = 50
WINDOW_SECONDS = 18
WINDOW_SAMPLES = 900

HR_MIN = 40
HR_MAX = 180


# ======================================================
# FUNCTIONS
# ======================================================

def zscore_signal(x):
    x = np.asarray(x, dtype=float)
    std = np.std(x)

    if std < 1e-8:
        return x * 0.0

    return (x - np.mean(x)) / std


def detect_r_peaks(ecg_window, fs=50):
    """
    Βρίσκει R-peaks σε ένα ECG window.
    Δοκιμάζει και θετική και αρνητική πολικότητα.
    """

    ecg_z = zscore_signal(ecg_window)

    min_distance_samples = int(0.35 * fs)
    prominence = 0.5

    candidates = []

    for polarity_name, signal_to_use in [
        ("positive", ecg_z),
        ("negative", -ecg_z),
    ]:
        peaks, _ = find_peaks(
            signal_to_use,
            distance=min_distance_samples,
            prominence=prominence
        )

        if len(peaks) >= 2:
            rr = np.diff(peaks) / fs
            hr = 60.0 / rr

            valid_hr = hr[(hr >= HR_MIN) & (hr <= HR_MAX)]

            candidates.append({
                "polarity": polarity_name,
                "peaks": peaks,
                "rr": rr,
                "hr": hr,
                "score": len(valid_hr),
            })

    if len(candidates) == 0:
        return {
            "success": False,
            "polarity": None,
            "peaks": np.array([]),
            "rr": np.array([]),
            "hr": np.array([]),
        }

    best = sorted(candidates, key=lambda d: d["score"], reverse=True)[0]

    return {
        "success": len(best["peaks"]) >= 4,
        "polarity": best["polarity"],
        "peaks": best["peaks"],
        "rr": best["rr"],
        "hr": best["hr"],
    }


def create_hr_timeseries_from_peaks(peaks, fs=50, n_samples=900):
    """
    Φτιάχνει HR time series μήκους 900 samples από R-peaks.

    HR values υπολογίζονται στα midpoints μεταξύ διαδοχικών peaks.
    Μετά γίνεται interpolation σε όλο το 18-sec window.
    """

    if len(peaks) < 2:
        return None

    peak_times = peaks / fs

    rr = np.diff(peak_times)
    hr = 60.0 / rr

    # κρατάμε μόνο φυσιολογικές HR τιμές
    valid_mask = (hr >= HR_MIN) & (hr <= HR_MAX)

    if valid_mask.sum() < 2:
        return None

    rr_mid_times = (peak_times[:-1] + peak_times[1:]) / 2.0

    valid_times = rr_mid_times[valid_mask]
    valid_hr = hr[valid_mask]

    if len(valid_hr) < 2:
        return None

    target_times = np.arange(n_samples) / fs

    # interpolation
    f = interp1d(
        valid_times,
        valid_hr,
        kind="linear",
        bounds_error=False,
        fill_value=(valid_hr[0], valid_hr[-1])
    )

    hr_series = f(target_times)

    return hr_series.astype(np.float32)


def fallback_hr_series(previous_hr_series=None, n_samples=900):
    """
    Αν αποτύχει το peak detection, βάζουμε fallback.
    Προτιμάμε previous valid HR series αν υπάρχει.
    Αλλιώς βάζουμε σταθερό 75 bpm.
    """
    if previous_hr_series is not None:
        return previous_hr_series.copy()

    return np.ones(n_samples, dtype=np.float32) * 75.0


# ======================================================
# MAIN
# ======================================================

def main():
    print("Ξεκινάω δημιουργία HR_X.npy από ECG_X.npy για όλα τα conditions...\n")

    for condition in CONDITIONS:
        print("=" * 70)
        print(f"Condition: {condition}")
        print("=" * 70)

        ecg_path = INPUT_ROOT / condition / "ECG_X.npy"
        metadata_path = INPUT_ROOT / condition / "ECG_metadata.csv"

        if not ecg_path.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε ECG file: {ecg_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε metadata file: {metadata_path}")

        ecg_data = np.load(ecg_path, mmap_mode="r")
        metadata = pd.read_csv(metadata_path)

        n_windows, n_samples = ecg_data.shape

        if n_samples != WINDOW_SAMPLES:
            raise ValueError(f"Expected 900 samples, got {n_samples}")

        print(f"ECG shape: {ecg_data.shape}")
        print(f"Metadata shape: {metadata.shape}")

        hr_data = np.zeros((n_windows, n_samples), dtype=np.float32)

        rows = []

        previous_valid_hr_series = None

        for i in range(n_windows):
            ecg_window = ecg_data[i]

            result = detect_r_peaks(ecg_window, fs=FS)

            hr_series = None

            if result["success"]:
                hr_series = create_hr_timeseries_from_peaks(
                    peaks=result["peaks"],
                    fs=FS,
                    n_samples=n_samples
                )

            used_fallback = False

            if hr_series is None:
                hr_series = fallback_hr_series(
                    previous_hr_series=previous_valid_hr_series,
                    n_samples=n_samples
                )
                used_fallback = True
            else:
                previous_valid_hr_series = hr_series.copy()

            hr_data[i] = hr_series

            valid_hr_values = hr_series[
                (hr_series >= HR_MIN) & (hr_series <= HR_MAX)
            ]

            rows.append({
                "row_id": i,
                "subject_id": metadata.iloc[i]["subject_id"],
                "part": metadata.iloc[i]["part"],
                "window_id": metadata.iloc[i]["window_id"],
                "success_peak_detection": result["success"],
                "used_fallback": used_fallback,
                "polarity": result["polarity"],
                "num_peaks": len(result["peaks"]),
                "mean_hr": float(np.mean(valid_hr_values)) if len(valid_hr_values) > 0 else np.nan,
                "std_hr": float(np.std(valid_hr_values)) if len(valid_hr_values) > 0 else np.nan,
                "min_hr": float(np.min(valid_hr_values)) if len(valid_hr_values) > 0 else np.nan,
                "max_hr": float(np.max(valid_hr_values)) if len(valid_hr_values) > 0 else np.nan,
            })

            if (i + 1) % 1000 == 0:
                print(f"{condition}: processed {i + 1}/{n_windows}")

        out_condition = OUTPUT_ROOT / condition
        out_condition.mkdir(parents=True, exist_ok=True)

        hr_path = out_condition / "HR_X.npy"
        summary_path = out_condition / "HR_extraction_summary.csv"

        np.save(hr_path, hr_data)

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSaved HR data:")
        print(hr_path)
        print(f"HR shape: {hr_data.shape}")

        print("\nSaved summary:")
        print(summary_path)

        print("\nPeak detection summary:")
        print("Total windows:", len(summary_df))
        print("Successful peak detection:", int(summary_df["success_peak_detection"].sum()))
        print("Used fallback:", int(summary_df["used_fallback"].sum()))
        print("Success ratio:", summary_df["success_peak_detection"].mean())

        print("\nMean HR summary:")
        print(summary_df["mean_hr"].describe())

        print()

    print("Ολοκληρώθηκε η δημιουργία HR_X.npy για όλα τα conditions.")


if __name__ == "__main__":
    main()