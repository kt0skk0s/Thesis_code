import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import warnings
import time

# Αγνοούμε τα warnings για καθαρή κονσόλα
warnings.filterwarnings("ignore")

# ==========================================
# 1. ΡΥΘΜΙΣΕΙΣ (ΓΙΑ ΟΛΟ ΤΟ DATASET)
# ==========================================
# Το βασικό folder που περιέχει ΟΛΟ το dataset
ROOT_DIR = Path(r"C:\Users\user\Dropbox\PC\Desktop\BIRAFFE2_DATASET") 

# Φάκελος εξόδου (Base folder)
OUTPUT_DIR = Path(r"E:\BIRAFFE2_DATASET\PROCESSED_FINAL")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Παράμετροι Επεξεργασίας
FS_TARGET = 50.0      # Στόχος: 50 Hz
NYQ = FS_TARGET / 2.0
NUMTAPS = 33          # Φίλτρο Order 32

# Παράμετροι Windowing
WINDOW_SECONDS = 18
SAMPLES_PER_WINDOW = int(WINDOW_SECONDS * FS_TARGET)  # 900 samples

print(f"Ξεκινάω MASTER Pipeline (Organized Folders)")
print(f"Input: {ROOT_DIR}")
print(f"Output Base: {OUTPUT_DIR}\n")

# ==========================================
# 2. ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ==========================================

def get_timestamps(proc_path):
    if not proc_path.exists():
        return None
    
    # Διάβασμα με ; και διόρθωση κεφαλαίων
    try:
        proc = pd.read_csv(proc_path, sep=";")
        proc.columns = proc.columns.str.strip().str.upper()
        
        events_map = {
            "BASELINE": ["BASELINE START", "BASELINE END"],
            "PRE_AVS":  ["STIMULI PART 1 START", "STIMULI PART 1 END"],
            "GAME":     ["GAME START", "GAME END"],
            "POST_AVS": ["STIMULI PART 2 START", "STIMULI PART 2 END"]
        }
        
        times = {}
        for phase, (start_evt, end_evt) in events_map.items():
            try:
                t0 = proc.loc[proc["EVENT"] == start_evt, "TIMESTAMP"].values[0]
                t1 = proc.loc[proc["EVENT"] == end_evt, "TIMESTAMP"].values[0]
                times[phase] = (t0, t1)
            except (IndexError, KeyError):
                times[phase] = None
        return times
    except Exception as e:
        print(f"   ⚠️ Error reading timestamps: {e}")
        return None

def apply_filters(data_series, signal_type):
    # Αντικατάσταση NaN με forward fill και μετά 0
    data_clean = data_series.fillna(method='ffill').fillna(0)
    
    if signal_type == "EDA":
        # Lowpass 2.0 Hz
        taps = signal.firwin(NUMTAPS, 2.0 / NYQ, pass_zero=True)
        return signal.lfilter(taps, 1.0, data_clean)
    else:
        # Highpass 0.05 Hz
        taps = signal.firwin(NUMTAPS, 0.05 / NYQ, pass_zero=False)
        return signal.lfilter(taps, 1.0, data_clean)

def process_segment(df_segment, signal_types, target_len):
    processed_data = {}
    
    # Χρόνος
    t_start = df_segment["TIMESTAMP"].iloc[0]
    t_end = df_segment["TIMESTAMP"].iloc[-1]
    processed_data["TIMESTAMP"] = np.linspace(t_start, t_end, target_len)
    
    for col, s_type in signal_types.items():
        if col not in df_segment.columns:
            continue
            
        # 1. Detrend
        series_clean = df_segment[col].fillna(method='ffill').fillna(0)
        detrended = signal.detrend(series_clean)
        
        # 2. Resample
        resampled = signal.resample(detrended, target_len)
        
        # 3. Filter
        filtered = apply_filters(pd.Series(resampled), s_type)
        
        # 4. Standardize (Z-score)
        std_val = np.std(filtered)
        if std_val != 0:
            standardized = (filtered - np.mean(filtered)) / std_val
        else:
            standardized = filtered
            
        processed_data[col] = standardized
        
    return pd.DataFrame(processed_data)

def add_windows(df):
    windowed_dfs = []
    num_windows = 0
    total_samples = len(df)
    
    for start_idx in range(0, total_samples, SAMPLES_PER_WINDOW):
        end_idx = start_idx + SAMPLES_PER_WINDOW
        
        if end_idx <= total_samples:
            win = df.iloc[start_idx:end_idx].copy()
            win["WINDOW_ID"] = num_windows
            windowed_dfs.append(win)
            num_windows += 1
            
    if windowed_dfs:
        return pd.concat(windowed_dfs, ignore_index=True)
    return pd.DataFrame()

# ==========================================
# 3. ΚΥΡΙΑ ΔΙΑΔΙΚΑΣΙΑ
# ==========================================
def main():
    # Βρίσκουμε όλα τα αρχεία Procedure
    proc_folder = ROOT_DIR / "metadata" / "BIRAFFE2-procedure"
    all_procs = sorted(list(proc_folder.glob("SUB*-Procedure.csv")))
    
    total_subs = len(all_procs)
    print(f"📊 Βρέθηκαν {total_subs} Subjects. Ξεκινάω...\n")
    
    start_time = time.time()
    
    for i, proc_path in enumerate(all_procs, 1):
        # Εξαγωγή ονόματος (π.χ. SUB100)
        sub_name = proc_path.stem.split("-")[0]
        
        # --- Δημιουργία φακέλου για τον χρήστη ---
        sub_output_folder = OUTPUT_DIR / sub_name
        sub_output_folder.mkdir(parents=True, exist_ok=True)
       
        
        print(f"[{i}/{total_subs}] Επεξεργασία: {sub_name}...", end=" ")
        
        # Paths
        path_bio = ROOT_DIR / "biosignals" / "BIRAFFE2-biosigs" / f"{sub_name}-BioSigs.csv"
        path_gp = ROOT_DIR / "gamepad" / "BIRAFFE2-gamepad" / f"{sub_name}-Gamepad.csv"
        
        if not path_bio.exists():
            print("❌ Skip (Missing BioSigs)")
            continue
            
        try:
            # 1. Timestamps
            times = get_timestamps(proc_path)
            if not times:
                print("⚠️ Skip (Bad Procedure)")
                continue

            # 2. Raw Data Load
            df_bio = pd.read_csv(path_bio)
            df_bio.columns = df_bio.columns.str.strip().str.upper()
            
            # Gamepad Load
            has_gp = False
            df_gp = pd.DataFrame()
            if path_gp.exists():
                try:
                    df_gp = pd.read_csv(path_gp, sep=";")
                    df_gp.columns = df_gp.columns.str.strip().str.upper()
                    has_gp = True
                except:
                    pass 

            # 3. Baseline Logic
            mean_ecg, mean_eda = 0, 0
            if times.get("BASELINE"):
                t0, t1 = times["BASELINE"]
                base_seg = df_bio[(df_bio["TIMESTAMP"] >= t0) & (df_bio["TIMESTAMP"] <= t1)]
                if not base_seg.empty:
                    mean_ecg = base_seg["ECG"].mean()
                    mean_eda = base_seg["EDA"].mean()

            # 4. Process Phases
            saved_count = 0
            phases = ["BASELINE", "PRE_AVS", "GAME", "POST_AVS"]
            
            for phase in phases:
                if not times.get(phase):
                    continue
                    
                t0, t1 = times[phase]
                duration = t1 - t0
                if duration < WINDOW_SECONDS:
                    continue
                    
                target_samples = int(duration * FS_TARGET)
                
                # --- BioSigs ---
                seg_bio = df_bio[(df_bio["TIMESTAMP"] >= t0) & (df_bio["TIMESTAMP"] <= t1)].copy()
                if seg_bio.empty: continue
                
                seg_bio["ECG"] = seg_bio["ECG"] - mean_ecg
                seg_bio["EDA"] = seg_bio["EDA"] - mean_eda
                
                final_bio = process_segment(seg_bio, {"ECG": "OTHER", "EDA": "EDA"}, target_samples)
                
                # --- Gamepad ---
                final_gp = pd.DataFrame()
                if has_gp and not df_gp.empty:
                    seg_gp = df_gp[(df_gp["TIMESTAMP"] >= t0) & (df_gp["TIMESTAMP"] <= t1)].copy()
                    if not seg_gp.empty:
                        gp_cols = [c for c in seg_gp.columns if "ACC" in c or "GYR" in c]
                        final_gp = process_segment(seg_gp, {c: "OTHER" for c in gp_cols}, target_samples)

                # --- Merge & Window ---
                if not final_bio.empty:
                    full_df = final_bio
                    if not final_gp.empty:
                        final_gp = final_gp.drop(columns=["TIMESTAMP"], errors="ignore")
                        full_df = pd.concat([final_bio, final_gp], axis=1)
                        
                    windowed_df = add_windows(full_df)
                    
                    if not windowed_df.empty:
                        # --- ΑΠΟΘΗΚΕΥΣΗ ΣΤΟΝ ΦΑΚΕΛΟ ΤΟΥ ΧΡΗΣΤΗ ---
                        out_name = sub_output_folder / f"{sub_name}_{phase}_preprocessed.csv"
                        windowed_df.to_csv(out_name, index=False)
                        saved_count += 1

            print(f"✅ Done -> Saved in /{sub_name}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

    total_time = time.time() - start_time
    print(f"\n🎉 ΟΛΟΚΛΗΡΩΘΗΚΕ! Χρόνος: {total_time/60:.2f} λεπτά.")

if __name__ == "__main__":
    main()