# Διπλωματική Εργασία – BIRAFFE2 Affective GNN

Το repository περιέχει τον βασικό κώδικα, τα δεδομένα που χρησιμοποιήθηκαν για την κατασκευή του graph dataset και τα κύρια αποτελέσματα των πειραμάτων της διπλωματικής εργασίας.

Η εργασία αφορά την αναγνώριση συναισθηματικής κατάστασης από πολυτροπικά φυσιολογικά σήματα του BIRAFFE2 dataset, με χρήση Graph Neural Networks και κλασικών μεθόδων Machine Learning.

---

# Δομή Repository

## `src/`

Περιέχει τον κώδικα που χρησιμοποιήθηκε για όλη τη διαδικασία.

### `src/preprocessing/`

#### `pipeline.py`

Κώδικας preprocessing των σημάτων.

Περιλαμβάνει:

- διαχωρισμό των φάσεων μέσω των timestamps του Procedure file
- detrending
- resampling στα 50 Hz
- filtering
- standardization
- διαχωρισμό σε παράθυρα διάρκειας 18 sec

Κάθε window περιέχει 900 samples ανά signal.

---

### `src/features/`

Περιέχει τον κώδικα για την εξαγωγή των features.

#### `pca50_features.py`

Εφαρμόζει PCA ξεχωριστά σε κάθε signal και διατηρεί 50 PCA components.

#### `ecg_to_hr.py`

Μετατρέπει το ECG σε Heart Rate μέσω:

- ανίχνευσης R-peaks
- υπολογισμού RR intervals
- υπολογισμού HR
- interpolation ώστε να δημιουργηθεί HR time series για κάθε window

#### `hr_pca.py`

Εφαρμόζει PCA στο HR signal και διατηρεί 50 components.

---

### `src/labels/`

Περιέχει τον κώδικα δημιουργίας και προετοιμασίας των labels.

#### `labels.py`

Δημιουργεί τα Valence και Arousal labels από τα πεδία:

- `ANS-VALENCE`
- `ANS-AROUSAL`

του Procedure file.

#### `prepare_two_task_splits.py`

Δημιουργεί τα δύο τελικά classification tasks:

**Task 1 – PRE/POST**

- Valence
- Arousal

**Task 2 – GAME**

- Positive Affect
- Negative Affect

Τα GAME labels υπολογίζονται από τα GEQ δεδομένα.

---

### `src/graphs/`

Περιέχει τον κώδικα κατασκευής του graph dataset.

#### `graph_dataset_with_hr.py`

Δημιουργεί ένα graph για κάθε 18-second window.

Κάθε graph περιέχει 8 nodes:

1. HR
2. EDA
3. GYR-X
4. GYR-Y
5. GYR-Z
6. ACC-X
7. ACC-Y
8. ACC-Z

Κάθε node έχει 50 PCA features.

Άρα:

`x.shape = [8, 50]`

Τα edges δημιουργούνται με βάση την Pearson correlation μεταξύ των raw signals.

Δημιουργείται edge όταν:

`|r| > 0.3`

Η τιμή της correlation αποθηκεύεται στο `edge_attr`.

Κάθε graph περιέχει επίσης:

- `subject_id`
- `condition`
- `window_id`
- timestamps
- label `y`

#### `create_subject_level_split.py`

Δημιουργεί τα Train / Validation / Test splits σε επίπεδο subject.

Δεν υπάρχει overlap subjects μεταξύ των splits.

Η συνολική αναλογία είναι περίπου:

- Train: 56%
- Validation: 24%
- Test: 20%

---

### `src/models/`

Περιέχει τους κώδικες εκπαίδευσης και αξιολόγησης των μοντέλων.

#### `train_weighted_gcn_two_task.py`

Weighted GCN για τα δύο binary outputs.

Χρησιμοποιήθηκε σε πειράματα:

- χωρίς `pos_weight`
- με `pos_weight`

#### `train_weighted_gcn_two_tasks_sampler.py`

Weighted GCN με `WeightedRandomSampler` για αντιμετώπιση του class imbalance.

#### `train_classical_baselines_binary.py`

Κλασικά Machine Learning baselines για σύγκριση με το GCN:

- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest
- MLP

Στα classical models τα node features μετατρέπονται από:

`[8, 50]`

σε:

`[400]`

και δεν χρησιμοποιείται η graph connectivity.

---

# `data/`

Περιέχει τα βασικά δεδομένα που ανέβηκαν για έλεγχο της τελικής διαδικασίας.

## `data/graph/`

### `graph_dataset_hr.pt`

Το τελικό graph dataset.

Περιέχει τα graph objects που χρησιμοποιήθηκαν στα επόμενα στάδια.

---

## `data/labels/prepost/`

Περιέχει τα Valence/Arousal labels για PRE και POST:

- `PRE_labels.npy`
- `PRE_labels_metadata.csv`
- `POST_labels.npy`
- `POST_labels_metadata.csv`

Τα metadata αρχεία περιέχουν την αντιστοίχιση των labels με subjects και windows.

---

## `data/labels/game/`

### `game_geq_pa_na_labels.csv`

Περιέχει τα τελικά GAME labels:

- Positive Affect score
- Negative Affect score
- Positive Affect binary label
- Negative Affect binary label

για τα EASY, HARD και OPTIMAL conditions.

---

## `data/split/`

### `subject_split_subjects.json`

Περιέχει τη λίστα των subjects που χρησιμοποιήθηκαν στα:

- Train
- Validation
- Test

splits.

---

# `results/`

Περιέχει τα βασικά αποτελέσματα των τελικών πειραμάτων.

## `results/prepost/`

Αποτελέσματα για:

**PRE/POST → Valence / Arousal**

Περιλαμβάνονται:

- `gcn_posweight.txt`
- `gcn_no_posweight.txt`
- `gcn_sampler.txt`
- `classical_baselines.csv`

---

## `results/game/`

Αποτελέσματα για:

**GAME → Positive Affect / Negative Affect**

Περιλαμβάνονται:

- `gcn_posweight.txt`
- `gcn_no_posweight.txt`
- `gcn_sampler.txt`
- `classical_baselines.csv`

---

# Συνολική Ροή

Η βασική διαδικασία που ακολουθήθηκε είναι:

Raw Signals  
↓  
Preprocessing  
↓  
18-second Windows  
↓  
PCA Feature Extraction  
↓  
ECG → HR  
↓  
Graph Construction  
↓  
Subject-level Train / Validation / Test Split  
↓  
PRE/POST και GAME Tasks  
↓  
Weighted GCN / Classical ML  
↓  
Evaluation