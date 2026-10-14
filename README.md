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

### `src/graph/`

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

#### `train_single_target_gcn.py`

Εκπαιδεύει ξεχωριστό Graph Convolutional Network (GCN) για κάθε binary classification target.

Σε αντίθεση με την αρχική προσέγγιση του Weighted GCN, όπου ένα κοινό μοντέλο προβλέπει δύο outputs ταυτόχρονα, στη νέα προσέγγιση κάθε target προβλέπεται από ανεξάρτητο μοντέλο.

Χρησιμοποιούνται τα ίδια graph features και τα subject-independent Train / Validation / Test splits.

Τα classification targets είναι:

**PRE/POST:**

- Valence
- Arousal

**GAME:**

- Positive Affect
- Negative Affect

Συνολικά εκπαιδεύονται 4 ανεξάρτητα GCN models, ένα για κάθε target.

---

#### `train_single_target_gat.py`

Εκπαιδεύει ξεχωριστό Graph Attention Network (GAT) για κάθε binary classification target.

Το GAT χρησιμοποιεί attention mechanisms για να μαθαίνει τη σχετική σημασία των γειτονικών nodes κατά τη διαδικασία του message passing.

Η graph connectivity προκύπτει από την Pearson correlation, ενώ το GAT μαθαίνει τα attention coefficients μεταξύ των συνδεδεμένων nodes.

Χρησιμοποιούνται τα ίδια graph datasets και subject-independent splits με το single-target GCN.

Πραγματοποιούνται 4 ανεξάρτητα classification experiments:

- PRE/POST → Valence
- PRE/POST → Arousal
- GAME → Positive Affect
- GAME → Negative Affect

---

#### `train_single_target_gin.py`

Εκπαιδεύει ξεχωριστό Graph Isomorphism Network (GIN) για κάθε binary classification target.

Το GIN χρησιμοποιεί aggregation των χαρακτηριστικών των γειτονικών nodes και MLPs για την εκμάθηση των graph representations.

Η Pearson correlation χρησιμοποιείται για τον καθορισμό της graph topology.

Χρησιμοποιούνται τα ίδια graph datasets και subject-independent splits με τα GCN και GAT.

Πραγματοποιούνται 4 ανεξάρτητα classification experiments:

- PRE/POST → Valence
- PRE/POST → Arousal
- GAME → Positive Affect
- GAME → Negative Affect

---

### Single-target Classification

Στην αρχική προσέγγιση χρησιμοποιήθηκε ένα Weighted GCN με δύο outputs ανά classification task:

- PRE/POST → Valence και Arousal
- GAME → Positive Affect και Negative Affect

Στη συνέχεια εξετάστηκε η προσέγγιση single-target classification, όπου κάθε συναισθηματική μεταβλητή προβλέπεται ανεξάρτητα.

Για τη σύγκριση χρησιμοποιήθηκαν τρεις διαφορετικές GNN αρχιτεκτονικές:

- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GIN (Graph Isomorphism Network)

Κάθε αρχιτεκτονική εκπαιδεύεται ξεχωριστά για τα 4 targets, δίνοντας συνολικά 12 single-target training experiments.

Για την αξιολόγηση χρησιμοποιούνται τα ίδια subject-independent splits, με βασική μετρική σύγκρισης το Macro-F1.

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

## `data/labels/games/`

### `game_geq_pa_na_labels.csv`

Περιέχει τα τελικά GAME labels:

- Positive Affect score
- Negative Affect score
- Positive Affect binary label
- Negative Affect binary label

για τα EASY, HARD και OPTIMAL conditions.

---

## `data/labels/data/`

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
Weighted GCN / Classical ML / Single-target GCN, GAT, GIN  
↓  
Evaluation