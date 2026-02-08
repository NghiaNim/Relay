# ThoughtLink — EEG Pipeline Engineering Specification

**From Brain to Robot, at the Speed of Thought**

*Riemannian Geometry + SVM + Hysteresis*
*Non-invasive EEG Intent Decoding for Real-Time Robot Control*

VC Track — Supported by Kernel & Dimensional

---

## 1. Architecture Overview

This document specifies the complete EEG-based intent decoding pipeline for the ThoughtLink project. The system decodes motor imagery signals from non-invasive EEG recordings and maps them to discrete robot commands in real time.

### 1.1 System Summary

The pipeline consists of five core stages: data loading, preprocessing, Riemannian feature extraction, SVM classification, and a hysteresis-based state machine for stable command output.

| Stage | Method | Output |
|---|---|---|
| 1. Data Loading | Parse .npz files, extract EEG & labels | Raw EEG (7499 × 6) + class labels per sample |
| 2. Preprocessing | Band-pass filter (8–30Hz), windowing | Windows of shape (250 × 6), ~85k training samples |
| 3. Features | Covariance matrix + Riemannian tangent space | Feature vectors of 21 dimensions per window |
| 4. Classification | Linear SVM with probability estimates | 5-class prediction + confidence scores |
| 5. Control | Hysteresis state machine | Stable robot commands (no jitter) |

### 1.2 Label-to-Command Mapping

The dataset contains five motor task labels. Each is mapped to a discrete robot command:

| Brain Signal (Label) | Robot Command | Note |
|---|---|---|
| Left Fist | MOVE LEFT | Dataset may say 'Left First' (typo) |
| Right Fist | MOVE RIGHT | |
| Both Fists | MOVE FORWARD | Dataset may say 'Both Firsts' (typo) |
| Tongue Tapping | MOVE BACKWARD | |
| Relax | STOP | Entire recording is rest state |

---

## 2. Data Loading & Exploration

Load all 1,000 .npz files from the dataset. Each file represents a 15-second chunk of data recorded using a Kernel Flow headset.

### 2.1 File Structure

Each .npz file contains three keys:

- **feature_eeg:** Shape (7499, 6). EEG signal at 500Hz across 6 channels. Values in microvolts (µV).
- **feature_moments:** Shape (72, 40, 3, 2, 3). TD-NIRS data (reserved for Phase 2).
- **label:** Dictionary with label, subject_id, session_id, and duration.

### 2.2 EEG Channel Layout

The 6 EEG channels and their 10-20 system positions:

| Ch 0 | Ch 1 | Ch 2 | Ch 3 | Ch 4 | Ch 5 |
|---|---|---|---|---|---|
| AFF6 | AFp2 | AFp1 | AFF5 | FCz | CPz |

FCz and CPz are located over the motor cortex and are the most critical channels for motor imagery classification.

### 2.3 Timing Structure

Each 15-second recording follows this timing structure:

- **t = 0s to 3s:** Rest period. Subject is idle.
- **t = 3s:** Stimulus presented. Subject begins motor task.
- **t = 3s + duration:** Stimulus removed. Subject returns to rest.

The duration field varies per trial (typically 5–12 seconds). Only the active stimulus window is used for training.

### 2.4 Data Catalog

After loading, catalog the following: number of samples per class (expect ~200 each), number of unique subjects (subject_id), and number of sessions per subject (session_id). This information is needed for stratified cross-validation splitting.

---

## 3. Preprocessing

### 3.1 Extract Active Segment

For each recording, crop the EEG to the active stimulus window only. The pre-stimulus rest period (0–3s) adds noise to covariance estimates and must be discarded.

```python
start_sample = int(3.0 * 500)        # = 1500
end_sample = int((3.0 + duration) * 500)
active_eeg = eeg[start_sample:end_sample]
```

**Exception for Relax class:** The entire recording is rest. Use a matched-length window from the middle of the recording to maintain consistent window sizes across classes.

### 3.2 Band-Pass Filter

Apply a 4th-order Butterworth causal band-pass filter to isolate the motor imagery frequency bands:

- **Low cutoff: 8Hz** (mu rhythm lower bound)
- **High cutoff: 30Hz** (beta rhythm upper bound)

This removes slow drift below 8Hz and high-frequency noise above 30Hz. The mu (8–12Hz) and beta (13–30Hz) rhythms are where motor imagery activity is concentrated, particularly at the FCz and CPz electrodes.

**Important:** Use `scipy.signal.butter` with `sosfilt` (not `filtfilt`). Causal filtering is required for real-time compatibility since `filtfilt` uses future data that is unavailable during live inference.

### 3.3 Windowing

Slice each active segment into overlapping windows. This serves two purposes: it generates multiple training samples per recording (data augmentation) and mirrors the real-time sliding window approach.

#### Default Parameters

| Parameter | Value | Samples at 500Hz |
|---|---|---|
| Window size | 0.5 seconds | 250 samples |
| Slide (step) | 0.1 seconds | 50 samples |
| Overlap | 80% | 200 samples |

A typical 9-second active segment yields approximately 85 windows. Across all 1,000 recordings, this produces roughly 85,000 training samples from the original 1,000. Each window inherits the label of its parent recording.

**These parameters are configurable.** Testing should compare multiple window sizes (0.3s, 0.5s, 1.0s) to quantify the latency-accuracy tradeoff.

---

## 4. Feature Extraction — Riemannian Geometry

Rather than using raw EEG values or hand-crafted frequency features, we represent each window as a spatial covariance matrix and operate on it using Riemannian geometry. This approach is robust to noise, works well with few channels, and generalizes across subjects.

### 4.1 Covariance Matrix Computation

For each window of shape (250, 6), compute the spatial covariance matrix:

```python
cov = np.cov(window.T)            # shape (6, 6)
cov = cov + 1e-6 * np.eye(6)      # regularization
```

The resulting 6×6 symmetric matrix captures how each pair of EEG channels co-varies during that window. Different motor tasks produce different covariance patterns because they activate different regions of the motor cortex.

**Regularization:** Adding a small identity matrix (1e-6) ensures the covariance stays strictly positive definite, which is required for Riemannian operations. Without this, ill-conditioned matrices from short windows or low-variance channels can cause numerical errors.

### 4.2 Why Riemannian Geometry

Covariance matrices are symmetric positive definite (SPD) matrices. They do not live in flat Euclidean space — they live on a curved surface called a manifold. Standard operations like averaging or measuring distance produce incorrect results if applied in Euclidean space.

Riemannian geometry provides the correct mathematical framework for operating on this manifold: proper distance metrics, proper means, and proper projections.

### 4.3 Tangent Space Projection

To use standard classifiers like SVM, we need to project the curved manifold data into flat Euclidean space. This is done via tangent space projection:

1. Compute the Riemannian mean (geometric mean) of all training covariance matrices. This becomes the reference point.
2. Project each covariance matrix onto the tangent plane at the reference point using a logarithmic map.
3. Extract the upper triangle of the projected matrix as a flat feature vector. For a 6×6 matrix, this yields 21 features.

```python
from pyriemann.tangentspace import TangentSpace

ts = TangentSpace()
ts.fit(all_train_covariances)         # learns reference point
features = ts.transform(covariances)  # shape (n, 21)
```

### 4.4 What Gets Saved

- **The TangentSpace object** (contains the Riemannian reference point, needed during real-time inference)
- **The feature matrix** (shape n_samples × 21) and corresponding labels for training

---

## 5. Classification — SVM

### 5.1 Train/Test Split by Subject

Do not split randomly. Group by subject_id and use leave-one-subject-out cross-validation. This is the only honest evaluation for BCI because it tests whether the model generalizes to brains it has never seen.

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
groups = subject_ids  # one per window, inherited from parent
```

**Why not 80/20 random split?** If the same subject appears in both train and test sets, the model memorizes individual brain patterns rather than learning generalizable motor imagery features. Accuracy will be inflated and will not reflect real-world performance.

#### Recommended Split for Hackathon (20 subjects)

```
Subjects:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 | 18 19 20
           |←──────── train + validation (17) ──────────────→|  |← test →|
```

- **Step 1:** Hold out 3 subjects completely (18, 19, 20). Never touch until final evaluation.
- **Step 2:** Use remaining 17 for training + validation via Leave-One-Subject-Out (17 folds). Use this to tune hyperparameters.
- **Step 3:** Once best parameters are found, retrain on all 17 subjects.
- **Step 4:** Final evaluation on the 3 held-out subjects. This is your honest, never-seen accuracy number.

### 5.2 SVM Training

Use a linear Support Vector Machine with probability estimates enabled:

```python
from sklearn.svm import SVC

clf = SVC(kernel='linear', C=1.0, probability=True)
clf.fit(X_train, y_train)
```

**`probability=True` is critical** — the hysteresis state machine requires confidence scores, not just class labels. Without probabilities, we cannot implement threshold-based triggering and release.

Also test `kernel='rbf'` and compare. If linear SVM achieves similar accuracy, keep it — linear SVM prediction is a single dot product and is significantly faster at inference.

### 5.3 Evaluation Metrics

Report the following per fold and averaged across all folds:

- Overall accuracy (percentage of correct predictions)
- Per-class precision, recall, and F1-score
- Confusion matrix (critical for identifying which commands get confused with each other)
- Inference time per prediction in milliseconds

---

## 6. Hysteresis State Machine

This is the real-time control layer that prevents jitter. It sits between the SVM predictions and the robot commands, ensuring that momentary noise in the EEG signal does not cause the robot to oscillate between states.

### 6.1 Core Concept

Hysteresis means it is harder to leave a state than to enter it. The system uses two separate confidence thresholds: a higher trigger threshold to start a command, and a lower release threshold to stop it. The gap between them is the hysteresis band.

### 6.2 State Machine Definition

The system has five possible states: IDLE, LEFT, RIGHT, FORWARD, and BACKWARD.

#### Transition Rules

1. **IF current state is IDLE:** Transition to a command state when the maximum confidence score exceeds the trigger threshold (0.70).
2. **IF current state is a command:** Return to IDLE only when confidence for the current command drops below the release threshold (0.35).
3. **IF in the hysteresis band:** Confidence is between 0.35 and 0.70. The system maintains its current state, preventing flicker.

### 6.3 Configurable Thresholds

| Parameter | Default | Description |
|---|---|---|
| TRIGGER_THRESHOLD | 0.70 | Confidence required to START a command |
| RELEASE_THRESHOLD | 0.35 | Confidence must drop below this to STOP |
| MIN_HOLD_TIME | 200ms | Minimum time before state can change |
| COOLDOWN | 150ms | Wait time after release before accepting new trigger |

**Wider hysteresis band (larger gap) =** more stable output but slower to release commands.

**Narrower hysteresis band =** faster release but more susceptible to jitter.

---

## 7. Real-Time Inference Loop

This section describes the complete real-time loop that ties all components together into a continuous inference system.

### 7.1 Sliding Window in Real Time

The system maintains a ring buffer of the most recent EEG samples. Every 50ms (the slide interval), it grabs the latest 0.5s window and runs the full pipeline:

1. Grab the latest 250 samples (0.5s) from the ring buffer
2. Apply causal band-pass filter (8–30Hz)
3. Compute 6×6 covariance matrix + regularization
4. Project into tangent space using saved reference point (yields 21 features)
5. Run SVM prediction to get 5-class confidence scores
6. Pass scores through hysteresis state machine
7. If state changed, send new command to robot

### 7.2 Latency Budget

| Step | Time |
|---|---|
| Grab window from buffer | ~0ms (already in memory) |
| Band-pass filter | ~1ms |
| Compute 6×6 covariance | ~0.1ms |
| Riemannian tangent space projection | ~0.1ms |
| SVM dot product | ~0.05ms |
| Hysteresis logic | ~0.01ms |
| **TOTAL per prediction** | **~1–2ms** |

With a 50ms slide, the system produces 20 predictions per second. Each prediction takes only 1–2ms of computation, leaving 48ms idle per cycle.

### 7.3 End-to-End Latency

From the moment a person begins a motor task to the robot receiving a command:

| Component | Latency |
|---|---|
| Brain signal formation (biology) | ~100–200ms |
| First full window available | ~500ms (window size) |
| Hysteresis stabilization | ~100–200ms |
| **TOTAL end-to-end** | **~300–500ms** |

This is faster than typical human reaction time (~250ms for simple motor response), meaning the system operates at effectively real-time speed for robot control applications.

---

## 8. Saved Artifacts & Trained Components

After training, the following components are serialized and loaded during real-time inference.

### Fixed After Training (loaded at startup)

| Artifact | Description |
|---|---|
| tangent_space.pkl | Fitted TangentSpace object containing the Riemannian reference point (geometric mean of training covariances) |
| svm_model.pkl | Trained SVM with learned weight vectors and bias terms |
| filter_coefficients | Butterworth filter second-order sections for band-pass (8–30Hz) |
| config.py | All configurable parameters: window size, slide, thresholds, etc. |

### Recomputed Every Window (live data)

| Component | Description |
|---|---|
| Covariance matrix | Fresh 6×6 matrix from the latest EEG window |
| Tangent space features | 21-dimensional vector projected relative to saved reference point |
| Confidence scores | 5-class probability distribution from SVM |
| Hysteresis state | Current state (IDLE/LEFT/RIGHT/FORWARD/BACKWARD) and timers |

---

## 9. Project Structure

```
project/
├── config.py               # All configurable parameters
├── data_loader.py           # Load .npz files, parse labels, catalog subjects
├── preprocessing.py         # Band-pass filter, windowing, segment extraction
├── feature_extraction.py    # Covariance computation, Riemannian tangent space
├── train.py                 # SVM training with leave-subject-out CV
├── hysteresis.py            # State machine with configurable thresholds
├── realtime_loop.py         # Inference loop connecting all components
├── evaluate.py              # Metrics, confusion matrix, latency benchmarks
└── models/
    ├── tangent_space.pkl    # Fitted Riemannian reference
    └── svm_model.pkl        # Trained SVM
```

### 9.1 Configuration (config.py)

| Parameter | Default | Notes |
|---|---|---|
| WINDOW_SIZE | 0.5s | Window duration in seconds |
| SLIDE | 0.1s | Step between consecutive windows |
| FILTER_LOW | 8 Hz | Band-pass lower cutoff |
| FILTER_HIGH | 30 Hz | Band-pass upper cutoff |
| SVM_KERNEL | linear | Also test 'rbf' for comparison |
| TRIGGER_THRESHOLD | 0.70 | Confidence to enter a command state |
| RELEASE_THRESHOLD | 0.35 | Confidence to return to IDLE |
| MIN_HOLD_TIME | 0.2s | Prevents ultra-short accidental triggers |
| COOLDOWN | 0.15s | Wait time after release |

---

## 10. Hackathon Deliverables

### Deliverable 1: Offline Evaluation Notebook

1. Complete data loading, preprocessing, feature extraction pipeline
2. Leave-one-subject-out cross-validation results with accuracy and confusion matrix
3. Comparison across window sizes (0.3s, 0.5s, 1.0s) showing latency-accuracy tradeoff
4. Latency benchmarks (milliseconds per prediction)

### Deliverable 2: Real-Time Simulation Demo

1. Full inference loop connected to the humanoid robot simulation
2. Live visualization of EEG window, SVM confidence bars, hysteresis state, and robot response
3. Demonstration of stable command output without jitter

### Deliverable 3: Analysis & Documentation

1. Latency vs accuracy tradeoff chart across different window sizes
2. Hysteresis parameter sweep showing stability improvement
3. Documented failure modes and open research questions
4. Comparison of linear vs RBF kernel performance

---

## 11. Phase 2 — NIRS Integration (Future Work)

The EEG pipeline is designed for clean extensibility. When ready, NIRS can be added as a parallel confirmation layer without modifying any existing EEG components.

### NIRS Role: Background Safety Monitor

- EEG triggers robot commands immediately (fast path, ~300–500ms)
- NIRS monitors hemodynamic response in the background (slow path, ~3–5s)
- If NIRS contradicts EEG for more than 2 seconds, issue an emergency stop
- NIRS uses standard feature extraction (statistical summaries) + separate SVM. No Riemannian geometry required.

This dual-modality architecture addresses the false trigger suppression and scalability evaluation criteria, making the system suitable for supervising large fleets of robots.
