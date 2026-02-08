"""Baseline logistic regression: EEG band-power features → 5-class intent.

Exports:
  - models/baseline_logreg.joblib  (trained model + scaler)
  - prints classification report + confusion matrix
"""
import numpy as np, glob, json, time, os, warnings
from numpy.fft import rfft, rfftfreq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

# ── Config ──
DATA_DIR = "data/robot_control/data"
FS = 500
STIM_ONSET = 3.0  # seconds
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
CH_NAMES = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
LABEL_MAP = {"Right Fist": "right", "Left Fist": "left", "Both Fists": "both", "Tongue Tapping": "tongue", "Relax": "rest"}
MODEL_PATH = "models/baseline_logreg.joblib"

# ── Feature extraction ──
def extract_features(eeg):
    """Extract band-power + stats features from stimulus window of EEG.
    
    Returns feature vector of length 6 channels × (5 bands + 2 stats) = 42.
    """
    cut = int(STIM_ONSET * FS)
    stim = eeg[cut:]  # stimulus period only

    # Skip if NaN
    if np.isnan(stim).any():
        stim = np.nan_to_num(stim, nan=0.0)

    feats = []
    freqs = rfftfreq(stim.shape[0], d=1 / FS)
    for ch in range(stim.shape[1]):
        sig = stim[:, ch]
        spec = np.abs(rfft(sig)) ** 2

        # Band powers (log10)
        for lo, hi in BANDS.values():
            mask = (freqs >= lo) & (freqs <= hi)
            bp = spec[mask].mean()
            feats.append(np.log10(bp + 1e-12))

        # Time-domain stats
        feats.append(sig.mean())
        feats.append(sig.std())

    return np.array(feats, dtype=np.float32)


def feature_names():
    names = []
    for ch in CH_NAMES:
        for band in BANDS:
            names.append(f"{ch}_{band}")
        names.append(f"{ch}_mean")
        names.append(f"{ch}_std")
    return names


# ── Load dataset ──
print("Loading dataset...")
files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
X, y, groups = [], [], []

for f in files:
    arr = np.load(f, allow_pickle=True)
    lab = arr["label"].item()
    label = LABEL_MAP.get(lab["label"])
    if label is None:
        continue
    X.append(extract_features(arr["feature_eeg"]))
    y.append(label)
    groups.append(lab["subject_id"])

X = np.stack(X)
y = np.array(y)
groups = np.array(groups)
print(f"Loaded {len(X)} samples, {len(set(groups))} subjects, {X.shape[1]} features")

# ── Train with subject-wise cross-validation ──
le = LabelEncoder()
y_enc = le.fit_transform(y)

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_preds = np.zeros_like(y_enc)

print("Training (5-fold subject-wise CV)...")
t0 = time.time()

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_enc, groups)):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
    ])
    pipe.fit(X[train_idx], y_enc[train_idx])
    all_preds[val_idx] = pipe.predict(X[val_idx])

train_time = time.time() - t0
print(f"CV done in {train_time:.1f}s")

# ── Results ──
target_names = le.classes_.tolist()
print("\n=== Classification Report (subject-wise CV) ===")
print(classification_report(y_enc, all_preds, target_names=target_names))

cm = confusion_matrix(y_enc, all_preds)
print("Confusion Matrix:")
print(f"{'':>8}", "  ".join(f"{n:>6}" for n in target_names))
for i, row in enumerate(cm):
    print(f"{target_names[i]:>8}", "  ".join(f"{v:>6}" for v in row))

# ── Train final model on all data + export ──
print("\nTraining final model on all data...")
final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
])
final_pipe.fit(X, y_enc)

os.makedirs("models", exist_ok=True)
export = {
    "pipeline": final_pipe,
    "label_encoder": le,
    "feature_names": feature_names(),
    "label_map": LABEL_MAP,
    "config": {"fs": FS, "stim_onset": STIM_ONSET, "bands": BANDS, "n_channels": 6},
}
joblib.dump(export, MODEL_PATH)
print(f"Saved → {MODEL_PATH}")

# ── Demo predict ──
def predict(eeg_window):
    """Predict intent from raw EEG window. Returns JSON-compatible dict."""
    t_start = time.time()
    feats = extract_features(eeg_window).reshape(1, -1)
    proba = final_pipe.predict_proba(feats)[0]
    pred_idx = proba.argmax()
    latency = (time.time() - t_start) * 1000

    classes = le.classes_.tolist()
    return {
        "predicted_class": classes[pred_idx],
        "confidence": round(float(proba[pred_idx]), 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "is_above_threshold": float(proba[pred_idx]) > 0.5,
        "timestamp_ms": int(time.time() * 1000),
        "inference_latency_ms": round(latency, 2),
    }

# Test on one sample
sample = np.load(files[0], allow_pickle=True)["feature_eeg"]
result = predict(sample)
print(f"\n=== Sample Prediction ===")
print(json.dumps(result, indent=2))
