"""Train multiple models for EEG intent classification.

Models:
  1. logreg   — Logistic Regression (baseline)
  2. rf       — Random Forest on same band-power features
  3. csp_lda  — CSP spatial filters + LDA (classic BCI)
  4. eegnet   — EEGNet compact CNN (if torch available)

All use subject-wise 5-fold CV. Exports to models/<name>.joblib
"""
import numpy as np, glob, json, time, os, warnings
from numpy.fft import rfft, rfftfreq
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

# ── Config ──
DATA_DIR = "../robot_control/data"
FS = 500
STIM_ONSET = 3.0
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
LABEL_MAP = {"Right Fist": "right", "Left Fist": "left", "Both Fists": "both",
             "Tongue Tapping": "tongue", "Relax": "rest"}


def extract_bandpower_features(eeg):
    """42-dim: 6ch × (5 bands + 2 stats)."""
    cut = int(STIM_ONSET * FS)
    stim = eeg[cut:]
    if np.isnan(stim).any():
        stim = np.nan_to_num(stim, nan=0.0)
    feats = []
    freqs = rfftfreq(stim.shape[0], d=1 / FS)
    for ch in range(stim.shape[1]):
        sig = stim[:, ch]
        spec = np.abs(rfft(sig)) ** 2
        for lo, hi in BANDS.values():
            mask = (freqs >= lo) & (freqs <= hi)
            feats.append(np.log10(spec[mask].mean() + 1e-12))
        feats.append(sig.mean())
        feats.append(sig.std())
    return np.array(feats, dtype=np.float32)


def extract_csp_epoch(eeg):
    """Return stimulus-period EEG matrix (samples, channels) for CSP."""
    cut = int(STIM_ONSET * FS)
    stim = eeg[cut:]
    if np.isnan(stim).any():
        stim = np.nan_to_num(stim, nan=0.0)
    return stim


# ── CSP implementation (simple, no mne dependency) ──
class SimpleCSP:
    """Common Spatial Patterns for multi-class via one-vs-rest."""
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    def _cov(self, X_class):
        """Average covariance across trials."""
        covs = []
        for trial in X_class:
            c = np.cov(trial.T)
            covs.append(c / np.trace(c))
        return np.mean(covs, axis=0)

    def fit(self, X, y):
        """X: list of (samples, channels) arrays. y: labels."""
        classes = np.unique(y)
        n_ch = X[0].shape[1]
        all_filters = []

        for cls in classes:
            idx_cls = np.where(y == cls)[0]
            idx_rest = np.where(y != cls)[0]
            cov_cls = self._cov([X[i] for i in idx_cls])
            cov_rest = self._cov([X[i] for i in idx_rest])

            # Solve generalized eigenvalue problem
            composite = cov_cls + cov_rest
            try:
                eigvals, eigvecs = np.linalg.eigh(np.linalg.solve(composite, cov_cls))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(np.eye(n_ch))

            # Take top and bottom components
            n = min(self.n_components // 2, n_ch // 2)
            idx_top = np.argsort(eigvals)[-n:][::-1]
            idx_bot = np.argsort(eigvals)[:n]
            sel = np.concatenate([idx_top, idx_bot])
            all_filters.append(eigvecs[:, sel])

        self.filters_ = np.hstack(all_filters)
        return self

    def transform(self, X):
        """X: list of (samples, channels) arrays → (n_trials, n_features)."""
        features = []
        for trial in X:
            projected = trial @ self.filters_
            # Log-variance features
            var = np.var(projected, axis=0)
            var = np.maximum(var, 1e-12)
            features.append(np.log(var))
        return np.array(features, dtype=np.float32)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ── Load dataset ──
print("Loading dataset...")
files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
bp_feats, epochs, labels, groups = [], [], [], []

for f in files:
    arr = np.load(f, allow_pickle=True)
    lab = arr["label"].item()
    label = LABEL_MAP.get(lab["label"])
    if label is None:
        continue
    eeg = arr["feature_eeg"]
    bp_feats.append(extract_bandpower_features(eeg))
    epochs.append(extract_csp_epoch(eeg))
    labels.append(label)
    groups.append(lab["subject_id"])

X_bp = np.stack(bp_feats)
y = np.array(labels)
groups = np.array(groups)
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Loaded {len(X_bp)} samples, {len(set(groups))} subjects")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(X_bp, y_enc, groups))

config = {"fs": FS, "stim_onset": STIM_ONSET, "bands": BANDS, "n_channels": 6}
os.makedirs("models", exist_ok=True)
results = {}


def train_and_eval(name, make_pipe, X, y_enc, splits):
    """Train with CV, print results, train final, save."""
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    all_preds = np.zeros_like(y_enc)
    t0 = time.time()
    for fold, (tr, val) in enumerate(splits):
        pipe = make_pipe()
        pipe.fit(X[tr], y_enc[tr])
        all_preds[val] = pipe.predict(X[val])
    cv_time = time.time() - t0

    acc = accuracy_score(y_enc, all_preds)
    print(f"CV Accuracy: {acc:.1%} (chance=20%) — {cv_time:.1f}s")
    print(classification_report(y_enc, all_preds, target_names=le.classes_.tolist()))

    # Final model on all data
    final = make_pipe()
    final.fit(X, y_enc)

    # Measure inference latency
    t0 = time.time()
    for _ in range(100):
        final.predict_proba(X[:1])
    latency = (time.time() - t0) / 100 * 1000

    export = {
        "pipeline": final,
        "label_encoder": le,
        "config": config,
        "cv_accuracy": round(acc, 4),
        "inference_ms": round(latency, 2),
    }
    path = f"models/{name}.joblib"
    joblib.dump(export, path)
    print(f"Saved → {path} (latency: {latency:.2f}ms)")

    results[name] = {"accuracy": round(acc, 4), "latency_ms": round(latency, 2)}
    return final


# ── 1. Logistic Regression (baseline) ──
train_and_eval("logreg", lambda: Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"))
]), X_bp, y_enc, splits)

# ── 2. Random Forest ──
train_and_eval("rf", lambda: Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
]), X_bp, y_enc, splits)

# ── 3. CSP + LDA ──
print(f"\n{'='*50}")
print("Training: csp_lda")
print(f"{'='*50}")

# CSP needs epoch-level data
all_preds_csp = np.zeros_like(y_enc)
t0 = time.time()
for fold, (tr, val) in enumerate(splits):
    csp = SimpleCSP(n_components=4)
    X_csp_tr = csp.fit_transform([epochs[i] for i in tr], y_enc[tr])
    X_csp_val = csp.transform([epochs[i] for i in val])

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())])
    pipe.fit(X_csp_tr, y_enc[tr])
    all_preds_csp[val] = pipe.predict(X_csp_val)

cv_time = time.time() - t0
acc_csp = accuracy_score(y_enc, all_preds_csp)
print(f"CV Accuracy: {acc_csp:.1%} (chance=20%) — {cv_time:.1f}s")
print(classification_report(y_enc, all_preds_csp, target_names=le.classes_.tolist()))

# Final CSP model
csp_final = SimpleCSP(n_components=4)
X_csp_all = csp_final.fit_transform(epochs, y_enc)
lda_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())])
lda_pipe.fit(X_csp_all, y_enc)

t0 = time.time()
for _ in range(100):
    x_test = csp_final.transform([epochs[0]])
    lda_pipe.predict_proba(x_test)
lat_csp = (time.time() - t0) / 100 * 1000

export_csp = {
    "csp": csp_final,
    "pipeline": lda_pipe,
    "label_encoder": le,
    "config": config,
    "cv_accuracy": round(acc_csp, 4),
    "inference_ms": round(lat_csp, 2),
}
joblib.dump(export_csp, "models/csp_lda.joblib")
print(f"Saved → models/csp_lda.joblib (latency: {lat_csp:.2f}ms)")
results["csp_lda"] = {"accuracy": round(acc_csp, 4), "latency_ms": round(lat_csp, 2)}

# ── 4. EEGNet (if torch available) ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class EEGNet(nn.Module):
        """Compact EEGNet for 6-channel EEG → 5 classes."""
        def __init__(self, n_channels=6, n_classes=5, n_samples=6000, F1=8, D=2, F2=16, dropout=0.25):
            super().__init__()
            # Block 1: temporal conv + depthwise spatial conv
            self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False)
            self.bn2 = nn.BatchNorm2d(F1*D)
            self.pool1 = nn.AvgPool2d((1, 8))
            self.drop1 = nn.Dropout(dropout)

            # Block 2: separable conv
            self.separable1 = nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False)
            self.bn3 = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)

            # Classifier
            self._fc_size = None
            self.fc = None
            self.n_classes = n_classes

            # Compute fc size
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_samples)
                out = self._features(dummy)
                self._fc_size = out.shape[1]
                self.fc = nn.Linear(self._fc_size, n_classes)

        def _features(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.depthwise(x)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.drop1(x)
            x = self.separable1(x)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pool2(x)
            x = self.drop2(x)
            return x.flatten(1)

        def forward(self, x):
            return self.fc(self._features(x))

    # Prepare data: (n_trials, 1, channels, samples)
    N_SAMPLES = 6000  # use first 6000 samples of stimulus period (12s at 500Hz)
    eeg_tensors = []
    for ep in epochs:
        s = ep[:N_SAMPLES]
        if s.shape[0] < N_SAMPLES:
            s = np.pad(s, ((0, N_SAMPLES - s.shape[0]), (0, 0)))
        eeg_tensors.append(s.T)  # (channels, samples)

    X_eeg = np.stack(eeg_tensors)[:, np.newaxis, :, :]  # (N, 1, 6, 6000)
    X_eeg_t = torch.tensor(X_eeg, dtype=torch.float32)
    y_t = torch.tensor(y_enc, dtype=torch.long)

    print(f"\n{'='*50}")
    print("Training: eegnet")
    print(f"{'='*50}")

    all_preds_eeg = np.zeros_like(y_enc)
    t0 = time.time()

    for fold, (tr, val) in enumerate(splits):
        model = EEGNet(n_channels=6, n_classes=5, n_samples=N_SAMPLES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        X_tr, y_tr = X_eeg_t[tr], y_t[tr]
        X_val, y_val = X_eeg_t[val], y_t[val]

        model.train()
        for epoch in range(30):
            # Mini-batch
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 32):
                idx = perm[i:i+32]
                out = model(X_tr[idx])
                loss = F.cross_entropy(out, y_tr[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(dim=1).numpy()
        all_preds_eeg[val] = preds

    cv_time = time.time() - t0
    acc_eeg = accuracy_score(y_enc, all_preds_eeg)
    print(f"CV Accuracy: {acc_eeg:.1%} (chance=20%) — {cv_time:.1f}s")
    print(classification_report(y_enc, all_preds_eeg, target_names=le.classes_.tolist()))

    # Train final EEGNet
    final_model = EEGNet(n_channels=6, n_classes=5, n_samples=N_SAMPLES)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    final_model.train()
    for epoch in range(40):
        perm = torch.randperm(len(X_eeg_t))
        for i in range(0, len(X_eeg_t), 32):
            idx = perm[i:i+32]
            out = final_model(X_eeg_t[idx])
            loss = F.cross_entropy(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    final_model.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            final_model(X_eeg_t[:1])
    lat_eeg = (time.time() - t0) / 100 * 1000

    torch.save(final_model.state_dict(), "models/eegnet.pt")
    # Save wrapper for predict
    export_eeg = {
        "model_class": "EEGNet",
        "state_dict_path": "models/eegnet.pt",
        "label_encoder": le,
        "config": config,
        "n_samples": N_SAMPLES,
        "cv_accuracy": round(acc_eeg, 4),
        "inference_ms": round(lat_eeg, 2),
    }
    joblib.dump(export_eeg, "models/eegnet.joblib")
    print(f"Saved → models/eegnet.joblib + models/eegnet.pt (latency: {lat_eeg:.2f}ms)")
    results["eegnet"] = {"accuracy": round(acc_eeg, 4), "latency_ms": round(lat_eeg, 2)}

except ImportError:
    print("\n[SKIP] EEGNet — torch not installed. Run: pip install torch")

# ── Summary ──
print(f"\n{'='*50}")
print("MODEL COMPARISON")
print(f"{'='*50}")
print(f"{'Model':<12} {'Accuracy':>10} {'Latency':>10}")
print("-" * 34)
for name, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
    print(f"{name:<12} {r['accuracy']:>9.1%} {r['latency_ms']:>8.2f}ms")
print(f"\nChance level: 20.0%")

# Also save the legacy baseline path for backward compat
import shutil
if os.path.exists("models/logreg.joblib"):
    shutil.copy("models/logreg.joblib", "models/baseline_logreg.joblib")
    print("Copied logreg → baseline_logreg.joblib (backward compat)")
