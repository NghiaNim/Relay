"""Generate a prediction stream JSON from dataset for the viz demo."""
import numpy as np, json, glob, os, joblib
from numpy.fft import rfft, rfftfreq

DATA_DIR = "data/robot_control/data"
MODEL_PATH = "models/baseline_logreg.joblib"
OUT_PATH = "viz/stream.json"

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
le = bundle["label_encoder"]
cfg = bundle["config"]
BANDS = cfg["bands"]
FS = cfg["fs"]
STIM_ONSET = cfg["stim_onset"]

# EEG class → robot action mapping
ACTION_MAP = {"left": "LEFT", "right": "RIGHT", "both": "FORWARD", "tongue": "BACKWARD", "rest": "STOP"}
CMD_VEL_MAP = {
    "FORWARD":  {"vx":  0.6, "vy": 0.0, "yaw_rate":  0.0},
    "BACKWARD": {"vx": -0.4, "vy": 0.0, "yaw_rate":  0.0},
    "LEFT":     {"vx":  0.0, "vy": 0.0, "yaw_rate":  1.5},
    "RIGHT":    {"vx":  0.0, "vy": 0.0, "yaw_rate": -1.5},
    "STOP":     {"vx":  0.0, "vy": 0.0, "yaw_rate":  0.0},
}


def extract_features(eeg):
    cut = int(STIM_ONSET * FS)
    stim = eeg[cut:] if eeg.shape[0] > cut else eeg
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


# Pick one session
files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
sessions = {}
for f in files:
    sid = f.split("/")[-1].split("-")[0]
    sessions.setdefault(sid, []).append(f)

session_id = max(sessions, key=lambda k: len(sessions[k]))
session_files = sorted(sessions[session_id], key=lambda f: int(f.split("-")[-1].replace(".npz", "")))

stream = []
for f in session_files:
    arr = np.load(f, allow_pickle=True)
    eeg = arr["feature_eeg"]
    label_info = arr["label"].item()

    feats = extract_features(eeg).reshape(1, -1)
    proba = pipeline.predict_proba(feats)[0]
    pred_idx = proba.argmax()
    classes = le.classes_.tolist()
    predicted = classes[pred_idx]
    above = float(proba[pred_idx]) > 0.5
    # Use ungated action for viz demo so robot actually moves
    robot_action = ACTION_MAP[predicted]

    # Downsample EEG heavily: every 50th sample → ~150 points
    eeg_ds = eeg[::50, :]
    eeg_ds = np.nan_to_num(eeg_ds, nan=0.0, posinf=0.0, neginf=0.0)
    eeg_norm = np.zeros_like(eeg_ds)
    for ch in range(6):
        col = eeg_ds[:, ch].copy()
        col = col - col.mean()
        mx = np.abs(col).max()
        if mx > 0:
            col = col / mx
        eeg_norm[:, ch] = col
    eeg_list = np.round(eeg_norm, 3).tolist()

    stream.append({
        "file": f.split("/")[-1],
        "true_label": label_info["label"],
        "predicted_class": predicted,
        "confidence": round(float(proba[pred_idx]), 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "is_above_threshold": above,
        "robot_action": robot_action,
        "cmd_vel": CMD_VEL_MAP[robot_action],
        "eeg": eeg_list,
    })

os.makedirs("viz", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(stream, f)

sz = os.path.getsize(OUT_PATH) / 1024
print(f"Generated {len(stream)} frames → {OUT_PATH} ({sz:.0f} KB)")
print(f"Labels: {[s['true_label'] for s in stream[:10]]}...")
print(f"Actions: {[s['robot_action'] for s in stream[:10]]}...")
