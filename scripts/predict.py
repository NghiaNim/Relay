"""Inference module — load baseline model and predict intent from EEG.

Usage:
    from scripts.predict import predict
    result = predict(eeg_window)  # eeg_window: np.ndarray (7499, 6) or (N, 6)
"""
import numpy as np, time, joblib
from numpy.fft import rfft, rfftfreq

_MODEL_PATH = "models/baseline_logreg.joblib"
_bundle = joblib.load(_MODEL_PATH)

pipeline = _bundle["pipeline"]
label_encoder = _bundle["label_encoder"]
config = _bundle["config"]
BANDS = config["bands"]
FS = config["fs"]
STIM_ONSET = config["stim_onset"]
CONFIDENCE_THRESHOLD = 0.5

# EEG class → robot action mapping
ACTION_MAP = {
    "left": "LEFT",
    "right": "RIGHT",
    "both": "FORWARD",
    "tongue": "BACKWARD",
    "rest": "STOP",
}

CMD_VEL_MAP = {
    "FORWARD":  {"vx":  0.6, "vy": 0.0, "yaw_rate":  0.0},
    "BACKWARD": {"vx": -0.4, "vy": 0.0, "yaw_rate":  0.0},
    "LEFT":     {"vx":  0.0, "vy": 0.0, "yaw_rate":  1.5},
    "RIGHT":    {"vx":  0.0, "vy": 0.0, "yaw_rate": -1.5},
    "STOP":     {"vx":  0.0, "vy": 0.0, "yaw_rate":  0.0},
}


def extract_features(eeg):
    """Band-power + stats features from stimulus window."""
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


def predict(eeg_window: np.ndarray, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Predict intent from raw EEG window.

    Args:
        eeg_window: shape (num_samples, 6) — 15s at 500Hz = (7499, 6)
        threshold: confidence threshold for is_above_threshold flag

    Returns:
        dict matching the output contract (see gameplan/output_contract.md)
    """
    t_start = time.time()
    feats = extract_features(eeg_window).reshape(1, -1)
    proba = pipeline.predict_proba(feats)[0]
    pred_idx = proba.argmax()
    latency = (time.time() - t_start) * 1000

    classes = label_encoder.classes_.tolist()
    predicted = classes[pred_idx]
    above = float(proba[pred_idx]) > threshold

    # Robot action: gated by confidence threshold
    robot_action = ACTION_MAP[predicted] if above else "STOP"

    return {
        "predicted_class": predicted,
        "confidence": round(float(proba[pred_idx]), 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "is_above_threshold": above,
        "robot_action": robot_action,
        "cmd_vel": CMD_VEL_MAP[robot_action],
        "timestamp_ms": int(time.time() * 1000),
        "inference_latency_ms": round(latency, 2),
    }


if __name__ == "__main__":
    import json
    sample = np.load("data/robot_control/data/0b2dbd41-10.npz", allow_pickle=True)["feature_eeg"]
    print(json.dumps(predict(sample), indent=2))
