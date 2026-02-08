"""
Predict using EEG-ML OneDCNN and return the Relay output contract.

Expects model trained with train_eegml_on_robot.py (saved as models/eegml_onedcnn.keras).
Input: eeg_window (num_samples, 6). Uses stimulus window (STIM_ONSET, WINDOW_SAMPLES),
pads to (801, 7) for the model, then maps prediction to output contract.

Usage:
  from scripts.predict_eegml import predict_eegml
  result = predict_eegml(eeg_window)
"""
from __future__ import annotations

import os
import time

import numpy as np

# Same contract as predict.py
CONFIDENCE_THRESHOLD = 0.5
ACTION_MAP = {"left": "LEFT", "right": "RIGHT", "both": "FORWARD", "tongue": "BACKWARD", "rest": "STOP"}
CMD_VEL_MAP = {
    "FORWARD": {"vx": 0.6, "vy": 0.0, "yaw_rate": 0.0},
    "BACKWARD": {"vx": -0.4, "vy": 0.0, "yaw_rate": 0.0},
    "LEFT": {"vx": 0.0, "vy": 0.0, "yaw_rate": 1.5},
    "RIGHT": {"vx": 0.0, "vy": 0.0, "yaw_rate": -1.5},
    "STOP": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0},
}

# Must match export_for_eegml / OneDCNN
EEGML_TIME = 801
EEGML_CHANNELS = 7
CLASS_ORDER = ["left", "right", "both", "tongue", "rest"]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RELAY_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_MODEL_PATH = os.path.join(_RELAY_ROOT, "models", "eegml_onedcnn.keras")

_model_cache = None


def _load_model(path: str):
    global _model_cache
    if _model_cache is None:
        import tensorflow as tf
        # compile=False avoids loading EEG-ML's custom optimizer (GradAugAdam) at inference
        _model_cache = tf.keras.models.load_model(path, compile=False)
    return _model_cache


def _eeg_to_model_input(eeg: np.ndarray, stim_onset_sec: float, window_samples: int) -> np.ndarray:
    """Convert (samples, 6) to (1, 801, 7) for OneDCNN."""
    fs = 500
    cut = int(stim_onset_sec * fs)
    stim = eeg[cut : cut + window_samples] if eeg.shape[0] > cut else eeg
    if stim.shape[0] < window_samples:
        stim = np.pad(stim, ((0, window_samples - stim.shape[0]), (0, 0)))
    else:
        stim = stim[:window_samples]
    if np.isnan(stim).any():
        stim = np.nan_to_num(stim, nan=0.0)
    # (150, 6) -> (6, 150) -> (7, 801)
    stim = stim.T  # (6, 150)
    pad_ch = np.zeros((1, stim.shape[1]), dtype=stim.dtype)
    stim = np.concatenate([stim, pad_ch], axis=0)  # (7, 150)
    pad_time = np.zeros((stim.shape[0], EEGML_TIME - stim.shape[1]), dtype=stim.dtype)
    stim = np.concatenate([stim, pad_time], axis=1)  # (7, 801)
    # (7, 801) -> (1, 801, 7)
    stim = np.transpose(stim, (1, 0))[np.newaxis, :, :].astype(np.float32)
    return stim


def predict_eegml(
    eeg_window: np.ndarray,
    model_path: str = _DEFAULT_MODEL_PATH,
    threshold: float = CONFIDENCE_THRESHOLD,
    stim_onset_sec: float = 3.0,
    window_samples: int = 250,
) -> dict:
    """
    Predict intent from raw EEG using EEG-ML OneDCNN; return output contract.

    Args:
        eeg_window: shape (num_samples, 6)
        model_path: path to .keras model
        threshold: confidence threshold for is_above_threshold
        stim_onset_sec: start of analysis window (seconds)
        window_samples: length of window (samples)

    Returns:
        dict matching gameplan/output_contract.md
    """
    t_start = time.time()
    model = _load_model(model_path)
    X = _eeg_to_model_input(eeg_window, stim_onset_sec, window_samples)
    proba = model.predict(X, verbose=0)[0]
    latency = (time.time() - t_start) * 1000

    pred_idx = int(np.argmax(proba))
    predicted = CLASS_ORDER[pred_idx]
    confidence = float(proba[pred_idx])
    above = confidence > threshold
    robot_action = ACTION_MAP[predicted] if above else "STOP"

    return {
        "predicted_class": predicted,
        "confidence": round(confidence, 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_ORDER, proba)},
        "is_above_threshold": above,
        "robot_action": robot_action,
        "cmd_vel": CMD_VEL_MAP[robot_action],
        "timestamp_ms": int(time.time() * 1000),
        "inference_latency_ms": round(latency, 2),
    }


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, _SCRIPT_DIR)
    from data_utils import DATA_DIR, STIM_ONSET, WINDOW_SAMPLES

    sample_path = os.path.join(os.path.dirname(_RELAY_ROOT), "robot_control", "data")
    if not os.path.isdir(sample_path):
        sample_path = os.path.join(_RELAY_ROOT, "..", "robot_control", "data")
    files = [f for f in os.listdir(sample_path) if f.endswith(".npz")]
    if not files:
        print("No .npz in robot_control/data")
        sys.exit(1)
    eeg = np.load(os.path.join(sample_path, files[0]), allow_pickle=True)["feature_eeg"]
    result = predict_eegml(eeg, stim_onset_sec=float(STIM_ONSET), window_samples=int(WINDOW_SAMPLES))
    print(json.dumps(result, indent=2))
