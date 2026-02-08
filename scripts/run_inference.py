"""Run inference and evaluation. Test-set outputs follow output_contract.md. Scaling metrics in separate dir."""

import argparse
import json
import os
import sys
import time

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_dataset,
    preprocess_eeg,
    extract_bandpower_features,
    extract_stimulus_epoch,
    bandpass_causal,
    get_riemannian_features,
    get_train_test_split,
    get_eeg_windows_for_nn,
    DATA_DIR,
    WINDOW_SAMPLES,
    STIM_ONSET,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    confusion_matrix,
)

N_INFERENCE_AT_SCALE = 1000
CONFIDENCE_THRESHOLD = 0.5

# Output contract: gameplan/output_contract.md
ACTION_MAP = {"left": "LEFT", "right": "RIGHT", "both": "FORWARD", "tongue": "BACKWARD", "rest": "STOP"}
CMD_VEL_MAP = {
    "FORWARD": {"vx": 0.6, "vy": 0.0, "yaw_rate": 0.0},
    "BACKWARD": {"vx": -0.4, "vy": 0.0, "yaw_rate": 0.0},
    "LEFT": {"vx": 0.0, "vy": 0.0, "yaw_rate": 1.5},
    "RIGHT": {"vx": 0.0, "vy": 0.0, "yaw_rate": -1.5},
    "STOP": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0},
}

RESULTS_TEST_DIR = "results/test_inference"
RESULTS_SCALE_DIR = "results/latency_at_scale"

# Random baseline (no trained model)
RANDOM_SEED = 42


def load_data_only(data_dir=None):
    """Load dataset and train/test split only. Returns X_test, y_test, le (no model)."""
    X_eeg, y, groups = load_dataset(data_dir or DATA_DIR)
    X_eeg = [preprocess_eeg(e) for e in X_eeg]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    train_idx, test_idx = get_train_test_split(
        np.arange(len(y)), y_enc, groups, test_size=0.2, random_state=42
    )
    # Use bandpower for consistent test set size; random baseline ignores features
    X = np.stack([extract_bandpower_features(e) for e in X_eeg])
    X_test = X[test_idx]
    y_test = y_enc[test_idx]
    return X_test, y_test, le


# EEG-ML OneDCNN: same class order as export_for_eegml / predict_eegml
EEGML_CLASS_ORDER = ["left", "right", "both", "tongue", "rest"]
EEGML_TIME, EEGML_CH = 801, 7


def _eeg_to_eegml_input(eeg, stim_onset_sec, window_samples):
    """One trial (samples, 6) -> (1, 801, 7)."""
    fs = 500
    cut = int(stim_onset_sec * fs)
    stim = eeg[cut : cut + window_samples] if eeg.shape[0] > cut else eeg
    if stim.shape[0] < window_samples:
        stim = np.pad(stim, ((0, window_samples - stim.shape[0]), (0, 0)))
    else:
        stim = stim[:window_samples]
    if np.isnan(stim).any():
        stim = np.nan_to_num(stim, nan=0.0)
    stim = stim.T  # (6, 150)
    pad_ch = np.zeros((1, stim.shape[1]), dtype=stim.dtype)
    stim = np.concatenate([stim, pad_ch], axis=0)
    pad_time = np.zeros((stim.shape[0], EEGML_TIME - stim.shape[1]), dtype=stim.dtype)
    stim = np.concatenate([stim, pad_time], axis=1)  # (7, 801)
    return np.transpose(stim, (1, 0))[np.newaxis, :, :].astype(np.float32)


def load_eegml_model_and_data(models_dir, data_dir=None):
    """Load Keras eegml_onedcnn and test set as (X_test (n,801,7), y_test, le)."""
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder

    model_path = os.path.join(models_dir, "eegml_onedcnn.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"EEG-ML model not found: {model_path}. Train with train_eegml_on_robot.py first.")
    model = tf.keras.models.load_model(model_path, compile=False)

    X_eeg, y, groups = load_dataset(data_dir or DATA_DIR)
    le = LabelEncoder()
    le.classes_ = np.array(EEGML_CLASS_ORDER)
    y_enc = le.transform(y)
    train_idx, test_idx = get_train_test_split(
        np.arange(len(y)), y_enc, groups, test_size=0.2, random_state=42
    )

    X_list = []
    for i in test_idx:
        inp = _eeg_to_eegml_input(
            X_eeg[i], float(STIM_ONSET), int(WINDOW_SAMPLES)
        )  # (1, 801, 7)
        X_list.append(inp[0])
    X_test = np.stack(X_list).astype(np.float32)  # (n, 801, 7)
    y_test = y_enc[test_idx]
    return model, X_test, y_test, le


def load_model_and_data(model_path, data_dir=None):
    """Load model bundle and test data for the model's input format."""
    bundle = joblib.load(model_path)
    X_eeg, y, groups = load_dataset(data_dir or DATA_DIR)
    if bundle.get("config", {}).get("preprocess"):
        X_eeg = [preprocess_eeg(e) for e in X_eeg]
    le = bundle["label_encoder"]
    input_format = bundle.get("input_format", "bandpower")
    y_enc = le.transform(y)

    train_idx, test_idx = get_train_test_split(
        np.arange(len(y)), y_enc, groups, test_size=0.2, random_state=42
    )

    if input_format == "bandpower":
        X = np.stack([extract_bandpower_features(e) for e in X_eeg])
    elif input_format == "riemannian":
        ts = bundle.get("tangent_space")
        if ts is None:
            raise ValueError("Riemannian model requires tangent_space in bundle")
        X, _ = get_riemannian_features(X_eeg, y_enc, groups, tangent_space=ts)
    else:
        X = get_eeg_windows_for_nn(
            X_eeg, n_samples=bundle.get("eeg_n_samples", WINDOW_SAMPLES)
        )

    X_test = X[test_idx]
    y_test = y_enc[test_idx]
    return bundle, X_test, y_test, le, input_format


def _get_predictor(bundle):
    """Return (predict_fn, is_torch). predict_fn(X) -> (y_pred, proba).
    Supports bundle['model'] (train.py) or bundle['pipeline'] (legacy)."""
    m = bundle.get("model") or bundle.get("pipeline")
    if m is None:
        raise KeyError("Bundle must have 'model' or 'pipeline'")
    if hasattr(m, "predict_proba"):
        return (lambda X: (m.predict(X), m.predict_proba(X))), False
    import torch
    model = m["model"]
    model.eval()

    def predict_fn(X):
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = model(X_t)
            proba = torch.softmax(logits, dim=1).numpy()
            y_pred = logits.argmax(dim=1).numpy()
        return y_pred, proba

    return predict_fn, True


def predict_batch(bundle, X, input_format):
    """Run prediction. Returns (y_pred, proba)."""
    pred_fn, _ = _get_predictor(bundle)
    return pred_fn(X)


def predict_one_contract(bundle, X_one, le, threshold=CONFIDENCE_THRESHOLD):
    """Single sample → output-schema dict (output_contract.md). Measures latency."""
    pred_fn, _ = _get_predictor(bundle)
    t0 = time.perf_counter()
    y_pred, proba = pred_fn(X_one)
    latency_ms = (time.perf_counter() - t0) * 1000
    idx = int(y_pred.flat[0])
    classes = le.classes_.tolist()
    pred_class = classes[idx]
    conf = float(proba.flat[idx])
    above = conf > threshold
    robot_action = ACTION_MAP[pred_class] if above else "STOP"
    return {
        "predicted_class": pred_class,
        "confidence": round(conf, 4),
        "all_probabilities": {c: round(float(proba.flat[i]), 4) for i, c in enumerate(classes)},
        "is_above_threshold": above,
        "robot_action": robot_action,
        "cmd_vel": CMD_VEL_MAP[robot_action],
        "timestamp_ms": int(time.time() * 1000),
        "inference_latency_ms": round(latency_ms, 2),
    }


def measure_latency(bundle, X_sample, n_runs=N_INFERENCE_AT_SCALE):
    """Measure inference latency over n_runs. Returns mean_ms, std_ms."""
    if len(X_sample.shape) == 2:
        X_sample = X_sample[:1]
    elif len(X_sample.shape) == 4:
        X_sample = X_sample[:1]

    pred_fn, is_torch = _get_predictor(bundle)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pred_fn(X_sample)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def random_predict_batch(n_samples, n_classes, rng):
    """Random baseline: (y_pred, proba) with uniform random class and dirichlet probs."""
    y_pred = rng.integers(0, n_classes, size=n_samples)
    proba = rng.dirichlet(np.ones(n_classes), size=n_samples).astype(np.float32)
    return y_pred, proba


def random_predict_one_contract(le, rng, threshold=CONFIDENCE_THRESHOLD):
    """One random prediction in output-contract dict."""
    classes = le.classes_.tolist()
    n_classes = len(classes)
    idx = rng.integers(0, n_classes)
    proba = rng.dirichlet(np.ones(n_classes)).astype(np.float64)
    pred_class = classes[idx]
    conf = float(proba[idx])
    above = conf > threshold
    robot_action = ACTION_MAP[pred_class] if above else "STOP"
    t0 = time.perf_counter()
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "predicted_class": pred_class,
        "confidence": round(conf, 4),
        "all_probabilities": {c: round(float(proba[i]), 4) for i, c in enumerate(classes)},
        "is_above_threshold": above,
        "robot_action": robot_action,
        "cmd_vel": CMD_VEL_MAP[robot_action],
        "timestamp_ms": int(time.time() * 1000),
        "inference_latency_ms": round(latency_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or path to .joblib")
    parser.add_argument("--models-dir", default="../models", help="Dir with saved models")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-inference", type=int, default=N_INFERENCE_AT_SCALE)
    parser.add_argument("--results-dir", default="./models/results", help="Parent for results/ (default: cwd)")
    args = parser.parse_args()

    model_name = args.model
    is_random_baseline = model_name.lower() == "random"

    root = args.results_dir or os.getcwd()
    test_dir = os.path.join(root, RESULTS_TEST_DIR)
    scale_dir = os.path.join(root, RESULTS_SCALE_DIR)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(scale_dir, exist_ok=True)

    if is_random_baseline:
        model_path = None
        print("Loading data for random baseline...")
        X_test, y_test, le = load_data_only(args.data_dir)
        rng = np.random.default_rng(RANDOM_SEED)
        n_classes = len(le.classes_)
        print("Evaluating random baseline on test set...")
        y_pred, proba = random_predict_batch(len(y_test), n_classes, rng)
    elif model_name.lower() == "eegml":
        model_name = "eegml"
        model_path = os.path.join(args.models_dir, "eegml_onedcnn.keras")
        print(f"Loading EEG-ML model from {model_path}...")
        eegml_model, X_test, y_test, le = load_eegml_model_and_data(args.models_dir, args.data_dir)
        print("Evaluating EEG-ML OneDCNN on test set...")
        proba = eegml_model.predict(X_test, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        bundle = {"_eegml_model": eegml_model, "input_format": "eegml"}
    else:
        model_path = model_name if model_name.endswith(".joblib") else os.path.join(args.models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            sys.exit(1)
        base = os.path.basename(model_path)
        model_name = base.replace(".joblib", "") if base.endswith(".joblib") else model_name

        print(f"Loading {model_path}...")
        bundle, X_test, y_test, le, _ = load_model_and_data(model_path, args.data_dir)

        print("Evaluating on test set...")
        y_pred, proba = predict_batch(bundle, X_test, bundle.get("input_format", "bandpower"))
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(le.classes_)), zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Performance ===")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision (macro): {prec_macro:.2%}")
    print(f"Recall (macro):    {rec_macro:.2%}")
    print("\nPer-class precision/recall:")
    for i, c in enumerate(le.classes_):
        print(f"  {c}: P={prec[i]:.2%} R={rec[i]:.2%}")
    print("\nConfusion matrix:")
    print(cm)

    # Test-set inference: one output-schema object per sample → results/test_inference/
    predictions = []
    if is_random_baseline:
        for i in range(len(y_test)):
            out = random_predict_one_contract(le, rng)
            out["true_class"] = le.classes_[y_test[i]]
            predictions.append(out)
    else:
        if bundle.get("_eegml_model"):
            for i in range(len(y_test)):
                idx = int(y_pred[i])
                pred_class = le.classes_[idx]
                conf = float(proba[i, idx])
                above = conf > CONFIDENCE_THRESHOLD
                robot_action = ACTION_MAP[pred_class] if above else "STOP"
                out = {
                    "predicted_class": pred_class,
                    "confidence": round(conf, 4),
                    "all_probabilities": {c: round(float(proba[i, j]), 4) for j, c in enumerate(le.classes_)},
                    "is_above_threshold": above,
                    "robot_action": robot_action,
                    "cmd_vel": CMD_VEL_MAP[robot_action],
                    "timestamp_ms": int(time.time() * 1000),
                    "inference_latency_ms": 0.0,
                }
                out["true_class"] = le.classes_[y_test[i]]
                predictions.append(out)
        else:
            for i in range(len(y_test)):
                if len(X_test.shape) == 2:
                    x_one = X_test[i : i + 1]
                else:
                    x_one = X_test[i : i + 1]
                out = predict_one_contract(bundle, x_one, le)
                out["true_class"] = le.classes_[y_test[i]]
                predictions.append(out)
    test_out = os.path.join(test_dir, f"{model_name}_predictions.json")
    with open(test_out, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nTest inference (output contract) → {test_out}")

    # Scaling experiment metrics → results/latency_at_scale/
    print("\n=== Efficiency (inference latency at scale) ===")
    if is_random_baseline:
        times = []
        for _ in range(args.n_inference):
            t0 = time.perf_counter()
            random_predict_one_contract(le, rng)
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms, std_ms = float(np.mean(times)), float(np.std(times))
        model_size_kb = 0.0
    elif bundle.get("_eegml_model"):
        eegml_model = bundle["_eegml_model"]
        X_one = X_test[:1] if len(X_test.shape) == 3 else X_test[0:1]
        times = []
        for _ in range(args.n_inference):
            t0 = time.perf_counter()
            eegml_model.predict(X_one, verbose=0)
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms, std_ms = float(np.mean(times)), float(np.std(times))
        model_size_kb = os.path.getsize(model_path) / 1024 if model_path and os.path.exists(model_path) else 0.0
    else:
        mean_ms, std_ms = measure_latency(bundle, X_test, n_runs=args.n_inference)
        model_size_kb = os.path.getsize(model_path) / 1024
    print(f"Mean inference time: {mean_ms:.4f} ms (±{std_ms:.4f}) over {args.n_inference} runs")
    print(f"Model size: {model_size_kb:.1f} KB")

    scale_results = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "inference_mean_ms": mean_ms,
        "inference_std_ms": std_ms,
        "n_inference_runs": args.n_inference,
        "model_size_kb": model_size_kb,
    }
    scale_out = os.path.join(scale_dir, f"{model_name}_eval.json")
    with open(scale_out, "w") as f:
        json.dump(scale_results, f, indent=2)
    print(f"Latency-at-scale metrics → {scale_out}")


if __name__ == "__main__":
    main()
