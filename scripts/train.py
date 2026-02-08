"""Train a single model. Usage: python train.py --model logreg|riemann_svm|knn|..."""

import argparse
import os
import sys
import time

import numpy as np
import joblib

np.random.seed(42)

# Add parent for imports when run as script
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
    FS,
    STIM_ONSET,
    BANDS,
    LABEL_MAP,
    WINDOW_SAMPLES,
)
from models_registry import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data-dir", default=None, help=f"Override data dir (default: {DATA_DIR})")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out-dir", default="../models")
    args = parser.parse_args()

    model_name = args.model
    train_fn, input_format = MODEL_REGISTRY[model_name]

    print(f"Loading dataset from {args.data_dir or DATA_DIR}...")
    X_eeg, y, groups = load_dataset(args.data_dir)
    # Signal enhancement: notch 50/60 Hz → bandpass 1–45 Hz → baseline correction
    X_eeg = [preprocess_eeg(e) for e in X_eeg]
    le = __import__("sklearn.preprocessing", fromlist=["LabelEncoder"]).LabelEncoder()
    y_enc = le.fit_transform(y)

    train_idx, test_idx = get_train_test_split(
        np.arange(len(y)), y_enc, groups, test_size=args.test_size, random_state=42
    )

    tangent_space = None
    # Prepare features by format
    if input_format == "bandpower":
        X = np.stack([extract_bandpower_features(e) for e in X_eeg])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    elif input_format == "riemannian":
        X_train, tangent_space = get_riemannian_features(
            [X_eeg[i] for i in train_idx],
            y_enc[train_idx],
            [groups[i] for i in train_idx],
        )
        y_train = y_enc[train_idx]
        from pyriemann.estimation import Covariances
        epochs_test = []
        for i in test_idx:
            ep = extract_stimulus_epoch(X_eeg[i])
            ep = bandpass_causal(ep)
            epochs_test.append(ep)
        epochs_test = np.asarray(epochs_test)
        epochs_test = np.transpose(epochs_test, (0, 2, 1))  # (n_trials, n_channels, n_times)
        covs_test = Covariances(estimator="lwf").fit_transform(epochs_test)
        X_test = tangent_space.transform(covs_test).astype(np.float32)
        y_test = y_enc[test_idx]
    else:  # raw_eeg
        X = get_eeg_windows_for_nn(X_eeg, n_samples=WINDOW_SAMPLES)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    print(f"Training {model_name}...")
    t0 = time.time()
    model_or_bundle = train_fn(X_train, y_train)
    train_time = time.time() - t0

    # Evaluate on held-out subjects
    if hasattr(model_or_bundle, "predict_proba"):
        y_pred = model_or_bundle.predict(X_test)
        proba = model_or_bundle.predict_proba(X_test)
    else:
        import torch
        bundle = model_or_bundle
        model = bundle["model"]
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            logits = model(X_t)
            proba = torch.softmax(logits, dim=1).numpy()
            y_pred = logits.argmax(dim=1).numpy()

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    tangent_space_ = tangent_space if input_format == "riemannian" else None
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{model_name}.joblib")
    export = {
        "model": model_or_bundle,
        "label_encoder": le,
        "label_map": LABEL_MAP,
        "config": {"fs": FS, "stim_onset": STIM_ONSET, "bands": BANDS, "n_channels": 6, "window_samples": WINDOW_SAMPLES, "preprocess": True},
        "input_format": input_format,
        "eeg_n_samples": WINDOW_SAMPLES if input_format == "raw_eeg" else None,
        "tangent_space": tangent_space_,
        "metrics": {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
            "train_time_s": float(train_time),
        },
    }

    joblib.dump(export, out_path)
    print(f"Saved {out_path}")
    print(f"Test accuracy: {acc:.2%} | precision: {prec:.2%} | recall: {rec:.2%}")


if __name__ == "__main__":
    main()
