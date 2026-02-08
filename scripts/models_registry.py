"""Model registry: factory functions for each model. Input format: bandpower | riemannian | raw_eeg."""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Config from training_spec
SEED = 42
LABEL_MAP = {
    "Right Fist": "right", "Left Fist": "left", "Left First": "left",
    "Both Fists": "both", "Both Firsts": "both",
    "Tongue Tapping": "tongue", "Relax": "rest",
}
CONFIG = {"fs": 500, "stim_onset": 3.0, "bands": {}, "n_channels": 6}


def _make_sklearn_pipe(clf, scaler=StandardScaler()):
    return Pipeline([("scaler", scaler), ("clf", clf)])


# ── 1. Logistic Regression ──
def train_logreg(X, y, **kwargs):
    pipe = _make_sklearn_pipe(
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=SEED)
    )
    pipe.fit(X, y)
    return pipe


# ── 2. Riemannian Geometry → SVM ──
def train_riemann_svm(X, y, **kwargs):
    pipe = _make_sklearn_pipe(SVC(kernel="linear", C=1.0, probability=True, random_state=SEED))
    pipe.fit(X, y)
    return pipe


# ── 3. k-NN ──
def train_knn(X, y, **kwargs):
    pipe = _make_sklearn_pipe(KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    pipe.fit(X, y)
    return pipe


# ── 4. LightGBM ──
def train_lightgbm(X, y, **kwargs):
    import lightgbm as lgb
    pipe = _make_sklearn_pipe(
        lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1, verbose=-1)
    )
    pipe.fit(X, y)
    return pipe


# ── 5. XGBoost ──
def train_xgboost(X, y, **kwargs):
    import xgboost as xgb
    pipe = _make_sklearn_pipe(
        xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1)
    )
    pipe.fit(X, y)
    return pipe


# ── 6. MLP ──
def train_mlp(X, y, **kwargs):
    pipe = _make_sklearn_pipe(
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED)
    )
    pipe.fit(X, y)
    return pipe


# ── Neural models (7–11): need raw EEG, PyTorch ──
def _ensure_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch required for neural models. pip install torch")


def train_1dcnn(X, y, **kwargs):
    import torch
    from nn_models import EEG1DCNN, train_torch_model

    torch.manual_seed(SEED)
    n_classes = len(np.unique(y))
    model = EEG1DCNN(n_channels=6, n_samples=X.shape[-1], n_classes=n_classes)
    return train_torch_model(model, X, y, **kwargs)


def train_tcn(X, y, **kwargs):
    from nn_models import EEGTCN, train_torch_model

    _ensure_torch()
    n_classes = len(np.unique(y))
    model = EEGTCN(n_channels=6, n_samples=X.shape[-1], n_classes=n_classes)
    return train_torch_model(model, X, y, **kwargs)


def train_lstm(X, y, **kwargs):
    from nn_models import EEGLSTM, train_torch_model

    _ensure_torch()
    n_classes = len(np.unique(y))
    model = EEGLSTM(n_channels=6, n_samples=X.shape[-1], n_classes=n_classes)
    return train_torch_model(model, X, y, **kwargs)


def train_transformer(X, y, **kwargs):
    from nn_models import EEGTransformer, train_torch_model

    _ensure_torch()
    n_classes = len(np.unique(y))
    model = EEGTransformer(n_channels=6, n_samples=X.shape[-1], n_classes=n_classes)
    return train_torch_model(model, X, y, **kwargs)


def train_mamba(X, y, **kwargs):
    from nn_models import EEGMamba, train_torch_model

    _ensure_torch()
    n_classes = len(np.unique(y))
    model = EEGMamba(n_channels=6, n_samples=X.shape[-1], n_classes=n_classes)
    return train_torch_model(model, X, y, **kwargs)


# Registry: model_name -> (train_fn, input_format)
MODEL_REGISTRY = {
    "logreg": (train_logreg, "bandpower"),
    "riemann_svm": (train_riemann_svm, "riemannian"),
    "knn": (train_knn, "bandpower"),
    "lightgbm": (train_lightgbm, "bandpower"),
    "xgboost": (train_xgboost, "bandpower"),
    "mlp": (train_mlp, "bandpower"),
    "1dcnn": (train_1dcnn, "raw_eeg"),
    "tcn": (train_tcn, "raw_eeg"),
    "lstm": (train_lstm, "raw_eeg"),
    "transformer": (train_transformer, "raw_eeg"),
    "mamba": (train_mamba, "raw_eeg"),
}
