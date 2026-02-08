"""Data loading, preprocessing, feature extraction for EEG intent classification."""

import glob
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Config
DATA_DIR = "../robot_control/data"
FS = 500
STIM_ONSET = 3  # seconds: start of analysis window (training_spec: t=3s)
WINDOW_DURATION_SEC = 0.5  # seconds: length of temporal window
WINDOW_SAMPLES = int(WINDOW_DURATION_SEC * FS)  # 250 at 500 Hz
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
CH_NAMES = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
# Handle typos in dataset: Left First, Both Firsts
LABEL_MAP = {
    "Right Fist": "right",
    "Left Fist": "left",
    "Left First": "left",
    "Both Fists": "both",
    "Both Firsts": "both",
    "Tongue Tapping": "tongue",
    "Relax": "rest",
}

# Preprocessing: filter + baseline + optional artifact clip
NOTCH_FREQS = (50, 60)  # Hz to notch (line noise)
BANDPASS_MAIN = (1, 45)  # Hz for bandpower / NN (task-relevant range)
ARTIFACT_CLIP_STD = 4.0  # clip trial to ± this many std per channel (None = no clip)

# Riemannian: window 0.5s, step 0.3s
RMAN_WINDOW_SAMPLES = int(0.5 * FS)  # 250
RMAN_SLIDE_SAMPLES = int(0.2 * FS)   # 150
RMAN_FILTER_LOW, RMAN_FILTER_HIGH = 8, 30  # Hz


def load_dataset(data_dir=None):
    """Load EEG data and labels. Returns X_eeg (list of arrays), y, groups."""
    data_dir = data_dir or DATA_DIR
    files = sorted(glob.glob(f"{data_dir}/*.npz"))
    X_eeg, y, groups = [], [], []
    for f in files:
        arr = np.load(f, allow_pickle=True)
        lab = arr["label"].item()
        label = LABEL_MAP.get(lab["label"])
        if label is None:
            continue
        eeg = arr["feature_eeg"]
        X_eeg.append(eeg)
        y.append(label)
        groups.append(lab["subject_id"])
    return X_eeg, np.array(y), np.array(groups)


def _notch_sos(freq_hz, quality=30):
    """SOS for a single notch at freq_hz. iirnotch returns (b, a); convert to SOS for sosfilt."""
    from scipy.signal import iirnotch, tf2sos
    w0 = freq_hz / (FS / 2)
    b, a = iirnotch(w0, quality, fs=FS)
    return tf2sos(b, a)


def notch_filter(eeg, freqs_hz=NOTCH_FREQS):
    """Remove line noise at given frequencies. eeg: (samples, channels)."""
    from scipy.signal import sosfilt
    out = eeg.astype(np.float64)
    for f in freqs_hz:
        if f >= FS / 2:
            continue
        sos = _notch_sos(f)
        out = sosfilt(sos, out, axis=0)
    return out.astype(np.float32)


def bandpass_main(eeg, low=BANDPASS_MAIN[0], high=BANDPASS_MAIN[1], order=4):
    """Band-pass 1–45 Hz (or custom). eeg: (samples, channels)."""
    from scipy.signal import sosfilt
    sos = _bandpass_sos(low, high, order=order)
    return sosfilt(sos, eeg.astype(np.float64), axis=0).astype(np.float32)


def baseline_correct(eeg, stim_onset_sec=STIM_ONSET):
    """Subtract pre-stimulus mean per channel. eeg: (samples, channels)."""
    n_pre = int(stim_onset_sec * FS)
    n_pre = min(n_pre, eeg.shape[0] - 1)
    if n_pre <= 0:
        return eeg
    baseline = np.mean(eeg[:n_pre], axis=0, keepdims=True)
    return (eeg - baseline).astype(np.float32)


def preprocess_eeg(eeg, notch=True, bandpass=True, baseline=True):
    """Full enhancement: notch 50/60 Hz → bandpass 1–45 Hz → baseline correction.
    Returns eeg (samples, channels). Leakage-safe (no fit from other trials)."""
    if np.isnan(eeg).any():
        eeg = np.nan_to_num(eeg, nan=0.0)
    if notch:
        eeg = notch_filter(eeg)
    if bandpass:
        eeg = bandpass_main(eeg)
    if baseline:
        eeg = baseline_correct(eeg)
    return eeg.astype(np.float32)


def _zscore_trial(epoch):
    """Per-trial z-score per channel. epoch: (samples, channels)."""
    mean = np.mean(epoch, axis=0, keepdims=True)
    std = np.std(epoch, axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    return ((epoch - mean) / std).astype(np.float32)


def extract_stimulus_epoch(eeg, zscore_trial=True, clip_std=ARTIFACT_CLIP_STD):
    """Return stimulus-period EEG (samples, channels). Length = WINDOW_SAMPLES.
    Optional: per-trial z-score, then clip to ±clip_std (None = no clip)."""
    cut = int(STIM_ONSET * FS)
    stim = eeg[cut:cut + WINDOW_SAMPLES]
    if np.isnan(stim).any():
        stim = np.nan_to_num(stim, nan=0.0)
    if stim.shape[0] < WINDOW_SAMPLES:
        stim = np.pad(stim, ((0, WINDOW_SAMPLES - stim.shape[0]), (0, 0)))
    if zscore_trial:
        stim = _zscore_trial(stim)
    if clip_std is not None and clip_std > 0:
        std = np.std(stim, axis=0, keepdims=True)
        std = np.where(std < 1e-9, 1.0, std)
        stim = np.clip(stim, -clip_std * std, clip_std * std)
    return stim.astype(np.float32)


def extract_bandpower_features(eeg, zscore_trial=True, clip_std=ARTIFACT_CLIP_STD):
    """42-dim: 6ch × (5 bands + 2 stats). Uses stimulus window; optional z-score + clip.
    Expects eeg already preprocessed (notch, bandpass, baseline) if using full pipeline."""
    from numpy.fft import rfft, rfftfreq

    stim = extract_stimulus_epoch(eeg, zscore_trial=zscore_trial, clip_std=clip_std)
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


def _bandpass_sos(low, high, order=4):
    from scipy.signal import butter
    return butter(order, [low, high], btype="band", fs=FS, output="sos")


def bandpass_causal(eeg, low=RMAN_FILTER_LOW, high=RMAN_FILTER_HIGH):
    """Causal band-pass filter (real-time compatible)."""
    from scipy.signal import sosfilt
    sos = _bandpass_sos(low, high)
    return sosfilt(sos, eeg, axis=0)


def get_riemannian_features(X_eeg, y, groups, tangent_space=None):
    """Per-trial: bandpass → covariance → tangent space. Returns (N, 21), tangent_space."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    epochs = []
    for eeg in X_eeg:
        epoch = extract_stimulus_epoch(eeg)
        epoch = bandpass_causal(epoch)
        epochs.append(epoch)
    epochs = np.asarray(epochs)
    # PyRiemann expects (n_trials, n_channels, n_times); we had (n_trials, n_times, n_channels)
    epochs = np.transpose(epochs, (0, 2, 1))

    covs = Covariances(estimator="lwf").fit_transform(epochs)
    if tangent_space is None:
        tangent_space = TangentSpace()
        tangent_space.fit(covs)
    feats = tangent_space.transform(covs)
    return feats.astype(np.float32), tangent_space


def get_train_test_split(X, y, groups, test_size=0.2, random_state=42):
    """80/20 split by subject_id. Returns train_idx, test_idx."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return train_idx, test_idx


def get_eeg_windows_for_nn(X_eeg, n_samples=None):
    """Return (N, 1, channels, samples) for neural nets. Uses stimulus window (WINDOW_SAMPLES), padded if needed."""
    if n_samples is None:
        n_samples = WINDOW_SAMPLES
    windows = []
    for eeg in X_eeg:
        epoch = extract_stimulus_epoch(eeg)
        if epoch.shape[0] >= n_samples:
            epoch = epoch[:n_samples]
        else:
            epoch = np.pad(epoch, ((0, n_samples - epoch.shape[0]), (0, 0)))
        windows.append(epoch.T)  # (channels, samples)
    X = np.stack(windows)[:, np.newaxis, :, :]  # (N, 1, 6, n_samples)
    return X.astype(np.float32)
