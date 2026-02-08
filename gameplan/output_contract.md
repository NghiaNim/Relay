# Model Output Contract

## `predict(eeg_window) → dict`

```python
from scripts.predict import predict
import numpy as np

eeg = np.load("data/robot_control/data/0b2dbd41-10.npz", allow_pickle=True)["feature_eeg"]
result = predict(eeg)
```

## Output Schema

```json
{
  "predicted_class": "left",
  "confidence": 0.87,
  "all_probabilities": {
    "both": 0.03,
    "left": 0.87,
    "rest": 0.02,
    "right": 0.04,
    "tongue": 0.04
  },
  "is_above_threshold": true,
  "timestamp_ms": 1702847362000,
  "inference_latency_ms": 7.2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted_class` | `str` | One of: `left`, `right`, `both`, `tongue`, `rest` |
| `confidence` | `float` | Probability of top class (0–1) |
| `all_probabilities` | `dict[str, float]` | All 5 class probabilities, sum to 1.0 |
| `is_above_threshold` | `bool` | `confidence > 0.5` (configurable) |
| `timestamp_ms` | `int` | Unix epoch ms at prediction time |
| `inference_latency_ms` | `float` | Wall-clock inference time in ms |

## Input

- `eeg_window`: `np.ndarray` shape `(7499, 6)` — 15s at 500Hz, 6 EEG channels
- Channels: `[AFF6, AFp2, AFp1, AFF5, FCz, CPz]` in µV

## Model Details

| Property | Value |
|----------|-------|
| Model | Logistic Regression (L2, C=1.0) |
| Features | 42 (6ch × 5 band-powers + 2 stats) |
| Bands | delta(1-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-45) Hz |
| CV Accuracy | 22% (5-fold subject-wise, chance=20%) |
| Inference | ~7ms |
| Weights | `models/baseline_logreg.joblib` |

## For Viz Engineer

```python
# In your dashboard loop:
from scripts.predict import predict
import numpy as np

# Load or stream EEG window
eeg_window = ...  # shape (7499, 6)
result = predict(eeg_window)

# Use result directly:
label = result["predicted_class"]       # → drives robot
conf = result["confidence"]             # → confidence bar
probs = result["all_probabilities"]     # → probability chart
gate = result["is_above_threshold"]     # → decision gate indicator
latency = result["inference_latency_ms"] # → latency display
```

## Notes

- Baseline is barely above chance (22% vs 20%). Next: bandpass filter, CSP features, or neural net.
- `is_above_threshold` will almost always be `false` at this accuracy. Still use it — it'll matter when model improves.
- Model loads once on import. `predict()` calls are stateless and fast.
