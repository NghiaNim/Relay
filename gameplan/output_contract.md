# Model Output Contract

## Quick Start

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
  "all_probabilities": {"both": 0.03, "left": 0.87, "rest": 0.02, "right": 0.04, "tongue": 0.04},
  "is_above_threshold": true,
  "robot_action": "LEFT",
  "cmd_vel": {"vx": 0.0, "vy": 0.0, "yaw_rate": 1.5},
  "timestamp_ms": 1702847362000,
  "inference_latency_ms": 7.2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted_class` | `str` | EEG intent: `left`, `right`, `both`, `tongue`, `rest` |
| `confidence` | `float` | Top-class probability (0–1) |
| `all_probabilities` | `dict[str, float]` | All 5 class probabilities, sum ≈ 1.0 |
| `is_above_threshold` | `bool` | `confidence > threshold` (default 0.5) |
| `robot_action` | `str` | Robot command: `FORWARD`, `BACKWARD`, `LEFT`, `RIGHT`, `STOP` |
| `cmd_vel` | `dict` | Velocity: `{vx, vy, yaw_rate}` — what the robot consumes |
| `timestamp_ms` | `int` | Unix epoch ms |
| `inference_latency_ms` | `float` | Wall-clock inference time (ms) |

## EEG → Robot Action Mapping

| Brain Task | EEG Class | Robot Action | CmdVel `(vx, vy, yaw)` | Rationale |
|------------|-----------|-------------|------------------------|-----------|
| Left fist clench | `left` | `LEFT` | `(0, 0, +1.5)` | Motor imagery maps to direction |
| Right fist clench | `right` | `RIGHT` | `(0, 0, -1.5)` | Motor imagery maps to direction |
| Both fists clench | `both` | `FORWARD` | `(0.6, 0, 0)` | Bilateral activation = "go" |
| Tongue tap | `tongue` | `BACKWARD` | `(-0.4, 0, 0)` | Distinct non-hand action = reverse |
| Relax | `rest` | `STOP` | `(0, 0, 0)` | No intent = idle |

**Gating:** If `is_above_threshold == false`, `robot_action` is forced to `STOP` regardless of `predicted_class`. This prevents jittery commands when the model is uncertain.

## Input

- `eeg_window`: `np.ndarray` shape `(7499, 6)` — 15s at 500Hz, 6 EEG channels
- Channels: `[AFF6, AFp2, AFp1, AFF5, FCz, CPz]` in µV
- First 3s is rest baseline, stimulus starts at t=3s

## Integration with Robot Sim (`bri`)

```python
from bri import Action, Controller

ACTION_MAP = {
    "left": Action.LEFT,
    "right": Action.RIGHT,
    "both": Action.FORWARD,
    "tongue": "BACKWARD",  # extend bri.Action for backward
    "rest": Action.STOP,
}

ctrl = Controller(backend="sim", hold_s=0.3, smooth_alpha=0.2)
ctrl.start()

# Prediction loop:
result = predict(eeg_window)
if result["is_above_threshold"]:
    ctrl.set_action(ACTION_MAP[result["predicted_class"]])
else:
    ctrl.set_action(Action.STOP)
```

The `bri` controller handles:
- **Velocity smoothing** via exponential moving average (`smooth_alpha=0.2`)
- **Auto-stop timeout** after `hold_s=0.3s` of no new action
- **MuJoCo sim** rendering at 50Hz control loop

## For Viz Engineer

Use the output JSON directly to drive the dashboard:

| Field | Dashboard Panel |
|-------|----------------|
| `all_probabilities` | Class probability bar chart |
| `robot_action` | Robot animation direction |
| `cmd_vel` | Velocity vector overlay |
| `confidence` + `is_above_threshold` | Decision gate indicator |
| `inference_latency_ms` | Latency display |

Stream format: `viz/stream.json` — array of frames, each containing the prediction output + downsampled EEG snippet for the waveform panel.

## Model Details

| Property | Value |
|----------|-------|
| Model | Logistic Regression (L2, C=1.0) |
| Features | 42 (6ch × 5 band-powers + 2 stats) |
| Bands | delta(1-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-45) Hz |
| CV Accuracy | 22% (5-fold subject-wise, chance=20%) |
| Inference | ~3-7ms |
| Weights | `models/baseline_logreg.joblib` |

## Next Steps

1. Improve model accuracy (bandpass filter, CSP features, or EEGNet)
2. Sliding window predictions (~2s at 2-5Hz) instead of full 15s windows
3. Temporal smoothing / hysteresis in prediction layer
4. End-to-end `bri` MuJoCo integration
