# Relay

EEG + TD-NIRS → robot control prediction. Real-time BCI pipeline for hackathon.

## Status

- [x] Dataset pulled: `KernelCo/robot_control`
- [x] Single-packet EDA complete
- [x] Baseline model trained (logistic regression, 22% acc, chance=20%)
- [x] Output contract defined for viz integration
- [x] Robot animation dashboard (EEG + confidence + decision gate)
- [x] Reference pipeline analyzed (`bri`) — discrepancies documented
- [x] Output contract updated: EEG class → robot action + cmd_vel mapping
- [ ] Improved model (bandpass + CSP or neural net)
- [ ] Temporal smoothing / sliding window predictions
- [ ] `bri` MuJoCo sim integration

## Quick Start

```bash
# Launch viz dashboard
cd viz && python3 -m http.server 8888
# Open http://localhost:8888

# Predict on a sample
python3 scripts/predict.py

# Retrain baseline
python3 scripts/train_baseline.py

# Regenerate viz stream data
python3 scripts/generate_stream.py
```

```python
# Use in code
from scripts.predict import predict
import numpy as np

eeg = np.load("data/robot_control/data/0b2dbd41-10.npz", allow_pickle=True)["feature_eeg"]
result = predict(eeg)  # → dict with predicted_class, confidence, all_probabilities, etc.
```

## Pipeline

```
EEG motor imagery → [ML Model] → Intent class → [Mapping] → Robot action → [bri Controller] → Robot moves
```

| Brain Task | EEG Class | Robot Action |
|------------|-----------|-------------|
| Left fist | `left` | `LEFT` (turn left) |
| Right fist | `right` | `RIGHT` (turn right) |
| Both fists | `both` | `FORWARD` (walk) |
| Tongue tap | `tongue` | `STOP` |
| Relax | `rest` | `STOP` |

See [`gameplan/output_contract.md`](gameplan/output_contract.md) for full spec.

## Dataset

**900 packets** | **6 subjects** | **5 classes** (balanced 20% each)

| Signal | Shape | Rate |
|--------|-------|------|
| EEG | (7499, 6) | 500Hz |
| TD-NIRS | (72, 40, 3, 2, 3) | 4.76Hz |

## File Structure

```
Relay/
├── README.md
├── .cursor/rules/         # Project rules
├── gameplan/
│   ├── gameplan.md        # Team plan & roles
│   └── output_contract.md # Model output spec for viz integration
├── scripts/
│   ├── analyze_packet.py  # Single-packet EDA
│   ├── train_baseline.py  # Train logistic regression
│   ├── predict.py         # Inference module (import this)
│   └── generate_stream.py # Generate viz stream JSON
├── viz/
│   ├── index.html         # Dashboard (robot + EEG + confidence)
│   └── stream.json        # Pre-generated prediction stream
├── models/
│   └── baseline_logreg.joblib
├── reference/
│   └── brain-robot-interface/  # bri — robot sim/control API
└── data/
    └── robot_control/     # HF dataset (900 .npz files)
```
