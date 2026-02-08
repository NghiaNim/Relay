# Relay

EEG + TD-NIRS → robot control prediction. Real-time BCI pipeline for hackathon.

## Status

- [x] Dataset pulled: `KernelCo/robot_control`
- [x] Single-packet EDA complete
- [x] Baseline model trained (logistic regression, 22% acc, chance=20%)
- [x] Output contract defined for viz integration
- [ ] Improved model (bandpass + CSP or neural net)
- [ ] Dashboard integration
- [ ] Robot simulation hookup

## Quick Start

```bash
# Predict on a sample
python3 scripts/predict.py

# Retrain baseline
python3 scripts/train_baseline.py
```

```python
# Use in code
from scripts.predict import predict
import numpy as np

eeg = np.load("data/robot_control/data/0b2dbd41-10.npz", allow_pickle=True)["feature_eeg"]
result = predict(eeg)  # → dict with predicted_class, confidence, all_probabilities, etc.
```

## Output Contract

See [`gameplan/output_contract.md`](gameplan/output_contract.md) for full spec. Key fields:

```json
{
  "predicted_class": "left",
  "confidence": 0.87,
  "all_probabilities": {"both": 0.03, "left": 0.87, "rest": 0.02, "right": 0.04, "tongue": 0.04},
  "is_above_threshold": true,
  "timestamp_ms": 1702847362000,
  "inference_latency_ms": 7.2
}
```

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
│   └── predict.py         # Inference module (import this)
├── models/
│   └── baseline_logreg.joblib
└── data/
    └── robot_control/     # HF dataset (900 .npz files)
```
