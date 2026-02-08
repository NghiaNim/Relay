# Relay

EEG → robot control prediction. Real-time BCI pipeline for hackathon.

## Status

- [x] Dataset: `robot_control/` (EEG only)
- [x] 11 models: logreg, riemann_svm, knn, lightgbm, xgboost, mlp, 1dcnn, tcn, lstm, transformer, mamba
- [x] 80/20 train/test split by subject
- [x] Performance: accuracy, precision, recall
- [x] Efficiency: inference latency at scale (5k runs)
- [x] Output contract for viz integration

## Quick Start

```bash
pip install -r requirements.txt

# Train one model
bash train.sh logreg

# Train all models
bash train.sh

# Run inference + evaluation (test-set outputs + latency at scale)
bash run_inference.sh logreg
# Writes: results/test_inference/<model>_predictions.json (output contract)
#         results/latency_at_scale/<model>_eval.json (metrics)
```

```bash
# Legacy models (Relay/models/): inference + results under models/test_inference, models/latency_at_scale
./run_legacy_inference.sh
# Or: python scripts/run_legacy_inference.py --models-dir models

# Viz dashboard
cd viz && python3 -m http.server 8888

# Legacy: predict on sample (uses models/baseline_logreg.joblib)
python3 scripts/predict.py
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
| Tongue tap | `tongue` | `BACKWARD` (reverse) |
| Relax | `rest` | `STOP` (idle) |

See [`gameplan/output_contract.md`](gameplan/output_contract.md) for full spec.

## Dataset

**900 packets** | **6 subjects** | **5 classes** (balanced 20% each)

| Signal | Shape | Rate |
|--------|-------|------|
| EEG | (7499, 6) | 500Hz |
| TD-NIRS | (72, 40, 3, 2, 3) | 4.76Hz |

## Models

| Model | Input | Description |
|-------|-------|-------------|
| logreg | Bandpower (42) | Logistic Regression |
| riemann_svm | Riemannian (21) | Covariance + tangent space + SVM |
| knn | Bandpower | k-Nearest Neighbors |
| lightgbm | Bandpower | LightGBM |
| xgboost | Bandpower | XGBoost |
| mlp | Bandpower | MLPClassifier |
| 1dcnn | Raw EEG | 1D CNN |
| tcn | Raw EEG | Temporal Convolutional Network |
| lstm | Raw EEG | LSTM |
| transformer | Raw EEG | Self-attention transformer |
| mamba | Raw EEG | SSM-style (Mamba-inspired) |
| **eegml** | Raw EEG (padded) | [EEG-ML](https://github.com/NeuroTechX/eeg-ml) OneDCNN trained on robot_control |

Data: 80/20 split by `subject_id`. EEG stimulus window (t≥3s), 6 channels.

### Training and inference with EEG-ML (external repo)

To train the [EEG-ML](https://github.com/NeuroTechX/eeg-ml) OneDCNN on robot_control data and run inference with the same output contract:

1. **Export data** (robot_control → EEG-ML format: 7 ch × 801 time, per-subject .npy):
   ```bash
   cd Relay && python scripts/export_for_eegml.py [--out-dir data/eegml_export]
   ```

2. **Train** (requires EEG-ML repo at `../EEG-ML` or set `EEGML_ROOT`):
   ```bash
   cd Relay && python scripts/train_eegml_on_robot.py [--export-dir data/eegml_export] [--k 5] [--epochs 30]
   ```
   Saves `models/eegml_onedcnn.keras`.

3. **Inference** (single window → output contract):
   ```bash
   python scripts/predict_eegml.py   # sample from robot_control/data
   ```
   Or in code: `from scripts.predict_eegml import predict_eegml; predict_eegml(eeg_window)`.

4. **Evaluation** (test set + latency at scale, same as other models):
   ```bash
  bash run_inference.sh eegml
   ```
   Writes `results/test_inference/eegml_predictions.json` and `results/latency_at_scale/eegml_eval.json`.

## File Structure

```
Relay/
├── README.md
├── requirements.txt
├── train.sh              # Train model(s)
├── run_inference.sh      # Inference + eval
├── gameplan/
│   ├── training_spec.md
│   └── output_contract.md
├── scripts/
│   ├── data_utils.py
│   ├── models_registry.py
│   ├── nn_models.py
│   ├── train.py
│   ├── run_inference.py   # Eval + latency at scale
│   ├── run_legacy_inference.py  # Legacy models (predict.py–compatible)
│   └── predict.py         # Load baseline, predict(eeg) -> output contract
├── models/               # Legacy .joblib; results in models/test_inference, models/latency_at_scale
├── run_legacy_inference.sh
├── results/
│   ├── test_inference/   # Per-sample output (output_contract schema)
│   └── latency_at_scale/ # Scaling metrics (accuracy, latency, etc.)
└── viz/
```
