# Training Spec

Standard config for all model experiments. Every model must use these settings for fair comparison.

## Data Split

| Set | Size | Method |
|-----|------|--------|
| Train | 80% | `StratifiedGroupKFold` or `GroupShuffleSplit` by `subject_id` |
| Test | 20% | Held-out subjects (never seen during training) |

**Why group split?** Subjects have different brain patterns. Splitting by subject prevents data leakage and measures true generalization.

## Random Seeds

| Purpose | Seed |
|---------|------|
| Data split | `42` |
| Model initialization | `42` |
| NumPy RNG | `np.random.seed(42)` |
| PyTorch (if used) | `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)` |
| sklearn | `random_state=42` everywhere |

## Cross-Validation

- **Method:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- **Groups:** `subject_id` (6 subjects → leave-~1-out per fold)
- **Stratify by:** class label (ensures balanced folds)

## Feature Extraction

| Parameter | Value |
|-----------|-------|
| Sample rate | 500 Hz |
| Stimulus onset | t=3s (sample index 1500) |
| Input window | t=3s → t=15s (stimulus period only) |
| Channels | 6 EEG (`AFF6, AFp2, AFp1, AFF5, FCz, CPz`) |
| Bands | delta (1–4Hz), theta (4–8Hz), alpha (8–13Hz), beta (13–30Hz), gamma (30–45Hz) |
| Features per channel | 5 band powers (log10) + mean + std = 7 |
| Total features | 6 × 7 = **42** |
| NaN handling | `np.nan_to_num(x, nan=0.0)` |

## Preprocessing

```python
# Standard for all models
scaler = StandardScaler()  # fit on train, transform both
```

No bandpass filter in baseline. Models that add filtering must document it.

## Metrics (report all)

| Metric | How |
|--------|-----|
| Accuracy | Overall + per-class |
| F1 | Macro-averaged |
| Confusion matrix | 5×5 |
| Inference latency | Mean ± std over 100 predictions (ms) |
| Model size | File size on disk |

## Export Format

All models saved as `.joblib` dict:

```python
{
    "pipeline": sklearn_pipeline,       # scaler + classifier
    "label_encoder": LabelEncoder,      # fitted
    "feature_names": list[str],         # 42 feature names
    "label_map": dict,                  # raw label → clean key
    "config": {
        "fs": 500,
        "stim_onset": 3.0,
        "bands": {...},
        "n_channels": 6,
    },
    "metrics": {
        "cv_accuracy": float,
        "cv_f1_macro": float,
        "inference_ms": float,
        "model_size_kb": float,
    },
}
```

Saved to `models/<model_name>.joblib`. The `predict.py` module loads whichever model is specified.

## Model Registry

| Model | File | Status |
|-------|------|--------|
| Logistic Regression | `baseline_logreg.joblib` | ✅ Trained |
| CSP + LDA | `csp_lda.joblib` | ✅ Trained |
| Logistic Regression v2 | `logreg.joblib` | ✅ Trained |
| Random Forest | `rf.joblib` | ✅ Trained |

## Reproducibility Checklist

- [ ] `random_state=42` in all sklearn objects
- [ ] `np.random.seed(42)` at top of script
- [ ] Split by `subject_id`, not random row split
- [ ] StandardScaler fit on train only
- [ ] Report all metrics from table above
- [ ] Save model using export format above
