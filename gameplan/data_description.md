# Data Description

Source: [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) — collected via Kernel Flow headset.

## At a Glance

| Property | Value |
|----------|-------|
| Total samples | 900 `.npz` files |
| Subjects | 6 unique participants |
| Classes | 5, balanced (~180 each) |
| Trial length | 15 seconds |
| Modalities | EEG + TD-NIRS |

## Trial Structure

```
t=0             t=3                              t=3+duration        t=15
│── rest ──────│── stimulus (cue shown) ────────│── rest ───────────│
```

- **t=0–3s**: Baseline rest period
- **t=3s**: Stimulus cue appears (e.g. "clench left fist")
- **t=3+duration**: Cue removed, participant returns to rest
- Duration varies per trial (~9–12s typical)

## Classes → Robot Actions

| Label in Dataset | Our Key | Robot Action | Description |
|-----------------|---------|-------------|-------------|
| `Left Fist` | `left` | `LEFT` | Clench left fist → turn left |
| `Right Fist` | `right` | `RIGHT` | Clench right fist → turn right |
| `Both Fists` | `both` | `FORWARD` | Clench both fists → walk forward |
| `Tongue Tapping` | `tongue` | `BACKWARD` | Tap tongue → reverse |
| `Relax` | `rest` | `STOP` | Relax → idle |

## File Format

Each `.npz` contains 3 arrays:

```python
arr = np.load("data/robot_control/data/0b2dbd41-10.npz", allow_pickle=True)
arr.keys()  # ['feature_eeg', 'feature_moments', 'label']
```

### `feature_eeg` — EEG signal

| Property | Value |
|----------|-------|
| Shape | `(7499, 6)` |
| Meaning | `(samples, channels)` |
| Sample rate | 500 Hz |
| Units | µV (microvolts) |
| Duration | 15s × 500Hz = 7499 samples (last sample truncated) |

**Channel layout:**

```
Index:  0     1     2     3     4    5
Name:   AFF6  AFp2  AFp1  AFF5  FCz  CPz
Region: front front front front mid  mid-back
```

- Channels 0–3 (AFF/AFp): Anterior frontal — prefrontal activity
- Channel 4 (FCz): Fronto-central — motor planning
- Channel 5 (CPz): Centro-parietal — motor execution/imagery

### `feature_moments` — TD-NIRS signal

| Property | Value |
|----------|-------|
| Shape | `(72, 40, 3, 2, 3)` |
| Meaning | `(samples, modules, SDS_ranges, wavelengths, moments)` |
| Sample rate | 4.76 Hz |

**Dimensions:**

| Dim | Size | Values |
|-----|------|--------|
| samples | 72 | 15s at 4.76Hz |
| modules | 40 | Physical positions on scalp |
| SDS ranges | 3 | short (0–10mm), medium (10–25mm), long (25–60mm) |
| wavelengths | 2 | 690nm (red), 905nm (infrared) |
| moments | 3 | log10(intensity), mean time-of-flight, variance |

### `label` — Metadata

```python
arr['label'].item()
# {
#   'label': 'Both Fists',
#   'subject_id': 'fa7e4026',
#   'session_id': 'bf56a42c',
#   'duration': 9.41
# }
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | str | One of the 5 class labels |
| `subject_id` | str | Unique participant ID (6 total) |
| `session_id` | str | Unique recording session ID |
| `duration` | float | Stimulus duration in seconds |

## What We Use

Currently **EEG only** (`feature_eeg`). The 6 EEG channels are the primary input to all models.

We extract the **stimulus window** (t=3s onward) and compute:
- Band power features: delta (1–4Hz), theta (4–8Hz), alpha (8–13Hz), beta (13–30Hz), gamma (30–45Hz)
- Time-domain stats: mean, std per channel
- Total: 6 channels × 7 features = **42 features**

TD-NIRS is available but unused — potential for multimodal fusion in future iterations.
