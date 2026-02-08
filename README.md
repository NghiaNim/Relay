# Relay

EEG + TD-NIRS → robot control prediction.

## Status

- [x] Dataset pulled: `KernelCo/robot_control` → `data/robot_control/`
- [x] Single-packet analysis complete
- [ ] Model training

## Dataset

**Source:** [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control)  
**900 packets** | **6 subjects** | **5 classes** (balanced 20% each)

| Signal | Shape | Rate | Channels |
|--------|-------|------|----------|
| EEG | (7499, 6) | 500Hz | AFF6, AFp2, AFp1, AFF5, FCz, CPz |
| TD-NIRS | (72, 40, 3, 2, 3) | 4.76Hz | 40 modules × 3 SDS × 2λ × 3 moments |

**Labels:** Right Fist, Left Fist, Both Fists, Tongue Tapping, Relax  
**Protocol:** 0–3s rest → 3–15s stimulus (~9–10s cue duration)

## Analysis Findings (packet `0b2dbd41-10.npz`, Left Fist)

### EEG
- **Large DC offset** (12k–57k µV) — needs baseline correction or high-pass filter.
- **Clear stimulus response:** mean amplitude shifts +78 to +248 µV on frontal channels, −234 µV on CPz (contralateral suppression for left fist).
- **Alpha ERD during stimulus:** +10–28% increase in log-power (note: alpha *increase* here likely due to DC-dominated spectrum; proper ERD requires bandpass first).
- **Power spectrum:** 1/f dominated. Delta >> theta >> alpha >> beta >> gamma. Standard.
- **35/900 files have EEG NaNs** — need handling.

### TD-NIRS
- **Medium SDS (10–25mm):** 0% NaN — best modality for modeling.
- **Long SDS (25–60mm):** 100% NaN in this packet (3.3% of total NIRS data) — sparse, exclude or impute.
- **Short SDS (0–10mm):** Clean, useful for systemic regression.
- **Minimal hemodynamic shift** in single trial (Δlog10sum ≈ −0.0001) — expected for single-trial fNIRS; need averaging or ML.
- **Top responding modules:** 14, 23, 6, 24, 22 — motor-adjacent regions.

### Key Takeaways
1. EEG is the stronger signal for single-trial classification (clear amplitude + spectral changes).
2. TD-NIRS adds spatial resolution but needs trial averaging or deep learning for single-trial use.
3. Preprocessing needed: bandpass filter EEG, handle NaNs, baseline-correct both modalities.
4. Balanced dataset is good for classification — no class weighting needed.

## File Structure

```
Relay/
├── README.md              # This file (project state)
├── .cursor/rules/         # Project rules (.mdc)
├── scripts/
│   └── analyze_packet.py  # Single-packet analysis
└── data/
    └── robot_control/     # HF dataset (900 .npz files)
```
