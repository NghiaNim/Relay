# ThoughtLink — Team Plan

Decode brain signals → high-level intent → control a humanoid robot in real time.

---

## What We're Building

A real-time pipeline: **EEG signal → ML classifier → decision gate → robot simulation**, wrapped in a dashboard that visualizes the entire loop.

---

## Team Roles

### Role 1: ML Engineer (Technical)

Build the intent decoder. Takes a windowed EEG signal, outputs a prediction.

**Model output contract (JSON per prediction cycle):**

```json
{
  "predicted_class": "left",
  "confidence": 0.87,
  "all_probabilities": {"left": 0.87, "right": 0.04, "both": 0.03, "tongue": 0.02, "rest": 0.04},
  "is_above_threshold": true,
  "robot_action": "LEFT",
  "cmd_vel": {"vx": 0.0, "vy": 0.0, "yaw_rate": 1.5},
  "timestamp_ms": 1702847362000,
  "inference_latency_ms": 4.2
}
```

**EEG → Robot mapping:** `left`→LEFT, `right`→RIGHT, `both`→FORWARD, `tongue`→BACKWARD, `rest`→STOP.
Gated by confidence: if `is_above_threshold == false`, force STOP.

**Deliverables:**
1. Preprocessing pipeline (windowing, filtering)
2. Trained model — start with logistic regression/SVM, then iterate
3. A `predict(signal_window) → JSON` function the viz engineer can call
4. Accuracy, latency, confusion matrix metrics
5. Comparison of simple vs. complex model with latency–accuracy tradeoff

---

### Role 2: Visualization Engineer (Technical)

Build a real-time dashboard that consumes the model output and drives the demo.

**Four panels:**

| Panel | Shows |
|-------|-------|
| Signal Viewer | Scrolling EEG waveform feeding into the model |
| Model Confidence | Live bar chart of class probabilities |
| Decision Gate | Current command + confidence + threshold indicator |
| Robot State | Simulation view showing robot position and actions |

**Deliverables:**
1. Working dashboard (React, Streamlit, or Dash)
2. Integrates with `predict()` JSON output
3. Integrates with provided robot simulation/execution script
4. Polished, demo-ready interface

---

### Role 3: Research & Strategy Lead (Non-Technical)

Own the depth that turns a demo into a winning submission.

**Deliverables:**
1. **Scalability argument** — 1-pager on how this scales to 100+ robots (latency budgets, parallel inference, attention bottlenecks)
2. **Competitive scan** — summarize existing BCI approaches (Neuralink, EEGNet, OpenBCI) and what's different about ours
3. **Failure mode analysis** — document signal noise, electrode drift, inter-subject variability; propose mitigations
4. **Metric narratives** — work with ML engineer to turn raw numbers into compelling charts and talking points

---

### Role 4: Presentation & Demo Lead (Non-Technical)

Own the pitch. Make judges understand and remember our work.

**Deliverables:**
1. **Pitch deck** — problem → approach → architecture → demo → results → scalability vision → open questions
2. **Demo script** — step-by-step walkthrough + backup video recording in case live demo fails
3. **Visual assets** — architecture diagrams, pipeline flow graphics
4. **Time management** — track milestones, flag blockers, keep the team on schedule

---

## Timeline (24–36hr hackathon)

| Phase | ML Engineer | Viz Engineer | Non-Technical |
|-------|------------|--------------|---------------|
| Hours 1–4 | Load data, explore, preprocess | Scaffold app, connect simulation | Research BCI landscape, start scalability doc |
| Hours 5–8 | Binary baseline, measure accuracy + latency | Build panels with mock data | Draft deck outline, architecture diagram |
| Hours 9–14 | Multi-class, compare models | Integrate real model output | Failure modes, refine scalability |
| Hours 15–20 | Add smoothing/hysteresis, optimize | Polish dashboard, end-to-end test | Finalize slides, write demo script |
| Hours 21+ | Freeze model, produce metrics | Bug fixes, record backup video | Full rehearsal, Q&A prep |

---

## How We Win

| Most teams will... | We will also... |
|--------------------|-----------------|
| Train a classifier | Compare models and quantify latency vs. accuracy |
| Plug into the simulation | Show a full dashboard: signal → model → decision → robot |
| Do single-frame predictions | Add temporal smoothing, hysteresis, confidence gating |
| Show it works | Document failure modes and propose mitigations |
| Build a demo | Deliver a rehearsed pitch with a scalability vision |

---

## Resources

- **Dataset:** [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control)
- **Reference pipeline:** [Nabla7/brain-robot-interface](https://github.com/Nabla7/brain-robot-interface)
- **Provided:** ~5hr brain dataset, humanoid simulation, execution script

> *A fast, simple model that works beats a slow, complex model that doesn't.*
