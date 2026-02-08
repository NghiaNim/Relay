"""Generate slide-ready visuals: EEG+Robot GIF, pipeline diagram, EEG panel."""
import numpy as np, json, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from PIL import Image

plt.rcParams.update({
    'figure.facecolor': '#0a0e17',
    'axes.facecolor': '#0a0e17',
    'text.color': '#e0e6ed',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'axes.edgecolor': '#1e293b',
    'font.family': 'monospace',
    'font.size': 10,
})

ACTION_COLORS = {'FORWARD':'#22c55e','BACKWARD':'#f472b6','LEFT':'#38bdf8','RIGHT':'#a78bfa','STOP':'#475569'}
CLASS_COLORS = {'left':'#38bdf8','right':'#a78bfa','both':'#22c55e','tongue':'#f472b6','rest':'#475569'}
CH_NAMES = ['AFF6','AFp2','AFp1','AFF5','FCz','CPz']
CH_COLORS = ['#38bdf8','#a78bfa','#f472b6','#fb923c','#22c55e','#facc15']

OUT_DIR = "viz/slides"
os.makedirs(OUT_DIR, exist_ok=True)

# Load stream
with open("viz/stream.json") as f:
    stream = json.load(f)
print(f"Loaded {len(stream)} frames")

# ═══════════════════════════════════════════════════
# 1. GIF: EEG + Robot Grid combined animation
# ═══════════════════════════════════════════════════
print("Generating GIF...")

GRID = 20
robotX, robotY, heading = 10.0, 10.0, -math.pi/2
trail = [(robotX, robotY)]
STEP = 0.8
TURN = math.pi / 4

def apply_action(action):
    global robotX, robotY, heading
    if action == 'FORWARD':
        robotX += math.cos(heading) * STEP
        robotY += math.sin(heading) * STEP
    elif action == 'BACKWARD':
        robotX -= math.cos(heading) * STEP * 0.6
        robotY -= math.sin(heading) * STEP * 0.6
    elif action == 'LEFT':
        heading -= TURN
    elif action == 'RIGHT':
        heading += TURN
    robotX = max(0.5, min(GRID-0.5, robotX))
    robotY = max(0.5, min(GRID-0.5, robotY))
    trail.append((robotX, robotY))
    if len(trail) > 60:
        trail.pop(0)

frames_pil = []
N_GIF = min(len(stream), 50)  # 50 frames for GIF

for idx in range(N_GIF):
    f = stream[idx]
    apply_action(f['robot_action'])

    fig, (ax_eeg, ax_grid) = plt.subplots(1, 2, figsize=(14, 5),
        gridspec_kw={'width_ratios': [1.3, 1]})

    # ── Left: EEG ──
    eeg = np.array(f['eeg'])
    N = eeg.shape[0]
    for ch in range(6):
        offset = (5 - ch) * 2.2
        ax_eeg.plot(np.linspace(0, 15, N), eeg[:, ch] + offset,
                    color=CH_COLORS[ch], linewidth=0.8, alpha=0.85)
        ax_eeg.text(-0.3, offset, CH_NAMES[ch], fontsize=8, color=CH_COLORS[ch],
                    ha='right', va='center')

    # Stim onset line
    ax_eeg.axvline(3, color='#334155', linestyle='--', linewidth=0.7)
    ax_eeg.text(3.2, 12.5, 'stim', fontsize=7, color='#475569')

    ax_eeg.set_xlim(-0.5, 15)
    ax_eeg.set_ylim(-2, 13.5)
    ax_eeg.set_xlabel('Time (s)', fontsize=9)
    ax_eeg.set_yticks([])
    ax_eeg.set_title(f'EEG — True: {f["true_label"]}', fontsize=11,
                     color=CLASS_COLORS.get(f.get('predicted_class','rest'), '#94a3b8'))
    ax_eeg.spines['top'].set_visible(False)
    ax_eeg.spines['right'].set_visible(False)
    ax_eeg.spines['left'].set_visible(False)

    # ── Right: Robot Grid ──
    ax_grid.set_xlim(0, GRID)
    ax_grid.set_ylim(0, GRID)
    ax_grid.set_aspect('equal')
    ax_grid.set_xticks(range(0, GRID+1, 5))
    ax_grid.set_yticks(range(0, GRID+1, 5))
    ax_grid.grid(True, color='#1a2332', linewidth=0.3)
    ax_grid.tick_params(labelsize=7)

    action = f['robot_action']
    color = ACTION_COLORS.get(action, '#475569')

    # Trail
    if len(trail) > 1:
        for i in range(1, len(trail)):
            alpha = (i / len(trail)) * 0.6
            ax_grid.plot([trail[i-1][0], trail[i][0]], [trail[i-1][1], trail[i][1]],
                        color=color, alpha=alpha, linewidth=1.5)

    # Robot triangle
    size = 0.9
    pts = np.array([
        [robotX + math.cos(heading) * size, robotY + math.sin(heading) * size],
        [robotX + math.cos(heading + 2.5) * size * 0.55, robotY + math.sin(heading + 2.5) * size * 0.55],
        [robotX + math.cos(heading - 2.5) * size * 0.55, robotY + math.sin(heading - 2.5) * size * 0.55],
    ])
    tri = patches.Polygon(pts, closed=True, facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.9)
    ax_grid.add_patch(tri)

    # Origin marker
    ax_grid.plot(10, 10, 'o', color='#1e293b', markersize=4)

    # Action label
    ax_grid.set_title(f'{action}  (conf: {f["confidence"]*100:.0f}%)',
                     fontsize=12, fontweight='bold', color=color)

    # Frame number
    fig.text(0.5, 0.01, f'Frame {idx+1}/{N_GIF}  |  Pred: {f["predicted_class"]}  |  ⚡ RELAY',
             ha='center', fontsize=8, color='#475569')

    plt.tight_layout(pad=1.5)

    # Save frame to PIL
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', (w, h), buf)
    frames_pil.append(img.convert('RGB'))
    plt.close(fig)

# Save GIF
gif_path = f"{OUT_DIR}/relay_brain_robot.gif"
frames_pil[0].save(gif_path, save_all=True, append_images=frames_pil[1:],
                    duration=400, loop=0, optimize=True)
sz = os.path.getsize(gif_path) / 1024
print(f"  → {gif_path} ({sz:.0f} KB, {len(frames_pil)} frames)")

# ═══════════════════════════════════════════════════
# 2. Static: EEG brainwave panel (single beautiful frame)
# ═══════════════════════════════════════════════════
print("Generating EEG panel...")
f = stream[5]  # Pick a motor imagery frame
eeg = np.array(f['eeg'])
N = eeg.shape[0]

fig, ax = plt.subplots(figsize=(12, 6))
for ch in range(6):
    offset = (5 - ch) * 2.4
    ax.plot(np.linspace(0, 15, N), eeg[:, ch] * 0.8 + offset,
            color=CH_COLORS[ch], linewidth=0.9, alpha=0.9)
    ax.text(-0.6, offset, CH_NAMES[ch], fontsize=11, color=CH_COLORS[ch],
            ha='right', va='center', fontweight='bold')

# Stim regions
ax.axvspan(0, 3, alpha=0.05, color='#64748b')
ax.axvspan(3, 15, alpha=0.04, color='#22c55e')
ax.axvline(3, color='#22c55e', linestyle='--', linewidth=1, alpha=0.5)
ax.text(1.5, 13.5, 'REST', fontsize=10, color='#64748b', ha='center', fontweight='bold')
ax.text(9, 13.5, 'STIMULUS', fontsize=10, color='#22c55e', ha='center', fontweight='bold')

ax.set_xlim(-1, 15.5)
ax.set_ylim(-2.5, 14.5)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_yticks([])
ax.set_title(f'6-Channel EEG — {f["true_label"]}', fontsize=14, pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
eeg_path = f"{OUT_DIR}/eeg_brainwave.png"
fig.savefig(eeg_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  → {eeg_path}")

# ═══════════════════════════════════════════════════
# 3. Pipeline diagram: EEG → Features → Model → Robot
# ═══════════════════════════════════════════════════
print("Generating pipeline diagram...")

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

boxes = [
    (1, 2, 'EEG Signal\n6ch x 500Hz\n15s window', '#38bdf8'),
    (4, 2, 'Feature\nExtraction\n42 features', '#a78bfa'),
    (7, 2, 'Classifier\nLogReg / GBM', '#f472b6'),
    (10, 2, 'Action Gate\nconf > 50%', '#fb923c'),
    (13, 2, 'Robot\nCmd Vel', '#22c55e'),
]

for x, y, text, color in boxes:
    rect = patches.FancyBboxPatch((x - 1.2, y - 1.1), 2.4, 2.2,
        boxstyle="round,pad=0.2", facecolor=color + '22',
        edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            color=color, fontweight='bold', linespacing=1.4)

# Arrows
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + 1.3
    x2 = boxes[i+1][0] - 1.3
    ax.annotate('', xy=(x2, 2), xytext=(x1, 2),
                arrowprops=dict(arrowstyle='->', color='#475569', lw=2))

# Title
ax.text(7, 3.7, 'RELAY  --  Brain to Robot Pipeline', ha='center', fontsize=14,
        fontweight='bold', color='#e0e6ed')

# Action mapping at bottom
mapping_text = 'left→LEFT  |  right→RIGHT  |  both→FORWARD  |  tongue→BACKWARD  |  rest→STOP'
ax.text(7, 0.15, mapping_text, ha='center', fontsize=8, color='#64748b')

plt.tight_layout()
pipe_path = f"{OUT_DIR}/pipeline_diagram.png"
fig.savefig(pipe_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  → {pipe_path}")

# ═══════════════════════════════════════════════════
# 4. Action distribution + confidence chart
# ═══════════════════════════════════════════════════
print("Generating stats figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Action distribution
actions = [f['robot_action'] for f in stream]
action_names = ['FORWARD','BACKWARD','LEFT','RIGHT','STOP']
counts = [actions.count(a) for a in action_names]
colors = [ACTION_COLORS[a] for a in action_names]
bars = ax1.bar(action_names, counts, color=colors, edgecolor='#1e293b', linewidth=0.5)
ax1.set_title('Action Distribution', fontsize=12)
ax1.set_ylabel('Count')
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(count), ha='center', fontsize=9, color='#94a3b8')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Confidence over time
confs = [f['confidence'] for f in stream]
pred_classes = [f['predicted_class'] for f in stream]
pred_colors = [CLASS_COLORS.get(c, '#475569') for c in pred_classes]
ax2.bar(range(len(confs)), confs, color=pred_colors, width=1.0, alpha=0.7)
ax2.axhline(0.5, color='#ef4444', linestyle='--', linewidth=1, alpha=0.6)
ax2.text(len(confs)+1, 0.5, 'threshold', fontsize=8, color='#ef4444', va='center')
ax2.set_title('Prediction Confidence per Trial', fontsize=12)
ax2.set_xlabel('Trial')
ax2.set_ylabel('Confidence')
ax2.set_ylim(0, 1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2)
stats_path = f"{OUT_DIR}/action_stats.png"
fig.savefig(stats_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  → {stats_path}")

# ═══════════════════════════════════════════════════
# 5. Robot path overview (full session on grid)
# ═══════════════════════════════════════════════════
print("Generating robot path overview...")

# Reset and replay full stream
rX, rY, rH = 10.0, 10.0, -math.pi/2
full_trail = [(rX, rY)]
trail_actions = ['STOP']
for f in stream:
    a = f['robot_action']
    if a == 'FORWARD':
        rX += math.cos(rH) * STEP; rY += math.sin(rH) * STEP
    elif a == 'BACKWARD':
        rX -= math.cos(rH) * STEP * 0.6; rY -= math.sin(rH) * STEP * 0.6
    elif a == 'LEFT':
        rH -= TURN
    elif a == 'RIGHT':
        rH += TURN
    rX = max(0.5, min(GRID-0.5, rX))
    rY = max(0.5, min(GRID-0.5, rY))
    full_trail.append((rX, rY))
    trail_actions.append(a)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, GRID)
ax.set_ylim(0, GRID)
ax.set_aspect('equal')
ax.grid(True, color='#1a2332', linewidth=0.3)
ax.set_xticks(range(0, GRID+1, 5))
ax.set_yticks(range(0, GRID+1, 5))

# Draw path segments colored by action
for i in range(1, len(full_trail)):
    color = ACTION_COLORS.get(trail_actions[i], '#475569')
    alpha = 0.3 + 0.7 * (i / len(full_trail))
    ax.plot([full_trail[i-1][0], full_trail[i][0]],
            [full_trail[i-1][1], full_trail[i][1]],
            color=color, alpha=alpha, linewidth=1.5)

# Start and end markers
ax.plot(10, 10, 'o', color='#22c55e', markersize=10, zorder=5)
ax.text(10.3, 10.3, 'START', fontsize=8, color='#22c55e', fontweight='bold')
ax.plot(full_trail[-1][0], full_trail[-1][1], 's', color='#ef4444', markersize=8, zorder=5)
ax.text(full_trail[-1][0]+0.3, full_trail[-1][1]+0.3, 'END', fontsize=8, color='#ef4444', fontweight='bold')

ax.set_title(f'Robot Path — {len(stream)} steps', fontsize=13, pad=10)

# Legend
for action, color in ACTION_COLORS.items():
    ax.plot([], [], color=color, linewidth=3, label=action)
ax.legend(loc='upper right', fontsize=8, framealpha=0.3, edgecolor='#1e293b')

plt.tight_layout()
path_img = f"{OUT_DIR}/robot_path.png"
fig.savefig(path_img, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  → {path_img}")

print(f"\n✅ All slides saved to {OUT_DIR}/")
print("Files:")
for fn in os.listdir(OUT_DIR):
    sz = os.path.getsize(f"{OUT_DIR}/{fn}") / 1024
    print(f"  {fn:30s} {sz:6.0f} KB")
