"""Single-packet deep analysis — KernelCo/robot_control dataset."""
import numpy as np, json, glob, os
from numpy.fft import rfft, rfftfreq

DATA_DIR = "data/robot_control/data"

# --- Pick a motor-task packet ---
files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
chosen = None
for f in files:
    lab = np.load(f, allow_pickle=True)['label'].item()
    if lab['label'] in ('Right Fist', 'Left Fist', 'Both Fists'):
        chosen = f
        break
if not chosen:
    chosen = files[0]

arr = np.load(chosen, allow_pickle=True)
label = arr['label'].item()
eeg = arr['feature_eeg']      # (7499, 6)
nirs = arr['feature_moments']  # (72, 40, 3, 2, 3)

CH = ['AFF6','AFp2','AFp1','AFF5','FCz','CPz']
SDS = ['short(0-10mm)','medium(10-25mm)','long(25-60mm)']
WL = ['690nm','905nm']
MOM = ['log10(sum)','mean_tof','variance']
FS_EEG, FS_NIRS = 500, 4.76
STIM_ONSET_S = 3.0

print(f"File: {os.path.basename(chosen)}")
print(f"Label: {label['label']} | Subject: {label['subject_id']} | Session: {label['session_id']} | Stim dur: {label['duration']:.1f}s")

# ── Dataset-wide quick scan ──
label_counts = {}
subjects = set()
nan_files = 0
for f in files:
    a = np.load(f, allow_pickle=True)
    l = a['label'].item()['label']
    label_counts[l] = label_counts.get(l, 0) + 1
    subjects.add(a['label'].item()['subject_id'])
    if np.isnan(a['feature_eeg']).any():
        nan_files += 1

print(f"\n=== DATASET OVERVIEW ({len(files)} packets, {len(subjects)} subjects) ===")
for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:<16} {v:>4} ({100*v/len(files):.0f}%)")
print(f"Files with EEG NaNs: {nan_files}")

# ── EEG Analysis ──
print(f"\n=== EEG ({eeg.shape}) ===")
cut = int(STIM_ONSET_S * FS_EEG)
rest, stim = eeg[:cut], eeg[cut:]

print(f"{'Chan':<6} {'µ_rest':>8} {'µ_stim':>8} {'σ_rest':>8} {'σ_stim':>8} {'Δµ':>8}")
for i, ch in enumerate(CH):
    r, s = rest[:, i], stim[:, i]
    print(f"{ch:<6} {r.mean():>8.1f} {s.mean():>8.1f} {r.std():>8.1f} {s.std():>8.1f} {s.mean()-r.mean():>+8.1f}")

# Band power
freqs = rfftfreq(eeg.shape[0], d=1/FS_EEG)
bands = {'delta(1-4)': (1,4), 'theta(4-8)': (4,8), 'alpha(8-13)': (8,13), 'beta(13-30)': (13,30), 'gamma(30-45)': (30,45)}
print(f"\n{'Chan':<6}", end='')
for b in bands: print(f" {b:>12}", end='')
print()
for i, ch in enumerate(CH):
    spec = np.abs(rfft(eeg[:, i]))**2
    print(f"{ch:<6}", end='')
    for b, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        print(f" {np.log10(spec[mask].mean()):>12.2f}", end='')
    print()

# Rest vs Stim band power (alpha ERD check)
print(f"\n=== Alpha ERD (8-13Hz power: rest→stim) ===")
freqs_r = rfftfreq(rest.shape[0], d=1/FS_EEG)
freqs_s = rfftfreq(stim.shape[0], d=1/FS_EEG)
alpha_r_mask = (freqs_r >= 8) & (freqs_r <= 13)
alpha_s_mask = (freqs_s >= 8) & (freqs_s <= 13)
for i, ch in enumerate(CH):
    pr = np.abs(rfft(rest[:, i]))**2
    ps = np.abs(rfft(stim[:, i]))**2
    a_rest = np.log10(pr[alpha_r_mask].mean())
    a_stim = np.log10(ps[alpha_s_mask].mean())
    erd = 100 * (a_stim - a_rest) / abs(a_rest)
    print(f"  {ch}: rest={a_rest:.2f} stim={a_stim:.2f} ERD={erd:+.1f}%")

# ── TD-NIRS Analysis (use medium SDS to avoid NaNs) ──
sds_idx = 1  # medium
print(f"\n=== TD-NIRS — medium SDS, 690nm ===")
nirs_slice = nirs[:, :, sds_idx, 0, :]  # (72, 40, 3)
nan_pct = 100 * np.isnan(nirs_slice).sum() / nirs_slice.size
print(f"NaN%: {nan_pct:.1f}%")

cut_n = int(STIM_ONSET_S * FS_NIRS)
rest_n, stim_n = nirs_slice[:cut_n], nirs_slice[cut_n:]

for m, mn in enumerate(MOM):
    rv = np.nanmean(rest_n[:, :, m])
    sv = np.nanmean(stim_n[:, :, m])
    print(f"  {mn:<12} rest={rv:.4f} stim={sv:.4f} Δ={sv-rv:+.4f}")

# Top responding modules
log_rest = np.nanmean(rest_n[:, :, 0], axis=0)
log_stim = np.nanmean(stim_n[:, :, 0], axis=0)
delta_mod = log_stim - log_rest
valid = ~np.isnan(delta_mod)
top5 = np.argsort(np.abs(delta_mod[valid]))[-5:][::-1]
valid_idx = np.where(valid)[0]
print(f"\nTop 5 modules by |Δlog10sum| (medium SDS, 690nm):")
for j in top5:
    mi = valid_idx[j]
    print(f"  Module {mi:2d}: Δ={delta_mod[mi]:+.4f}")

# Data quality summary
print(f"\n=== DATA QUALITY ===")
print(f"EEG: {eeg.dtype}, NaN={np.isnan(eeg).sum()}/{eeg.size}")
print(f"NIRS: {nirs.dtype}, NaN={np.isnan(nirs).sum()}/{nirs.size} ({100*np.isnan(nirs).sum()/nirs.size:.1f}%)")
print(f"NIRS NaN by SDS: short={np.isnan(nirs[:,:,0,:,:]).sum()}, med={np.isnan(nirs[:,:,1,:,:]).sum()}, long={np.isnan(nirs[:,:,2,:,:]).sum()}")
