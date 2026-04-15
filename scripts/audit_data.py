"""Audit preprocessed data quality: mel, f0, content — find bad files."""
import numpy as np
import glob
import os

prep = "data/vc/aishell3/preprocessed"

mel_files = sorted(glob.glob(os.path.join(prep, "*.mel.npy")))
print(f"Checking {len(mel_files)} mel files...")

bad = []
stats = {"nan": 0, "inf": 0, "huge": 0, "tiny_std": 0, "ok": 0}
all_mins, all_maxs, all_means, all_stds = [], [], [], []

for i, mf in enumerate(mel_files):
    m = np.load(mf)
    mn, mx, mean, std = m.min(), m.max(), m.mean(), m.std()
    all_mins.append(mn)
    all_maxs.append(mx)
    all_means.append(mean)
    all_stds.append(std)

    if np.isnan(m).any():
        stats["nan"] += 1
        bad.append(("NaN", os.path.basename(mf), m.shape))
    elif np.isinf(m).any():
        stats["inf"] += 1
        bad.append(("Inf", os.path.basename(mf), m.shape))
    elif abs(mx) > 20 or abs(mn) > 20:
        stats["huge"] += 1
        bad.append(("huge", os.path.basename(mf), f"range=[{mn:.2f},{mx:.2f}]"))
    elif std < 0.01:
        stats["tiny_std"] += 1
        bad.append(("tiny_std", os.path.basename(mf), f"std={std:.6f}"))
    else:
        stats["ok"] += 1

    if (i + 1) % 10000 == 0:
        print(f"  [{i+1}/{len(mel_files)}] ok={stats['ok']} bad={sum(v for k,v in stats.items() if k!='ok')}")

print(f"\n=== MEL STATS ===")
print(f"  ok={stats['ok']} nan={stats['nan']} inf={stats['inf']} huge={stats['huge']} tiny_std={stats['tiny_std']}")
print(f"  Bad files: {len(bad)}")
for kind, name, info in bad[:15]:
    print(f"    {kind}: {name} — {info}")
print(f"\n  Overall distribution (all files):")
print(f"    min:  p0={np.min(all_mins):.2f}  p50={np.median(all_mins):.2f}  p100={np.max(all_mins):.2f}")
print(f"    max:  p0={np.min(all_maxs):.2f}  p50={np.median(all_maxs):.2f}  p100={np.max(all_maxs):.2f}")
print(f"    mean: p0={np.min(all_means):.2f} p50={np.median(all_means):.2f} p100={np.max(all_means):.2f}")
print(f"    std:  p0={np.min(all_stds):.2f}  p50={np.median(all_stds):.2f}  p100={np.max(all_stds):.2f}")

# Also check a few f0 and content files
print(f"\n=== F0 SAMPLE ===")
f0_files = sorted(glob.glob(os.path.join(prep, "*.f0.npy")))[:100]
f0_voiced = []
for ff in f0_files:
    f = np.load(ff)
    vr = np.mean(f > 0)
    f0_voiced.append(vr)
print(f"  Voiced ratio (first 100): mean={np.mean(f0_voiced):.2f} min={np.min(f0_voiced):.2f}")

print(f"\n=== CONTENT SAMPLE ===")
ct_files = sorted(glob.glob(os.path.join(prep, "*.content.npy")))[:100]
ct_ranges = []
for cf in ct_files:
    c = np.load(cf)
    ct_ranges.append((c.min(), c.max(), c.mean(), c.std()))
mins, maxs, means, stds = zip(*ct_ranges)
print(f"  range: [{np.min(mins):.2f}, {np.max(maxs):.2f}]")
print(f"  mean: {np.mean(means):.4f}, std: {np.mean(stds):.4f}")
