"""Test different batch sizes + segment lengths to find optimal config.
Also filters out single-utterance speakers."""
import torch, os, sys, yaml, time, numpy as np, json
sys.path.insert(0, ".")
from munch import Munch

with open("checkpoints/config.yaml") as f:
    args = Munch.fromDict(yaml.safe_load(f))
args.s2mel.length_regulator.in_channels = 256
args.s2mel.length_regulator.f0_condition = True
args.s2mel.length_regulator.n_f0_bins = 256
args.s2mel.DiT.f0_condition = True
args.training = Munch(pretrained_s2mel="checkpoints/s2mel.pth", reset_cond_projection=False)

from indextts.vc_train.model_adapter import build_vc_model
from indextts.vc_train.train import vc_train_step
from indextts.vc_train.dataset import VCDataset, vc_collate_fn
from indextts.vc_train.manifest import read_jsonl
from indextts.vc.kmeans_quantizer import KMeansQuantizer

model, rp, fp = build_vc_model(args)
model = model.cuda().train()
for g in rp + fp:
    for p in g["params"]:
        p.requires_grad_(True)

DATA_DIR = os.environ.get("VC_DATA_DIR", "L:/DATASET/vc_training/preprocessed_aishell3")
kmeans = KMeansQuantizer(n_clusters=200)
kmeans.load(os.path.join(DATA_DIR, "kmeans_codebook.pt"))

# Build entries, FILTER OUT single-utterance speakers
entries_raw = read_jsonl(os.path.join(DATA_DIR, "manifest.jsonl"))
# Count speakers
spk_count = {}
for m in entries_raw:
    spk_count[m.speaker_id] = spk_count.get(m.speaker_id, 0) + 1

entries = []
filtered_spk = 0
for m in entries_raw:
    if spk_count[m.speaker_id] < 2:
        filtered_spk += 1
        continue
    if m.duration_s > 6.0:  # Crop to 6s max for higher batch
        continue
    stem = os.path.splitext(os.path.basename(m.audio_path))[0]
    e = m.to_dict()
    e["feature_base"] = os.path.join(DATA_DIR, "preprocessed", stem)
    entries.append(e)

print(f"Dataset after filtering: {len(entries)} (removed {filtered_spk} single-spk utts, cropped to <=6s)")

ds = VCDataset(entries=entries, f0_strategy="source_contour")

# Test different batch sizes
results = []
for bs in [10, 16, 20, 24, 28, 32]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, collate_fn=vc_collate_fn,
        drop_last=True, num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    t0 = time.time()
    ok = True
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = vc_train_step(model, batch, device="cuda")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  batch={bs}: OOM at step {i}")
                ok = False
                torch.cuda.empty_cache()
                break
            raise
    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1024**3

    if ok:
        sps = 5 / elapsed
        samples_per_sec = sps * bs
        print(f"  batch={bs}: {sps:.1f} step/s, {samples_per_sec:.0f} samples/s, GPU {mem:.1f}GB, loss={loss.item():.4f}")
        results.append((bs, sps, samples_per_sec, mem))
    else:
        results.append((bs, 0, 0, mem))

print("\n=== Summary ===")
print(f"{'batch':>6} {'step/s':>8} {'samples/s':>10} {'GPU_GB':>8}")
for bs, sps, samp, mem in results:
    status = f"{sps:.1f}" if sps > 0 else "OOM"
    print(f"{bs:>6} {status:>8} {samp:>10.0f} {mem:>8.1f}")

# Recommend
best = max(results, key=lambda x: x[2])
print(f"\nRecommended: batch_size={best[0]} ({best[2]:.0f} samples/s, {best[3]:.1f}GB)")
