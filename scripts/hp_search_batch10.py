"""Quick HP search for batch_size=10 on mini real data.
Search: lr × warmup_steps (phase/reset already fixed from earlier)."""
import torch, os, sys, time, copy, numpy as np, itertools, math
sys.path.insert(0, ".")
from munch import Munch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

with open("checkpoints/config.yaml") as f:
    BASE = Munch.fromDict(yaml.safe_load(f))
BASE.s2mel.length_regulator.in_channels = 256
BASE.s2mel.length_regulator.f0_condition = True
BASE.s2mel.length_regulator.n_f0_bins = 256
BASE.s2mel.DiT.f0_condition = True
BASE.training = Munch(pretrained_s2mel="checkpoints/s2mel.pth", reset_cond_projection=False)

from indextts.vc_train.model_adapter import build_vc_model
from indextts.vc_train.train import vc_train_step
from indextts.vc_train.dataset import VCDataset, vc_collate_fn
from indextts.vc_train.manifest import read_jsonl
from indextts.vc.kmeans_quantizer import KMeansQuantizer

# Mini real dataset
kmeans = KMeansQuantizer(n_clusters=20)
kmeans.load("data/vc/mini_real/kmeans_codebook.pt")
entries = []
for m in read_jsonl("data/vc/mini_real/manifest.jsonl"):
    if m.duration_s > 6.0:
        continue
    stem = os.path.splitext(os.path.basename(m.audio_path))[0]
    e = m.to_dict()
    e["feature_base"] = os.path.join("data/vc/mini_real/preprocessed", stem)
    entries.append(e)
ds = VCDataset(entries=entries, f0_strategy="source_contour")
print(f"Dataset: {len(ds)} utterances")

# Search space
LRS = [2e-4, 5e-4, 1e-3, 2e-3]
WARMUPS = [500, 1000, 2000]
BATCH = 10
MAX_STEPS = 200
TOTAL_STEPS = 300000  # for cosine schedule

results = []
total = len(LRS) * len(WARMUPS)
idx = 0

for lr, warmup in itertools.product(LRS, WARMUPS):
    idx += 1
    name = f"lr{lr:.0e}_wu{warmup}"
    print(f"\n[{idx}/{total}] {name}")

    model, rp, fp = build_vc_model(copy.deepcopy(BASE))
    model = model.cuda().train()
    for g in rp + fp:
        for p in g["params"]:
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(rp + fp, lr=lr, weight_decay=1e-2)
    def make_sched(warmup_s):
        def fn(step):
            if step < warmup_s:
                return step / max(1, warmup_s)
            progress = (step - warmup_s) / max(1, TOTAL_STEPS - warmup_s)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return fn
    scheduler = LambdaLR(optimizer, lr_lambda=make_sched(warmup))

    loader = torch.utils.data.DataLoader(
        ds, batch_size=min(BATCH, len(ds)), shuffle=True,
        collate_fn=vc_collate_fn, drop_last=True,
    )
    writer = SummaryWriter(f"runs/hp_batch10/{name}")

    losses = []
    step = 0
    t0 = time.time()
    for epoch in range(MAX_STEPS):
        for batch in loader:
            if step >= MAX_STEPS:
                break
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = vc_train_step(model, batch, device="cuda")
            if not torch.isfinite(loss):
                step += 1
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
        if step >= MAX_STEPS:
            break

    writer.close()
    avg = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
    nan_pct = 100 * (MAX_STEPS - len(losses)) / MAX_STEPS
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f} avg_last50={avg:.4f} nan={nan_pct:.0f}%")
    results.append({"name": name, "lr": lr, "warmup": warmup, "avg": avg, "nan_pct": nan_pct})
    del model, optimizer
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("RESULTS (sorted by avg_last50)")
print("=" * 60)
results.sort(key=lambda r: r["avg"])
for i, r in enumerate(results):
    mark = " <-- BEST" if i == 0 else ""
    print(f"  {r['name']:25s} avg={r['avg']:.4f} nan={r['nan_pct']:.0f}%{mark}")

best = results[0]
print(f"\nBest: lr={best['lr']}, warmup={best['warmup']}")
print("TensorBoard: runs/hp_batch10/")
