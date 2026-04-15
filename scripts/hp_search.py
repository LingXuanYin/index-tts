"""Hyperparameter grid search on mini smoke data.

Searches: lr × phase_strategy × reset_cond_projection
Each config runs 50 steps. Results logged to TensorBoard (runs/hp_search/).
"""
import torch
import os
import sys
import copy
import numpy as np
import itertools

sys.path.insert(0, ".")
from munch import Munch
import yaml
from torch.utils.tensorboard import SummaryWriter

# ---------- Base config ----------
with open("checkpoints/config.yaml") as f:
    BASE_CFG = Munch.fromDict(yaml.safe_load(f))
BASE_CFG.s2mel.length_regulator.in_channels = 256
BASE_CFG.s2mel.length_regulator.f0_condition = True
BASE_CFG.s2mel.length_regulator.n_f0_bins = 256
BASE_CFG.s2mel.DiT.f0_condition = True
if not hasattr(BASE_CFG, "training"):
    BASE_CFG.training = Munch()
BASE_CFG.training.pretrained_s2mel = "checkpoints/s2mel.pth"

# ---------- Dataset (shared across all runs) ----------
from indextts.vc_train.dataset import VCDataset, vc_collate_fn
from indextts.vc_train.manifest import read_jsonl
from indextts.vc.kmeans_quantizer import KMeansQuantizer

kmeans = KMeansQuantizer(n_clusters=5)
kmeans.load("data/vc/smoke/kmeans_codebook.pt")

entries_raw = read_jsonl("data/vc/smoke/manifest.jsonl")
entries = []
for m in entries_raw:
    stem = os.path.splitext(os.path.basename(m.audio_path))[0]
    e = m.to_dict()
    e["feature_base"] = os.path.join("data/vc/smoke/preprocessed", stem)
    entries.append(e)

ds = VCDataset(entries=entries, f0_strategy="source_contour")
print(f"Dataset: {len(ds)} samples")

# ---------- Search space ----------
LRS = [5e-5, 1e-4, 2e-4]
PHASE_STRATEGIES = ["two_phase", "single_phase"]
RESET_COND_PROJ = [False, True]
MAX_STEPS = 50

from indextts.vc_train.model_adapter import build_vc_model
from indextts.vc_train.train import vc_train_step

results = []
total_configs = len(LRS) * len(PHASE_STRATEGIES) * len(RESET_COND_PROJ)
run_idx = 0

for lr, phase_strat, reset_cp in itertools.product(LRS, PHASE_STRATEGIES, RESET_COND_PROJ):
    run_idx += 1
    run_name = f"lr{lr:.0e}_{phase_strat}_cp{'R' if reset_cp else 'K'}"
    print(f"\n[{run_idx}/{total_configs}] {run_name}")

    # Fresh model for each run
    cfg = copy.deepcopy(BASE_CFG)
    cfg.training.reset_cond_projection = reset_cp
    model, reset_params, finetune_params = build_vc_model(cfg)
    model = model.cuda().train()

    reset_tensors = [p for g in reset_params for p in g["params"]]
    finetune_tensors = [p for g in finetune_params for p in g["params"]]

    if phase_strat == "two_phase":
        # Phase 1: only reset params
        for p in finetune_tensors:
            p.requires_grad_(False)
        for p in reset_tensors:
            p.requires_grad_(True)
        optimizer = torch.optim.AdamW(reset_params, lr=lr)
    else:
        # Single phase: all params
        for p in finetune_tensors:
            p.requires_grad_(True)
        for p in reset_tensors:
            p.requires_grad_(True)
        all_groups = reset_params + finetune_params
        optimizer = torch.optim.AdamW(all_groups, lr=lr)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=True, collate_fn=vc_collate_fn, drop_last=True
    )
    writer = SummaryWriter(f"runs/hp_search/{run_name}")

    losses = []
    global_step = 0
    for epoch in range(MAX_STEPS):
        for batch in loader:
            if global_step >= MAX_STEPS:
                break
            batch = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            loss = vc_train_step(model, batch, device="cuda")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            losses.append(loss_val)
            writer.add_scalar("train/loss", loss_val, global_step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
            global_step += 1
        if global_step >= MAX_STEPS:
            break

    writer.close()

    avg_last10 = np.mean(losses[-10:])
    result = {
        "run": run_name,
        "lr": lr,
        "phase": phase_strat,
        "reset_cp": reset_cp,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "avg_last10": avg_last10,
        "all_finite": all(np.isfinite(l) for l in losses),
    }
    results.append(result)
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f} (avg_last10={avg_last10:.4f})")

    # Free GPU memory
    del model, optimizer
    torch.cuda.empty_cache()

# ---------- Summary ----------
print("\n" + "=" * 70)
print("HYPERPARAMETER SEARCH RESULTS")
print("=" * 70)
results.sort(key=lambda r: r["avg_last10"])
for i, r in enumerate(results):
    marker = " <-- BEST" if i == 0 else ""
    print(
        f"  {r['run']:40s} avg_last10={r['avg_last10']:.4f} "
        f"({r['loss_first']:.4f}->{r['loss_last']:.4f}){marker}"
    )

best = results[0]
print(f"\nBest config: lr={best['lr']}, phase={best['phase']}, "
      f"reset_cond_projection={best['reset_cp']}")
print(f"Best avg_last10 loss: {best['avg_last10']:.4f}")
print(f"\nAll results logged to runs/hp_search/ — check TensorBoard.")
