"""Diagnose grad=0.0 issue: check if gradients flow through the model."""
import torch, sys, os, yaml, numpy as np
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

model, reset_params, finetune_params = build_vc_model(args)
model = model.cuda().train()

# Unlock all params
for g in reset_params + finetune_params:
    for p in g["params"]:
        p.requires_grad_(True)

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {n_train}/{n_total} ({100*n_train/n_total:.1f}%)")

# Load mini data
kmeans = KMeansQuantizer(n_clusters=20)
kmeans.load("data/vc/mini_real/kmeans_codebook.pt")
entries_raw = read_jsonl("data/vc/mini_real/manifest.jsonl")
entries = []
for m in entries_raw:
    stem = os.path.splitext(os.path.basename(m.audio_path))[0]
    e = m.to_dict()
    e["feature_base"] = os.path.join("data/vc/mini_real/preprocessed", stem)
    entries.append(e)
ds = VCDataset(entries=entries[:10], f0_strategy="source_contour")
loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=vc_collate_fn, drop_last=True)
batch = next(iter(loader))
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# Test 1: WITHOUT autocast
print("\n--- Test 1: NO autocast, NO scaler ---")
loss = vc_train_step(model, batch, device="cuda")
print(f"loss = {loss.item():.4f}")
loss.backward()
grads = []
for n, p in model.named_parameters():
    if p.grad is not None and p.grad.norm().item() > 0:
        grads.append((n, p.grad.norm().item()))
print(f"Params with nonzero grad: {len(grads)} / {sum(1 for p in model.parameters() if p.requires_grad)}")
if grads:
    top5 = sorted(grads, key=lambda x: -x[1])[:5]
    for name, gn in top5:
        print(f"  {name}: {gn:.6f}")
else:
    print("  ALL GRADS ZERO!")
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Params with grad tensor (even zero): {has_grad}")
total_grad = sum(g for _, g in grads)
print(f"Total grad norm: {total_grad:.6f}")
model.zero_grad()

# Test 2: WITH autocast fp16
print("\n--- Test 2: WITH autocast fp16 ---")
with torch.amp.autocast("cuda", dtype=torch.float16):
    loss2 = vc_train_step(model, batch, device="cuda")
print(f"loss = {loss2.item():.4f}, dtype = {loss2.dtype}")
loss2.backward()
grads2 = []
for n, p in model.named_parameters():
    if p.grad is not None and p.grad.norm().item() > 0:
        grads2.append((n, p.grad.norm().item()))
print(f"Params with nonzero grad: {len(grads2)}")
if grads2:
    top5 = sorted(grads2, key=lambda x: -x[1])[:5]
    for name, gn in top5:
        print(f"  {name}: {gn:.6f}")
else:
    print("  ALL GRADS ZERO with autocast!")

# Test 3: WITH scaler (full pipeline)
model.zero_grad()
print("\n--- Test 3: WITH autocast + GradScaler ---")
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    loss3 = vc_train_step(model, batch, device="cuda")
scaler.scale(loss3).backward()
scaler.unscale_(torch.optim.AdamW(model.parameters(), lr=1e-4))
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
print(f"loss = {loss3.item():.4f}, grad_norm after unscale+clip = {grad_norm.item()}")
print(f"scaler scale = {scaler.get_scale()}")
