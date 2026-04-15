"""Quick verification of v2 training: 100 steps on mini real data."""
import torch, os, sys, yaml, numpy as np, math, time
sys.path.insert(0, ".")
from munch import Munch
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR

# --- Setup ---
accelerator = Accelerator(mixed_precision="bf16")
device = accelerator.device
print(f"Device: {device}, bf16: {accelerator.mixed_precision}")

with open("checkpoints/config.yaml") as f:
    cfg = Munch.fromDict(yaml.safe_load(f))
cfg.s2mel.length_regulator.in_channels = 256
cfg.s2mel.length_regulator.f0_condition = True
cfg.s2mel.length_regulator.n_f0_bins = 256
cfg.s2mel.DiT.f0_condition = True
cfg.training = Munch(pretrained_s2mel="checkpoints/s2mel.pth", reset_cond_projection=False)

# --- Model + EMA ---
from indextts.vc_train.model_adapter import build_vc_model
from indextts.vc_train.train import vc_train_step
model, rp, fp = build_vc_model(cfg)

try:
    from ema_pytorch import EMA
    # Remove weight_norm before EMA deepcopy (weight_norm breaks deepcopy)
    from torch.nn.utils import remove_weight_norm
    for module in model.modules():
        try:
            remove_weight_norm(module)
        except ValueError:
            pass
    ema = EMA(model, beta=0.9999, update_every=1)
    print("EMA: enabled (weight_norm removed for deepcopy compatibility)")
except ImportError:
    ema = None
    print("EMA: NOT installed!")
except Exception as e:
    ema = None
    print(f"EMA: failed ({e}), training without EMA")

# --- Dataset ---
from indextts.vc_train.dataset import VCDataset, vc_collate_fn
from indextts.vc_train.manifest import read_jsonl
from indextts.vc.kmeans_quantizer import KMeansQuantizer

kmeans = KMeansQuantizer(n_clusters=20)
kmeans.load("data/vc/mini_real/kmeans_codebook.pt")
entries = []
for m in read_jsonl("data/vc/mini_real/manifest.jsonl"):
    stem = os.path.splitext(os.path.basename(m.audio_path))[0]
    e = m.to_dict()
    e["feature_base"] = os.path.join("data/vc/mini_real/preprocessed", stem)
    entries.append(e)

ds = VCDataset(entries=entries[:20], f0_strategy="source_contour")
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=vc_collate_fn, drop_last=True)

# --- Optimizer ---
for g in rp + fp:
    for p in g["params"]:
        p.requires_grad_(True)
optimizer = torch.optim.AdamW(rp + fp, lr=2e-4, weight_decay=1e-2)

def warmup_cosine(step):
    if step < 20:
        return step / 20
    return 0.5 * (1 + math.cos(math.pi * (step - 20) / 80))
scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)

# --- Prepare ---
model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

# --- Train 100 steps ---
model.train()
t0 = time.time()
step = 0
nan_count = 0

print("\n--- Training 100 steps (verify v2) ---")
for epoch in range(100):
    for batch in loader:
        if step >= 100:
            break

        with accelerator.accumulate(model):
            loss = vc_train_step(accelerator.unwrap_model(model), batch, device=str(device))

            if not torch.isfinite(loss):
                nan_count += 1
                optimizer.zero_grad()
                step += 1
                continue

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), 10.0)
            else:
                grad_norm = torch.tensor(0.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if ema and accelerator.sync_gradients:
            ema.update()

        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        if step % 20 == 0:
            print(f"Step {step:3d}: loss={loss.item():.4f} grad={gn:.2f} lr={optimizer.param_groups[0]['lr']:.2e}")
        step += 1
    if step >= 100:
        break

elapsed = time.time() - t0
print(f"\n--- Verification complete ---")
print(f"  100 steps in {elapsed:.1f}s ({100/elapsed:.1f} step/s)")
print(f"  Final loss: {loss.item():.4f}")
print(f"  NaN skipped: {nan_count}")
print(f"  EMA: {'active' if ema else 'disabled'}")

# Check EMA weights differ from model weights
if ema:
    model_w = list(accelerator.unwrap_model(model).parameters())[0].data
    ema_w = list(ema.ema_model.parameters())[0].data.to(model_w.device)
    diff = (model_w - ema_w).abs().mean().item()
    print(f"  EMA vs model weight diff: {diff:.6f} (should be small but > 0)")

print("\nVERIFICATION PASSED" if nan_count < 10 else f"\nWARNING: {nan_count} NaN steps")
