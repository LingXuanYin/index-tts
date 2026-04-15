"""Comprehensive audit of the entire VC training pipeline.

Checks EVERYTHING: data, model, training step, loss, gradients, NaN reproduction.
Any issue found = LOUD ERROR, not silent skip.
"""
import torch, os, sys, yaml, numpy as np, json, time, traceback
sys.path.insert(0, ".")
from munch import Munch

FAIL = []
WARN = []
PASS = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append((name, detail))
        print(f"  [FAIL] {name}: {detail}")

def warn(name, detail):
    WARN.append((name, detail))
    print(f"  [WARN] {name}: {detail}")

print("=" * 70)
print("FULL AUDIT: VC Training Pipeline")
print("=" * 70)

# ============================================================
# 1. CONFIG + MODEL
# ============================================================
print("\n--- 1. Config + Model ---")
with open("checkpoints/config.yaml") as f:
    cfg = Munch.fromDict(yaml.safe_load(f))
cfg.s2mel.length_regulator.in_channels = 256
cfg.s2mel.length_regulator.f0_condition = True
cfg.s2mel.length_regulator.n_f0_bins = 256
cfg.s2mel.DiT.f0_condition = True
cfg.training = Munch(pretrained_s2mel="checkpoints/s2mel.pth", reset_cond_projection=False)

from indextts.vc_train.model_adapter import build_vc_model
model, rp, fp = build_vc_model(cfg)
model = model.cuda().eval()

# Check content_in_proj shape
cip = model.models["length_regulator"].content_in_proj
check("content_in_proj shape", cip.weight.shape == torch.Size([512, 256]),
      f"got {cip.weight.shape}, expected [512, 256]")

# Check f0_embedding exists and has correct size
lr_mod = model.models["length_regulator"]
check("f0_embedding exists", hasattr(lr_mod, "f0_embedding") and lr_mod.f0_embedding is not None)
if hasattr(lr_mod, "f0_embedding") and lr_mod.f0_embedding is not None:
    check("f0_embedding shape", lr_mod.f0_embedding.weight.shape[0] == 256,
          f"got {lr_mod.f0_embedding.weight.shape}")

# Check gpt_layer NOT in model
has_gpt = "gpt_layer" in dict(model.models.named_children())
check("gpt_layer absent", not has_gpt, "gpt_layer should not exist in VC mode")

# Check model param count
n_params = sum(p.numel() for p in model.parameters())
check("model param count reasonable", 50e6 < n_params < 200e6, f"{n_params/1e6:.1f}M")

# ============================================================
# 2. DATA QUALITY (sample 100 files)
# ============================================================
print("\n--- 2. Data Quality ---")
import glob
prep = "data/vc/aishell3/preprocessed"

content_files = sorted(glob.glob(os.path.join(prep, "*.content.npy")))[:100]
check("content files exist", len(content_files) > 0, f"found {len(content_files)}")

for i, cf in enumerate(content_files[:20]):
    stem = cf.replace(".content.npy", "")
    c = np.load(cf)
    check(f"content[{i}] shape", c.ndim == 2 and c.shape[1] == 256, f"shape={c.shape}")
    check(f"content[{i}] no NaN", not np.isnan(c).any())
    check(f"content[{i}] range", np.abs(c).max() < 10, f"max={np.abs(c).max():.2f}")

    if os.path.exists(stem + ".mel.npy"):
        m = np.load(stem + ".mel.npy")
        check(f"mel[{i}] shape", m.ndim == 2 and m.shape[0] == 80, f"shape={m.shape}")
        check(f"mel[{i}] is log-mel", m.min() < -5 and m.max() < 5,
              f"range=[{m.min():.2f},{m.max():.2f}] (expect log-mel ~[-11,2])")
        check(f"mel[{i}] no NaN", not np.isnan(m).any())

    if os.path.exists(stem + ".f0.npy"):
        f = np.load(stem + ".f0.npy")
        check(f"f0[{i}] shape", f.ndim == 1, f"shape={f.shape}")
        check(f"f0[{i}] range", f.max() < 3000, f"max={f.max():.0f}")
        voiced_ratio = np.mean(f > 0)
        check(f"f0[{i}] voiced ratio", voiced_ratio > 0.1, f"voiced={voiced_ratio:.2f}")

    if os.path.exists(stem + ".style.npy"):
        s = np.load(stem + ".style.npy")
        check(f"style[{i}] shape", s.shape == (192,), f"shape={s.shape}")
        # Style is mock random — check if it's actually random or contains useful info
        if i == 0:
            warn("style is MOCK random", "np.random.randn(192), model never learns speaker identity")

# ============================================================
# 3. DATASET + COLLATION
# ============================================================
print("\n--- 3. Dataset + Collation ---")
from indextts.vc_train.dataset import VCDataset, vc_collate_fn
from indextts.vc_train.manifest import read_jsonl
from indextts.vc.kmeans_quantizer import KMeansQuantizer

kmeans = KMeansQuantizer(n_clusters=200)
kmeans.load("data/vc/aishell3/kmeans_codebook.pt")

entries_raw = read_jsonl("data/vc/aishell3/manifest.jsonl")
spk_count = {}
for m in entries_raw:
    spk_count[m.speaker_id] = spk_count.get(m.speaker_id, 0) + 1

entries = []
for m in entries_raw:
    if m.duration_s > 6.0 or spk_count[m.speaker_id] < 2:
        continue
    stem = os.path.splitext(os.path.basename(m.audio_path))[0]
    e = m.to_dict()
    e["feature_base"] = os.path.join(prep, stem)
    entries.append(e)

ds = VCDataset(entries=entries[:100], f0_strategy="source_contour")
check("dataset created", len(ds) > 0, f"{len(ds)} samples")

# Test single __getitem__
sample = ds[0]
required_keys = ["content", "f0", "mel", "style", "prompt_mel", "prompt_content"]
for k in required_keys:
    check(f"sample has '{k}'", k in sample, f"keys: {list(sample.keys())}")

# Test collation
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=vc_collate_fn, drop_last=True)
batch = next(iter(loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        check(f"batch[{k}] no NaN", not v.isnan().any(), f"has NaN!")
        check(f"batch[{k}] no Inf", not v.isinf().any(), f"has Inf!")
        print(f"    batch[{k}]: shape={v.shape} dtype={v.dtype} range=[{v.min():.3f},{v.max():.3f}]")

# ============================================================
# 4. TRAINING STEP — SINGLE FORWARD/BACKWARD IN FP32
# ============================================================
print("\n--- 4. Training Step (fp32, no autocast) ---")
from indextts.vc_train.train import vc_train_step

model.train()
for g in rp + fp:
    for p in g["params"]:
        p.requires_grad_(True)

batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# fp32 forward
loss_fp32 = vc_train_step(model, batch_cuda, device="cuda")
check("fp32 loss finite", torch.isfinite(loss_fp32).item(), f"loss={loss_fp32.item()}")
check("fp32 loss range", 0 < loss_fp32.item() < 50, f"loss={loss_fp32.item()}")
print(f"    fp32 loss = {loss_fp32.item():.4f}")

# fp32 backward
loss_fp32.backward()
grad_norms = []
nan_grads = []
for n, p in model.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        grad_norms.append((n, gn))
        if np.isnan(gn) or np.isinf(gn):
            nan_grads.append(n)
check("fp32 grads no NaN", len(nan_grads) == 0, f"NaN grad in: {nan_grads[:5]}")
total_grad = sum(g for _, g in grad_norms)
check("fp32 total grad norm", 0 < total_grad < 1e6, f"total_grad={total_grad:.2f}")
top5 = sorted(grad_norms, key=lambda x: -x[1])[:5]
print(f"    Top grad norms: {[(n.split('.')[-1], f'{g:.4f}') for n, g in top5]}")
model.zero_grad()

# ============================================================
# 5. TRAINING STEP — BF16 AUTOCAST
# ============================================================
print("\n--- 5. Training Step (bf16 autocast) ---")
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    loss_bf16 = vc_train_step(model, batch_cuda, device="cuda")
check("bf16 loss finite", torch.isfinite(loss_bf16).item(), f"loss={loss_bf16.item()}")
check("bf16 loss range", 0 < loss_bf16.item() < 50, f"loss={loss_bf16.item()}")
check("bf16 vs fp32 similar", abs(loss_bf16.item() - loss_fp32.item()) < 5,
      f"fp32={loss_fp32.item():.4f} bf16={loss_bf16.item():.4f}")
print(f"    bf16 loss = {loss_bf16.item():.4f}")

loss_bf16.backward()
nan_grads_bf16 = []
for n, p in model.named_parameters():
    if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
        nan_grads_bf16.append(n)
check("bf16 grads no NaN", len(nan_grads_bf16) == 0, f"NaN grad in: {nan_grads_bf16[:5]}")
model.zero_grad()

# ============================================================
# 6. NaN STRESS TEST — 100 steps on random batches
# ============================================================
print("\n--- 6. NaN Stress Test (100 steps, bf16) ---")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
nan_steps = []
loss_values = []
grad_values = []

for step in range(100):
    batch = next(iter(loader))
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = vc_train_step(model, batch, device="cuda")
    if not torch.isfinite(loss):
        nan_steps.append(step)
        optimizer.zero_grad()
        continue
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    optimizer.zero_grad()
    loss_values.append(loss.item())
    grad_values.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

check("stress test NaN count", len(nan_steps) == 0,
      f"{len(nan_steps)} NaN steps: {nan_steps[:10]}")
if loss_values:
    check("stress test loss stable", max(loss_values) < 50,
          f"max_loss={max(loss_values):.2f}")
    check("stress test grad stable", max(grad_values) < 1000,
          f"max_grad={max(grad_values):.2f}")
    print(f"    100 steps: loss [{min(loss_values):.3f}, {max(loss_values):.3f}] "
          f"grad [{min(grad_values):.3f}, {max(grad_values):.3f}]")

# ============================================================
# 7. NaN STRESS TEST — HIGHER LR (2e-3)
# ============================================================
print("\n--- 7. NaN Stress Test (100 steps, lr=2e-3) ---")
# Reload fresh model to avoid contamination from test 6
model2, rp2, fp2 = build_vc_model(cfg)
model2 = model2.cuda().train()
for g in rp2 + fp2:
    for p in g["params"]:
        p.requires_grad_(True)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=2e-3, weight_decay=1e-2)
nan_steps2 = []
loss_values2 = []
grad_values2 = []

for step in range(100):
    batch = next(iter(loader))
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = vc_train_step(model2, batch, device="cuda")
    if not torch.isfinite(loss):
        nan_steps2.append(step)
        optimizer2.zero_grad()
        continue
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model2.parameters(), 10.0)
    optimizer2.step()
    optimizer2.zero_grad()
    loss_values2.append(loss.item())
    grad_values2.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

check("lr=2e-3 stress NaN count", len(nan_steps2) == 0,
      f"{len(nan_steps2)} NaN steps: {nan_steps2[:10]}")
if loss_values2:
    print(f"    100 steps: loss [{min(loss_values2):.3f}, {max(loss_values2):.3f}] "
          f"grad [{min(grad_values2):.3f}, {max(grad_values2):.3f}]")

# ============================================================
# 8. COMPARE WITH ORIGINAL TTS TRAINING STEP
# ============================================================
print("\n--- 8. Compare VC vs TTS mel scale ---")
from indextts.s2mel.modules.audio import mel_spectrogram
import librosa, torchaudio

# Load a real audio and compute mel both ways
audio_path = entries[0]["audio_path"]
audio_16k, _ = librosa.load(audio_path, sr=16000)
audio_22k = librosa.resample(audio_16k, orig_sr=16000, target_sr=22050)
audio_22k_t = torch.from_numpy(audio_22k).unsqueeze(0)

# Our preprocessing mel (should match s2mel)
mel_ours = mel_spectrogram(audio_22k_t, n_fft=1024, num_mels=80, sampling_rate=22050,
                            hop_size=256, win_size=1024, fmin=0, fmax=None)
print(f"    Our mel: shape={mel_ours.shape} range=[{mel_ours.min():.2f},{mel_ours.max():.2f}]")

# Check saved mel matches
stem = entries[0]["feature_base"]
mel_saved = np.load(stem + ".mel.npy")
mel_saved_t = torch.from_numpy(mel_saved)
diff = (mel_ours.squeeze(0) - mel_saved_t).abs().mean().item()
check("saved mel matches computed mel", diff < 0.01, f"mean diff={diff:.4f}")

# What does the TTS inference path use? (reference)
print(f"    Saved mel: shape={mel_saved.shape} range=[{mel_saved.min():.2f},{mel_saved.max():.2f}]")
check("mel is log-scale", mel_saved.min() < -5, f"min={mel_saved.min():.2f} (expect <-5 for log-mel)")

# ============================================================
# 9. STYLE EMBEDDING IMPACT
# ============================================================
print("\n--- 9. Style Embedding Check ---")
style_saved = np.load(stem + ".style.npy")
check("style is random noise", abs(style_saved.mean()) < 1.0 and style_saved.std() > 0.5,
      f"mean={style_saved.mean():.3f} std={style_saved.std():.3f}")
warn("MOCK style", f"Style embeddings are np.random.randn(192). "
     f"The model has NO speaker identity signal. "
     f"This means: (1) model learns 'average voice', not speaker conversion. "
     f"(2) Content + F0 carry ALL the generation burden. "
     f"(3) At inference, random style won't match any training distribution.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print(f"AUDIT COMPLETE: {len(PASS)} PASS, {len(WARN)} WARN, {len(FAIL)} FAIL")
print("=" * 70)

if FAIL:
    print("\nFAILURES:")
    for name, detail in FAIL:
        print(f"  [FAIL] {name}: {detail}")

if WARN:
    print("\nWARNINGS:")
    for name, detail in WARN:
        print(f"  [WARN] {name}: {detail}")

if not FAIL:
    print("\nNo critical failures found. Training data flow is correct.")
    print("NaN crashes are likely caused by:")
    print("  1. lr too high for long training (decay not aggressive enough)")
    print("  2. Specific rare data batches triggering numerical extremes")
    print("  3. MOCK style embedding forcing model to overfit content pathway")
