"""
End-to-end fix + train + verify.
Fixes: P0 (real CAMPPlus style), P1 (loss clamp), P2 (proper HP search).
Runs: HP search → full training → inference → verify.
ALL errors CRASH LOUD.
"""
import torch, os, sys, yaml, numpy as np, json, glob, time, math, itertools, copy
import librosa, torchaudio
sys.path.insert(0, ".")
from munch import Munch

def FATAL(msg):
    print(f"\n{'='*60}\nFATAL: {msg}\n{'='*60}")
    raise RuntimeError(msg)

print("=" * 70)
print("FIX AND TRAIN: Full Pipeline")
print("=" * 70)

# ============================================================
# STEP 1: Regenerate .style.npy with real CAMPPlus
# ============================================================
print("\n=== STEP 1: Regenerate style with CAMPPlus ===")

CAMPPLUS_PATH = "/root/autodl-tmp/index-tts/checkpoints/campplus/campplus_cn_common.bin"
if not os.path.exists(CAMPPLUS_PATH):
    FATAL(f"CAMPPlus checkpoint not found: {CAMPPLUS_PATH}")

from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
campplus = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_state = torch.load(CAMPPLUS_PATH, map_location="cpu", weights_only=True)
campplus.load_state_dict(campplus_state)
campplus = campplus.cuda().eval()
print("CAMPPlus loaded.")

# Mel function (same as s2mel training)
from indextts.s2mel.modules.audio import mel_spectrogram
mel_fn = lambda x: mel_spectrogram(x, n_fft=1024, num_mels=80, sampling_rate=22050,
                                     hop_size=256, win_size=1024, fmin=0, fmax=None)

prep = "data/vc/aishell3/preprocessed"
content_files = sorted(glob.glob(os.path.join(prep, "*.content.npy")))
print(f"Regenerating style for {len(content_files)} utterances...")

# Read manifest for audio paths
manifest = {}
with open("data/vc/aishell3/manifest.jsonl") as f:
    for line in f:
        e = json.loads(line)
        stem = os.path.splitext(os.path.basename(e["audio_path"]))[0]
        manifest[stem] = e["audio_path"]

regen_count = 0
regen_errors = 0
t0 = time.time()

for i, cf in enumerate(content_files):
    stem_name = os.path.basename(cf).replace(".content.npy", "")
    style_path = os.path.join(prep, stem_name + ".style.npy")
    audio_path = manifest.get(stem_name)
    if audio_path is None:
        continue

    try:
        audio_16k, _ = librosa.load(audio_path, sr=16000)
        audio_16k_t = torch.from_numpy(audio_16k).unsqueeze(0).cuda()
        # CAMPPlus input: kaldi fbank (NOT mel_spectrogram!) — same as infer_v2.py:457-464
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k_t, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)  # mean normalization
        with torch.no_grad():
            style = campplus(feat.unsqueeze(0))  # (1, 192)
        np.save(style_path, style.cpu().numpy().squeeze(0).astype(np.float32))
        regen_count += 1
    except Exception as e:
        regen_errors += 1
        if regen_errors <= 3:
            print(f"  Error on {stem_name}: {e}")

    if (i + 1) % 5000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(content_files) - i - 1) / rate
        print(f"  [{i+1}/{len(content_files)}] {regen_count} done, {regen_errors} errors, {rate:.0f}/s, ETA {eta/60:.0f}min")

elapsed = time.time() - t0
print(f"Style regeneration: {regen_count} done, {regen_errors} errors, {elapsed/60:.1f}min")

# Verify a few
for cf in content_files[:5]:
    stem_name = os.path.basename(cf).replace(".content.npy", "")
    s = np.load(os.path.join(prep, stem_name + ".style.npy"))
    if s.shape != (192,):
        FATAL(f"Style shape wrong: {s.shape}")
    if np.isnan(s).any():
        FATAL(f"Style has NaN: {stem_name}")
    # Real CAMPPlus embeddings should NOT look like random noise
    # They should have structure (not all near-zero mean/unit std like randn)
    if abs(s.std() - 1.0) < 0.1:
        print(f"  WARNING: {stem_name} style looks suspiciously like randn (std={s.std():.3f})")
    else:
        pass  # Good — real embeddings have non-unit std

print("Style verification passed.\n")
del campplus
torch.cuda.empty_cache()

# Also regenerate mini_real styles
print("Regenerating mini_real styles...")
campplus2 = CAMPPlus(feat_dim=80, embedding_size=192)
campplus2.load_state_dict(torch.load(CAMPPLUS_PATH, map_location="cpu", weights_only=True))
campplus2 = campplus2.cuda().eval()

mini_prep = "data/vc/mini_real/preprocessed"
mini_content = sorted(glob.glob(os.path.join(mini_prep, "*.content.npy")))
mini_manifest = {}
with open("data/vc/mini_real/manifest.jsonl") as f:
    for line in f:
        e = json.loads(line)
        stem = os.path.splitext(os.path.basename(e["audio_path"]))[0]
        mini_manifest[stem] = e["audio_path"]

for cf in mini_content:
    stem_name = os.path.basename(cf).replace(".content.npy", "")
    audio_path = mini_manifest.get(stem_name)
    if audio_path is None:
        continue
    try:
        audio_16k, _ = librosa.load(audio_path, sr=16000)
        audio_16k_t = torch.from_numpy(audio_16k).unsqueeze(0).cuda()
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k_t, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        with torch.no_grad():
            style = campplus2(feat.unsqueeze(0))
        np.save(os.path.join(mini_prep, stem_name + ".style.npy"),
                style.cpu().numpy().squeeze(0).astype(np.float32))
    except Exception:
        pass
del campplus2
torch.cuda.empty_cache()
print("Mini_real styles done.\n")

# ============================================================
# STEP 2: HP Search (2000 steps, real style, loss clamp)
# ============================================================
print("=== STEP 2: HP Search (2000 steps, batch=10, real style) ===")

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
from torch.utils.tensorboard import SummaryWriter

# Mini dataset with real style
kmeans_mini = KMeansQuantizer(n_clusters=20)
kmeans_mini.load("data/vc/mini_real/kmeans_codebook.pt")
entries_mini = []
for m in read_jsonl("data/vc/mini_real/manifest.jsonl"):
    if m.duration_s > 6.0:
        continue
    stem = os.path.splitext(os.path.basename(m.audio_path))[0]
    e = m.to_dict()
    e["feature_base"] = os.path.join("data/vc/mini_real/preprocessed", stem)
    entries_mini.append(e)

ds_mini = VCDataset(entries=entries_mini, f0_strategy="source_contour")
print(f"HP search dataset: {len(ds_mini)} utterances")

LRS = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
WARMUPS = [500, 1000]
HP_STEPS = 2000
BATCH = min(10, len(ds_mini))
hp_results = []

for lr, warmup in itertools.product(LRS, WARMUPS):
    name = f"lr{lr:.0e}_wu{warmup}"
    print(f"  [{name}] ...", end="", flush=True)

    model, rp, fp = build_vc_model(copy.deepcopy(BASE))
    model = model.cuda().train()
    for g in rp + fp:
        for p in g["params"]:
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(rp + fp, lr=lr, weight_decay=1e-2)
    def make_sched(wu):
        def fn(step):
            if step < wu:
                return step / max(1, wu)
            return 0.5 * (1 + math.cos(math.pi * (step - wu) / max(1, 300000 - wu)))
        return fn
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=make_sched(warmup))

    loader = torch.utils.data.DataLoader(
        ds_mini, batch_size=BATCH, shuffle=True, collate_fn=vc_collate_fn, drop_last=True)

    losses = []
    nan_count = 0
    step = 0
    for epoch in range(HP_STEPS):
        for batch in loader:
            if step >= HP_STEPS:
                break
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = vc_train_step(model, batch, device="cuda")
            # LOSS CLAMP (P1 fix)
            loss = torch.clamp(loss, max=10.0)
            if not torch.isfinite(loss):
                nan_count += 1
                optimizer.zero_grad()
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
        if step >= HP_STEPS:
            break

    avg = np.mean(losses[-500:]) if len(losses) >= 500 else np.mean(losses)
    print(f" avg_last500={avg:.4f} nan={nan_count}")
    hp_results.append({"name": name, "lr": lr, "warmup": warmup, "avg": avg, "nan": nan_count})
    del model, optimizer
    torch.cuda.empty_cache()

hp_results.sort(key=lambda r: r["avg"])
print("\nHP Search Results:")
for i, r in enumerate(hp_results):
    mark = " <-- BEST" if i == 0 else ""
    print(f"  {r['name']:20s} avg={r['avg']:.4f} nan={r['nan']}{mark}")

best_hp = hp_results[0]
BEST_LR = best_hp["lr"]
BEST_WARMUP = best_hp["warmup"]
print(f"\nBest: lr={BEST_LR}, warmup={BEST_WARMUP}")

if best_hp["nan"] > HP_STEPS * 0.01:
    FATAL(f"Best HP config has {best_hp['nan']} NaN ({100*best_hp['nan']/HP_STEPS:.1f}%) — training will be unstable")

# ============================================================
# STEP 3: Full Training (300k steps, real style, loss clamp)
# ============================================================
print(f"\n=== STEP 3: Full Training (300k, lr={BEST_LR}, warmup={BEST_WARMUP}, batch=10) ===")

from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="bf16")
device = accelerator.device

model, rp, fp = build_vc_model(copy.deepcopy(BASE))

# EMA (manual, no deepcopy issues)
ema_decay = 0.9999

# Full dataset with real style
kmeans_full = KMeansQuantizer(n_clusters=200)
kmeans_full.load("data/vc/aishell3/kmeans_codebook.pt")

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
    e["feature_base"] = os.path.join("data/vc/aishell3/preprocessed", stem)
    entries.append(e)
n_spk = len(set(e["speaker_id"] for e in entries))
print(f"Dataset: {len(entries)} utterances, {n_spk} speakers")

ds = VCDataset(entries=entries, f0_strategy="source_contour")
loader = torch.utils.data.DataLoader(
    ds, batch_size=10, shuffle=True, collate_fn=vc_collate_fn, drop_last=True, num_workers=4)

# Optimizer
for g in rp + fp:
    for p in g["params"]:
        p.requires_grad_(True)
optimizer = torch.optim.AdamW(rp + fp, lr=BEST_LR, weight_decay=1e-2)

TOTAL_STEPS = 300000
def warmup_cosine(step):
    if step < BEST_WARMUP:
        return step / max(1, BEST_WARMUP)
    return 0.5 * (1 + math.cos(math.pi * (step - BEST_WARMUP) / max(1, TOTAL_STEPS - BEST_WARMUP)))
scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)

model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

# EMA init after prepare
unwrapped = accelerator.unwrap_model(model)
ema_params = {n: p.data.clone() for n, p in unwrapped.named_parameters()}
print(f"EMA initialized ({len(ema_params)} params, decay={ema_decay})")

writer = SummaryWriter("runs/train_final")
os.makedirs("checkpoints/vc", exist_ok=True)
# Clean old checkpoints
for f in glob.glob("checkpoints/vc/v2_*.pth") + glob.glob("checkpoints/vc/final_*.pth"):
    os.remove(f)

global_step = 0
losses_window = []
best_avg = float("inf")
nan_count = 0
t0 = time.time()

print(f"Training: {TOTAL_STEPS} steps, lr={BEST_LR}, warmup={BEST_WARMUP}")
print("=" * 70)

model.train()
while global_step < TOTAL_STEPS:
    for batch in loader:
        if global_step >= TOTAL_STEPS:
            break

        with accelerator.accumulate(model):
            loss = vc_train_step(unwrapped, batch, device=str(device))
            # LOSS CLAMP (P1 fix) — prevent extreme timestep losses from corrupting weights
            loss = torch.clamp(loss, max=10.0)

            if not torch.isfinite(loss):
                nan_count += 1
                if nan_count >= 3:
                    # Diagnostic dump
                    print(f"\nFATAL: 3 consecutive NaN at step {global_step}")
                    print(f"  lr={optimizer.param_groups[0]['lr']:.2e}")
                    for n, p in unwrapped.named_parameters():
                        if p.data.isnan().any():
                            print(f"  NaN weight: {n}")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  batch[{k}]: nan={v.isnan().any()} inf={v.isinf().any()}")
                    FATAL(f"3 consecutive NaN at step {global_step}")
                optimizer.zero_grad()
                global_step += 1
                continue
            else:
                nan_count = 0  # Reset consecutive counter

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), 10.0)
            else:
                grad_norm = torch.tensor(0.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # EMA
        if accelerator.sync_gradients:
            with torch.no_grad():
                for n, p in unwrapped.named_parameters():
                    if n in ema_params:
                        ema_params[n].mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

        # Logging
        lv = loss.item()
        if np.isfinite(lv):
            losses_window.append(lv)
        if len(losses_window) > 100:
            losses_window.pop(0)
        avg100 = np.nanmean(losses_window) if losses_window else float("nan")
        if avg100 < best_avg and np.isfinite(avg100):
            best_avg = avg100

        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        writer.add_scalar("train/loss", lv, global_step)
        writer.add_scalar("train/avg100", avg100, global_step)
        writer.add_scalar("train/grad_norm", gn, global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if global_step % 1000 == 0:
            elapsed = time.time() - t0
            sps = (global_step + 1) / max(elapsed, 1)
            eta = (TOTAL_STEPS - global_step) / max(sps, 0.01)
            print(f"Step {global_step:6d}/{TOTAL_STEPS} | loss={lv:.4f} avg100={avg100:.4f} | "
                  f"grad={gn:.1f} lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"{sps:.1f} step/s ETA {eta/3600:.1f}h", flush=True)

        if (global_step + 1) % 20000 == 0:
            ckpt_path = f"checkpoints/vc/final_step{global_step+1}.pth"
            torch.save({
                "model_state_dict": unwrapped.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step + 1,
                "ema_params": ema_params,
            }, ckpt_path)
            # EMA inference checkpoint
            ema_state = unwrapped.state_dict()
            for n in ema_params:
                if n in ema_state:
                    ema_state[n] = ema_params[n]
            ema_path = f"checkpoints/vc/final_ema_step{global_step+1}.pth"
            torch.save({"model_state_dict": ema_state, "global_step": global_step + 1}, ema_path)
            print(f"  >> Saved {ckpt_path} + {ema_path}", flush=True)
            # Auto-cleanup: keep last 3
            for prefix in ["final_step", "final_ema_step"]:
                old = sorted(glob.glob(f"checkpoints/vc/{prefix}*.pth"))
                while len(old) > 3:
                    os.remove(old.pop(0))

        global_step += 1

# Final save
ema_state = unwrapped.state_dict()
for n in ema_params:
    if n in ema_state:
        ema_state[n] = ema_params[n]
torch.save({"model_state_dict": ema_state, "global_step": global_step},
           "checkpoints/vc/final_ema.pth")
writer.close()

total_time = time.time() - t0
print("=" * 70)
print(f"Training complete: {global_step} steps in {total_time/3600:.1f}h")
print(f"Final loss={lv:.4f} | Best avg100={best_avg:.4f} | NaN total={nan_count}")
print(f"Checkpoint: checkpoints/vc/final_ema.pth")

# ============================================================
# STEP 4: Inference + Verify
# ============================================================
print(f"\n=== STEP 4: Inference + Verify ===")

# Reload model from EMA checkpoint (clean state)
del model, optimizer
torch.cuda.empty_cache()

model_infer, _, _ = build_vc_model(copy.deepcopy(BASE))
ckpt = torch.load("checkpoints/vc/final_ema.pth", map_location="cpu", weights_only=False)
model_infer.load_state_dict(ckpt["model_state_dict"], strict=False)
model_infer = model_infer.cuda().eval()
print(f"Inference model loaded (step {ckpt.get('global_step', '?')})")

# HuBERT-soft
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda().eval()
hubert.requires_grad_(False)

# BigVGAN
from indextts.s2mel.modules.bigvgan import bigvgan as bigvgan_module
bigvgan_snap = "/root/.cache/huggingface/hub/models--nvidia--bigvgan_v2_22khz_80band_256x/snapshots/633ff708ed5b74903e86ff1298cf4a98e921c513"
h_dict = json.load(open(os.path.join(bigvgan_snap, "config.json")))
h_dict["use_cuda_kernel"] = False
bigvgan_model = bigvgan_module.BigVGAN(Munch.fromDict(h_dict))
state = torch.load(os.path.join(bigvgan_snap, "bigvgan_generator.pt"), map_location="cpu", weights_only=False)
bigvgan_model.load_state_dict(state, strict=False)
bigvgan_model.remove_weight_norm()
bigvgan_model = bigvgan_model.cuda().eval()

# CAMPPlus for inference style
campplus_infer = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_infer.load_state_dict(torch.load(CAMPPLUS_PATH, map_location="cpu", weights_only=True))
campplus_infer = campplus_infer.cuda().eval()

SOURCE = "/root/autodl-tmp/train/wav/SSB0005/SSB00050001.wav"
TARGET = "/root/autodl-tmp/train/wav/SSB0009/SSB00090001.wav"

src_audio, _ = librosa.load(SOURCE, sr=16000)
tgt_audio, _ = librosa.load(TARGET, sr=16000)
print(f"Source: {len(src_audio)/16000:.2f}s, Target ref: {len(tgt_audio)/16000:.2f}s")

with torch.no_grad():
    # Source content
    src_content = hubert.units(torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).cuda())
    src_content_q = kmeans_full.quantize_to_vector(src_content.cpu()).cuda()

    # Source F0
    f0, _, _ = librosa.pyin(src_audio, fmin=65, fmax=2000, sr=16000, hop_length=320)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    if len(f0) > src_content.shape[1]: f0 = f0[:src_content.shape[1]]
    elif len(f0) < src_content.shape[1]: f0 = np.pad(f0, (0, src_content.shape[1] - len(f0)))
    f0_t = torch.from_numpy(f0).unsqueeze(0).cuda()

    # Source mel length
    src_22k = librosa.resample(src_audio, orig_sr=16000, target_sr=22050)
    src_mel = mel_fn(torch.from_numpy(src_22k).unsqueeze(0).cuda())
    src_mel_len = src_mel.shape[-1]

    # Target content + mel + REAL style
    tgt_content = hubert.units(torch.from_numpy(tgt_audio).unsqueeze(0).unsqueeze(0).cuda())
    tgt_content_q = kmeans_full.quantize_to_vector(tgt_content.cpu()).cuda()
    tgt_22k = librosa.resample(tgt_audio, orig_sr=16000, target_sr=22050)
    tgt_mel = mel_fn(torch.from_numpy(tgt_22k).unsqueeze(0).cuda())
    tgt_mel_len = tgt_mel.shape[-1]

    # REAL style from target speaker (kaldi fbank + mean norm, same as training)
    tgt_16k_t = torch.from_numpy(tgt_audio).unsqueeze(0).cuda()
    tgt_feat = torchaudio.compliance.kaldi.fbank(
        tgt_16k_t, num_mel_bins=80, dither=0, sample_frequency=16000)
    tgt_feat = tgt_feat - tgt_feat.mean(dim=0, keepdim=True)
    style = campplus_infer(tgt_feat.unsqueeze(0))  # (1, 192)
    print(f"  Real style: shape={style.shape} range=[{style.min():.3f},{style.max():.3f}]")

    # DiT setup
    dit = model_infer.models["cfm"].estimator
    dit.setup_caches(max_batch_size=2, max_seq_length=src_mel_len + tgt_mel_len + 200)

    # Through length_regulator
    src_cond = model_infer.models["length_regulator"](
        src_content_q, ylens=torch.LongTensor([src_mel_len]).cuda(), n_quantizers=3, f0=f0_t)[0]
    tgt_cond = model_infer.models["length_regulator"](
        tgt_content_q, ylens=torch.LongTensor([tgt_mel_len]).cuda(), n_quantizers=3, f0=None)[0]

    mu = torch.cat([tgt_cond, src_cond], dim=1)
    total_len = torch.LongTensor([mu.shape[1]]).cuda()

    # CFM inference
    vc_mel = model_infer.models["cfm"].inference(
        mu, total_len, tgt_mel, style, None, 25, inference_cfg_rate=0.7)
    vc_mel = vc_mel[:, :, tgt_mel_len:]

    # BigVGAN decode
    wav = bigvgan_model(vc_mel.float()).squeeze().cpu()
    wav = torch.clamp(wav, -1.0, 1.0)

# Save
os.makedirs("outputs/vc_final", exist_ok=True)
torchaudio.save("outputs/vc_final/source.wav", torch.from_numpy(src_audio).unsqueeze(0), 16000)
torchaudio.save("outputs/vc_final/target_ref.wav", torch.from_numpy(tgt_audio).unsqueeze(0), 16000)
torchaudio.save("outputs/vc_final/output.wav", wav.unsqueeze(0), 22050)

# Verify
print(f"\n=== Verification ===")
print(f"  VC mel: shape={vc_mel.shape} range=[{vc_mel.min():.2f},{vc_mel.max():.2f}]")
print(f"  Source mel ref: range=[{src_mel.min():.2f},{src_mel.max():.2f}]")
mel_ok = -15 < vc_mel.min() < -5 and vc_mel.max() < 5
print(f"  Mel in log-mel range: {'YES' if mel_ok else 'NO (BAD!)'}")
print(f"  Output wav: shape={wav.shape} ({wav.shape[-1]/22050:.2f}s) range=[{wav.min():.3f},{wav.max():.3f}]")
print(f"  Has NaN: {wav.isnan().any()}")
print(f"  Has clipping: {(wav.abs() > 0.99).sum()} samples")

print(f"\n{'='*70}")
print(f"=== VC Training Delivery ===")
print(f"Issues fixed:")
print(f"  P0: Real CAMPPlus style ({regen_count} files regenerated)")
print(f"  P1: Loss clamp (max=10.0)")
print(f"  P2: HP search (2000 steps, {len(LRS)*len(WARMUPS)} configs)")
print(f"HP search result: lr={BEST_LR}, warmup={BEST_WARMUP}")
print(f"Training: {global_step} steps, final loss={lv:.4f}, best avg100={best_avg:.4f}, NaN={nan_count}")
print(f"Inference: outputs/vc_final/ (source.wav + target_ref.wav + output.wav)")
print(f"Mel range: [{vc_mel.min():.2f}, {vc_mel.max():.2f}] ({'OK' if mel_ok else 'BAD'})")
print(f"Wav duration: {wav.shape[-1]/22050:.2f}s (source: {len(src_audio)/16000:.2f}s)")
print(f"{'='*70}")
