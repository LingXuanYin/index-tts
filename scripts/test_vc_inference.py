"""Test VC inference with trained checkpoint."""
import sys, os, time
sys.path.insert(0, ".")

import torch
import torchaudio
import librosa
import numpy as np
from munch import Munch
import yaml

# ---- Config ----
SOURCE_AUDIO = "/root/autodl-tmp/train/wav/SSB0005/SSB00050001.wav"
TARGET_REF = "/root/autodl-tmp/train/wav/SSB0009/SSB00090001.wav"
CHECKPOINT = "checkpoints/vc/v2_ema_step100000.pth"
KMEANS_PATH = "data/vc/aishell3/kmeans_codebook.pt"
OUTPUT_PATH = "outputs/vc_test_output.wav"

os.makedirs("outputs", exist_ok=True)

# ---- Load config ----
with open("checkpoints/config.yaml") as f:
    args = Munch.fromDict(yaml.safe_load(f))
args.s2mel.length_regulator.in_channels = 256
args.s2mel.length_regulator.f0_condition = True
args.s2mel.length_regulator.n_f0_bins = 256
args.s2mel.DiT.f0_condition = True
if not hasattr(args, "training"):
    args.training = Munch()
args.training.pretrained_s2mel = "checkpoints/s2mel.pth"
args.training.reset_cond_projection = False

# ---- Load model ----
print("Loading VC model...")
from indextts.vc_train.model_adapter import build_vc_model
model, _, _ = build_vc_model(args)

# Load trained weights
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model = model.cuda().eval()
print(f"Model loaded from {CHECKPOINT} (step {ckpt.get('global_step', '?')})")

# ---- Load content encoder ----
print("Loading HuBERT-soft...")
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
hubert = hubert.cuda().eval()
hubert.requires_grad_(False)

# ---- Load k-means ----
from indextts.vc.kmeans_quantizer import KMeansQuantizer
kmeans = KMeansQuantizer(n_clusters=200)
kmeans.load(KMEANS_PATH)

# ---- Load BigVGAN ----
print("Loading BigVGAN...")
# Manual BigVGAN loading (avoid from_pretrained API mismatch)
from indextts.s2mel.modules.bigvgan import bigvgan as bigvgan_module
import json
from types import SimpleNamespace

bigvgan_snap = "/root/.cache/huggingface/hub/models--nvidia--bigvgan_v2_22khz_80band_256x/snapshots/633ff708ed5b74903e86ff1298cf4a98e921c513"
with open(os.path.join(bigvgan_snap, "config.json")) as f:
    h_dict = json.load(f)
h_dict["use_cuda_kernel"] = False
# BigVGAN expects h to support both attribute and item access
from munch import Munch
h = Munch.fromDict(h_dict)
bigvgan_model = bigvgan_module.BigVGAN(h)
state = torch.load(os.path.join(bigvgan_snap, "bigvgan_generator.pt"), map_location="cpu", weights_only=False)
# Try different key formats
if "generator" in state:
    state = state["generator"]
elif "state_dict" in state:
    state = state["state_dict"]
missing, unexpected = bigvgan_model.load_state_dict(state, strict=False)
if missing:
    print(f"  BigVGAN missing keys: {len(missing)}")
if unexpected:
    print(f"  BigVGAN unexpected keys: {len(unexpected)}")
bigvgan_model.remove_weight_norm()
bigvgan_model = bigvgan_model.cuda().eval()
print("BigVGAN loaded")

# ---- Process source audio ----
print(f"\nSource: {SOURCE_AUDIO}")
src_audio, _ = librosa.load(SOURCE_AUDIO, sr=16000)
print(f"  Duration: {len(src_audio)/16000:.2f}s")

# Content features
src_t = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).cuda()
with torch.no_grad():
    src_content = hubert.units(src_t)  # (1, T, 256)
src_content_q = kmeans.quantize_to_vector(src_content.cpu()).cuda()  # (1, T, 256)

# F0 (source contour)
f0, _, _ = librosa.pyin(src_audio, fmin=65, fmax=2000, sr=16000, hop_length=16000//50)
f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
# Align to content frames
if len(f0) > src_content.shape[1]:
    f0 = f0[:src_content.shape[1]]
elif len(f0) < src_content.shape[1]:
    f0 = np.pad(f0, (0, src_content.shape[1] - len(f0)))
src_f0 = torch.from_numpy(f0).unsqueeze(0).cuda()  # (1, T)

# Source mel (for length)
src_22k = librosa.resample(src_audio, orig_sr=16000, target_sr=22050)
# Use the SAME log-mel as s2mel training
from indextts.s2mel.modules.audio import mel_spectrogram as _mel_spec
mel_fn = lambda x: _mel_spec(x, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=None)
src_mel = mel_fn(torch.from_numpy(src_22k).unsqueeze(0))  # (1, 80, T_mel)
src_mel_len = src_mel.shape[-1]

# ---- Process target reference ----
print(f"Target ref: {TARGET_REF}")
tgt_audio, _ = librosa.load(TARGET_REF, sr=16000)
print(f"  Duration: {len(tgt_audio)/16000:.2f}s")

# Target content (for prompt_condition)
tgt_t = torch.from_numpy(tgt_audio).unsqueeze(0).unsqueeze(0).cuda()
with torch.no_grad():
    tgt_content = hubert.units(tgt_t)
tgt_content_q = kmeans.quantize_to_vector(tgt_content.cpu()).cuda()

# Target mel (for ref_mel)
tgt_22k = librosa.resample(tgt_audio, orig_sr=16000, target_sr=22050)
tgt_mel = mel_fn(torch.from_numpy(tgt_22k).unsqueeze(0)).cuda()  # (1, 80, T_mel_tgt)
tgt_mel_len = tgt_mel.shape[-1]

# Target style (mock - should be CAMPPlus but we don't have it loaded)
style = torch.randn(1, 192).cuda()

# ---- Run through model ----
print("\nRunning VC inference...")
with torch.no_grad():
    # Source content -> length_regulator (with F0)
    src_cond = model.models["length_regulator"](
        src_content_q, ylens=torch.LongTensor([src_mel_len]).cuda(),
        n_quantizers=3, f0=src_f0
    )[0]  # (1, T_src_mel, 512)

    # Target content -> length_regulator (prompt, no F0)
    tgt_cond = model.models["length_regulator"](
        tgt_content_q, ylens=torch.LongTensor([tgt_mel_len]).cuda(),
        n_quantizers=3, f0=None
    )[0]  # (1, T_tgt_mel, 512)

    # Concatenate
    mu = torch.cat([tgt_cond, src_cond], dim=1)  # (1, T_total, 512)
    total_len = torch.LongTensor([mu.shape[1]]).cuda()
    prompt_len = tgt_mel_len

    print(f"  src_cond: {src_cond.shape}, tgt_cond: {tgt_cond.shape}, mu: {mu.shape}")

    # Setup DiT caches (required before forward/inference)
    dit = model.models["cfm"].estimator
    dit.setup_caches(max_batch_size=2, max_seq_length=mu.shape[1] + 100)

    # CFM inference
    vc_mel = model.models["cfm"].inference(
        mu, total_len, tgt_mel, style, None, 25,
        inference_cfg_rate=0.7
    )  # (1, 80, T_total)

    # Strip prompt
    vc_mel = vc_mel[:, :, prompt_len:]
    print(f"  VC mel: {vc_mel.shape}")

    # BigVGAN decode
    wav = bigvgan_model(vc_mel.float())
    wav = wav.squeeze().cpu()
    print(f"  Output wav: {wav.shape} ({wav.shape[-1]/22050:.2f}s)")

# Save
wav = torch.clamp(wav, -1.0, 1.0)
torchaudio.save(OUTPUT_PATH, wav.unsqueeze(0), 22050)
print(f"\nSaved to {OUTPUT_PATH}")

# Also save source and target for comparison
torchaudio.save("outputs/vc_test_source.wav",
                torch.from_numpy(src_audio).unsqueeze(0), 16000)
torchaudio.save("outputs/vc_test_target_ref.wav",
                torch.from_numpy(tgt_audio).unsqueeze(0), 16000)
print("Also saved source and target reference for comparison")
print("Done!")
