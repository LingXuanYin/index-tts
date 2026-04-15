"""Diagnose inference issues: check weights, mel output, wav range."""
import torch, os, sys, yaml, numpy as np
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
model, _, _ = build_vc_model(args)

# === CHECK 1: content_in_proj BEFORE loading trained ckpt ===
cip_before = model.models["length_regulator"].content_in_proj.weight.data.clone()
print("=== CHECK 1: content_in_proj BEFORE trained ckpt ===")
print(f"  shape: {cip_before.shape}")
print(f"  mean: {cip_before.mean():.6f}, std: {cip_before.std():.6f}")
print(f"  norm: {cip_before.norm():.4f}")

# === Load trained checkpoint ===
ckpt = torch.load("checkpoints/vc/trained_vc.pth", map_location="cpu", weights_only=False)
print(f"\n=== Trained ckpt info ===")
print(f"  step: {ckpt.get('global_step')}")
print(f"  keys in state_dict: {len(ckpt['model_state_dict'])}")

# Check if content_in_proj is in the checkpoint
cip_key = None
for k in ckpt["model_state_dict"]:
    if "content_in_proj" in k:
        cip_key = k
        v = ckpt["model_state_dict"][k]
        print(f"  FOUND: {k} shape={v.shape} mean={v.mean():.6f} std={v.std():.6f}")

if cip_key is None:
    print("  !!! content_in_proj NOT FOUND in checkpoint !!!")

# Load trained weights
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"  missing keys: {len(missing)}")
print(f"  unexpected keys: {len(unexpected)}")
if missing:
    print(f"  first 5 missing: {missing[:5]}")

# === CHECK 2: content_in_proj AFTER loading trained ckpt ===
cip_after = model.models["length_regulator"].content_in_proj.weight.data
print(f"\n=== CHECK 2: content_in_proj AFTER trained ckpt ===")
print(f"  shape: {cip_after.shape}")
print(f"  mean: {cip_after.mean():.6f}, std: {cip_after.std():.6f}")
print(f"  norm: {cip_after.norm():.4f}")
print(f"  CHANGED: {not torch.equal(cip_before, cip_after)}")

# === CHECK 3: Quick forward to see mel output range ===
model = model.cuda().eval()
import librosa, torchaudio
audio, _ = librosa.load("/root/autodl-tmp/train/wav/SSB0005/SSB00050001.wav", sr=16000)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda().eval()
from indextts.vc.kmeans_quantizer import KMeansQuantizer
kmeans = KMeansQuantizer(n_clusters=200)
kmeans.load("data/vc/aishell3/kmeans_codebook.pt")

with torch.no_grad():
    content = hubert.units(torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).cuda())
    content_q = kmeans.quantize_to_vector(content.cpu()).cuda()

    # F0
    f0, _, _ = librosa.pyin(audio, fmin=65, fmax=2000, sr=16000, hop_length=320)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    if len(f0) > content.shape[1]:
        f0 = f0[:content.shape[1]]
    elif len(f0) < content.shape[1]:
        f0 = np.pad(f0, (0, content.shape[1] - len(f0)))
    f0_t = torch.from_numpy(f0).unsqueeze(0).cuda()

    mel_fn = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80)
    src_22k = librosa.resample(audio, orig_sr=16000, target_sr=22050)
    src_mel = mel_fn(torch.from_numpy(src_22k).unsqueeze(0)).cuda()
    src_mel_len = src_mel.shape[-1]

    # Through length_regulator
    src_cond = model.models["length_regulator"](
        content_q, ylens=torch.LongTensor([src_mel_len]).cuda(),
        n_quantizers=3, f0=f0_t
    )[0]

    print(f"\n=== CHECK 3: intermediate tensors ===")
    print(f"  content_q: shape={content_q.shape}, range=[{content_q.min():.3f}, {content_q.max():.3f}]")
    print(f"  f0: voiced_ratio={np.mean(f0>0):.2f}, mean_voiced={np.mean(f0[f0>0]):.1f}Hz")
    print(f"  src_cond: shape={src_cond.shape}, range=[{src_cond.min():.3f}, {src_cond.max():.3f}], mean={src_cond.mean():.4f}")
    print(f"  src_mel: shape={src_mel.shape}, range=[{src_mel.min():.3f}, {src_mel.max():.3f}]")

    # Style - compare random vs zeros
    style_rand = torch.randn(1, 192).cuda()
    style_zero = torch.zeros(1, 192).cuda()

    # Use same speaker reference for prompt
    tgt_audio, _ = librosa.load("/root/autodl-tmp/train/wav/SSB0009/SSB00090001.wav", sr=16000)
    tgt_content = hubert.units(torch.from_numpy(tgt_audio).unsqueeze(0).unsqueeze(0).cuda())
    tgt_cq = kmeans.quantize_to_vector(tgt_content.cpu()).cuda()
    tgt_22k = librosa.resample(tgt_audio, orig_sr=16000, target_sr=22050)
    tgt_mel = mel_fn(torch.from_numpy(tgt_22k).unsqueeze(0)).cuda()
    tgt_mel_len = tgt_mel.shape[-1]

    tgt_cond = model.models["length_regulator"](
        tgt_cq, ylens=torch.LongTensor([tgt_mel_len]).cuda(),
        n_quantizers=3, f0=None
    )[0]

    mu = torch.cat([tgt_cond, src_cond], dim=1)
    total_len = torch.LongTensor([mu.shape[1]]).cuda()

    dit = model.models["cfm"].estimator
    dit.setup_caches(max_batch_size=2, max_seq_length=mu.shape[1] + 100)

    vc_mel = model.models["cfm"].inference(
        mu, total_len, tgt_mel, style_rand, None, 25, inference_cfg_rate=0.7
    )
    vc_mel = vc_mel[:, :, tgt_mel_len:]

    print(f"\n=== CHECK 4: VC output mel ===")
    print(f"  shape: {vc_mel.shape}")
    print(f"  range: [{vc_mel.min():.3f}, {vc_mel.max():.3f}]")
    print(f"  mean: {vc_mel.mean():.4f}, std: {vc_mel.std():.4f}")
    print(f"  has NaN: {torch.isnan(vc_mel).any()}")
    print(f"  has Inf: {torch.isinf(vc_mel).any()}")

    # Compare with source mel
    print(f"\n=== CHECK 5: source mel for reference ===")
    print(f"  range: [{src_mel.min():.3f}, {src_mel.max():.3f}]")
    print(f"  mean: {src_mel.mean():.4f}, std: {src_mel.std():.4f}")
