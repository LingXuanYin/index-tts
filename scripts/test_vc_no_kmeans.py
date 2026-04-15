"""Test VC inference WITHOUT k-means (raw HuBERT-soft features).
Compare with k-means version to isolate content scrambling source."""
import torch, os, sys, yaml, json, numpy as np
import librosa, torchaudio
sys.path.insert(0, ".")
from munch import Munch

CHECKPOINT = "checkpoints/vc/v2_ema_step100000.pth"
SOURCE = "/root/autodl-tmp/train/wav/SSB0005/SSB00050001.wav"
TARGET = "/root/autodl-tmp/train/wav/SSB0009/SSB00090001.wav"

# Config + model
with open("checkpoints/config.yaml") as f:
    args = Munch.fromDict(yaml.safe_load(f))
args.s2mel.length_regulator.in_channels = 256
args.s2mel.length_regulator.f0_condition = True
args.s2mel.length_regulator.n_f0_bins = 256
args.s2mel.DiT.f0_condition = True
args.training = Munch(pretrained_s2mel="checkpoints/s2mel.pth", reset_cond_projection=False)

from indextts.vc_train.model_adapter import build_vc_model
model, _, _ = build_vc_model(args)
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model = model.cuda().eval()
print(f"Model loaded (step {ckpt.get('global_step', '?')})")

# HuBERT-soft
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).cuda().eval()
hubert.requires_grad_(False)

# Mel function (log mel)
from indextts.s2mel.modules.audio import mel_spectrogram as _mel_spec
mel_fn = lambda x: _mel_spec(x, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=None)

# BigVGAN
from indextts.s2mel.modules.bigvgan import bigvgan as bigvgan_module
bigvgan_snap = "/root/.cache/huggingface/hub/models--nvidia--bigvgan_v2_22khz_80band_256x/snapshots/633ff708ed5b74903e86ff1298cf4a98e921c513"
h_dict = json.load(open(os.path.join(bigvgan_snap, "config.json")))
h_dict["use_cuda_kernel"] = False
from munch import Munch as M
bigvgan_model = bigvgan_module.BigVGAN(M.fromDict(h_dict))
state = torch.load(os.path.join(bigvgan_snap, "bigvgan_generator.pt"), map_location="cpu", weights_only=False)
bigvgan_model.load_state_dict(state, strict=False)
bigvgan_model.remove_weight_norm()
bigvgan_model = bigvgan_model.cuda().eval()

# K-means (for comparison)
from indextts.vc.kmeans_quantizer import KMeansQuantizer
kmeans = KMeansQuantizer(n_clusters=200)
kmeans.load("data/vc/aishell3/kmeans_codebook.pt")

# Source audio
src_audio, _ = librosa.load(SOURCE, sr=16000)
src_t = torch.from_numpy(src_audio).unsqueeze(0).unsqueeze(0).cuda()

with torch.no_grad():
    content_raw = hubert.units(src_t)  # (1, T, 256)
    content_kmeans = kmeans.quantize_to_vector(content_raw.cpu()).cuda()

    # F0
    f0, _, _ = librosa.pyin(src_audio, fmin=65, fmax=2000, sr=16000, hop_length=320)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    if len(f0) > content_raw.shape[1]: f0 = f0[:content_raw.shape[1]]
    elif len(f0) < content_raw.shape[1]: f0 = np.pad(f0, (0, content_raw.shape[1] - len(f0)))
    f0_t = torch.from_numpy(f0).unsqueeze(0).cuda()

    src_22k = librosa.resample(src_audio, orig_sr=16000, target_sr=22050)
    src_mel = mel_fn(torch.from_numpy(src_22k).unsqueeze(0).cuda())
    src_mel_len = src_mel.shape[-1]

    # Target
    tgt_audio, _ = librosa.load(TARGET, sr=16000)
    tgt_t = torch.from_numpy(tgt_audio).unsqueeze(0).unsqueeze(0).cuda()
    tgt_content_raw = hubert.units(tgt_t)
    tgt_content_km = kmeans.quantize_to_vector(tgt_content_raw.cpu()).cuda()
    tgt_22k = librosa.resample(tgt_audio, orig_sr=16000, target_sr=22050)
    tgt_mel = mel_fn(torch.from_numpy(tgt_22k).unsqueeze(0).cuda())
    tgt_mel_len = tgt_mel.shape[-1]
    style = torch.randn(1, 192).cuda()

    dit = model.models["cfm"].estimator
    os.makedirs("outputs", exist_ok=True)

    for name, src_c, tgt_c in [
        ("with_kmeans", content_kmeans, tgt_content_km),
        ("no_kmeans", content_raw, tgt_content_raw),
    ]:
        src_cond = model.models["length_regulator"](
            src_c, ylens=torch.LongTensor([src_mel_len]).cuda(), n_quantizers=3, f0=f0_t
        )[0]
        tgt_cond = model.models["length_regulator"](
            tgt_c, ylens=torch.LongTensor([tgt_mel_len]).cuda(), n_quantizers=3, f0=None
        )[0]
        mu = torch.cat([tgt_cond, src_cond], dim=1)
        dit.setup_caches(max_batch_size=2, max_seq_length=mu.shape[1] + 100)

        vc_mel = model.models["cfm"].inference(
            mu, torch.LongTensor([mu.shape[1]]).cuda(), tgt_mel, style, None, 25, inference_cfg_rate=0.7
        )
        vc_mel = vc_mel[:, :, tgt_mel_len:]

        wav = bigvgan_model(vc_mel.float()).squeeze().cpu()
        wav = torch.clamp(wav, -1.0, 1.0)
        out_path = f"outputs/vc_test_{name}.wav"
        torchaudio.save(out_path, wav.unsqueeze(0), 22050)
        print(f"{name}: mel range [{vc_mel.min():.2f}, {vc_mel.max():.2f}], wav range [{wav.min():.3f}, {wav.max():.3f}] -> {out_path}")

print("Done! Compare with_kmeans vs no_kmeans to isolate content issue.")
