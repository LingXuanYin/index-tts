"""
indextts/vc/preprocessing.py

Offline preprocessing pipeline for VC training data.

For each utterance, extracts and saves:
  - <stem>.content.npy  : HuBERT-soft features (T_content, 256) at 50 Hz
  - <stem>.f0.npy       : F0 array (T_content,) in Hz at 50 Hz (decimated from 100 Hz)
  - <stem>.mel.npy      : Mel spectrogram (80, T_mel) at 22050/80-band/hop=256
  - <stem>.style.npy    : CAMPPlus speaker embedding (192,)

Filtering:
  - voiced_ratio < 0.1 → utterance is skipped (returns False, saves nothing)

Mel spec constants (s2mel spec, MUST use 22050Hz/80-band/hop=256 for BigVGAN):
  MEL_SAMPLE_RATE = 22050
  MEL_N_MELS = 80
  MEL_HOP_LENGTH = 256

CLI usage:
  python -m indextts.vc.preprocessing \\
      --manifest data/vc/train.jsonl \\
      --output_dir data/vc/preprocessed \\
      --content_model_name bshall/hubert-soft \\
      --f0_model_path checkpoints/rmvpe/model.pt \\
      --device cpu

Design notes (design.md D4, D5):
  - mel must be 22050/80/256 (s2mel spec) to match BigVGAN input requirements
  - content at 50 Hz; f0 at 100 Hz decimated to 50 Hz to align with content
  - voiced_ratio < 0.1 utterances filtered (too little pitch signal = noise)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Mel spectrogram constants (s2mel spec — MUST NOT use dataset spec 24000/100)
# ---------------------------------------------------------------------------
MEL_SAMPLE_RATE = 22050    # BigVGAN input sample rate
MEL_N_MELS = 80            # 80 mel bands (NOT 100 from dataset config)
MEL_HOP_LENGTH = 256       # hop length in samples
MEL_N_FFT = 1024           # FFT size
MEL_WIN_LENGTH = 1024      # window length
MEL_FMIN = 0               # min frequency
MEL_FMAX = None            # max frequency (None = Nyquist)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_voiced_ratio(f0: np.ndarray) -> float:
    """Compute fraction of frames that are voiced (f0 > 0).

    Args:
        f0: 1D numpy float array, Hz. Unvoiced = 0.

    Returns:
        float in [0, 1].
    """
    if len(f0) == 0:
        return 0.0
    return float(np.sum(f0 > 0)) / float(len(f0))


def should_filter_utterance(
    f0: np.ndarray,
    min_voiced_ratio: float = 0.1,
) -> bool:
    """Return True if utterance should be filtered out (voiced_ratio < threshold).

    Args:
        f0: 1D numpy float array, Hz. Unvoiced = 0.
        min_voiced_ratio: Minimum fraction of voiced frames (design D4). Default 0.1.

    Returns:
        True  → skip utterance (voiced_ratio < min_voiced_ratio)
        False → keep utterance
    """
    return compute_voiced_ratio(f0) < min_voiced_ratio


def compute_mel(
    audio_22k: np.ndarray,
) -> np.ndarray:
    """Compute 22050/80-band/hop=256 mel spectrogram.

    Mel spec must use s2mel spec (22050Hz/80-band/hop=256) for BigVGAN compatibility.
    This is DIFFERENT from the dataset mel spec (24000Hz/100-band).

    Args:
        audio_22k: 1D numpy float array at 22050 Hz.

    Returns:
        numpy array of shape (80, T_frames).
    """
    from indextts.s2mel.modules.audio import mel_spectrogram as _mel_spectrogram

    # Convert to torch tensor, shape (1, T)
    audio_tensor = torch.from_numpy(audio_22k).float().unsqueeze(0)

    mel = _mel_spectrogram(
        y=audio_tensor,
        n_fft=MEL_N_FFT,
        num_mels=MEL_N_MELS,
        sampling_rate=MEL_SAMPLE_RATE,
        hop_size=MEL_HOP_LENGTH,
        win_size=MEL_WIN_LENGTH,
        fmin=MEL_FMIN,
        fmax=8000,  # common setting; None causes issues in librosa_mel_fn
    )  # (1, 80, T_frames)

    return mel.squeeze(0).numpy()  # (80, T_frames)


def write_manifest_line(
    manifest_path: str,
    audio_path: str,
    speaker_id: str,
    duration_s: float,
    language: str,
    output_stem: str,
    **extra_fields,
) -> None:
    """Append one JSONL entry to the manifest file.

    Format matches cases.jsonl convention:
      {"audio_path": "...", "speaker_id": "...", "duration_s": ..., "language": "...", ...}

    Args:
        manifest_path: Path to the JSONL manifest file (append mode).
        audio_path: Path to the source audio file.
        speaker_id: Speaker identifier string.
        duration_s: Duration in seconds.
        language: Language code ("zh" or "en").
        output_stem: Stem path where preprocessed .npy files are saved.
        **extra_fields: Optional additional fields to include in the JSON entry.
    """
    entry = {
        "audio_path": audio_path,
        "speaker_id": speaker_id,
        "duration_s": duration_s,
        "language": language,
        "output_stem": output_stem,
        **extra_fields,
    }
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def preprocess_utterance(
    audio_16k: np.ndarray,
    audio_22k: np.ndarray,
    output_stem: str,
    content_encoder,
    f0_encoder,
    style_encoder,
    min_voiced_ratio: float = 0.1,
) -> bool:
    """Preprocess a single utterance and save features to disk.

    Args:
        audio_16k: 1D float numpy array at 16kHz (for HuBERT-soft + RMVPE + CAMPPlus).
        audio_22k: 1D float numpy array at 22050Hz (for mel spectrogram).
        output_stem: Output file stem. Files are <stem>.content.npy etc.
        content_encoder: ContentEncoder instance (or mock). extract(audio) -> (1, T, 256)
        f0_encoder: F0Encoder instance (or mock). extract(audio) -> np.ndarray 100Hz
        style_encoder: Callable. style_encoder(audio_tensor) -> (1, 192) tensor.
        min_voiced_ratio: Filter threshold. Utterances below this are skipped.

    Returns:
        True  → preprocessing succeeded; files saved.
        False → utterance filtered (voiced_ratio too low); no files written.
    """
    # Step 1: Extract F0 (100 Hz) and check voiced ratio
    f0_100hz = f0_encoder.extract(audio_16k)
    if should_filter_utterance(f0_100hz, min_voiced_ratio=min_voiced_ratio):
        vr = compute_voiced_ratio(f0_100hz)
        warnings.warn(
            f"Utterance filtered: voiced_ratio={vr:.3f} < {min_voiced_ratio}. "
            f"Stem: {output_stem}",
            UserWarning,
        )
        return False

    # Step 2: Extract content features (HuBERT-soft, 50 Hz)
    audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0)  # (1, T)
    content_features = content_encoder.extract(audio_tensor)  # (1, T_content, 256)
    T_content = content_features.shape[1]

    # Step 3: Decimate F0 from 100 Hz to content frame rate (50 Hz ≈ T_content frames)
    from indextts.vc.f0_encoder import align_to_content_framerate
    f0_50hz = align_to_content_framerate(f0_100hz, target_frames=T_content)
    if isinstance(f0_50hz, torch.Tensor):
        f0_50hz_np = f0_50hz.numpy()
    else:
        f0_50hz_np = np.asarray(f0_50hz, dtype=np.float32)

    # Step 4: Mel spectrogram (22050/80/hop=256 — s2mel spec)
    mel = compute_mel(audio_22k)  # (80, T_mel)

    # Step 5: Speaker style embedding (CAMPPlus)
    style = style_encoder(audio_tensor)  # (1, 192) or similar
    if isinstance(style, torch.Tensor):
        style_np = style.detach().squeeze(0).numpy()  # (192,)
    else:
        style_np = np.asarray(style, dtype=np.float32).reshape(-1)

    # Step 6: Save all features
    content_np = content_features.detach().squeeze(0).numpy()  # (T_content, 256)

    np.save(output_stem + ".content.npy", content_np)
    np.save(output_stem + ".f0.npy", f0_50hz_np)
    np.save(output_stem + ".mel.npy", mel)
    np.save(output_stem + ".style.npy", style_np)

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline preprocessing for VC training data."
    )
    parser.add_argument("--manifest", required=True,
                        help="Input JSONL manifest file (audio_path, speaker_id, ...)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write .npy feature files")
    parser.add_argument("--output_manifest", default=None,
                        help="Output JSONL manifest for successfully preprocessed items")
    parser.add_argument("--content_model_name", default="bshall/hubert-soft",
                        help="Content encoder model name")
    parser.add_argument("--f0_model_path", required=True,
                        help="Path to RMVPE checkpoint (.pt)")
    parser.add_argument("--device", default="cpu",
                        help="Device (cpu / cuda:0 / cuda:1 etc.)")
    parser.add_argument("--min_voiced_ratio", type=float, default=0.1,
                        help="Minimum voiced ratio for filtering (default 0.1)")
    parser.add_argument("--sample_rate_16k", type=int, default=16000)
    parser.add_argument("--sample_rate_22k", type=int, default=22050)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_manifest = args.output_manifest or os.path.join(
        args.output_dir, "preprocessed.jsonl"
    )

    # Lazy-import heavy models
    from indextts.vc.content_encoder import ContentEncoder
    from indextts.vc.f0_encoder import F0Encoder

    content_enc = ContentEncoder(model_name=args.content_model_name, device=args.device)
    f0_enc = F0Encoder(model_path=args.f0_model_path, device=args.device)

    # CAMPPlus style encoder — reuse infer.py's CamPPlus loading pattern
    try:
        from indextts.s2mel.modules.campplus.cam_plus import CAMPPlus
        style_enc = CAMPPlus(feat_dim=80, embedding_size=192)
        style_enc.to(args.device).eval()
    except Exception as e:
        print(f"[WARNING] CAMPPlus load failed: {e}. Style will be random zeros.")
        def style_enc(x):
            return torch.zeros(x.shape[0], 192)

    # Process manifest
    import torchaudio
    n_total = 0
    n_filtered = 0

    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = item["audio_path"]
            speaker_id = item.get("speaker_id", "unknown")
            language = item.get("language", "zh")

            # Load and resample audio
            try:
                wav, sr = torchaudio.load(audio_path)
            except Exception as e:
                print(f"[WARNING] Failed to load {audio_path}: {e}")
                continue

            wav = wav.mean(dim=0)  # to mono

            # 16kHz version
            if sr != 16000:
                import torchaudio.functional as AF
                wav_16k = AF.resample(wav, sr, 16000)
            else:
                wav_16k = wav
            audio_16k_np = wav_16k.numpy()

            # 22050Hz version
            if sr != 22050:
                import torchaudio.functional as AF
                wav_22k = AF.resample(wav, sr, 22050)
            else:
                wav_22k = wav
            audio_22k_np = wav_22k.numpy()

            duration_s = len(audio_16k_np) / 16000.0
            rel_path = os.path.relpath(audio_path, os.path.dirname(args.manifest))
            stem_name = rel_path.replace(os.sep, "_").replace("/", "_")
            if stem_name.endswith(".wav"):
                stem_name = stem_name[:-4]
            output_stem = os.path.join(args.output_dir, stem_name)

            n_total += 1
            ok = preprocess_utterance(
                audio_16k=audio_16k_np,
                audio_22k=audio_22k_np,
                output_stem=output_stem,
                content_encoder=content_enc,
                f0_encoder=f0_enc,
                style_encoder=style_enc,
                min_voiced_ratio=args.min_voiced_ratio,
            )

            if ok:
                write_manifest_line(
                    manifest_path=output_manifest,
                    audio_path=audio_path,
                    speaker_id=speaker_id,
                    duration_s=duration_s,
                    language=language,
                    output_stem=output_stem,
                )
            else:
                n_filtered += 1

    print(f"Done: {n_total} total, {n_filtered} filtered, "
          f"{n_total - n_filtered} kept.")
    filter_rate = n_filtered / max(n_total, 1)
    if filter_rate > 0.2:
        print(f"[WARNING] Filter rate {filter_rate:.1%} > 20%. "
              f"Check RMVPE checkpoint and audio quality.")


if __name__ == "__main__":
    main()
