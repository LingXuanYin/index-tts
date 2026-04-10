"""
indextts/infer_vc_trained.py

VC inference entry-point for trained Voice Conversion (add-voice-conversion-v2).

Design rationale (why a standalone module rather than a method on IndexTTS2):
  - IndexTTS2.__init__ already does heavy work loading TTS models. Adding VC
    model loading there would increase startup time for TTS-only users.
  - A standalone module keeps the VC inference path isolated and independently
    testable without instantiating IndexTTS2.
  - The public API is `infer_vc_trained(tts, ...)` which takes an IndexTTS2
    instance, so it behaves like a method but lives in its own file.
  - infer_v2.py is not modified (TTS paths unchanged, spec requirement).

Public API:
    from indextts.infer_vc_trained import infer_vc_trained
    wav = infer_vc_trained(tts, "source.wav", "speaker_ref.wav", output_path="out.wav")

Inference data flow (design D1 + Group 5 spec):
  source_audio → HuBERT-soft → kmeans → content_quantized (B, T_src, 256)
  source_audio → RMVPE → F0 → strategy → align_to_content_framerate
  speaker_ref  → HuBERT-soft → kmeans → prompt_content_quantized (B, T_ref, 256)
  speaker_ref  → mel → ref_mel (1, 80, T_ref_mel)
  speaker_ref  → CAMPPlus fbank → style (1, 192)

  cond_src    = length_regulator(content_quantized, ylens=T_tgt_mel, f0=f0_processed)
  cond_prompt = length_regulator(prompt_content_quantized, ylens=T_ref_mel, f0=None)
  mu          = cat([cond_prompt, cond_src], dim=1)  # time-first (B, T, 512)
  vc_out      = cfm.inference(mu, x_lens, ref_mel, style, f0=None, steps, cfg_rate)
  vc_out      = vc_out[:, :, ref_mel.size(-1):]     # strip prompt prefix
  wav         = bigvgan(vc_out)

MUST NOT call: self.gpt.*, self.semantic_model, self.semantic_codec,
               self.asr_model, torchaudio.functional.pitch_shift
cfm.inference f0 argument = None (F0 is injected via length_regulator, not cfm)
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi

from indextts.vc.f0_encoder import (
    VALID_F0_STRATEGIES,
    align_to_content_framerate,
    apply_source_contour,
    apply_source_plus_shift,
    apply_target_median,
    apply_manual,
    estimate_speaker_median_log_f0,
    STRATEGY_SOURCE_CONTOUR,
    STRATEGY_SOURCE_PLUS_SHIFT,
    STRATEGY_TARGET_MEDIAN,
    STRATEGY_MANUAL,
)


# ─────────────────────────────────────────────── private loader helpers ──────


def _load_vc_checkpoint(tts, checkpoint_path: str):
    """Load VC checkpoint and return the MyModel.

    The checkpoint must contain:
      - "model": state dict or MyModel weights
      - "metadata": dict with at least {"f0_strategy": str}

    Attaches `_vc_checkpoint_metadata` to the returned model object.
    """
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2, recursive_munch
    from omegaconf import OmegaConf

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"VC checkpoint not found: {checkpoint_path!r}. "
            "Train a VC model first with indextts/vc_train/train.py."
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    metadata = ckpt.get("metadata", {})

    # Load VC config embedded in checkpoint (or fall back to config_vc.yaml)
    vc_config_dict = ckpt.get("config", None)
    if vc_config_dict is None:
        # Fallback: load from default location
        default_config_path = os.path.join(
            os.path.dirname(__file__), "vc_train", "config_vc.yaml"
        )
        vc_config_dict = OmegaConf.to_container(OmegaConf.load(default_config_path), resolve=True)

    args = recursive_munch(vc_config_dict["s2mel"])
    model = MyModel(args, use_gpt_latent=False)

    # Load state dict — use the model state dict directly
    state_dict = ckpt.get("model", ckpt)
    if isinstance(state_dict, dict) and "models" in state_dict:
        # Checkpoint saved via torch.save({"model": model.state_dict(), ...})
        model.load_state_dict(state_dict["models"] if "models" in state_dict else state_dict,
                              strict=False)
    else:
        # Direct state dict
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(tts.device)
    model._vc_checkpoint_metadata = metadata
    return model


def _load_content_encoder(tts):
    """Lazy-load HuBERT-soft ContentEncoder onto tts.device."""
    from indextts.vc.content_encoder import ContentEncoder
    enc = ContentEncoder(device=tts.device)
    return enc


def _load_kmeans(tts, codebook_path: str):
    """Lazy-load KMeansQuantizer from saved codebook."""
    from indextts.vc.kmeans_quantizer import KMeansQuantizer
    q = KMeansQuantizer()
    q.load(codebook_path)
    return q


def _load_f0_encoder(tts, model_path: str):
    """Lazy-load F0Encoder (RMVPE wrapper)."""
    from indextts.vc.f0_encoder import F0Encoder
    # Normalise device: RMVPE cannot handle bare "cuda"
    device = tts.device
    if device == "cuda":
        device = "cuda:0"
    is_half = tts.use_fp16 if hasattr(tts, "use_fp16") else False
    enc = F0Encoder(model_path=model_path, device=device, is_half=is_half)
    return enc


# ─────────────────────────────────────────────── core helpers ────────────────


def _ensure_vc_loaded(
    tts,
    checkpoint_path: str,
    f0_strategy: str,
    kmeans_codebook_path: Optional[str] = None,
    rmvpe_model_path: Optional[str] = None,
) -> None:
    """Lazy-load all VC components into tts.vc_* attributes.

    This is a no-op if tts.vc_model is already set (idempotent).

    Sets:
        tts.vc_model          — MyModel (VC finetuned)
        tts.vc_content_encoder — ContentEncoder (HuBERT-soft)
        tts.vc_kmeans         — KMeansQuantizer
        tts.vc_f0_encoder     — F0Encoder (RMVPE)
        tts.vc_f0_strategy    — str from checkpoint metadata
        tts.vc_config         — dict (embedded config or default)
    """
    if tts.vc_model is not None:
        return  # already loaded

    print(">> [VC] Loading VC model components (first call)...")

    # Default paths
    if kmeans_codebook_path is None:
        kmeans_codebook_path = "checkpoints/vc/kmeans_200.pt"
    if rmvpe_model_path is None:
        rmvpe_model_path = "checkpoints/rmvpe/model.pt"

    # Load in order: checkpoint first (to read metadata), then encoders
    model = _load_vc_checkpoint(tts, checkpoint_path)
    tts.vc_model = model

    metadata = getattr(model, "_vc_checkpoint_metadata", {})
    ckpt_strategy = metadata.get("f0_strategy", f0_strategy)
    tts.vc_f0_strategy = ckpt_strategy

    tts.vc_content_encoder = _load_content_encoder(tts)
    tts.vc_kmeans = _load_kmeans(tts, kmeans_codebook_path)
    tts.vc_f0_encoder = _load_f0_encoder(tts, rmvpe_model_path)
    tts.vc_config = getattr(model, "_vc_config", None)

    print(f">> [VC] VC model loaded. F0 strategy from checkpoint: {ckpt_strategy!r}")
    print(">> [VC] VC model will be loaded on first use (lazy-loading complete)")


def _assert_f0_strategy_consistent(tts, requested_strategy: str) -> None:
    """Assert that requested_strategy matches the checkpoint's stored strategy.

    Raises:
        ValueError: If strategy name is invalid or mismatches checkpoint metadata.
    """
    if requested_strategy not in VALID_F0_STRATEGIES:
        raise ValueError(
            f"Invalid f0_strategy {requested_strategy!r}. "
            f"Valid options: {sorted(VALID_F0_STRATEGIES)}"
        )
    ckpt_strategy = tts.vc_f0_strategy
    if ckpt_strategy is not None and ckpt_strategy != requested_strategy:
        raise ValueError(
            f"F0 strategy mismatch: checkpoint was trained with {ckpt_strategy!r} "
            f"but caller requested {requested_strategy!r}. "
            "Use the same strategy or retrain the VC model."
        )


def _load_audio_16k(path: str) -> np.ndarray:
    """Load audio file and resample to 16kHz mono. Returns 1D float32 numpy array."""
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return audio.astype(np.float32)


def _load_audio_22k(path: str) -> torch.Tensor:
    """Load audio file and resample to 22050 Hz. Returns (1, T) float tensor."""
    audio, sr = librosa.load(path, sr=22050, mono=True)
    return torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)


def _apply_f0_strategy(
    f0_raw: np.ndarray,
    strategy: str,
    manual_semitones: float = 0.0,
    src_median_log: float = 0.0,
    tgt_median_log: float = 0.0,
) -> np.ndarray:
    """Apply the selected F0 strategy and return processed F0 array."""
    if strategy == STRATEGY_SOURCE_CONTOUR:
        return apply_source_contour(f0_raw)
    elif strategy == STRATEGY_SOURCE_PLUS_SHIFT:
        return apply_source_plus_shift(f0_raw, src_median_log, tgt_median_log)
    elif strategy == STRATEGY_TARGET_MEDIAN:
        return apply_target_median(f0_raw, tgt_median_log)
    elif strategy == STRATEGY_MANUAL:
        return apply_manual(f0_raw, manual_semitones)
    else:
        raise ValueError(f"Unknown F0 strategy: {strategy!r}")


# ─────────────────────────────────────────────── public API ─────────────────


def infer_vc_trained(
    tts,
    source_audio: Union[str, os.PathLike],
    speaker_refs: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
    output_path: Optional[str] = None,
    *,
    checkpoint_path: str = "checkpoints/vc/trained_vc.pth",
    f0_strategy: str = "source_plus_shift",
    manual_semitones: float = 0.0,
    diffusion_steps: int = 25,
    inference_cfg_rate: float = 0.7,
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    """Run trained VC inference: convert source audio to target speaker voice.

    Args:
        tts: IndexTTS2 instance (provides shared infrastructure: device, bigvgan,
             campplus_model, mel_fn). VC-specific models are lazily loaded into
             tts.vc_* attributes on first call.
        source_audio: Path to source speech audio file (the speech to convert).
        speaker_refs: Path (or list of paths) to target speaker reference audio.
                      Only the first reference is used for prompt_condition;
                      if multiple, they are averaged for style.
        output_path: If provided, save WAV to this path and return None.
                     If None, return the waveform as a float tensor (1, T).
        checkpoint_path: Path to the trained VC checkpoint (.pth).
        f0_strategy: One of "source_contour", "source_plus_shift",
                     "target_median", "manual". Must match checkpoint metadata.
        manual_semitones: Semitone shift for "manual" strategy. Positive = raise.
        diffusion_steps: Number of ODE steps for CFM inference (default 25).
        inference_cfg_rate: Classifier-free guidance scale (default 0.7).
        verbose: If True, print timing and shape info.

    Returns:
        torch.Tensor of shape (1, T_samples) at 22050 Hz if output_path is None,
        else None after saving to output_path.

    Raises:
        ValueError: If f0_strategy is invalid or mismatches checkpoint metadata.
        FileNotFoundError: If checkpoint or audio files are not found.

    IMPORTANT — forbidden call-sites (spec red-line):
        This function MUST NOT call:
          tts.gpt.*            (GPT autoregressive model)
          tts.semantic_model   (w2v-bert feature extractor)
          tts.semantic_codec   (MaskGCT VQ codec)
          tts.asr_model        (ASR model)
          torchaudio.functional.pitch_shift

        cfm.inference() f0 argument is always None here.
        F0 is injected via length_regulator (additive embedding), not cfm.
    """
    # ── 1. Lazy-load VC models ────────────────────────────────────────────────
    _ensure_vc_loaded(tts, checkpoint_path, f0_strategy)

    # ── 2. Validate F0 strategy consistency ──────────────────────────────────
    _assert_f0_strategy_consistent(tts, f0_strategy)

    # Normalise speaker_refs to list
    if isinstance(speaker_refs, (str, os.PathLike)):
        speaker_refs = [str(speaker_refs)]
    else:
        speaker_refs = [str(p) for p in speaker_refs]

    device = tts.device

    # ── 3. Source audio: HuBERT-soft → kmeans → content ──────────────────────
    src_audio_16k = _load_audio_16k(str(source_audio))
    src_tensor_16k = torch.from_numpy(src_audio_16k).unsqueeze(0).to(device)  # (1, T)

    src_content = tts.vc_content_encoder.extract(src_tensor_16k)  # (1, T_src, 256)
    src_content_q = tts.vc_kmeans.quantize_to_vector(src_content)  # (1, T_src, 256)

    if verbose:
        print(f"[VC] src_content shape: {src_content.shape}")

    # ── 4. Source F0: RMVPE → strategy → align to content framerate ──────────
    f0_raw = tts.vc_f0_encoder.extract(src_audio_16k)  # 1D np at 100 Hz
    f0_aligned_tensor = align_to_content_framerate(f0_raw, src_content.shape[1])  # (T_src,)
    f0_aligned = f0_aligned_tensor.numpy()  # back to numpy for strategy functions

    # Compute source median (for shift strategies)
    src_median_log = estimate_speaker_median_log_f0(f0_aligned)

    if verbose:
        print(f"[VC] f0 raw shape: {f0_raw.shape}, aligned to {f0_aligned.shape}")

    # ── 5. Speaker reference: HuBERT-soft + kmeans + mel + style ─────────────
    # Use first reference for prompt_condition; all references averaged for style
    ref_path = speaker_refs[0]
    ref_audio_16k = _load_audio_16k(ref_path)
    ref_tensor_16k = torch.from_numpy(ref_audio_16k).unsqueeze(0).to(device)  # (1, T_ref)
    ref_audio_22k = _load_audio_22k(ref_path).to(device)  # (1, T_ref_22k)

    # Content for prompt_condition
    ref_content = tts.vc_content_encoder.extract(ref_tensor_16k)  # (1, T_ref, 256)
    ref_content_q = tts.vc_kmeans.quantize_to_vector(ref_content)  # (1, T_ref, 256)

    # Mel spectrogram for ref_mel (22050 Hz / 80-band / hop=256 via tts.mel_fn)
    ref_mel = tts.mel_fn(ref_audio_22k.float())  # (1, 80, T_ref_mel)
    ref_mel = ref_mel.to(device)
    ref_mel_lens = torch.LongTensor([ref_mel.size(2)]).to(device)

    # CAMPPlus style extraction (speaker identity vector)
    feat = torchaudio.compliance.kaldi.fbank(
        ref_tensor_16k.float(),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    style = tts.campplus_model(feat.unsqueeze(0))  # (1, 192)

    if verbose:
        print(f"[VC] ref_content shape: {ref_content.shape}, ref_mel: {ref_mel.shape}")

    # Compute target speaker median F0 (for shift strategies)
    ref_f0_raw = tts.vc_f0_encoder.extract(ref_audio_16k)
    tgt_median_log = estimate_speaker_median_log_f0(ref_f0_raw)

    # Apply F0 strategy
    f0_processed = _apply_f0_strategy(
        f0_aligned,
        strategy=f0_strategy,
        manual_semitones=manual_semitones,
        src_median_log=src_median_log,
        tgt_median_log=tgt_median_log,
    )
    f0_tensor = torch.from_numpy(f0_processed).float().unsqueeze(0).to(device)  # (1, T_src)

    # ── 6. Length regulator: source content → target mel frames ──────────────
    # Estimate target mel length from source audio duration
    src_duration_s = len(src_audio_16k) / 16000.0
    # Target mel rate = sr / hop = 22050 / 256 ≈ 86.1 Hz
    target_mel_len = max(1, int(round(src_duration_s * 22050 / 256)))
    target_mel_lens = torch.LongTensor([target_mel_len]).to(device)

    length_regulator = tts.vc_model.models["length_regulator"]
    cfm = tts.vc_model.models["cfm"]

    # source cond: inject F0
    cond_src = length_regulator(
        src_content_q,          # (1, T_src, 256)
        ylens=target_mel_lens,  # (1,) target mel length
        f0=f0_tensor,           # (1, T_src) F0 in Hz — injected in length_regulator
    )[0]  # (1, T_tgt_mel, 512)

    # ── 7. Length regulator: prompt content (no F0) ───────────────────────────
    cond_prompt = length_regulator(
        ref_content_q,          # (1, T_ref, 256)
        ylens=ref_mel_lens,     # (1,) ref mel length
        f0=None,                # prompt uses f0_mask placeholder (design D1)
    )[0]  # (1, T_ref_mel, 512)

    if verbose:
        print(f"[VC] cond_src: {cond_src.shape}, cond_prompt: {cond_prompt.shape}")

    # ── 8. Concatenate mu (time-first) ───────────────────────────────────────
    # mu is time-first (B, T, 512) when passed to cfm.inference
    # cfm.inference internally transposes / handles as needed
    # Design note: flow_matching.py uses mu[bib, :, :prompt_lens[bib]] with
    # channel-first indexing during training (cfm.forward). However for inference
    # the cat is done before cfm, and cfm.inference receives cat_condition directly.
    # Following the TTS pattern at infer_v2.py:932-941:
    #   cat_condition = cat([prompt_cond, cond], dim=1)  ← time-first (B, T, 512)
    #   cfm.inference(cat_condition, ...)
    mu = torch.cat([cond_prompt, cond_src], dim=1)  # (1, T_prompt+T_tgt, 512)
    x_lens = torch.LongTensor([mu.size(1)]).to(device)

    if verbose:
        print(f"[VC] mu shape: {mu.shape}")

    # ── 9. CFM inference ─────────────────────────────────────────────────────
    # IMPORTANT: f0=None here. F0 was already injected in length_regulator above.
    # This follows the TTS pattern (infer_v2.py:937-941).
    with torch.no_grad():
        vc_out = cfm.inference(
            mu,           # (1, T_total, 512) — time-first, matching TTS usage
            x_lens,       # (1,) total length
            ref_mel,      # (1, 80, T_ref_mel) prompt mel
            style,        # (1, 192) speaker style
            None,         # f0 = None (injected via length_regulator, not cfm)
            diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )  # → (1, 80, T_total)

    # ── 10. Strip prompt prefix ───────────────────────────────────────────────
    vc_out = vc_out[:, :, ref_mel.size(-1):]  # (1, 80, T_tgt)

    if verbose:
        print(f"[VC] vc_out after strip: {vc_out.shape}")

    # ── 11. BigVGAN decode ────────────────────────────────────────────────────
    with torch.no_grad():
        wav = tts.bigvgan(vc_out.float()).squeeze(1)  # (1, T_samples)

    wav = torch.clamp(wav, -1.0, 1.0)

    if verbose:
        print(f"[VC] wav shape: {wav.shape}")

    # ── 12. Save or return ────────────────────────────────────────────────────
    if output_path is not None:
        if os.path.isfile(output_path):
            os.remove(output_path)
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, wav.cpu(), 22050)
        if verbose:
            print(f"[VC] Saved to: {output_path}")
        return None

    return wav.cpu()
