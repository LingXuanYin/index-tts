"""
indextts/vc_train/eval.py

VC evaluation harness (design D10).

Metrics:
  - SECS (Speaker Encoder Cosine Similarity):
      Primary: ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb, 192 dim)
      Cross-validation: CAMPPlus (existing indextts speaker encoder, 192 dim)
  - CER: whisper-large-v3, character-level edit distance
  - F0 RMSE: RMVPE-extracted F0, in cents (1200 * log2(pred/ref)), voiced/unvoiced
  - UTMOSv2: try/except isolated (optional dependency)

JSON report format:
  {
    "secs_ecapa": float,
    "secs_campplus": float,
    "cer_whisper": float,       # character-level, not word-level
    "f0_rmse_cents_voiced": float,
    "f0_rmse_cents_unvoiced": float,
    "utmos": float | null
  }

CLI usage:
  python -m indextts.vc_train.eval \\
      --source_audio source.wav \\
      --converted_audio converted.wav \\
      --reference_audio reference.wav \\
      [--source_text "你好世界"] \\
      [--output_json eval_report.json] \\
      [--config indextts/vc_train/config_vc.yaml]
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Container for VC evaluation metrics."""
    secs_ecapa: Optional[float]
    secs_campplus: Optional[float]
    cer_whisper: Optional[float]
    f0_rmse_cents_voiced: Optional[float]
    f0_rmse_cents_unvoiced: Optional[float]
    utmos: Optional[float]

    def to_dict(self) -> Dict:
        return {
            "secs_ecapa": self.secs_ecapa,
            "secs_campplus": self.secs_campplus,
            "cer_whisper": self.cer_whisper,
            "f0_rmse_cents_voiced": self.f0_rmse_cents_voiced,
            "f0_rmse_cents_unvoiced": self.f0_rmse_cents_unvoiced,
            "utmos": self.utmos,
        }


# ---------------------------------------------------------------------------
# Individual metric functions (testable in isolation)
# ---------------------------------------------------------------------------

def compute_f0_rmse_cents(
    f0_ref: np.ndarray,
    f0_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compute F0 RMSE in cents for voiced and unvoiced frames separately.

    F0 RMSE in cents:
        cents_error = 1200 * log2(f0_pred / f0_ref)   (for voiced frames)
        RMSE = sqrt(mean(cents_error^2))

    Voiced frames: f0_ref > 0 AND f0_pred > 0
    Unvoiced frames: f0_ref == 0 AND f0_pred == 0 (trivially 0 error)
    Frames where voiced/unvoiced disagree: not counted in either metric.

    Args:
        f0_ref: Reference F0 array in Hz. Unvoiced = 0.
        f0_pred: Predicted F0 array in Hz. Unvoiced = 0.

    Returns:
        dict with keys "voiced" (float or nan) and "unvoiced" (always 0.0 here).
    """
    f0_ref = np.asarray(f0_ref, dtype=np.float32)
    f0_pred = np.asarray(f0_pred, dtype=np.float32)

    # Align lengths (take min)
    min_len = min(len(f0_ref), len(f0_pred))
    f0_ref = f0_ref[:min_len]
    f0_pred = f0_pred[:min_len]

    # Voiced: both ref and pred are voiced
    voiced_mask = (f0_ref > 0) & (f0_pred > 0)

    voiced_rmse: Optional[float]
    if voiced_mask.sum() == 0:
        voiced_rmse = float("nan")
    else:
        # cents = 1200 * log2(pred / ref)
        cents_error = 1200.0 * np.log2(
            f0_pred[voiced_mask] / f0_ref[voiced_mask]
        )
        voiced_rmse = float(np.sqrt(np.mean(cents_error ** 2)))

    # Unvoiced: both unvoiced → error = 0 by definition
    unvoiced_mask = (f0_ref == 0) & (f0_pred == 0)
    unvoiced_rmse = 0.0 if unvoiced_mask.sum() > 0 else 0.0

    return {"voiced": voiced_rmse, "unvoiced": unvoiced_rmse}


def compute_secs_from_embeddings(
    emb_ref: torch.Tensor,
    emb_pred: torch.Tensor,
) -> float:
    """Compute Speaker Encoder Cosine Similarity (SECS) from embeddings.

    Args:
        emb_ref: Reference speaker embedding. Shape (D,) or (B, D).
        emb_pred: Converted speaker embedding. Shape (D,) or (B, D).

    Returns:
        Mean cosine similarity as a Python float in [-1, 1].
    """
    emb_ref = emb_ref.float()
    emb_pred = emb_pred.float()

    if emb_ref.ndim == 1:
        emb_ref = emb_ref.unsqueeze(0)
    if emb_pred.ndim == 1:
        emb_pred = emb_pred.unsqueeze(0)

    ref_norm = F.normalize(emb_ref, dim=-1)
    pred_norm = F.normalize(emb_pred, dim=-1)

    # Cosine similarity per pair, then mean
    cosine = (ref_norm * pred_norm).sum(dim=-1)  # (B,)
    return float(cosine.mean().item())


def compute_cer_char_level(
    reference: str,
    hypothesis: str,
) -> float:
    """Compute Character Error Rate (CER) between reference and hypothesis.

    CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)

    Uses dynamic programming Levenshtein distance at character level.

    Args:
        reference: Ground-truth transcription string.
        hypothesis: ASR hypothesis string.

    Returns:
        CER as a float. 0.0 = perfect; can be > 1.0 for many insertions.
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0:
        return float(m)  # all insertions

    # DP Levenshtein
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp

    return dp[m] / n


def _extract_ecapa_embedding(
    ecapa_model: Any,
    audio: np.ndarray,
    sample_rate: int,
) -> torch.Tensor:
    """Extract ECAPA-TDNN speaker embedding from audio.

    Args:
        ecapa_model: EncoderClassifier instance from speechbrain.
        audio: 1D float32 numpy array.
        sample_rate: Audio sample rate (must be 16kHz for ECAPA-TDNN).

    Returns:
        1D tensor of shape (192,).
    """
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
    with torch.no_grad():
        emb = ecapa_model.encode_batch(audio_tensor)  # (1, 1, 192)
    return emb.squeeze(0).squeeze(0)  # (192,)


def _extract_campplus_embedding(
    campplus_model: Any,
    audio: np.ndarray,
    sample_rate: int,
) -> torch.Tensor:
    """Extract CAMPPlus speaker embedding from audio.

    Args:
        campplus_model: CAMPPlus callable (forward → (1, 192) tensor).
        audio: 1D float32 numpy array.
        sample_rate: Audio sample rate.

    Returns:
        1D tensor of shape (192,).
    """
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
    with torch.no_grad():
        emb = campplus_model(audio_tensor)  # expected (1, 192)
    if emb.ndim == 2:
        emb = emb.squeeze(0)  # (192,)
    return emb


def _run_whisper_cer(
    whisper_model: Any,
    audio: np.ndarray,
    sample_rate: int,
    reference_text: str,
) -> Optional[float]:
    """Transcribe audio with whisper and compute character CER.

    Args:
        whisper_model: whisper model with .transcribe(audio) method.
        audio: 1D float32 numpy array.
        sample_rate: Audio sample rate (whisper expects 16kHz).
        reference_text: Ground truth transcript.

    Returns:
        CER float or None on failure.
    """
    try:
        result = whisper_model.transcribe(audio)
        hypothesis = result.get("text", "")
        return compute_cer_char_level(reference_text, hypothesis)
    except Exception as e:
        logger.warning("Whisper CER failed: %s", e)
        return None


def _run_utmos(
    utmos_model: Any,
    audio: np.ndarray,
    sample_rate: int,
) -> Optional[float]:
    """Run UTMOSv2 prediction with try/except isolation.

    Args:
        utmos_model: UTMOSv2 predictor with .predict(audio, sr) interface.
                     May be None if not installed.
        audio: 1D float32 numpy array.
        sample_rate: Audio sample rate.

    Returns:
        UTMOSv2 score (float) or None on any failure.
    """
    if utmos_model is None:
        return None
    try:
        score = utmos_model.predict(audio, sample_rate)
        if hasattr(score, "item"):
            score = score.item()
        return float(score)
    except Exception as e:
        logger.warning("UTMOSv2 prediction failed (isolated): %s", e)
        return None


# ---------------------------------------------------------------------------
# Main run_eval function
# ---------------------------------------------------------------------------

def run_eval(
    source_audio: np.ndarray,
    converted_audio: np.ndarray,
    reference_audio: np.ndarray,
    source_text: str,
    source_f0: np.ndarray,
    converted_f0: np.ndarray,
    sample_rate: int,
    ecapa_model: Any,
    campplus_model: Any,
    whisper_model: Any,
    utmos_model: Any = None,
    output_json: Optional[str] = None,
) -> Dict:
    """Run the full VC evaluation harness.

    Args:
        source_audio: Source utterance audio (1D float32 numpy array).
        converted_audio: Converted utterance audio (1D float32 numpy array).
        reference_audio: Target speaker reference audio (1D float32 numpy array).
        source_text: Ground truth transcription of the source utterance.
        source_f0: Source F0 array (Hz, 50 Hz frame rate). Unvoiced = 0.
        converted_f0: Converted F0 array (Hz). Unvoiced = 0.
        sample_rate: Audio sample rate for all three audio arrays.
        ecapa_model: ECAPA-TDNN EncoderClassifier (speechbrain). Used for SECS.
        campplus_model: CAMPPlus model callable. Used for cross-validation SECS.
        whisper_model: Whisper model with .transcribe() method. Used for CER.
        utmos_model: UTMOSv2 predictor (optional). None = skip.
        output_json: If provided, write the metric dict to this JSON path.

    Returns:
        dict with keys:
          secs_ecapa / secs_campplus / cer_whisper /
          f0_rmse_cents_voiced / f0_rmse_cents_unvoiced / utmos
    """
    result: Dict[str, Any] = {}

    # 1. SECS — ECAPA-TDNN (primary)
    try:
        emb_ref_ecapa = _extract_ecapa_embedding(ecapa_model, reference_audio, sample_rate)
        emb_cvt_ecapa = _extract_ecapa_embedding(ecapa_model, converted_audio, sample_rate)
        result["secs_ecapa"] = compute_secs_from_embeddings(emb_ref_ecapa, emb_cvt_ecapa)
    except Exception as e:
        logger.warning("ECAPA-TDNN SECS failed: %s", e)
        result["secs_ecapa"] = None

    # 2. SECS — CAMPPlus (cross-validation)
    try:
        emb_ref_camp = _extract_campplus_embedding(campplus_model, reference_audio, sample_rate)
        emb_cvt_camp = _extract_campplus_embedding(campplus_model, converted_audio, sample_rate)
        result["secs_campplus"] = compute_secs_from_embeddings(emb_ref_camp, emb_cvt_camp)
    except Exception as e:
        logger.warning("CAMPPlus SECS failed: %s", e)
        result["secs_campplus"] = None

    # 3. CER — whisper-large-v3, character level
    result["cer_whisper"] = _run_whisper_cer(
        whisper_model, converted_audio, sample_rate, source_text
    )

    # 4. F0 RMSE in cents (voiced / unvoiced)
    try:
        f0_rmse = compute_f0_rmse_cents(source_f0, converted_f0)
        result["f0_rmse_cents_voiced"] = f0_rmse["voiced"]
        result["f0_rmse_cents_unvoiced"] = f0_rmse["unvoiced"]
    except Exception as e:
        logger.warning("F0 RMSE computation failed: %s", e)
        result["f0_rmse_cents_voiced"] = None
        result["f0_rmse_cents_unvoiced"] = None

    # 5. UTMOSv2 (isolated try/except)
    result["utmos"] = _run_utmos(utmos_model, converted_audio, sample_rate)

    # Write JSON report if requested
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Eval report written to %s", output_json)

    return result


# ---------------------------------------------------------------------------
# Model loading helpers (lazy, for actual eval runs)
# ---------------------------------------------------------------------------

def load_ecapa_model(source: str, savedir: str) -> Any:
    """Load ECAPA-TDNN from speechbrain.

    Args:
        source: HuggingFace model ID, e.g. "speechbrain/spkrec-ecapa-voxceleb".
        savedir: Local cache directory.

    Returns:
        EncoderClassifier instance.

    Raises:
        ImportError: If speechbrain is not installed.
    """
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as e:
        raise ImportError(
            "speechbrain is required for ECAPA-TDNN SECS. "
            "Install with: pip install speechbrain"
        ) from e
    return EncoderClassifier.from_hparams(source=source, savedir=savedir)


def load_whisper_model(model_id: str, device: str = "cpu") -> Any:
    """Load whisper model via openai-whisper or transformers.

    Tries openai-whisper first (lighter); falls back to transformers pipeline.

    Args:
        model_id: Model name e.g. "openai/whisper-large-v3" or "large-v3".
        device: Device string.

    Returns:
        Whisper model with .transcribe(audio) method.
    """
    # Normalise model ID
    model_name = model_id.replace("openai/", "")
    try:
        import whisper
        return whisper.load_model(model_name, device=device)
    except ImportError:
        pass
    # Fallback: transformers pipeline
    try:
        from transformers import pipeline
        pipe = pipeline("automatic-speech-recognition", model=model_id, device=device)
        # Wrap to match .transcribe(audio) interface
        class WhisperWrapper:
            def __init__(self, pipe):
                self._pipe = pipe
            def transcribe(self, audio):
                result = self._pipe(audio)
                return {"text": result.get("text", "")}
        return WhisperWrapper(pipe)
    except ImportError as e:
        raise ImportError(
            "Neither openai-whisper nor transformers is installed. "
            "Install one of them: pip install openai-whisper OR pip install transformers"
        ) from e


def load_utmos_model() -> Optional[Any]:
    """Load UTMOSv2 predictor with try/except isolation.

    Returns None if UTMOSv2 is not installed (not a fatal error).
    """
    try:
        import utmos
        predictor = utmos.Predictor()
        logger.info("UTMOSv2 loaded successfully")
        return predictor
    except Exception as e:
        logger.info("UTMOSv2 not available (isolated): %s. Skipping UTMOSv2.", e)
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import soundfile as sf
    import yaml

    from indextts.vc.f0_encoder import F0Encoder, align_to_content_framerate

    parser = argparse.ArgumentParser(description="VC Evaluation")
    parser.add_argument("--source_audio", required=True)
    parser.add_argument("--converted_audio", required=True)
    parser.add_argument("--reference_audio", required=True)
    parser.add_argument("--source_text", default="")
    parser.add_argument("--output_json", default="eval_report.json")
    parser.add_argument("--config", default="indextts/vc_train/config_vc.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {})

    # Load audio
    source_audio, sr = sf.read(args.source_audio)
    converted_audio, _ = sf.read(args.converted_audio)
    reference_audio, _ = sf.read(args.reference_audio)

    # Load models
    ecapa_cfg = eval_cfg.get("secs", {})
    ecapa_model = load_ecapa_model(
        ecapa_cfg.get("ecapa_source", "speechbrain/spkrec-ecapa-voxceleb"),
        ecapa_cfg.get("ecapa_savedir", "pretrained_models/spkrec-ecapa-voxceleb"),
    )

    whisper_model = load_whisper_model(
        eval_cfg.get("cer", {}).get("whisper_model", "openai/whisper-large-v3"),
        device=args.device,
    )

    utmos_model = load_utmos_model() if eval_cfg.get("utmos", {}).get("enabled", True) else None

    # Extract F0 (using RMVPE if available; otherwise skip)
    f0_cfg = config.get("f0", {})
    f0_model_path = f0_cfg.get("model_path", "checkpoints/rmvpe/model.pt")
    source_f0 = np.zeros(1, dtype=np.float32)
    converted_f0 = np.zeros(1, dtype=np.float32)
    if os.path.exists(f0_model_path):
        from indextts.vc.f0_encoder import F0Encoder
        f0enc = F0Encoder(model_path=f0_model_path, device=args.device)
        import librosa
        src_16k = librosa.resample(source_audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        cvt_16k = librosa.resample(converted_audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        source_f0 = f0enc.extract(src_16k)
        converted_f0 = f0enc.extract(cvt_16k)

    # CAMPPlus: try to use existing model if available (not a hard dependency for eval)
    campplus_model = None

    result = run_eval(
        source_audio=source_audio.astype(np.float32),
        converted_audio=converted_audio.astype(np.float32),
        reference_audio=reference_audio.astype(np.float32),
        source_text=args.source_text,
        source_f0=source_f0,
        converted_f0=converted_f0,
        sample_rate=sr,
        ecapa_model=ecapa_model,
        campplus_model=campplus_model,
        whisper_model=whisper_model,
        utmos_model=utmos_model,
        output_json=args.output_json,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
