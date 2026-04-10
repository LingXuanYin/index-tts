"""
indextts/vc/f0_encoder.py

F0 extraction and strategy functions for Voice Conversion.

Key components:
  - RMVPE import (from indextts.s2mel.modules.rmvpe)
  - F0Encoder: wraps RMVPE with safe device handling
  - 4 pure strategy functions (design D6):
      apply_source_contour / apply_source_plus_shift / apply_target_median / apply_manual
  - estimate_speaker_median_log_f0: voiced median in log Hz space
  - align_to_content_framerate: resample F0 from 100 Hz to target frame count

Design notes (from Group 1 research, task 1.5):
  - RMVPE output: 1D numpy array, 100 Hz frame rate, unvoiced = 0 Hz
  - RMVPE has hardcoded device issue: passing device="cuda" → forced to "cuda:0"
    Mitigation: always pass explicit "cpu" or "cuda:N" string from config
  - VC training: F0 is precomputed at 100 Hz, then decimated to 50 Hz (content
    encoder rate) via align_to_content_framerate
"""
from __future__ import annotations

import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

# Strategy name constants (D6) — used in checkpoint metadata assertions
STRATEGY_SOURCE_CONTOUR = "source_contour"
STRATEGY_SOURCE_PLUS_SHIFT = "source_plus_shift"
STRATEGY_TARGET_MEDIAN = "target_median"
STRATEGY_MANUAL = "manual"

# Set of all valid F0 strategy names (for validation in train.py / infer)
VALID_F0_STRATEGIES = frozenset({
    STRATEGY_SOURCE_CONTOUR,
    STRATEGY_SOURCE_PLUS_SHIFT,
    STRATEGY_TARGET_MEDIAN,
    STRATEGY_MANUAL,
})

# Import RMVPE lazily to allow patching in tests
# The actual import happens inside F0Encoder.__init__
from indextts.s2mel.modules.rmvpe import RMVPE


# ---------------------------------------------------------------------------
# Pure strategy functions (train and inference share these exact functions)
# ---------------------------------------------------------------------------

def apply_source_contour(f0: np.ndarray) -> np.ndarray:
    """F0 strategy: keep source F0 contour unchanged.

    Args:
        f0: 1D numpy array in Hz. Unvoiced frames are 0.

    Returns:
        Copy of input f0 (unmodified).
    """
    return f0.copy()


def apply_source_plus_shift(
    f0: np.ndarray,
    src_median_log: float,
    tgt_median_log: float,
) -> np.ndarray:
    """F0 strategy: shift source contour by (target_median - source_median) in log space.

    All voiced frames are scaled by exp(tgt_median_log - src_median_log).
    Unvoiced frames (f0 == 0) remain 0.

    This is a global pitch shift that preserves prosodic contour shape while
    aligning the pitch center to the target speaker's median.

    Note: uses MEDIAN (not mean) for robustness to outlier frames (design D6).

    Args:
        f0: Source F0 array in Hz. Unvoiced = 0.
        src_median_log: log(median_voiced_Hz) of source speaker.
        tgt_median_log: log(median_voiced_Hz) of target speaker.

    Returns:
        Shifted F0 array in Hz. Unvoiced frames remain 0.
    """
    shift_factor = float(np.exp(tgt_median_log - src_median_log))
    result = f0.copy()
    voiced = result > 0
    result[voiced] *= shift_factor
    return result


def apply_target_median(
    f0: np.ndarray,
    tgt_median_log: float,
) -> np.ndarray:
    """F0 strategy: replace all voiced frames with a constant = exp(tgt_median_log).

    Useful when source prosody should not be transferred; only speaker F0
    register matters (e.g. extreme male↔female conversion).

    Unvoiced frames remain 0.

    Args:
        f0: Source F0 array in Hz. Unvoiced = 0.
        tgt_median_log: log(median_voiced_Hz) of target speaker.

    Returns:
        F0 array where all voiced frames = exp(tgt_median_log).
    """
    target_hz = float(np.exp(tgt_median_log))
    result = f0.copy()
    voiced = result > 0
    result[voiced] = target_hz
    return result


def apply_manual(
    f0: np.ndarray,
    semitone_offset: float,
) -> np.ndarray:
    """F0 strategy: shift source contour by a fixed semitone offset.

    Formula: f0_out = f0_in * 2^(semitone_offset / 12)
    12 semitones up  → 2x frequency
    12 semitones down → 0.5x frequency

    Unvoiced frames remain 0.

    Args:
        f0: Source F0 array in Hz. Unvoiced = 0.
        semitone_offset: Positive = raise pitch, negative = lower pitch.

    Returns:
        Pitch-shifted F0 array in Hz. Unvoiced frames remain 0.
    """
    shift_factor = float(2.0 ** (semitone_offset / 12.0))
    result = f0.copy()
    voiced = result > 0
    result[voiced] *= shift_factor
    return result


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def estimate_speaker_median_log_f0(f0: np.ndarray) -> float:
    """Compute the median log F0 over voiced frames.

    Uses MEDIAN (design D6) for robustness to pitch outliers.
    Voiced frames: f0 > 0.

    Args:
        f0: 1D numpy float array in Hz. Unvoiced = 0.

    Returns:
        float: median(log(f0[voiced])). Returns 0.0 if no voiced frames.
    """
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return 0.0
    return float(np.median(np.log(voiced)))


def align_to_content_framerate(
    f0_100hz: np.ndarray,
    target_frames: int,
) -> torch.Tensor:
    """Resample F0 from 100 Hz to target_frames using nearest-neighbor interpolation.

    HuBERT-soft runs at ~50 Hz; RMVPE at 100 Hz. Preprocessing decimates
    from 100 Hz to content encoder frame rate (target_frames ≈ T_samples / 320).

    Args:
        f0_100hz: 1D numpy array at 100 Hz in Hz.
        target_frames: Target number of frames (e.g. n_content_frames at 50 Hz).

    Returns:
        1D torch.Tensor of shape (target_frames,) in Hz.
    """
    # Use F.interpolate for consistent behavior with length_regulator internal ops
    f0_tensor = torch.from_numpy(f0_100hz).float()
    # Reshape to (1, 1, T) for interpolate
    f0_4d = f0_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T_100hz)
    f0_resampled = F.interpolate(f0_4d, size=target_frames, mode="nearest")
    return f0_resampled.squeeze(0).squeeze(0)  # (target_frames,)


# ---------------------------------------------------------------------------
# F0Encoder class (RMVPE wrapper)
# ---------------------------------------------------------------------------

class F0Encoder:
    """RMVPE-based F0 extractor with safe device handling.

    RMVPE has a hardcoded device issue: passing device="cuda" gets forced to
    "cuda:0". This wrapper enforces that callers always pass a fully-qualified
    device string ("cpu" or "cuda:N").

    Args:
        model_path: Path to the RMVPE checkpoint (.pt file, state_dict format).
                    Must exist; raises FileNotFoundError otherwise.
        device: Fully-qualified device string: "cpu" or "cuda:0", "cuda:1", etc.
                Never pass bare "cuda" — the RMVPE internals may misbehave.
        is_half: Use fp16 weights. Default False for CPU; set True for CUDA.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        is_half: bool = False,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RMVPE checkpoint not found: {model_path!r}. "
                "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI "
                "(file: rmvpe.pt). See checkpoints/rmvpe/README.md for details."
            )
        if device == "cuda":
            raise ValueError(
                "Do not pass bare 'cuda' as device — RMVPE will force it to 'cuda:0'. "
                "Pass 'cuda:0' (or 'cuda:N') explicitly."
            )
        self._device = device
        self._rmvpe = RMVPE(model_path=model_path, is_half=is_half, device=device)

    def extract(self, audio_16k: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """Extract F0 from 16kHz audio.

        Args:
            audio_16k: 1D numpy float array at 16kHz.
            thred: Voiced/unvoiced threshold for RMVPE. Default 0.03.

        Returns:
            1D numpy float32 array at 100 Hz in Hz. Unvoiced frames = 0.
        """
        return self._rmvpe.infer_from_audio(audio_16k, thred=thred)

    def estimate_speaker_median_log_f0(self, f0: np.ndarray) -> float:
        """Delegate to module-level estimate_speaker_median_log_f0."""
        return estimate_speaker_median_log_f0(f0)

    @property
    def device(self) -> str:
        return self._device
