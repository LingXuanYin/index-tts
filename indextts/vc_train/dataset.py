"""
indextts/vc_train/dataset.py

VCDataset: PyTorch Dataset for VC training.

Key design points (design D8):
  - Builds speaker_id → [utterance_index_list] mapping at init
  - __getitem__ samples prompt from same speaker, different utterance
  - Falls back to self when speaker has only 1 utterance (logs warning once)
  - Filters out utterances with voiced_ratio < min_voiced_ratio at init
  - F0 strategy applied via indextts/vc/f0_encoder.py pure functions
  - collate function: zero-pads sequences, returns length tensors

Feature file naming convention:
  <feature_base>.content.npy   : (T_content, 256)
  <feature_base>.f0.npy        : (T_content,) in Hz, 50 Hz
  <feature_base>.mel.npy       : (80, T_mel) at 22050/80/hop=256
  <feature_base>.style.npy     : (192,) CAMPPlus speaker embedding
"""
from __future__ import annotations

import logging
import random
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from indextts.vc.f0_encoder import (
    STRATEGY_MANUAL,
    STRATEGY_SOURCE_CONTOUR,
    STRATEGY_SOURCE_PLUS_SHIFT,
    STRATEGY_TARGET_MEDIAN,
    apply_manual,
    apply_source_contour,
    apply_source_plus_shift,
    apply_target_median,
    estimate_speaker_median_log_f0,
)

logger = logging.getLogger(__name__)

# Voiced ratio threshold (design D4)
MIN_VOICED_RATIO = 0.1

# Valid F0 strategies
VALID_F0_STRATEGIES = {
    STRATEGY_SOURCE_CONTOUR,
    STRATEGY_SOURCE_PLUS_SHIFT,
    STRATEGY_TARGET_MEDIAN,
    STRATEGY_MANUAL,
}


# ---------------------------------------------------------------------------
# Helper: load preprocessed features from disk
# ---------------------------------------------------------------------------

def _load_features(feature_base: str) -> Optional[Dict]:
    """Load .content / .f0 / .mel / .style npy files for one utterance.

    Returns None if any file is missing or unreadable.

    Args:
        feature_base: Absolute path without extension, e.g.
                      "/data/vc/preprocessed/SSB0001/SSB0001_001"

    Returns:
        dict with keys: content (T,256) / f0 (T,) / mel (80,T_mel) / style (192,)
        or None on failure.
    """
    try:
        content = np.load(feature_base + ".content.npy")  # (T, 256)
        f0 = np.load(feature_base + ".f0.npy")            # (T,)
        mel = np.load(feature_base + ".mel.npy")           # (80, T_mel)
        style = np.load(feature_base + ".style.npy")       # (192,)
    except (FileNotFoundError, OSError):
        return None
    return {"content": content, "f0": f0, "mel": mel, "style": style}


def _compute_voiced_ratio(f0: np.ndarray) -> float:
    if len(f0) == 0:
        return 0.0
    return float(np.sum(f0 > 0)) / float(len(f0))


# ---------------------------------------------------------------------------
# VCDataset
# ---------------------------------------------------------------------------

class VCDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for VC finetuning.

    Args:
        entries: List of dicts, each with at minimum:
            - "speaker_id": str
            - "feature_base": str  (absolute path prefix for .content/.f0/.mel/.style)
            - (any other manifest fields are ignored)
        f0_strategy: One of "source_contour" / "source_plus_shift" /
                     "target_median" / "manual".
        manual_semitone_offset: Used only when f0_strategy == "manual".
        min_voiced_ratio: Utterances with voiced_ratio < this are filtered at init.
        seed: Random seed for prompt sampling (per-epoch determinism not guaranteed;
              sampling is done with random.Random per __getitem__ call).
        prompt_style_from_feature: If True, load prompt speaker style from its own
                                   .style.npy (default). Set False to reuse source style.
    """

    def __init__(
        self,
        entries: List[Dict],
        f0_strategy: str = STRATEGY_SOURCE_CONTOUR,
        manual_semitone_offset: float = 0.0,
        min_voiced_ratio: float = MIN_VOICED_RATIO,
        seed: int = 42,
        prompt_style_from_feature: bool = True,
    ):
        if f0_strategy not in VALID_F0_STRATEGIES:
            raise ValueError(
                f"Unknown f0_strategy: {f0_strategy!r}. "
                f"Must be one of {VALID_F0_STRATEGIES}."
            )
        self.f0_strategy = f0_strategy
        self.manual_semitone_offset = manual_semitone_offset
        self.min_voiced_ratio = min_voiced_ratio
        self.prompt_style_from_feature = prompt_style_from_feature
        self._seed = seed
        self._rng = random.Random(seed)

        # Filter entries
        valid_entries = []
        n_filtered = 0
        _warned_once: set = set()  # track missing feature warnings

        for entry in entries:
            feature_base = entry.get("feature_base", "")
            f0_path = feature_base + ".f0.npy"
            try:
                f0 = np.load(f0_path)
                voiced_ratio = _compute_voiced_ratio(f0)
                if voiced_ratio < min_voiced_ratio:
                    n_filtered += 1
                    logger.debug(
                        "Filtered (voiced_ratio=%.3f < %.3f): %s",
                        voiced_ratio, min_voiced_ratio, feature_base,
                    )
                    continue
            except (FileNotFoundError, OSError):
                # If features don't exist yet, include the entry provisionally
                # (will fail at __getitem__ if not found)
                if feature_base not in _warned_once:
                    logger.debug("F0 file not found (entry kept): %s", f0_path)
                    _warned_once.add(feature_base)

            valid_entries.append(entry)

        if n_filtered > 0:
            logger.info(
                "VCDataset: filtered %d/%d utterances with voiced_ratio < %.2f",
                n_filtered, len(entries), min_voiced_ratio,
            )

        self.entries = valid_entries

        # Build speaker → [index_list] mapping
        self._speaker_to_indices: Dict[str, List[int]] = {}
        for idx, entry in enumerate(self.entries):
            spk = entry["speaker_id"]
            self._speaker_to_indices.setdefault(spk, []).append(idx)

        # Track speakers with only 1 utterance (for single-warning)
        self._single_utt_warned: set = set()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        """Return one training sample.

        Returns:
            dict with:
              content      (T_content, 256) float32 tensor — kmeans-quantized
              f0           (T_content,) float32 tensor — after strategy applied
              mel          (80, T_mel) float32 tensor — source target mel
              style        (192,) float32 tensor — source speaker style
              prompt_mel   (80, T_pmt_mel) float32 tensor
              prompt_content (T_pmt_content, 256) float32 tensor
              speaker_id   str
              source_idx   int
              prompt_idx   int
        """
        entry = self.entries[idx]
        feature_base = entry["feature_base"]
        speaker_id = entry["speaker_id"]

        # Load source features
        feats = _load_features(feature_base)
        if feats is None:
            raise RuntimeError(
                f"Failed to load features from {feature_base!r}. "
                "Run preprocessing first."
            )

        # Sample prompt (design D8: same speaker, different utterance if possible)
        prompt_idx = self._sample_prompt_idx(idx, speaker_id)
        prompt_entry = self.entries[prompt_idx]
        prompt_feats = _load_features(prompt_entry["feature_base"])
        if prompt_feats is None:
            raise RuntimeError(
                f"Failed to load prompt features from {prompt_entry['feature_base']!r}."
            )

        # Apply F0 strategy to source F0
        f0_processed = self._apply_f0_strategy(
            source_f0=feats["f0"],
            prompt_f0=prompt_feats["f0"],
        )

        return {
            "content": torch.from_numpy(feats["content"]).float(),
            "f0": torch.from_numpy(f0_processed).float(),
            "mel": torch.from_numpy(feats["mel"]).float(),
            "style": torch.from_numpy(feats["style"]).float(),
            "prompt_mel": torch.from_numpy(prompt_feats["mel"]).float(),
            "prompt_content": torch.from_numpy(prompt_feats["content"]).float(),
            "speaker_id": speaker_id,
            "source_idx": idx,
            "prompt_idx": prompt_idx,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_prompt_idx(self, source_idx: int, speaker_id: str) -> int:
        """Sample a prompt utterance index from the same speaker.

        Design D8:
          - Pick a random utterance from the same speaker != source_idx
          - Fallback to source_idx if speaker has only 1 utterance (warn once)

        Args:
            source_idx: Index of the source utterance in self.entries.
            speaker_id: Speaker ID of the source.

        Returns:
            Utterance index to use as prompt.
        """
        candidates = self._speaker_to_indices.get(speaker_id, [])
        other_candidates = [i for i in candidates if i != source_idx]

        if not other_candidates:
            # Fallback: single-utterance speaker
            if speaker_id not in self._single_utt_warned:
                warnings.warn(
                    f"Speaker {speaker_id!r} has only 1 utterance. "
                    "Falling back to using source as its own prompt (self-reconstruction). "
                    "This is expected for very small speakers but degrades training quality.",
                    UserWarning,
                    stacklevel=2,
                )
                self._single_utt_warned.add(speaker_id)
            return source_idx

        return self._rng.choice(other_candidates)

    def _apply_f0_strategy(
        self,
        source_f0: np.ndarray,
        prompt_f0: np.ndarray,
    ) -> np.ndarray:
        """Apply the configured F0 strategy to the source F0 contour.

        Args:
            source_f0: Source utterance F0 (Hz, unvoiced=0). shape (T,)
            prompt_f0: Prompt utterance F0 (Hz, unvoiced=0). Used for target stats.

        Returns:
            Processed F0 array (Hz, unvoiced=0). shape (T,)
        """
        strategy = self.f0_strategy

        if strategy == STRATEGY_SOURCE_CONTOUR:
            return apply_source_contour(source_f0)

        elif strategy == STRATEGY_SOURCE_PLUS_SHIFT:
            src_median_log = estimate_speaker_median_log_f0(source_f0)
            tgt_median_log = estimate_speaker_median_log_f0(prompt_f0)
            if src_median_log == 0.0 or tgt_median_log == 0.0:
                # No voiced frames; fall back to source contour
                return apply_source_contour(source_f0)
            return apply_source_plus_shift(source_f0, src_median_log, tgt_median_log)

        elif strategy == STRATEGY_TARGET_MEDIAN:
            tgt_median_log = estimate_speaker_median_log_f0(prompt_f0)
            if tgt_median_log == 0.0:
                return apply_source_contour(source_f0)
            return apply_target_median(source_f0, tgt_median_log)

        elif strategy == STRATEGY_MANUAL:
            return apply_manual(source_f0, self.manual_semitone_offset)

        else:
            raise ValueError(f"Unknown F0 strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def vc_collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of VCDataset items into a padded batch.

    Padding:
      - content: (B, T_content_max, 256), zero-padded on time axis
      - f0:      (B, T_content_max), zero-padded
      - mel:     (B, 80, T_mel_max), zero-padded on time axis
      - style:   (B, 192) — no padding needed
      - prompt_mel:     (B, 80, T_pmt_mel_max)
      - prompt_content: (B, T_pmt_content_max, 256)

    Length tensors (long, shape (B,)):
      - content_lens, mel_lens, prompt_mel_lens, prompt_content_lens

    Args:
        batch: List of dicts from VCDataset.__getitem__

    Returns:
        Single dict with batched tensors.
    """
    B = len(batch)

    # Content: (T_content, D) → pad to max T_content
    content_list = [item["content"] for item in batch]          # [(T_i, D)]
    content_lens = torch.tensor([c.shape[0] for c in content_list], dtype=torch.long)
    content_padded = _pad_sequence_batch(content_list, pad_dim=0)  # (B, T_max, D)

    # F0: (T_content,) → pad to max T_content
    f0_list = [item["f0"] for item in batch]
    f0_padded = _pad_1d_batch(f0_list)                              # (B, T_max)

    # Mel: (80, T_mel) → pad on time axis
    mel_list = [item["mel"] for item in batch]
    mel_lens = torch.tensor([m.shape[1] for m in mel_list], dtype=torch.long)
    mel_padded = _pad_mel_batch(mel_list)                            # (B, 80, T_max)

    # Style: (192,) — no padding
    style = torch.stack([item["style"] for item in batch], dim=0)   # (B, 192)

    # Prompt mel: (80, T_pmt_mel) → pad on time axis
    prompt_mel_list = [item["prompt_mel"] for item in batch]
    prompt_mel_lens = torch.tensor([m.shape[1] for m in prompt_mel_list], dtype=torch.long)
    prompt_mel_padded = _pad_mel_batch(prompt_mel_list)              # (B, 80, T_max)

    # Prompt content: (T_pmt_content, D) → pad to max
    prompt_content_list = [item["prompt_content"] for item in batch]
    prompt_content_lens = torch.tensor(
        [c.shape[0] for c in prompt_content_list], dtype=torch.long
    )
    prompt_content_padded = _pad_sequence_batch(prompt_content_list, pad_dim=0)

    return {
        "content": content_padded,                  # (B, T_content_max, D)
        "content_lens": content_lens,               # (B,)
        "f0": f0_padded,                            # (B, T_content_max)
        "mel": mel_padded,                          # (B, 80, T_mel_max)
        "mel_lens": mel_lens,                       # (B,)
        "style": style,                             # (B, 192)
        "prompt_mel": prompt_mel_padded,            # (B, 80, T_pmt_mel_max)
        "prompt_mel_lens": prompt_mel_lens,         # (B,)
        "prompt_content": prompt_content_padded,    # (B, T_pmt_content_max, D)
        "prompt_content_lens": prompt_content_lens, # (B,)
        "speaker_ids": [item["speaker_id"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def _pad_sequence_batch(
    sequences: List[torch.Tensor],
    pad_dim: int = 0,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad a list of 2D tensors along pad_dim and stack into (B, T_max, D).

    Args:
        sequences: List of (T_i, D) tensors.
        pad_dim: Dimension to pad along (0 = time).
        pad_value: Fill value for padding.

    Returns:
        (B, T_max, D) tensor.
    """
    T_max = max(s.shape[0] for s in sequences)
    D = sequences[0].shape[1]
    B = len(sequences)
    out = torch.full((B, T_max, D), fill_value=pad_value)
    for b, s in enumerate(sequences):
        out[b, : s.shape[0], :] = s
    return out


def _pad_1d_batch(
    sequences: List[torch.Tensor],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad a list of 1D tensors to max length, return (B, T_max)."""
    T_max = max(s.shape[0] for s in sequences)
    B = len(sequences)
    out = torch.full((B, T_max), fill_value=pad_value)
    for b, s in enumerate(sequences):
        out[b, : s.shape[0]] = s
    return out


def _pad_mel_batch(
    mels: List[torch.Tensor],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad a list of (80, T_i) mel tensors to (B, 80, T_max)."""
    T_max = max(m.shape[1] for m in mels)
    B = len(mels)
    bands = mels[0].shape[0]
    out = torch.full((B, bands, T_max), fill_value=pad_value)
    for b, m in enumerate(mels):
        out[b, :, : m.shape[1]] = m
    return out
