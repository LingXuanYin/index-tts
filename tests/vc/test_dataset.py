"""
tests/vc/test_dataset.py

TDD tests for indextts/vc_train/dataset.py (task 4.3a).

Tests:
  - __getitem__ returns dict with all required fields
  - prompt != source constraint (design D8)
  - prompt and source same speaker (design D8)
  - single-utterance speaker fallback (self-copy)
  - voiced_ratio < 0.1 entries are filtered out at dataset construction
  - collate function: pads + builds length masks
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List

import numpy as np
import pytest
import torch

from indextts.vc_train.dataset import VCDataset, vc_collate_fn


# ---------------------------------------------------------------------------
# Fixtures: build synthetic preprocessed feature files
# ---------------------------------------------------------------------------

CONTENT_DIM = 256
STYLE_DIM = 192
MEL_BANDS = 80
N_F0_BINS = 256  # not used by dataset but kept for reference

def _write_features(
    out_dir: str,
    stem: str,
    n_content_frames: int,
    n_mel_frames: int,
    voiced_ratio: float = 0.8,
) -> str:
    """Write .content.npy / .f0.npy / .mel.npy / .style.npy for a fake utterance.

    Returns the absolute base path (without extension).
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem)

    # content features
    content = np.random.randn(n_content_frames, CONTENT_DIM).astype(np.float32)
    np.save(base + ".content.npy", content)

    # f0: set voiced_ratio fraction to non-zero values
    f0 = np.zeros(n_content_frames, dtype=np.float32)
    n_voiced = max(0, int(voiced_ratio * n_content_frames))
    if n_voiced > 0:
        f0[:n_voiced] = np.random.uniform(80, 300, size=n_voiced).astype(np.float32)
    np.save(base + ".f0.npy", f0)

    # mel
    mel = np.random.randn(MEL_BANDS, n_mel_frames).astype(np.float32)
    np.save(base + ".mel.npy", mel)

    # style
    style = np.random.randn(STYLE_DIM).astype(np.float32)
    np.save(base + ".style.npy", style)

    return base


@pytest.fixture
def feature_dir(tmp_path):
    """
    Build a dataset with:
      - SPK001: utterances 0, 1, 2  (3 utterances)
      - SPK002: utterance 0          (1 utterance — single-speaker fallback)
      - SPK003: utterances 0, 1     (2 utterances, voiced_ratio=0.05 → should be filtered)
    """
    entries = []

    # SPK001 — 3 utterances (good voiced_ratio)
    for i in range(3):
        stem = f"SPK001_utt{i:02d}"
        _write_features(str(tmp_path), stem, n_content_frames=50, n_mel_frames=86)
        entries.append({
            "audio_path": f"{stem}.wav",   # dummy path (features already saved)
            "feature_base": str(tmp_path / stem),
            "speaker_id": "SPK001",
            "language": "zh",
            "duration_s": 1.0,
            "text": None,
        })

    # SPK002 — 1 utterance only
    stem = "SPK002_utt00"
    _write_features(str(tmp_path), stem, n_content_frames=40, n_mel_frames=70)
    entries.append({
        "audio_path": f"{stem}.wav",
        "feature_base": str(tmp_path / stem),
        "speaker_id": "SPK002",
        "language": "zh",
        "duration_s": 0.8,
        "text": None,
    })

    # SPK003 — 2 utterances, both with low voiced_ratio → should be filtered
    for i in range(2):
        stem = f"SPK003_utt{i:02d}"
        _write_features(
            str(tmp_path), stem,
            n_content_frames=50, n_mel_frames=86,
            voiced_ratio=0.05,   # below 0.1 threshold
        )
        entries.append({
            "audio_path": f"{stem}.wav",
            "feature_base": str(tmp_path / stem),
            "speaker_id": "SPK003",
            "language": "zh",
            "duration_s": 1.0,
            "text": None,
        })

    return tmp_path, entries


@pytest.fixture
def dataset_spk001_spk002(feature_dir):
    """VCDataset with SPK001 (3 utts) and SPK002 (1 utt), no SPK003."""
    tmp_path, all_entries = feature_dir
    # Only include SPK001 and SPK002 (exclude SPK003 low voiced_ratio)
    entries = [e for e in all_entries if e["speaker_id"] in ("SPK001", "SPK002")]
    return VCDataset(entries=entries, f0_strategy="source_contour")


@pytest.fixture
def dataset_with_filtered(feature_dir):
    """VCDataset with all entries (SPK003 should be filtered out at init)."""
    tmp_path, all_entries = feature_dir
    return VCDataset(entries=all_entries, f0_strategy="source_contour")


# ---------------------------------------------------------------------------
# 1. __getitem__ returns required fields
# ---------------------------------------------------------------------------

class TestGetItemFields:
    REQUIRED_KEYS = {
        "content",
        "f0",
        "mel",
        "style",
        "prompt_mel",
        "prompt_content",
        "speaker_id",
        "source_idx",
        "prompt_idx",
    }

    def test_all_required_keys_present(self, dataset_spk001_spk002):
        ds = dataset_spk001_spk002
        item = ds[0]
        for key in self.REQUIRED_KEYS:
            assert key in item, f"Missing key: {key!r}"

    def test_content_is_float_tensor(self, dataset_spk001_spk002):
        item = dataset_spk001_spk002[0]
        assert isinstance(item["content"], torch.Tensor)
        assert item["content"].dtype == torch.float32

    def test_content_shape(self, dataset_spk001_spk002):
        item = dataset_spk001_spk002[0]
        T, D = item["content"].shape
        assert D == CONTENT_DIM, f"Expected content dim {CONTENT_DIM}, got {D}"

    def test_f0_is_float_tensor(self, dataset_spk001_spk002):
        item = dataset_spk001_spk002[0]
        assert isinstance(item["f0"], torch.Tensor)
        assert item["f0"].dtype == torch.float32
        assert item["f0"].ndim == 1

    def test_mel_shape(self, dataset_spk001_spk002):
        item = dataset_spk001_spk002[0]
        bands, T = item["mel"].shape
        assert bands == MEL_BANDS, f"Expected {MEL_BANDS} mel bands, got {bands}"

    def test_style_shape(self, dataset_spk001_spk002):
        item = dataset_spk001_spk002[0]
        assert item["style"].shape == (STYLE_DIM,)

    def test_prompt_mel_shape(self, dataset_spk001_spk002):
        item = dataset_spk001_spk002[0]
        bands, T = item["prompt_mel"].shape
        assert bands == MEL_BANDS

    def test_prompt_content_shape(self, dataset_spk001_spk002):
        item = dataset_spk001_spk002[0]
        T, D = item["prompt_content"].shape
        assert D == CONTENT_DIM

    def test_speaker_id_is_string(self, dataset_spk001_spk002):
        item = dataset_spk001_spk002[0]
        assert isinstance(item["speaker_id"], str)
        assert len(item["speaker_id"]) > 0


# ---------------------------------------------------------------------------
# 2. Prompt != source constraint (design D8)
# ---------------------------------------------------------------------------

class TestPromptDifferentFromSource:
    def test_prompt_idx_ne_source_idx_for_multi_utterance_speaker(self, dataset_spk001_spk002):
        """For SPK001 (3 utts), prompt must be a different utterance."""
        ds = dataset_spk001_spk002
        # Collect indices belonging to SPK001
        spk001_indices = [
            i for i in range(len(ds))
            if ds[i]["speaker_id"] == "SPK001"
        ]
        assert len(spk001_indices) >= 2, "Need at least 2 SPK001 utterances"

        for src_idx in spk001_indices:
            item = ds[src_idx]
            assert item["source_idx"] != item["prompt_idx"], (
                f"Prompt and source are the same utterance at source_idx={src_idx}! "
                "This violates design D8."
            )

    def test_prompt_content_not_identical_to_source_content(self, dataset_spk001_spk002):
        """Prompt content features must come from a different utterance (different values)."""
        ds = dataset_spk001_spk002
        # Find a SPK001 item (which has 3 utterances → can always pick different)
        for i in range(len(ds)):
            item = ds[i]
            if item["speaker_id"] == "SPK001":
                # source and prompt content should differ in shape or values
                # (synthetic features are random, so they differ unless same file loaded)
                src_content = item["content"]
                pmt_content = item["prompt_content"]
                # Different shapes OR different values → not the same utterance
                shape_diff = src_content.shape != pmt_content.shape
                if not shape_diff:
                    value_diff = not torch.allclose(src_content, pmt_content)
                    assert value_diff, (
                        "Prompt content is identical to source content — "
                        "looks like the same file was loaded for both."
                    )
                break

    def test_many_samples_always_different_idx(self, feature_dir):
        """Over 50 random accesses to SPK001, prompt_idx always != source_idx."""
        tmp_path, all_entries = feature_dir
        entries = [e for e in all_entries if e["speaker_id"] in ("SPK001",)]
        ds = VCDataset(entries=entries, f0_strategy="source_contour")

        for _ in range(50):
            for i in range(len(ds)):
                item = ds[i]
                assert item["source_idx"] != item["prompt_idx"], (
                    "Prompt equals source — D8 violation."
                )


# ---------------------------------------------------------------------------
# 3. Prompt and source same speaker (design D8)
# ---------------------------------------------------------------------------

class TestPromptSameSpeaker:
    def test_prompt_same_speaker_as_source(self, dataset_spk001_spk002):
        """Every item's prompt must be from the same speaker as the source."""
        ds = dataset_spk001_spk002
        for i in range(len(ds)):
            item = ds[i]
            src_spk = item["speaker_id"]
            pmt_spk = ds.entries[item["prompt_idx"]]["speaker_id"]
            assert src_spk == pmt_spk, (
                f"Prompt speaker {pmt_spk!r} != source speaker {src_spk!r} at idx={i}"
            )


# ---------------------------------------------------------------------------
# 4. Single-utterance speaker fallback
# ---------------------------------------------------------------------------

class TestSingleUtteranceFallback:
    def test_single_utt_does_not_raise(self, dataset_spk001_spk002):
        """SPK002 has only 1 utterance; __getitem__ must not raise."""
        ds = dataset_spk001_spk002
        spk002_indices = [
            i for i in range(len(ds))
            if ds[i]["speaker_id"] == "SPK002"
        ]
        assert len(spk002_indices) == 1

        # Should not raise
        item = ds[spk002_indices[0]]
        assert "content" in item

    def test_single_utt_prompt_equals_source(self, dataset_spk001_spk002):
        """For SPK002 (1 utt), prompt fallback = source itself."""
        ds = dataset_spk001_spk002
        spk002_indices = [
            i for i in range(len(ds))
            if ds[i]["speaker_id"] == "SPK002"
        ]
        item = ds[spk002_indices[0]]
        # Fallback: source_idx == prompt_idx (can't pick different)
        assert item["source_idx"] == item["prompt_idx"], (
            "Single-utterance speaker should fall back to source == prompt."
        )


# ---------------------------------------------------------------------------
# 5. voiced_ratio < 0.1 filtering
# ---------------------------------------------------------------------------

class TestVoicedRatioFiltering:
    def test_low_voiced_ratio_entries_filtered(self, dataset_with_filtered):
        """SPK003 utterances have voiced_ratio=0.05 (<0.1) → excluded from dataset."""
        ds = dataset_with_filtered
        for i in range(len(ds)):
            item = ds[i]
            assert item["speaker_id"] != "SPK003", (
                f"SPK003 entry at idx={i} should have been filtered out "
                f"(voiced_ratio < 0.1)."
            )

    def test_dataset_length_after_filtering(self, dataset_with_filtered):
        """After filtering SPK003 (2 entries), only SPK001 (3) + SPK002 (1) remain."""
        ds = dataset_with_filtered
        assert len(ds) == 4, f"Expected 4 entries after filtering, got {len(ds)}"


# ---------------------------------------------------------------------------
# 6. Collate function
# ---------------------------------------------------------------------------

class TestCollate:
    def _make_items(self, n: int = 3) -> List[Dict]:
        """Make n mock items with varying lengths."""
        items = []
        for i in range(n):
            t_content = 30 + i * 10
            t_mel = 50 + i * 15
            t_pmt_content = 25 + i * 5
            t_pmt_mel = 40 + i * 10
            items.append({
                "content": torch.randn(t_content, CONTENT_DIM),
                "f0": torch.zeros(t_content),
                "mel": torch.randn(MEL_BANDS, t_mel),
                "style": torch.randn(STYLE_DIM),
                "prompt_mel": torch.randn(MEL_BANDS, t_pmt_mel),
                "prompt_content": torch.randn(t_pmt_content, CONTENT_DIM),
                "speaker_id": f"SPK{i:03d}",
                "source_idx": i,
                "prompt_idx": i,
            })
        return items

    def test_collate_returns_dict(self):
        items = self._make_items(3)
        batch = vc_collate_fn(items)
        assert isinstance(batch, dict)

    def test_collate_content_shape(self):
        items = self._make_items(3)
        batch = vc_collate_fn(items)
        B, T_max, D = batch["content"].shape
        assert B == 3
        assert D == CONTENT_DIM
        # T_max should be max across items
        expected_T_max = max(it["content"].shape[0] for it in items)
        assert T_max == expected_T_max

    def test_collate_mel_shape(self):
        items = self._make_items(3)
        batch = vc_collate_fn(items)
        B, bands, T_max = batch["mel"].shape
        assert B == 3
        assert bands == MEL_BANDS

    def test_collate_has_length_fields(self):
        items = self._make_items(3)
        batch = vc_collate_fn(items)
        assert "content_lens" in batch
        assert "mel_lens" in batch
        assert "prompt_mel_lens" in batch
        assert "prompt_content_lens" in batch

    def test_collate_content_lens_correct(self):
        items = self._make_items(3)
        batch = vc_collate_fn(items)
        for b, item in enumerate(items):
            assert batch["content_lens"][b] == item["content"].shape[0]

    def test_collate_style_shape(self):
        items = self._make_items(3)
        batch = vc_collate_fn(items)
        assert batch["style"].shape == (3, STYLE_DIM)

    def test_collate_single_item(self):
        """Single-item batch should still work."""
        items = self._make_items(1)
        batch = vc_collate_fn(items)
        assert batch["content"].shape[0] == 1
