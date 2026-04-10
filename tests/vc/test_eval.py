"""
tests/vc/test_eval.py

TDD tests for indextts/vc_train/eval.py (task 4.6a).

All external models (ECAPA-TDNN, whisper, UTMOSv2) are mocked —
tests verify metric CALCULATION LOGIC, not model loading.

Tests:
  - run_eval returns dict with all required metric keys
  - SECS uses ECAPA-TDNN (separate from CAMPPlus)
  - F0 RMSE computed in cents (not Hz)
  - CER is character-level (not word-level)
  - UTMOSv2 failure is silently caught (try/except isolated)
  - JSON report is written correctly
"""
from __future__ import annotations

import json
import os
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


from indextts.vc_train.eval import (
    compute_f0_rmse_cents,
    compute_secs_from_embeddings,
    compute_cer_char_level,
    run_eval,
    EvalResult,
)


# ---------------------------------------------------------------------------
# Unit tests for individual metric functions
# ---------------------------------------------------------------------------

class TestF0RmseCents:
    def test_identical_f0_zero_rmse(self):
        """Identical voiced F0 arrays should give RMSE = 0."""
        f0 = np.array([100.0, 150.0, 200.0, 0.0, 250.0], dtype=np.float32)
        result = compute_f0_rmse_cents(f0, f0)
        assert result["voiced"] == pytest.approx(0.0, abs=1e-4)

    def test_octave_shift_gives_1200_cents(self):
        """One octave up (2x frequency) should give exactly 1200 cents RMSE."""
        f0_ref = np.array([100.0, 100.0, 100.0], dtype=np.float32)  # all voiced
        f0_pred = f0_ref * 2.0  # one octave up
        result = compute_f0_rmse_cents(f0_ref, f0_pred)
        assert result["voiced"] == pytest.approx(1200.0, abs=1e-2)

    def test_unit_is_cents_not_hz(self):
        """Verify the result is in cents (scaled by 1200/log(2)), not Hz."""
        f0_ref = np.array([100.0, 200.0], dtype=np.float32)
        f0_pred = np.array([110.0, 220.0], dtype=np.float32)  # ~165 cents difference
        result = compute_f0_rmse_cents(f0_ref, f0_pred)
        # In Hz: diff would be ~15 Hz; in cents: ~165 cents
        assert result["voiced"] > 50, "Expected cents-scale result (> 50), got Hz-scale?"

    def test_unvoiced_frames_excluded_from_voiced_rmse(self):
        """Frames where f0_ref == 0 should not affect voiced RMSE."""
        f0_ref = np.array([100.0, 0.0, 200.0], dtype=np.float32)
        f0_pred = np.array([100.0, 999.0, 200.0], dtype=np.float32)  # 999 is in unvoiced frame
        result = compute_f0_rmse_cents(f0_ref, f0_pred)
        assert result["voiced"] == pytest.approx(0.0, abs=1e-4), (
            "Unvoiced frames must be excluded from voiced RMSE"
        )

    def test_returns_dict_with_voiced_and_unvoiced_keys(self):
        """compute_f0_rmse_cents must return dict with 'voiced' and 'unvoiced' keys."""
        f0 = np.array([100.0, 0.0, 150.0], dtype=np.float32)
        result = compute_f0_rmse_cents(f0, f0)
        assert "voiced" in result
        assert "unvoiced" in result

    def test_no_voiced_frames_returns_nan_or_zero(self):
        """If no voiced frames, voiced RMSE should be NaN or 0 (not raise)."""
        f0_ref = np.zeros(10, dtype=np.float32)
        f0_pred = np.zeros(10, dtype=np.float32)
        result = compute_f0_rmse_cents(f0_ref, f0_pred)
        # Should not raise; NaN or 0 is acceptable
        assert "voiced" in result


class TestSecsFromEmbeddings:
    def test_identical_embeddings_give_cosine_1(self):
        """Same embedding vector → cosine similarity = 1.0."""
        emb = torch.randn(192)
        emb_norm = emb / emb.norm()
        secs = compute_secs_from_embeddings(emb_norm, emb_norm)
        assert secs == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_embeddings_give_cosine_0(self):
        """Orthogonal embeddings → cosine similarity ≈ 0."""
        emb_a = torch.zeros(192)
        emb_a[0] = 1.0
        emb_b = torch.zeros(192)
        emb_b[1] = 1.0
        secs = compute_secs_from_embeddings(emb_a, emb_b)
        assert secs == pytest.approx(0.0, abs=1e-5)

    def test_output_is_scalar_float(self):
        emb_a = torch.randn(192)
        emb_b = torch.randn(192)
        secs = compute_secs_from_embeddings(emb_a, emb_b)
        assert isinstance(secs, float)

    def test_batch_embeddings(self):
        """Accepts (B, D) batch tensors, returns mean cosine similarity."""
        emb_a = torch.randn(4, 192)
        emb_b = torch.randn(4, 192)
        secs = compute_secs_from_embeddings(emb_a, emb_b)
        assert isinstance(secs, float)
        assert -1.0 <= secs <= 1.0


class TestCerCharLevel:
    def test_identical_text_zero_cer(self):
        """Same reference and hypothesis → CER = 0."""
        cer = compute_cer_char_level("你好世界", "你好世界")
        assert cer == pytest.approx(0.0, abs=1e-6)

    def test_completely_different_text(self):
        """Completely different text → CER = 1.0 (or possibly > 1.0 for insertions)."""
        cer = compute_cer_char_level("abc", "xyz")
        assert cer >= 0.9, f"Expected high CER, got {cer}"

    def test_one_substitution(self):
        """One substitution in 4-char string → CER = 1/4 = 0.25."""
        cer = compute_cer_char_level("abcd", "abce")
        assert cer == pytest.approx(1.0 / 4.0, abs=0.01)

    def test_cer_is_character_not_word(self):
        """CER must use character edit distance (not word-level WER)."""
        # 'hello world' vs 'hello world !' — 2 char insertion / 11 chars = ~0.18
        # If word-level: 1 word diff / 2 words = 0.5
        cer = compute_cer_char_level("hello world", "hello world !")
        # Character-level: ~2/11; Word-level: ~1/2 = 0.5
        # If CER > 0.4, it's likely word-level
        assert cer < 0.4, (
            f"CER={cer:.3f} looks like word-level, expected character-level (<0.4)"
        )

    def test_empty_reference_returns_nan_or_one(self):
        """Empty reference string should not crash."""
        cer = compute_cer_char_level("", "hello")
        assert cer >= 0.0


# ---------------------------------------------------------------------------
# Integration test: run_eval with mocked models
# ---------------------------------------------------------------------------

class TestRunEval:
    """Tests for run_eval function with all external models mocked."""

    REQUIRED_KEYS = {
        "secs_ecapa",
        "secs_campplus",
        "cer_whisper",
        "f0_rmse_cents_voiced",
        "f0_rmse_cents_unvoiced",
        "utmos",
    }

    def _make_mock_ecapa(self):
        """Mock ECAPA-TDNN that returns random 192-dim embeddings."""
        mock = MagicMock()
        mock.encode_batch.return_value = torch.randn(1, 1, 192)
        return mock

    def _make_mock_campplus(self):
        """Mock CAMPPlus that returns random 192-dim embeddings."""
        mock = MagicMock()
        mock.return_value = torch.randn(1, 192)
        return mock

    def _make_mock_whisper(self):
        """Mock Whisper that returns a transcription string."""
        mock = MagicMock()
        mock.transcribe.return_value = {"text": "你好世界"}
        return mock

    def _make_eval_inputs(self):
        """Synthetic audio arrays and F0 for testing."""
        sr = 16000
        duration = 1.0
        n = int(sr * duration)
        return {
            "source_audio": np.random.randn(n).astype(np.float32),
            "converted_audio": np.random.randn(n).astype(np.float32),
            "reference_audio": np.random.randn(n).astype(np.float32),
            "source_text": "你好世界",
            "source_f0": np.random.uniform(80, 300, 100).astype(np.float32),
            "converted_f0": np.random.uniform(80, 300, 100).astype(np.float32),
            "sample_rate": sr,
        }

    def test_run_eval_returns_dict_with_required_keys(self, tmp_path):
        """run_eval must return dict with all required metric keys."""
        inputs = self._make_eval_inputs()
        mock_ecapa = self._make_mock_ecapa()
        mock_campplus = self._make_mock_campplus()
        mock_whisper = self._make_mock_whisper()

        result = run_eval(
            source_audio=inputs["source_audio"],
            converted_audio=inputs["converted_audio"],
            reference_audio=inputs["reference_audio"],
            source_text=inputs["source_text"],
            source_f0=inputs["source_f0"],
            converted_f0=inputs["converted_f0"],
            sample_rate=inputs["sample_rate"],
            ecapa_model=mock_ecapa,
            campplus_model=mock_campplus,
            whisper_model=mock_whisper,
            utmos_model=None,  # UTMOSv2 absent → utmos=None
        )

        assert isinstance(result, dict), "run_eval must return a dict"
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key!r}"

    def test_secs_ecapa_uses_ecapa_model(self, tmp_path):
        """secs_ecapa must come from ECAPA-TDNN, not CAMPPlus."""
        inputs = self._make_eval_inputs()
        mock_ecapa = self._make_mock_ecapa()
        mock_campplus = self._make_mock_campplus()
        mock_whisper = self._make_mock_whisper()

        run_eval(
            source_audio=inputs["source_audio"],
            converted_audio=inputs["converted_audio"],
            reference_audio=inputs["reference_audio"],
            source_text=inputs["source_text"],
            source_f0=inputs["source_f0"],
            converted_f0=inputs["converted_f0"],
            sample_rate=inputs["sample_rate"],
            ecapa_model=mock_ecapa,
            campplus_model=mock_campplus,
            whisper_model=mock_whisper,
        )
        # ECAPA model must have been called (not zero times)
        assert mock_ecapa.encode_batch.call_count > 0, (
            "ECAPA-TDNN encode_batch must be called for secs_ecapa computation"
        )

    def test_f0_rmse_is_cents_scale(self, tmp_path):
        """f0_rmse_cents_voiced should be in cents scale (not Hz)."""
        inputs = self._make_eval_inputs()
        # Source F0 doubled → should give ~1200 cents RMSE
        doubled_f0 = inputs["source_f0"] * 2.0
        mock_ecapa = self._make_mock_ecapa()
        mock_campplus = self._make_mock_campplus()
        mock_whisper = self._make_mock_whisper()

        result = run_eval(
            source_audio=inputs["source_audio"],
            converted_audio=inputs["converted_audio"],
            reference_audio=inputs["reference_audio"],
            source_text=inputs["source_text"],
            source_f0=inputs["source_f0"],
            converted_f0=doubled_f0,
            sample_rate=inputs["sample_rate"],
            ecapa_model=mock_ecapa,
            campplus_model=mock_campplus,
            whisper_model=mock_whisper,
        )
        # 1 octave → ~1200 cents
        assert result["f0_rmse_cents_voiced"] > 500, (
            f"F0 RMSE {result['f0_rmse_cents_voiced']:.1f} looks like Hz, not cents. "
            "Expected ~1200 for 1-octave difference."
        )

    def test_utmos_none_when_model_absent(self):
        """When utmos_model=None, utmos key should be None (not raise)."""
        inputs = self._make_eval_inputs()
        result = run_eval(
            source_audio=inputs["source_audio"],
            converted_audio=inputs["converted_audio"],
            reference_audio=inputs["reference_audio"],
            source_text=inputs["source_text"],
            source_f0=inputs["source_f0"],
            converted_f0=inputs["converted_f0"],
            sample_rate=inputs["sample_rate"],
            ecapa_model=self._make_mock_ecapa(),
            campplus_model=self._make_mock_campplus(),
            whisper_model=self._make_mock_whisper(),
            utmos_model=None,
        )
        assert result["utmos"] is None

    def test_utmos_exception_is_caught(self):
        """If utmos_model raises, utmos key should be None (try/except isolated)."""
        inputs = self._make_eval_inputs()
        bad_utmos = MagicMock()
        bad_utmos.predict.side_effect = RuntimeError("UTMOSv2 unavailable")

        result = run_eval(
            source_audio=inputs["source_audio"],
            converted_audio=inputs["converted_audio"],
            reference_audio=inputs["reference_audio"],
            source_text=inputs["source_text"],
            source_f0=inputs["source_f0"],
            converted_f0=inputs["converted_f0"],
            sample_rate=inputs["sample_rate"],
            ecapa_model=self._make_mock_ecapa(),
            campplus_model=self._make_mock_campplus(),
            whisper_model=self._make_mock_whisper(),
            utmos_model=bad_utmos,
        )
        assert result["utmos"] is None, (
            "UTMOSv2 exception must be caught; utmos key should be None"
        )

    def test_json_report_written(self, tmp_path):
        """run_eval with output_json should write a valid JSON file."""
        inputs = self._make_eval_inputs()
        output_path = str(tmp_path / "eval_report.json")

        run_eval(
            source_audio=inputs["source_audio"],
            converted_audio=inputs["converted_audio"],
            reference_audio=inputs["reference_audio"],
            source_text=inputs["source_text"],
            source_f0=inputs["source_f0"],
            converted_f0=inputs["converted_f0"],
            sample_rate=inputs["sample_rate"],
            ecapa_model=self._make_mock_ecapa(),
            campplus_model=self._make_mock_campplus(),
            whisper_model=self._make_mock_whisper(),
            output_json=output_path,
        )

        assert os.path.exists(output_path), "JSON report file must be created"
        with open(output_path) as f:
            report = json.load(f)
        for key in self.REQUIRED_KEYS:
            assert key in report


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_to_dict_has_all_keys(self):
        er = EvalResult(
            secs_ecapa=0.72,
            secs_campplus=0.68,
            cer_whisper=0.05,
            f0_rmse_cents_voiced=45.0,
            f0_rmse_cents_unvoiced=0.0,
            utmos=3.7,
        )
        d = er.to_dict()
        for key in (
            "secs_ecapa", "secs_campplus", "cer_whisper",
            "f0_rmse_cents_voiced", "f0_rmse_cents_unvoiced", "utmos"
        ):
            assert key in d

    def test_none_utmos_serialisable(self):
        er = EvalResult(
            secs_ecapa=0.6,
            secs_campplus=0.5,
            cer_whisper=0.1,
            f0_rmse_cents_voiced=60.0,
            f0_rmse_cents_unvoiced=0.0,
            utmos=None,
        )
        import json
        d = er.to_dict()
        # Must be JSON-serialisable
        json.dumps(d)
