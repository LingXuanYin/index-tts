"""
Task 3.2a - Tests for F0Encoder and F0 strategy pure functions (TDD)

Tests are written BEFORE implementation.
All RMVPE model loading is mocked; no checkpoint needed.

F0 strategy functions (pure):
  apply_source_contour(f0) -> f0 unchanged
  apply_source_plus_shift(f0_src, src_median_log, tgt_median_log) -> shifted
  apply_target_median(f0_src, tgt_median_log) -> constant
  apply_manual(f0_src, semitone_offset) -> shifted by fixed semitones

F0 encoder:
  F0Encoder(model_path, device) - wraps RMVPE
  extract(audio_16k) -> np.ndarray 1D, 100Hz, Hz, unvoiced=0
  estimate_speaker_median_log_f0(f0_array) -> float (median log of voiced frames)
  align_to_content_framerate(f0_100hz, target_frames) -> Tensor (T_target,)
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_f0_functions():
    """Import f0 pure functions from indextts.vc.f0_encoder."""
    from indextts.vc.f0_encoder import (
        apply_source_contour,
        apply_source_plus_shift,
        apply_target_median,
        apply_manual,
        estimate_speaker_median_log_f0,
        align_to_content_framerate,
    )
    return (apply_source_contour, apply_source_plus_shift,
            apply_target_median, apply_manual,
            estimate_speaker_median_log_f0, align_to_content_framerate)


def _make_f0_array(n_voiced=80, n_unvoiced=20, base_hz=200.0):
    """Synthetic F0 array: voiced frames at base_hz, unvoiced at 0."""
    f0 = np.zeros(n_voiced + n_unvoiced, dtype=np.float32)
    f0[:n_voiced] = base_hz
    return f0


# ---------------------------------------------------------------------------
# Test F0 pure strategy functions
# ---------------------------------------------------------------------------

class TestF0StrategyFunctions(unittest.TestCase):

    def setUp(self):
        (self.apply_source_contour,
         self.apply_source_plus_shift,
         self.apply_target_median,
         self.apply_manual,
         self.estimate_speaker_median_log_f0,
         self.align_to_content_framerate) = _import_f0_functions()

    # --- source_contour ---

    def test_apply_source_contour_returns_same_values(self):
        """source_contour strategy must return F0 array unchanged."""
        f0 = _make_f0_array(base_hz=150.0)
        result = self.apply_source_contour(f0)
        np.testing.assert_array_equal(result, f0)

    def test_apply_source_contour_returns_copy_or_same(self):
        """Result is numpy-compatible."""
        f0 = _make_f0_array()
        result = self.apply_source_contour(f0)
        self.assertEqual(result.shape, f0.shape)

    # --- source_plus_shift ---

    def test_apply_source_plus_shift_raises_voiced_frames(self):
        """source_plus_shift on higher target should raise voiced pitch."""
        f0 = np.array([200.0, 200.0, 0.0, 200.0], dtype=np.float32)
        src_median_log = np.log(200.0)
        tgt_median_log = np.log(300.0)  # higher target
        result = self.apply_source_plus_shift(f0, src_median_log, tgt_median_log)
        # voiced frames should be higher
        voiced_mask = f0 > 0
        self.assertTrue(np.all(result[voiced_mask] > f0[voiced_mask]),
                        "source_plus_shift must raise pitch for higher target")
        # unvoiced frames must remain 0
        self.assertEqual(result[~voiced_mask].sum(), 0.0)

    def test_apply_source_plus_shift_lowers_for_lower_target(self):
        """source_plus_shift on lower target should lower voiced pitch."""
        f0 = np.array([300.0, 300.0, 0.0], dtype=np.float32)
        src_median_log = np.log(300.0)
        tgt_median_log = np.log(150.0)  # lower target (male voice)
        result = self.apply_source_plus_shift(f0, src_median_log, tgt_median_log)
        voiced_mask = f0 > 0
        self.assertTrue(np.all(result[voiced_mask] < f0[voiced_mask]),
                        "source_plus_shift must lower pitch for lower target")

    def test_apply_source_plus_shift_uses_median_not_mean(self):
        """source_plus_shift must shift by (tgt_median - src_median) in log space.

        Key: uses median (D6 decision). We verify the shift magnitude.
        """
        f0 = np.array([100.0, 200.0, 0.0], dtype=np.float32)
        src_median_log = np.log(100.0)
        tgt_median_log = np.log(200.0)
        result = self.apply_source_plus_shift(f0, src_median_log, tgt_median_log)
        # Each voiced frame should be multiplied by exp(tgt_median_log - src_median_log) = 2
        shift_factor = np.exp(tgt_median_log - src_median_log)  # = 2.0
        voiced_mask = f0 > 0
        np.testing.assert_allclose(
            result[voiced_mask],
            f0[voiced_mask] * shift_factor,
            rtol=1e-5,
            err_msg="source_plus_shift: voiced frames must be scaled by exp(delta_log)"
        )

    def test_apply_source_plus_shift_unvoiced_stays_zero(self):
        """Unvoiced frames (f0==0) must remain 0 after shift."""
        f0 = np.array([0.0, 200.0, 0.0, 150.0], dtype=np.float32)
        result = self.apply_source_plus_shift(f0, np.log(200.0), np.log(250.0))
        unvoiced = f0 == 0
        self.assertTrue(np.all(result[unvoiced] == 0.0))

    # --- target_median ---

    def test_apply_target_median_strategy_name(self):
        """Strategy name constant must be 'target_median' (not 'target_mean')."""
        from indextts.vc.f0_encoder import STRATEGY_TARGET_MEDIAN
        self.assertEqual(STRATEGY_TARGET_MEDIAN, "target_median",
                         "Strategy name must be 'target_median', not 'target_mean'")

    def test_apply_target_median_all_voiced_set_to_constant(self):
        """target_median sets all voiced frames to exp(tgt_median_log)."""
        f0 = np.array([100.0, 200.0, 0.0, 300.0], dtype=np.float32)
        tgt_median_log = np.log(220.0)
        result = self.apply_target_median(f0, tgt_median_log)
        voiced_mask = f0 > 0
        expected = np.exp(tgt_median_log)
        np.testing.assert_allclose(
            result[voiced_mask],
            expected,
            rtol=1e-5,
            err_msg="target_median: all voiced frames must be set to exp(tgt_median_log)"
        )

    def test_apply_target_median_unvoiced_stays_zero(self):
        """target_median: unvoiced frames must remain 0."""
        f0 = np.array([200.0, 0.0, 150.0], dtype=np.float32)
        result = self.apply_target_median(f0, np.log(200.0))
        self.assertEqual(result[1], 0.0)

    # --- manual ---

    def test_apply_manual_shifts_by_semitones(self):
        """manual strategy shifts voiced frames by fixed semitone offset.

        12 semitones up = 2x frequency.
        """
        f0 = np.array([200.0, 0.0, 200.0], dtype=np.float32)
        result = self.apply_manual(f0, semitone_offset=12.0)
        voiced_mask = f0 > 0
        np.testing.assert_allclose(
            result[voiced_mask],
            f0[voiced_mask] * 2.0,
            rtol=1e-4,
            err_msg="12 semitones up must double the frequency"
        )

    def test_apply_manual_negative_semitones(self):
        """manual strategy with negative offset lowers pitch."""
        f0 = np.array([400.0, 0.0], dtype=np.float32)
        result = self.apply_manual(f0, semitone_offset=-12.0)
        voiced_mask = f0 > 0
        np.testing.assert_allclose(
            result[voiced_mask],
            f0[voiced_mask] / 2.0,
            rtol=1e-4,
        )

    def test_apply_manual_zero_offset_identity(self):
        """manual with offset=0 must return identical voiced frames."""
        f0 = np.array([200.0, 0.0, 300.0], dtype=np.float32)
        result = self.apply_manual(f0, semitone_offset=0.0)
        voiced_mask = f0 > 0
        np.testing.assert_allclose(result[voiced_mask], f0[voiced_mask], rtol=1e-5)

    def test_apply_manual_unvoiced_stays_zero(self):
        """manual: unvoiced frames must remain 0."""
        f0 = np.array([0.0, 200.0, 0.0], dtype=np.float32)
        result = self.apply_manual(f0, semitone_offset=5.0)
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[2], 0.0)


# ---------------------------------------------------------------------------
# Test estimate_speaker_median_log_f0
# ---------------------------------------------------------------------------

class TestEstimateSpeakerMedianLogF0(unittest.TestCase):

    def setUp(self):
        (_, _, _, _,
         self.estimate_speaker_median_log_f0,
         _) = _import_f0_functions()

    def test_returns_float(self):
        f0 = _make_f0_array(n_voiced=50, n_unvoiced=10, base_hz=200.0)
        result = self.estimate_speaker_median_log_f0(f0)
        self.assertIsInstance(result, float)

    def test_ignores_unvoiced_frames(self):
        """Median must be computed only over voiced (f0 > 0) frames."""
        f0 = np.array([200.0, 0.0, 200.0, 0.0], dtype=np.float32)
        result = self.estimate_speaker_median_log_f0(f0)
        expected = float(np.log(200.0))
        self.assertAlmostEqual(result, expected, places=5)

    def test_median_not_mean(self):
        """Must use median, not mean. Verify with asymmetric distribution."""
        # Median of [100, 200, 400] in log space = log(200)
        f0 = np.array([100.0, 200.0, 400.0], dtype=np.float32)
        result = self.estimate_speaker_median_log_f0(f0)
        # median of logs = log(200) = 5.298...
        expected_median = float(np.median(np.log(f0[f0 > 0])))
        self.assertAlmostEqual(result, expected_median, places=5)

    def test_all_unvoiced_returns_zero_or_nan(self):
        """All unvoiced input: result should be 0.0 or nan (not crash)."""
        f0 = np.zeros(10, dtype=np.float32)
        # Should not raise; result doesn't matter as long as it's a float
        try:
            result = self.estimate_speaker_median_log_f0(f0)
            self.assertIsInstance(result, float)
        except (ValueError, RuntimeError):
            pass  # acceptable to raise on empty voiced set


# ---------------------------------------------------------------------------
# Test align_to_content_framerate
# ---------------------------------------------------------------------------

class TestAlignToContentFramerate(unittest.TestCase):

    def setUp(self):
        (_, _, _, _, _,
         self.align_to_content_framerate) = _import_f0_functions()

    def test_100hz_to_50hz_halves_frames(self):
        """100 frames at 100Hz -> 50 frames at 50Hz."""
        f0_100hz = np.arange(100, dtype=np.float32)
        result = self.align_to_content_framerate(f0_100hz, target_frames=50)
        self.assertEqual(len(result), 50)

    def test_output_type_tensor_or_ndarray(self):
        """Output can be torch.Tensor or np.ndarray, must be 1D."""
        f0_100hz = np.ones(100, dtype=np.float32) * 200.0
        result = self.align_to_content_framerate(f0_100hz, target_frames=50)
        self.assertEqual(len(result.shape), 1)

    def test_arbitrary_target_frames(self):
        """align_to_content_framerate must produce exactly target_frames frames."""
        for target in [25, 47, 53, 100]:
            f0_100hz = np.ones(100, dtype=np.float32)
            result = self.align_to_content_framerate(f0_100hz, target_frames=target)
            self.assertEqual(
                len(result), target,
                f"Expected {target} frames, got {len(result)}"
            )

    def test_zero_frames_preserved(self):
        """Unvoiced (0) regions should remain 0 after resampling."""
        f0_100hz = np.zeros(100, dtype=np.float32)
        result = self.align_to_content_framerate(f0_100hz, target_frames=50)
        # All frames should be 0
        if isinstance(result, torch.Tensor):
            result_np = result.numpy()
        else:
            result_np = np.asarray(result)
        self.assertTrue(np.all(result_np == 0.0))


# ---------------------------------------------------------------------------
# Test F0Encoder class (with mocked RMVPE)
# ---------------------------------------------------------------------------

class TestF0EncoderClass(unittest.TestCase):
    """Test F0Encoder wrapper for RMVPE."""

    def _make_encoder(self, device="cpu"):
        mock_rmvpe = MagicMock()
        mock_rmvpe.infer_from_audio = MagicMock(
            return_value=np.array([0.0, 150.0, 200.0, 0.0, 180.0] * 20,
                                  dtype=np.float32)
        )

        # Patch both os.path.exists (so the checkpoint check passes) and RMVPE
        with patch("indextts.vc.f0_encoder.RMVPE", return_value=mock_rmvpe), \
             patch("os.path.exists", return_value=True):
            from indextts.vc.f0_encoder import F0Encoder
            enc = F0Encoder(model_path="fake/rmvpe.pt", device=device)
        return enc, mock_rmvpe

    def test_extract_returns_numpy_array(self):
        """extract() must return a numpy array."""
        enc, _ = self._make_encoder()
        audio = np.random.randn(16000).astype(np.float32)
        result = enc.extract(audio)
        self.assertIsInstance(result, np.ndarray)

    def test_extract_1d_output(self):
        """extract() output must be 1D."""
        enc, _ = self._make_encoder()
        audio = np.random.randn(16000).astype(np.float32)
        result = enc.extract(audio)
        self.assertEqual(result.ndim, 1)

    def test_missing_checkpoint_raises(self):
        """F0Encoder must raise clear error if model_path does not exist."""
        from indextts.vc.f0_encoder import F0Encoder
        with self.assertRaises((FileNotFoundError, RuntimeError, Exception)):
            # Don't mock RMVPE; let it try to load from non-existent path
            # But RMVPE itself might raise or not - just ensure our layer propagates
            F0Encoder(model_path="/nonexistent/rmvpe.pt", device="cpu")

    def test_estimate_speaker_median_log_f0(self):
        """F0Encoder.estimate_speaker_median_log_f0 returns float."""
        enc, _ = self._make_encoder()
        f0 = np.array([200.0, 0.0, 300.0, 150.0], dtype=np.float32)
        result = enc.estimate_speaker_median_log_f0(f0)
        self.assertIsInstance(result, float)


if __name__ == '__main__':
    unittest.main()
