"""
Task 3.4a - Tests for preprocessing.py (TDD - written BEFORE implementation)

preprocessing.py is an offline preprocessing pipeline that for each utterance:
  1. Extracts content features via HuBERT-soft (16kHz input)
  2. Extracts F0 via RMVPE (16kHz input), decimates to content frame rate
  3. Extracts mel spectrogram at 22050Hz/80-band/hop=256 (s2mel spec, NOT dataset spec)
  4. Extracts speaker style embedding via CAMPPlus (16kHz input)
  5. Saves .content.npy / .f0.npy / .mel.npy / .style.npy

Filtering:
  - voiced_ratio < 0.1 → skip utterance (print warning, do not save)

Tests use mock to avoid downloading real models.
"""
import sys
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_dummy_audio_16k(duration_s: float = 2.0) -> np.ndarray:
    """Create a short random audio array at 16kHz."""
    n = int(duration_s * 16000)
    return np.random.randn(n).astype(np.float32)


def _make_dummy_audio_22k(duration_s: float = 2.0) -> np.ndarray:
    """Create a short random audio array at 22050Hz."""
    n = int(duration_s * 22050)
    return np.random.randn(n).astype(np.float32)


def _make_voiced_f0(n_frames=100, voiced_ratio=0.8, base_hz=200.0) -> np.ndarray:
    """Synthetic 100Hz F0: voiced_ratio of frames at base_hz, rest 0."""
    f0 = np.zeros(n_frames, dtype=np.float32)
    n_voiced = int(n_frames * voiced_ratio)
    f0[:n_voiced] = base_hz
    return f0


# ---------------------------------------------------------------------------
# Test voiced_ratio filtering logic
# ---------------------------------------------------------------------------

class TestVoicedRatioFilter(unittest.TestCase):
    """Test the voiced_ratio < 0.1 filtering logic."""

    def _get_compute_voiced_ratio(self):
        from indextts.vc.preprocessing import compute_voiced_ratio
        return compute_voiced_ratio

    def test_all_voiced_ratio_is_1(self):
        f = self._get_compute_voiced_ratio()
        f0 = np.ones(100, dtype=np.float32) * 200.0
        self.assertAlmostEqual(f(f0), 1.0, places=5)

    def test_all_unvoiced_ratio_is_0(self):
        f = self._get_compute_voiced_ratio()
        f0 = np.zeros(100, dtype=np.float32)
        self.assertAlmostEqual(f(f0), 0.0, places=5)

    def test_half_voiced(self):
        f = self._get_compute_voiced_ratio()
        f0 = np.array([0.0, 200.0] * 50, dtype=np.float32)
        self.assertAlmostEqual(f(f0), 0.5, places=5)

    def test_should_filter_below_threshold(self):
        """voiced_ratio < 0.1 should be flagged for filtering."""
        from indextts.vc.preprocessing import should_filter_utterance
        f0_low = _make_voiced_f0(n_frames=100, voiced_ratio=0.05)
        self.assertTrue(
            should_filter_utterance(f0_low, min_voiced_ratio=0.1),
            "voiced_ratio=0.05 should trigger filtering"
        )

    def test_should_not_filter_above_threshold(self):
        """voiced_ratio >= 0.1 should NOT be filtered."""
        from indextts.vc.preprocessing import should_filter_utterance
        f0_ok = _make_voiced_f0(n_frames=100, voiced_ratio=0.2)
        self.assertFalse(
            should_filter_utterance(f0_ok, min_voiced_ratio=0.1),
            "voiced_ratio=0.2 should NOT be filtered"
        )

    def test_exactly_at_threshold_not_filtered(self):
        """voiced_ratio == 0.1 should NOT be filtered (boundary inclusive)."""
        from indextts.vc.preprocessing import should_filter_utterance
        f0_boundary = _make_voiced_f0(n_frames=100, voiced_ratio=0.1)
        self.assertFalse(
            should_filter_utterance(f0_boundary, min_voiced_ratio=0.1),
            "voiced_ratio==0.1 should NOT be filtered (boundary inclusive)"
        )


# ---------------------------------------------------------------------------
# Test mel spectrogram spec
# ---------------------------------------------------------------------------

class TestMelSpec(unittest.TestCase):
    """Test that mel extraction uses s2mel spec (22050/80/hop=256)."""

    def test_mel_spec_sample_rate_22050(self):
        """Mel must use 22050Hz sample rate (s2mel spec, not dataset 24000Hz)."""
        from indextts.vc.preprocessing import MEL_SAMPLE_RATE
        self.assertEqual(MEL_SAMPLE_RATE, 22050,
                         "Mel sample rate must be 22050 (s2mel spec)")

    def test_mel_spec_n_mels_80(self):
        """Mel must have 80 bands (s2mel spec, not dataset 100)."""
        from indextts.vc.preprocessing import MEL_N_MELS
        self.assertEqual(MEL_N_MELS, 80,
                         "Mel must use 80 bands (s2mel spec, not dataset 100-band)")

    def test_mel_spec_hop_length_256(self):
        """Mel must use hop_length=256 (s2mel spec)."""
        from indextts.vc.preprocessing import MEL_HOP_LENGTH
        self.assertEqual(MEL_HOP_LENGTH, 256)

    def test_mel_output_first_dim_is_80(self):
        """Mel output tensor must have shape[0] == 80, NOT 100."""
        from indextts.vc.preprocessing import compute_mel
        audio_22k = _make_dummy_audio_22k(duration_s=1.0)
        mel = compute_mel(audio_22k)
        # mel should be (80, T_frames)
        self.assertEqual(
            mel.shape[0], 80,
            f"Mel first dim must be 80 (n_mels), got {mel.shape[0]}. "
            "Ensure using s2mel mel spec (22050/80/hop=256), not dataset spec (24000/100)."
        )


# ---------------------------------------------------------------------------
# Test output file structure (offline preprocessing)
# ---------------------------------------------------------------------------

class TestPreprocessingOutputFiles(unittest.TestCase):
    """Test that preprocessing saves correct .npy files for a dummy utterance."""

    def _make_mocks(self):
        """Return mock replacements for all heavy models."""
        mock_content_enc = MagicMock()
        mock_content_enc.extract.return_value = torch.zeros(1, 100, 256)  # (B, T, 256)

        mock_f0_enc = MagicMock()
        mock_f0_enc.extract.return_value = _make_voiced_f0(n_frames=200, voiced_ratio=0.8)

        mock_style_enc = MagicMock()
        mock_style_enc.return_value = torch.zeros(1, 192)  # (B, 192)

        return mock_content_enc, mock_f0_enc, mock_style_enc

    def test_preprocess_utterance_saves_four_files(self):
        """preprocess_utterance must save .content.npy, .f0.npy, .mel.npy, .style.npy."""
        from indextts.vc.preprocessing import preprocess_utterance

        content_enc, f0_enc, style_enc = self._make_mocks()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy audio file
            audio_path = os.path.join(tmpdir, "test_utt.wav")
            # We'll pass the audio directly as numpy (bypass file loading in test)
            audio_16k = _make_dummy_audio_16k(duration_s=1.0)
            audio_22k = _make_dummy_audio_22k(duration_s=1.0)

            output_stem = os.path.join(tmpdir, "test_utt")

            preprocess_utterance(
                audio_16k=audio_16k,
                audio_22k=audio_22k,
                output_stem=output_stem,
                content_encoder=content_enc,
                f0_encoder=f0_enc,
                style_encoder=style_enc,
                min_voiced_ratio=0.1,
            )

            # Check all 4 files exist
            for suffix in [".content.npy", ".f0.npy", ".mel.npy", ".style.npy"]:
                expected = output_stem + suffix
                self.assertTrue(
                    os.path.exists(expected),
                    f"Expected output file {expected} not created"
                )

    def test_filtered_utterance_saves_no_files(self):
        """Utterance with voiced_ratio < 0.1 must not save any files."""
        from indextts.vc.preprocessing import preprocess_utterance

        content_enc, f0_enc, style_enc = self._make_mocks()
        # Override f0 to be mostly unvoiced
        f0_enc.extract.return_value = _make_voiced_f0(n_frames=200, voiced_ratio=0.05)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_16k = _make_dummy_audio_16k(duration_s=1.0)
            audio_22k = _make_dummy_audio_22k(duration_s=1.0)
            output_stem = os.path.join(tmpdir, "filtered_utt")

            result = preprocess_utterance(
                audio_16k=audio_16k,
                audio_22k=audio_22k,
                output_stem=output_stem,
                content_encoder=content_enc,
                f0_encoder=f0_enc,
                style_encoder=style_enc,
                min_voiced_ratio=0.1,
            )

            # Should return False (filtered) and save no files
            self.assertFalse(result, "Should return False for filtered utterance")
            for suffix in [".content.npy", ".f0.npy", ".mel.npy", ".style.npy"]:
                unexpected = output_stem + suffix
                self.assertFalse(
                    os.path.exists(unexpected),
                    f"Filtered utterance must NOT save {unexpected}"
                )

    def test_content_npy_shape(self):
        """Saved content.npy must have shape (T_content, 256)."""
        from indextts.vc.preprocessing import preprocess_utterance

        T_content = 50
        content_enc = MagicMock()
        content_enc.extract.return_value = torch.zeros(1, T_content, 256)

        f0_enc = MagicMock()
        f0_enc.extract.return_value = _make_voiced_f0(n_frames=200, voiced_ratio=0.8)

        style_enc = MagicMock()
        style_enc.return_value = torch.zeros(1, 192)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_16k = _make_dummy_audio_16k(duration_s=1.0)
            audio_22k = _make_dummy_audio_22k(duration_s=1.0)
            output_stem = os.path.join(tmpdir, "test_utt")

            preprocess_utterance(
                audio_16k=audio_16k,
                audio_22k=audio_22k,
                output_stem=output_stem,
                content_encoder=content_enc,
                f0_encoder=f0_enc,
                style_encoder=style_enc,
                min_voiced_ratio=0.1,
            )

            content = np.load(output_stem + ".content.npy")
            self.assertEqual(content.shape[1], 256, "content.npy must have dim=256")
            self.assertEqual(content.shape[0], T_content)

    def test_style_npy_shape(self):
        """Saved style.npy must have shape (192,)."""
        from indextts.vc.preprocessing import preprocess_utterance

        content_enc = MagicMock()
        content_enc.extract.return_value = torch.zeros(1, 50, 256)

        f0_enc = MagicMock()
        f0_enc.extract.return_value = _make_voiced_f0(n_frames=200, voiced_ratio=0.8)

        style_enc = MagicMock()
        style_enc.return_value = torch.zeros(1, 192)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_16k = _make_dummy_audio_16k(duration_s=1.0)
            audio_22k = _make_dummy_audio_22k(duration_s=1.0)
            output_stem = os.path.join(tmpdir, "test_utt")

            preprocess_utterance(
                audio_16k=audio_16k,
                audio_22k=audio_22k,
                output_stem=output_stem,
                content_encoder=content_enc,
                f0_encoder=f0_enc,
                style_encoder=style_enc,
                min_voiced_ratio=0.1,
            )

            style = np.load(output_stem + ".style.npy")
            self.assertEqual(style.shape, (192,), f"style.npy must have shape (192,), got {style.shape}")


# ---------------------------------------------------------------------------
# Test JSONL manifest output
# ---------------------------------------------------------------------------

class TestManifestOutput(unittest.TestCase):
    """Test that JSONL manifest output has correct format."""

    def test_jsonl_output_valid_json(self):
        """Each line of output manifest must be valid JSON."""
        from indextts.vc.preprocessing import write_manifest_line

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                         delete=False, encoding='utf-8') as f:
            tmp_path = f.name

        try:
            write_manifest_line(
                manifest_path=tmp_path,
                audio_path="/data/test.wav",
                speaker_id="SSB0001",
                duration_s=2.5,
                language="zh",
                output_stem="/output/test",
            )

            with open(tmp_path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
            self.assertTrue(len(line) > 0, "Manifest line must not be empty")
            data = json.loads(line)
            self.assertIn("audio_path", data)
            self.assertIn("speaker_id", data)
        finally:
            os.unlink(tmp_path)

    def test_jsonl_output_required_fields(self):
        """Manifest entry must contain: audio_path, speaker_id, duration_s, language."""
        from indextts.vc.preprocessing import write_manifest_line

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                         delete=False, encoding='utf-8') as f:
            tmp_path = f.name

        try:
            write_manifest_line(
                manifest_path=tmp_path,
                audio_path="/data/test.wav",
                speaker_id="SSB0001",
                duration_s=2.5,
                language="zh",
                output_stem="/output/test",
            )
            with open(tmp_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.readline())

            required = ["audio_path", "speaker_id", "duration_s", "language"]
            for field in required:
                self.assertIn(field, data, f"Manifest must contain field: {field!r}")
        finally:
            os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()
