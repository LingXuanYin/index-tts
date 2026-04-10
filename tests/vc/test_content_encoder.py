"""
Task 3.1a - Tests for ContentEncoder (TDD - written BEFORE implementation)

ContentEncoder wraps HuBERT-soft (via torch.hub.load) and provides:
  - extract(audio_16k) -> (B, T, 256)
  - output_dim property -> 256
  - model is frozen (.eval() + requires_grad_(False))

Tests use mock to avoid downloading the real model.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _make_mock_hubert(output_dim=256, frames_per_second=50):
    """Create a mock HuBERT-soft model that returns realistic-shaped output."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.requires_grad_.return_value = mock_model

    def fake_units(x):
        # x: (B, T_samples) at 16kHz
        B = x.shape[0]
        T_samples = x.shape[-1]
        T_frames = T_samples // 320  # 16000 / 50 = 320 samples per frame
        return torch.zeros(B, T_frames, output_dim)

    mock_model.units = fake_units
    return mock_model


class TestContentEncoderInterface(unittest.TestCase):
    """Test ContentEncoder public API contract (no real model loading)."""

    def _make_encoder(self, device="cpu"):
        """Helper: create ContentEncoder with mocked torch.hub.load."""
        mock_model = _make_mock_hubert()
        with patch("torch.hub.load", return_value=mock_model):
            from indextts.vc.content_encoder import ContentEncoder
            enc = ContentEncoder(device=device)
        return enc, mock_model

    def test_output_dim_is_256(self):
        """output_dim property must return 256."""
        enc, _ = self._make_encoder()
        self.assertEqual(enc.output_dim, 256)

    def test_extract_returns_correct_shape(self):
        """extract(audio_16k) must return (B, T_frames, 256)."""
        enc, _ = self._make_encoder()
        B, T_samples = 2, 16000  # 1 second at 16kHz
        audio = torch.randn(B, T_samples)
        out = enc.extract(audio)
        self.assertEqual(len(out.shape), 3, "Output must be 3D (B, T, D)")
        self.assertEqual(out.shape[0], B)
        self.assertEqual(out.shape[2], 256, "Feature dim must be 256")

    def test_extract_framerate_approx_50hz(self):
        """extract output frame count should be approx T_samples / 320 (50Hz)."""
        enc, _ = self._make_encoder()
        T_seconds = 2
        T_samples = T_seconds * 16000
        audio = torch.randn(1, T_samples)
        out = enc.extract(audio)
        T_frames = out.shape[1]
        expected_frames = T_samples // 320  # 50Hz
        # Allow ±5 frames tolerance for edge effects
        self.assertAlmostEqual(T_frames, expected_frames, delta=5,
                               msg=f"Expected ~{expected_frames} frames at 50Hz, got {T_frames}")

    def test_model_is_frozen_after_init(self):
        """Model must be in eval mode and gradients frozen after __init__."""
        mock_model = _make_mock_hubert()
        with patch("torch.hub.load", return_value=mock_model):
            from indextts.vc.content_encoder import ContentEncoder
            enc = ContentEncoder(device="cpu")
        mock_model.eval.assert_called()
        mock_model.requires_grad_.assert_called_with(False)

    def test_extract_no_grad(self):
        """extract() must not accumulate gradients (torch.no_grad context)."""
        enc, _ = self._make_encoder()
        audio = torch.randn(1, 16000, requires_grad=False)
        out = enc.extract(audio)
        self.assertFalse(out.requires_grad, "Output of extract() must not require grad")

    def test_hub_load_called_with_correct_args(self):
        """torch.hub.load must be called with the bshall/hubert:main entry point."""
        mock_model = _make_mock_hubert()
        with patch("torch.hub.load", return_value=mock_model) as mock_load:
            from indextts.vc.content_encoder import ContentEncoder
            ContentEncoder(device="cpu")
            call_args = mock_load.call_args
            # First positional: repo
            self.assertIn("bshall/hubert", call_args[0][0])
            # Second positional: model name
            self.assertIn("hubert_soft", call_args[0][1])

    def test_extract_single_sample(self):
        """extract works with B=1."""
        enc, _ = self._make_encoder()
        audio = torch.randn(1, 8000)
        out = enc.extract(audio)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[2], 256)


class TestContentEncoderLazyLoad(unittest.TestCase):
    """Test lazy loading behavior."""

    def test_model_loaded_after_init(self):
        """After __init__ (with mock), model should be loaded (not lazy by default)."""
        mock_model = _make_mock_hubert()
        with patch("torch.hub.load", return_value=mock_model) as mock_load:
            # Reimport to avoid cached module state
            import importlib
            import indextts.vc.content_encoder as ce_mod
            importlib.reload(ce_mod)
            enc = ce_mod.ContentEncoder(device="cpu")
            # hub.load must have been called during __init__
            mock_load.assert_called_once()


if __name__ == '__main__':
    unittest.main()
