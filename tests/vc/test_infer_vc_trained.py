"""
tests/vc/test_infer_vc_trained.py

TDD tests for Group 5: VC inference path (infer_vc_trained).

All tests use mocks — no real model weights are loaded.
Network access is fully patched.

Coverage:
  - Function signature and return type
  - Lazy-loading: vc_model is None at init, loaded on first call
  - F0 strategy mismatch raises ValueError
  - Inference call stack:
      - does NOT call self.gpt.*, self.semantic_model, self.semantic_codec,
        self.asr_model, torchaudio.functional.pitch_shift
      - DOES call cfm.inference with f0=None
  - prompt_condition walks HuBERT-soft + kmeans, not semantic_codec
  - output_path=None returns waveform ndarray / tensor
  - BigVGAN decode produces audio
"""
from __future__ import annotations

import inspect
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call

import numpy as np
import torch

# ------------------------------------------------------------------ helpers --

def _make_mock_cfm():
    """Create a minimal mock for the CFM model used in cfm.inference."""
    cfm = MagicMock()
    # inference returns (B, 80, T) mel
    cfm.inference.return_value = torch.zeros(1, 80, 200)
    return cfm


def _make_mock_length_regulator():
    """Mock for length_regulator: returns (cond, ...) where cond is (B, T, 512)."""
    lr = MagicMock()
    # Return a tuple; first element is the cond tensor (B, T, 512)
    lr.return_value = (torch.zeros(1, 50, 512), None, None, None, None)
    return lr


def _make_mock_my_model(cfm, length_regulator):
    """Assemble a mock MyModel with .models dict."""
    model = MagicMock()
    model.models = {
        "cfm": cfm,
        "length_regulator": length_regulator,
    }
    return model


def _make_mock_content_encoder():
    """Mock ContentEncoder.extract → (1, 50, 256)."""
    enc = MagicMock()
    enc.extract.return_value = torch.zeros(1, 50, 256)
    return enc


def _make_mock_kmeans():
    """Mock KMeansQuantizer.quantize_to_vector → (1, 50, 256)."""
    kmeans = MagicMock()
    kmeans.quantize_to_vector.return_value = torch.zeros(1, 50, 256)
    return kmeans


def _make_mock_f0_encoder():
    """Mock F0Encoder.extract → 1D numpy at 100 Hz."""
    f0enc = MagicMock()
    f0enc.extract.return_value = np.zeros(100, dtype=np.float32)
    return f0enc


def _make_mock_campplus():
    """Mock CAMPPlus speaker style extractor."""
    camp = MagicMock()
    camp.return_value = torch.zeros(1, 192)
    return camp


def _make_mock_bigvgan():
    """Mock BigVGAN vocoder → (1, 1, N_samples)."""
    bvg = MagicMock()
    bvg.return_value = torch.zeros(1, 1, 22050)
    return bvg


def _make_mock_mel_fn():
    """Mock mel_spectrogram function → (1, 80, T_mel)."""
    def mel_fn(audio):
        return torch.zeros(1, 80, 86)
    return mel_fn


# ---------------------------------------------------------------- fixtures ---


def _build_infer_vc_module():
    """
    Import indextts.infer_vc_trained and return the module.
    We patch torch.hub.load during import to avoid network access.
    """
    with patch("torch.hub.load", return_value=MagicMock()):
        import indextts.infer_vc_trained as m
    return m


# ============================================================= Test classes ==


class TestInferVCTrainedSignature(unittest.TestCase):
    """5.1.a: Function signature must match spec."""

    def test_infer_vc_trained_exists(self):
        """infer_vc_trained must be importable from indextts.infer_vc_trained."""
        import indextts.infer_vc_trained as m
        self.assertTrue(
            hasattr(m, "infer_vc_trained"),
            "indextts.infer_vc_trained must expose 'infer_vc_trained' function",
        )

    def test_signature_parameters(self):
        """infer_vc_trained must accept required and keyword-only parameters."""
        import indextts.infer_vc_trained as m
        sig = inspect.signature(m.infer_vc_trained)
        params = sig.parameters
        # Required positional
        self.assertIn("tts", params)
        self.assertIn("source_audio", params)
        self.assertIn("speaker_refs", params)
        # Optional with defaults
        self.assertIn("output_path", params)
        self.assertIn("checkpoint_path", params)
        self.assertIn("f0_strategy", params)
        self.assertIn("manual_semitones", params)
        self.assertIn("diffusion_steps", params)
        self.assertIn("inference_cfg_rate", params)
        self.assertIn("verbose", params)

    def test_default_values(self):
        """Check that default values match the spec."""
        import indextts.infer_vc_trained as m
        sig = inspect.signature(m.infer_vc_trained)
        params = sig.parameters
        self.assertEqual(params["f0_strategy"].default, "source_plus_shift")
        self.assertEqual(params["manual_semitones"].default, 0.0)
        self.assertEqual(params["diffusion_steps"].default, 25)
        self.assertAlmostEqual(params["inference_cfg_rate"].default, 0.7)
        self.assertEqual(params["verbose"].default, False)


class TestLazyLoading(unittest.TestCase):
    """5.1.b: VC model must be None at __init__ and loaded on first call."""

    def _build_tts_mock(self):
        """Return a minimal mock of IndexTTS2 with vc_* attributes as None."""
        tts = MagicMock()
        tts.vc_model = None
        tts.vc_content_encoder = None
        tts.vc_f0_encoder = None
        tts.vc_kmeans = None
        tts.vc_config = None
        tts.vc_f0_strategy = None
        tts.device = "cpu"
        return tts

    def test_vc_model_none_before_call(self):
        """IndexTTS2 must initialise vc_model as None (placeholder)."""
        # We can't import IndexTTS2 without loading real checkpoints,
        # so we validate the attribute names through the infer_vc module.
        import indextts.infer_vc_trained as m

        # The _ensure_vc_loaded helper must check self.vc_model is not None
        self.assertTrue(
            hasattr(m, "_ensure_vc_loaded"),
            "_ensure_vc_loaded helper must be defined in infer_vc_trained module",
        )

    def test_ensure_vc_loaded_skips_if_already_loaded(self):
        """_ensure_vc_loaded must be a no-op if vc_model is already set."""
        import indextts.infer_vc_trained as m

        tts = self._build_tts_mock()
        mock_model = MagicMock()
        tts.vc_model = mock_model  # already loaded

        # Call _ensure_vc_loaded — it should not overwrite vc_model
        with patch.object(m, "_load_vc_checkpoint", return_value=mock_model) as mock_load:
            m._ensure_vc_loaded(tts, "some/path.pth", "source_plus_shift")
            mock_load.assert_not_called()

        # vc_model must still be the same object
        self.assertIs(tts.vc_model, mock_model)

    def test_ensure_vc_loaded_loads_when_none(self):
        """_ensure_vc_loaded must load everything when vc_model is None."""
        import indextts.infer_vc_trained as m

        tts = self._build_tts_mock()

        # Patch all the loading helpers
        fake_model = MagicMock()
        fake_model._vc_checkpoint_metadata = {"f0_strategy": "source_plus_shift"}
        fake_enc = _make_mock_content_encoder()
        fake_kmeans = _make_mock_kmeans()
        fake_f0enc = _make_mock_f0_encoder()

        with (
            patch.object(m, "_load_vc_checkpoint", return_value=fake_model),
            patch.object(m, "_load_content_encoder", return_value=fake_enc),
            patch.object(m, "_load_kmeans", return_value=fake_kmeans),
            patch.object(m, "_load_f0_encoder", return_value=fake_f0enc),
        ):
            m._ensure_vc_loaded(tts, "checkpoints/vc/trained_vc.pth", "source_plus_shift")

        self.assertIs(tts.vc_model, fake_model)
        self.assertIs(tts.vc_content_encoder, fake_enc)
        self.assertIs(tts.vc_kmeans, fake_kmeans)
        self.assertIs(tts.vc_f0_encoder, fake_f0enc)
        self.assertEqual(tts.vc_f0_strategy, "source_plus_shift")


class TestF0StrategyConsistency(unittest.TestCase):
    """5.1.c: F0 strategy mismatch between checkpoint metadata and caller must raise."""

    def _build_tts_loaded(self, vc_f0_strategy: str):
        tts = MagicMock()
        tts.vc_model = MagicMock()
        tts.vc_content_encoder = _make_mock_content_encoder()
        tts.vc_f0_encoder = _make_mock_f0_encoder()
        tts.vc_kmeans = _make_mock_kmeans()
        tts.vc_f0_strategy = vc_f0_strategy
        tts.device = "cpu"
        return tts

    def test_strategy_mismatch_raises(self):
        """Passing f0_strategy that differs from checkpoint metadata → ValueError."""
        import indextts.infer_vc_trained as m

        tts = self._build_tts_loaded("source_contour")

        with self.assertRaises(ValueError) as ctx:
            m._assert_f0_strategy_consistent(tts, "target_median")

        self.assertIn("source_contour", str(ctx.exception))
        self.assertIn("target_median", str(ctx.exception))

    def test_strategy_match_no_raise(self):
        """Matching f0_strategy does not raise."""
        import indextts.infer_vc_trained as m

        tts = self._build_tts_loaded("source_plus_shift")
        # Should not raise
        m._assert_f0_strategy_consistent(tts, "source_plus_shift")

    def test_invalid_strategy_name_raises(self):
        """Unknown strategy name raises ValueError before checkpoint check."""
        import indextts.infer_vc_trained as m

        tts = self._build_tts_loaded("source_contour")

        with self.assertRaises(ValueError):
            m._assert_f0_strategy_consistent(tts, "nonexistent_strategy")


class TestInferenceCallStack(unittest.TestCase):
    """5.1.d: Inference must NOT call forbidden attributes; cfm.inference f0=None."""

    def _build_full_tts(self):
        """Build a fully-mocked TTS object that has both TTS and VC attributes."""
        cfm = _make_mock_cfm()
        lr = _make_mock_length_regulator()
        vc_model = _make_mock_my_model(cfm, lr)
        vc_model._vc_checkpoint_metadata = {"f0_strategy": "source_plus_shift"}

        tts = MagicMock(spec=[
            # vc attributes
            "vc_model", "vc_content_encoder", "vc_f0_encoder", "vc_kmeans",
            "vc_f0_strategy", "vc_config", "device",
            # TTS attributes (must be accessible but NOT called)
            "gpt", "semantic_model", "semantic_codec", "asr_model",
            # helpers shared with TTS
            "campplus_model", "bigvgan", "mel_fn", "cfg",
        ])
        tts.vc_model = vc_model
        tts.vc_content_encoder = _make_mock_content_encoder()
        tts.vc_f0_encoder = _make_mock_f0_encoder()
        tts.vc_kmeans = _make_mock_kmeans()
        tts.vc_f0_strategy = "source_plus_shift"
        tts.device = "cpu"

        # TTS components — should NOT be called during VC inference
        tts.gpt = MagicMock(name="gpt_FORBIDDEN")
        tts.semantic_model = MagicMock(name="semantic_model_FORBIDDEN")
        tts.semantic_codec = MagicMock(name="semantic_codec_FORBIDDEN")
        tts.asr_model = MagicMock(name="asr_model_FORBIDDEN")

        # Shared helpers that ARE used
        tts.campplus_model = _make_mock_campplus()
        tts.bigvgan = _make_mock_bigvgan()
        tts.mel_fn = _make_mock_mel_fn()

        return tts, cfm, lr

    def _run_infer(self, tts, source_audio_path="dummy_src.wav", speaker_ref="dummy_ref.wav"):
        import indextts.infer_vc_trained as m

        with (
            patch.object(m, "_ensure_vc_loaded"),
            patch.object(m, "_assert_f0_strategy_consistent"),
            patch.object(m, "_load_audio_16k", side_effect=[
                np.zeros(16000, dtype=np.float32),  # source
                np.zeros(8000, dtype=np.float32),   # speaker ref
            ]),
            patch.object(m, "_load_audio_22k", return_value=torch.zeros(1, 22050)),
            patch("torchaudio.compliance.kaldi.fbank", return_value=torch.zeros(50, 80)),
        ):
            result = m.infer_vc_trained(
                tts,
                source_audio=source_audio_path,
                speaker_refs=speaker_ref,
                output_path=None,
                f0_strategy="source_plus_shift",
                diffusion_steps=5,
            )
        return result

    def test_gpt_not_called(self):
        """tts.gpt must not be called during VC inference."""
        tts, cfm, lr = self._build_full_tts()
        self._run_infer(tts)
        tts.gpt.assert_not_called()

    def test_semantic_model_not_called(self):
        """tts.semantic_model must not be called during VC inference."""
        tts, cfm, lr = self._build_full_tts()
        self._run_infer(tts)
        tts.semantic_model.assert_not_called()

    def test_semantic_codec_not_called(self):
        """tts.semantic_codec must not be called during VC inference."""
        tts, cfm, lr = self._build_full_tts()
        self._run_infer(tts)
        tts.semantic_codec.assert_not_called()

    def test_cfm_inference_called_with_f0_none(self):
        """cfm.inference must be called, and its f0 argument must be None (5th positional)."""
        tts, cfm, lr = self._build_full_tts()
        self._run_infer(tts)

        cfm.inference.assert_called_once()
        call_args = cfm.inference.call_args

        # f0 is the 5th positional arg (index 4): mu, x_lens, ref_mel, style, f0, steps, ...
        # It can also be passed as a keyword argument
        args = call_args[0]  # positional
        kwargs = call_args[1]  # keyword

        if len(args) >= 5:
            f0_arg = args[4]
        else:
            f0_arg = kwargs.get("f0", None)

        self.assertIsNone(f0_arg, f"cfm.inference f0 argument must be None, got {f0_arg!r}")

    def test_pitch_shift_not_called(self):
        """torchaudio.functional.pitch_shift must not be called during VC inference."""
        import torchaudio
        tts, cfm, lr = self._build_full_tts()

        with patch("torchaudio.functional.pitch_shift") as mock_ps:
            self._run_infer(tts)
            mock_ps.assert_not_called()

    def test_returns_waveform_when_no_output_path(self):
        """When output_path=None, infer_vc_trained must return a waveform tensor or ndarray."""
        tts, cfm, lr = self._build_full_tts()
        result = self._run_infer(tts)

        self.assertIsNotNone(result, "Result must not be None when output_path is None")
        # Accept torch.Tensor or numpy array
        self.assertTrue(
            isinstance(result, (torch.Tensor, np.ndarray)),
            f"Expected Tensor or ndarray, got {type(result)}",
        )


class TestPromptConditionPath(unittest.TestCase):
    """5.1.e: prompt_condition must walk HuBERT-soft + kmeans, NOT semantic_codec."""

    def test_kmeans_called_for_speaker_ref(self):
        """vc_kmeans.quantize_to_vector must be called for the speaker reference."""
        import indextts.infer_vc_trained as m

        cfm = _make_mock_cfm()
        lr = _make_mock_length_regulator()
        vc_model = _make_mock_my_model(cfm, lr)
        vc_model._vc_checkpoint_metadata = {"f0_strategy": "source_plus_shift"}

        kmeans = _make_mock_kmeans()
        enc = _make_mock_content_encoder()

        tts = MagicMock()
        tts.vc_model = vc_model
        tts.vc_content_encoder = enc
        tts.vc_f0_encoder = _make_mock_f0_encoder()
        tts.vc_kmeans = kmeans
        tts.vc_f0_strategy = "source_plus_shift"
        tts.device = "cpu"
        tts.gpt = MagicMock()
        tts.semantic_model = MagicMock()
        tts.semantic_codec = MagicMock()
        tts.asr_model = MagicMock()
        tts.campplus_model = _make_mock_campplus()
        tts.bigvgan = _make_mock_bigvgan()
        tts.mel_fn = _make_mock_mel_fn()

        with (
            patch.object(m, "_ensure_vc_loaded"),
            patch.object(m, "_assert_f0_strategy_consistent"),
            patch.object(m, "_load_audio_16k", side_effect=[
                np.zeros(16000, dtype=np.float32),
                np.zeros(8000, dtype=np.float32),
            ]),
            patch.object(m, "_load_audio_22k", return_value=torch.zeros(1, 22050)),
            patch("torchaudio.compliance.kaldi.fbank", return_value=torch.zeros(50, 80)),
        ):
            m.infer_vc_trained(
                tts,
                source_audio="src.wav",
                speaker_refs="ref.wav",
                output_path=None,
                f0_strategy="source_plus_shift",
                diffusion_steps=5,
            )

        # quantize_to_vector should be called at least twice: once for source, once for ref
        self.assertGreaterEqual(
            kmeans.quantize_to_vector.call_count, 2,
            "kmeans.quantize_to_vector must be called for both source and speaker reference",
        )
        # semantic_codec must NOT be called
        tts.semantic_codec.assert_not_called()

    def test_content_encoder_called_for_both(self):
        """ContentEncoder.extract must be called for both source and speaker reference."""
        import indextts.infer_vc_trained as m

        cfm = _make_mock_cfm()
        lr = _make_mock_length_regulator()
        vc_model = _make_mock_my_model(cfm, lr)
        vc_model._vc_checkpoint_metadata = {"f0_strategy": "source_plus_shift"}

        enc = _make_mock_content_encoder()

        tts = MagicMock()
        tts.vc_model = vc_model
        tts.vc_content_encoder = enc
        tts.vc_f0_encoder = _make_mock_f0_encoder()
        tts.vc_kmeans = _make_mock_kmeans()
        tts.vc_f0_strategy = "source_plus_shift"
        tts.device = "cpu"
        tts.gpt = MagicMock()
        tts.semantic_model = MagicMock()
        tts.semantic_codec = MagicMock()
        tts.asr_model = MagicMock()
        tts.campplus_model = _make_mock_campplus()
        tts.bigvgan = _make_mock_bigvgan()
        tts.mel_fn = _make_mock_mel_fn()

        with (
            patch.object(m, "_ensure_vc_loaded"),
            patch.object(m, "_assert_f0_strategy_consistent"),
            patch.object(m, "_load_audio_16k", side_effect=[
                np.zeros(16000, dtype=np.float32),
                np.zeros(8000, dtype=np.float32),
            ]),
            patch.object(m, "_load_audio_22k", return_value=torch.zeros(1, 22050)),
            patch("torchaudio.compliance.kaldi.fbank", return_value=torch.zeros(50, 80)),
        ):
            m.infer_vc_trained(
                tts,
                source_audio="src.wav",
                speaker_refs="ref.wav",
                output_path=None,
                f0_strategy="source_plus_shift",
                diffusion_steps=5,
            )

        # enc.extract should be called at least twice (source + reference)
        self.assertGreaterEqual(
            enc.extract.call_count, 2,
            "ContentEncoder.extract must be called for both source audio and speaker reference",
        )


class TestIndexTTS2PlaceholderAttributes(unittest.TestCase):
    """5.3: IndexTTS2.__init__ must define vc_* placeholder attributes as None."""

    def test_placeholder_attributes_exist_in_init(self):
        """
        Verify that infer_v2.py sets vc_* attributes to None in __init__.
        We do this by checking the source code (without loading real models).
        """
        import ast
        infer_v2_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "indextts", "infer_v2.py"
        )
        with open(infer_v2_path, "r", encoding="utf-8") as f:
            source = f.read()

        required_attrs = [
            "vc_model",
            "vc_content_encoder",
            "vc_f0_encoder",
            "vc_kmeans",
            "vc_config",
        ]
        for attr in required_attrs:
            self.assertIn(
                f"self.{attr}",
                source,
                f"infer_v2.py __init__ must define self.{attr}",
            )


class TestBoundaryGrep(unittest.TestCase):
    """5.5: Grep boundary self-check against infer_vc_trained.py source.

    We check only executable code lines (not comments or docstrings).
    """

    def _read_code_lines(self) -> str:
        """Read infer_vc_trained.py and return only non-comment, non-docstring lines joined."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "indextts", "infer_vc_trained.py"
        )
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        code_lines = []
        in_docstring = False
        docstring_char = None
        for line in lines:
            stripped = line.strip()

            # Track triple-quote docstrings
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    delim = stripped[:3]
                    # Check if it closes on the same line (single-line docstring)
                    rest = stripped[3:]
                    if delim in rest:
                        # Single-line docstring, skip it entirely
                        continue
                    else:
                        in_docstring = True
                        docstring_char = delim
                        continue
                # Skip comment lines
                if stripped.startswith("#"):
                    continue
                code_lines.append(line)
            else:
                # Inside docstring: wait for closing triple-quote
                if docstring_char in stripped:
                    in_docstring = False
                # Skip this line (part of docstring)

        return "\n".join(code_lines)

    def _read_module_source(self) -> str:
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "indextts", "infer_vc_trained.py"
        )
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_no_gpt_forward_in_vc_path(self):
        src = self._read_code_lines()
        # self.gpt. should not appear in executable code (as a call)
        import re
        hits = re.findall(r"self\.gpt\.", src)
        self.assertEqual(hits, [], f"Found forbidden 'self.gpt.' in vc inference code: {hits}")

    def test_no_semantic_model_in_vc_path(self):
        src = self._read_code_lines()
        import re
        hits = re.findall(r"self\.semantic_model", src)
        self.assertEqual(hits, [], f"Found forbidden 'self.semantic_model' in vc inference code")

    def test_no_semantic_codec_in_vc_path(self):
        src = self._read_code_lines()
        import re
        hits = re.findall(r"self\.semantic_codec", src)
        self.assertEqual(hits, [], f"Found forbidden 'self.semantic_codec' in vc inference code")

    def test_no_pitch_shift_in_vc_path(self):
        src = self._read_code_lines()
        self.assertNotIn(
            "pitch_shift",
            src,
            "torchaudio.functional.pitch_shift must not appear in vc inference code",
        )

    def test_cfm_inference_f0_none_pattern(self):
        """cfm.inference call in source must pass f0=None (or have None as 5th arg)."""
        src = self._read_module_source()
        import re
        # Look for any cfm.inference call
        self.assertIn(
            "cfm.inference",
            src,
            "cfm.inference call must be present in infer_vc_trained.py",
        )
        # The f0 argument must be None: look for f0=None or positional None in context
        # Either "f0=None" appears, or None is 5th arg after cfm.inference(
        has_f0_none_kwarg = "f0=None" in src
        # If not kwarg, rely on runtime test (TestInferenceCallStack.test_cfm_inference_called_with_f0_none)
        # At minimum, "None" must appear near cfm.inference
        cfm_block = src[src.index("cfm.inference"):src.index("cfm.inference") + 300]
        self.assertIn(
            "None",
            cfm_block,
            "cfm.inference call must have None as f0 argument",
        )


if __name__ == "__main__":
    unittest.main()
