"""
indextts/vc/content_encoder.py

ContentEncoder: wraps HuBERT-soft (bshall/hubert:main) to extract
speaker-invariant content features from 16kHz audio.

Interface:
    enc = ContentEncoder(device="cpu")
    features = enc.extract(audio_16k)   # (B, T_samples) -> (B, T_frames, 256)
    dim = enc.output_dim                 # 256

Key facts (from Group 1 research, task 1.1):
- Load via: torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
- Extraction: model.units(audio_16k)  # NOT model.forward()
- Output dim: 256 (soft probability distribution over 256 speech units)
- Frame rate: ~50 Hz (16kHz / CNN stride 320 ≈ 50 frames/sec)
- Model is frozen (eval + requires_grad_(False)); never fine-tuned

Note: real usage requires torch.hub download (~378 MB) from bshall/hubert GitHub.
Tests use unittest.mock.patch("torch.hub.load", ...) to avoid network access.
"""
import torch
import torch.nn as nn


class ContentEncoder:
    """Frozen HuBERT-soft content feature extractor.

    Args:
        model_name: Unused (reserved for future ContentVec support).
                    Actual model is always bshall/hubert-soft via torch.hub.
        device: Target device string, e.g. "cpu", "cuda:0". Defaults to "cpu".
                Never pass bare "cuda" – prefer "cuda:0" to avoid RMVPE-style
                device string issues.
    """

    _DIM = 256  # HuBERT-soft output dimensionality (fixed)

    def __init__(
        self,
        model_name: str = "bshall/hubert-soft",
        device: str = "cpu",
    ):
        self._device = device
        self._model = torch.hub.load(
            "bshall/hubert:main",
            "hubert_soft",
            trust_repo=True,
        )
        self._model.to(device)
        self._model.eval()
        self._model.requires_grad_(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Extract content features from 16kHz audio waveform.

        Args:
            audio_16k: Float tensor of shape (B, T_samples), sampled at 16 kHz.
                       Single-sample tensors of shape (T_samples,) are also accepted
                       and will be unsqueezed to (1, T_samples).

        Returns:
            Tensor of shape (B, T_frames, 256), where T_frames ≈ T_samples / 320
            (approximately 50 Hz frame rate). Values are soft HuBERT probabilities.
        """
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)

        audio_16k = audio_16k.to(self._device)

        with torch.no_grad():
            features = self._model.units(audio_16k)  # (B, T, 256)

        return features

    @property
    def output_dim(self) -> int:
        """Dimensionality of extracted features (always 256 for HuBERT-soft)."""
        return self._DIM

    @property
    def device(self) -> str:
        return self._device
