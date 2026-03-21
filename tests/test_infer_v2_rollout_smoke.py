import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import indextts.infer_v2 as infer_v2
from indextts.infer_v2 import IndexTTS2


class _FakeTokenizer:
    unk_token_id = -1

    def tokenize(self, text):
        return list(text)

    def split_segments(self, tokens, max_text_tokens_per_segment, quick_streaming_tokens=0):
        return [tokens]

    def convert_tokens_to_ids(self, tokens):
        return list(range(1, len(tokens) + 1))

    def decode(self, ids):
        return "decoded-segment"


class _FakeGPT:
    def merge_emovec(self, *args, **kwargs):
        return torch.zeros(1, 1, 4)

    def inference_speech(self, *args, **kwargs):
        return torch.tensor([[1, 2, 3]], dtype=torch.long), torch.zeros(1, 3, 4)

    def __call__(self, *args, **kwargs):
        return torch.zeros(1, 3, 4)


class _FakeCFM:
    def inference(self, *args, **kwargs):
        return torch.zeros(1, 80, 9)


def _bundle_for(path: str):
    scale = 1.0 if "a" in path.lower() else 2.0
    return {
        "spk_cond_emb": torch.full((1, 3, 4), scale),
        "speech_conditioning_latent": torch.full((1, 3, 4), scale),
        "prompt_condition": torch.full((1, 4, 4), scale),
        "style": torch.full((1, 192), scale),
        "ref_mel": torch.full((1, 80, 4), scale),
        "cond_length": torch.tensor([3]),
        "source_path": path,
    }


def build_stub_tts() -> IndexTTS2:
    tts = IndexTTS2.__new__(IndexTTS2)
    tts.device = "cpu"
    tts.dtype = None
    tts.stop_mel_token = 999
    tts.gr_progress = None
    tts.reference_conditioning_cache = {}
    tts.emotion_conditioning_cache = {}
    tts.last_inference_metadata = {}
    tts.cache_spk_cond = torch.zeros(1, 3, 4)
    tts.cache_s2mel_style = torch.zeros(1, 192)
    tts.cache_s2mel_prompt = torch.zeros(1, 4, 4)
    tts.cache_spk_audio_prompt = "speaker_a.wav"
    tts.cache_emo_cond = None
    tts.cache_emo_audio_prompt = None
    tts.cache_mel = torch.zeros(1, 80, 4)
    tts.tokenizer = _FakeTokenizer()
    tts.gpt = _FakeGPT()
    tts.s2mel = SimpleNamespace(
        models={
            "gpt_layer": lambda latent: latent,
            "length_regulator": lambda *args, **kwargs: [torch.zeros(1, 5, 4)],
            "cfm": _FakeCFM(),
        }
    )
    tts.semantic_codec = SimpleNamespace(
        quantizer=SimpleNamespace(vq2emb=lambda codes: torch.zeros(1, 4, codes.shape[-1]))
    )
    tts.bigvgan = lambda x: torch.zeros(1, 1, 32)
    tts._get_reference_conditioning = lambda path, **kwargs: _bundle_for(path)
    tts._get_emotion_conditioning = lambda path, **kwargs: torch.full((1, 3, 4), 1.0 if "a" in path.lower() else 2.0)
    return tts


def test_rollout_smoke_writes_audio_and_metadata(tmp_path, monkeypatch):
    tts = build_stub_tts()
    output_path = tmp_path / "multiref.wav"
    metadata_path = tmp_path / "multiref.json"

    def fake_save(path, tensor, sample_rate):
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr(infer_v2.torchaudio, "save", fake_save)

    result = tts.infer(
        spk_audio_prompt="speaker_a.wav",
        speaker_references=["speaker_b.wav"],
        speaker_fusion_mode="default",
        emo_audio_prompt="emotion_a.wav",
        emotion_references=["emotion_b.wav"],
        emotion_fusion_mode="default",
        text="hello",
        output_path=str(output_path),
        metadata_output_path=str(metadata_path),
    )

    assert result == str(output_path)
    assert output_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["speaker_prompt"] == "speaker_a.wav"
    assert metadata["speaker_references"] == ["speaker_a.wav", "speaker_b.wav"]
    assert metadata["speaker_fusion_mode"] == "default"
    assert metadata["emotion_prompt"] == "emotion_a.wav"
    assert metadata["emotion_references"] == ["emotion_a.wav", "emotion_b.wav"]
    assert metadata["emotion_fusion_mode"] == "default"
    assert metadata["fusion"]["recipe"]["metadata"]["rollout"]["speaker_fusion_mode"] == "default"
    assert metadata["fusion"]["recipe"]["metadata"]["rollout"]["emotion_fusion_mode"] == "default"
    assert metadata["segments"]
