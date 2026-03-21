import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import indextts.cli as cli


def test_cli_routes_multiref_arguments_to_indextts2(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    voice_path = tmp_path / "voice.wav"
    voice_ref_path = tmp_path / "voice_ref.wav"
    emotion_ref_path = tmp_path / "emotion_ref.wav"
    output_path = tmp_path / "out.wav"
    config_path.write_text("{}", encoding="utf-8")
    model_dir.mkdir()
    voice_path.write_text("voice", encoding="utf-8")
    voice_ref_path.write_text("voice_ref", encoding="utf-8")
    emotion_ref_path.write_text("emotion_ref", encoding="utf-8")

    captured = {}

    class FakeIndexTTS2:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def infer(self, **kwargs):
            captured["infer"] = kwargs
            Path(kwargs["output_path"]).write_text("generated", encoding="utf-8")
            return kwargs["output_path"]

    monkeypatch.setitem(sys.modules, "indextts.infer_v2", types.SimpleNamespace(IndexTTS2=FakeIndexTTS2))

    args = cli.build_parser().parse_args(
        [
            "hello world",
            "--voice",
            str(voice_path),
            "--voice-ref",
            str(voice_ref_path),
            "--emotion-ref",
            str(emotion_ref_path),
            "--config",
            str(config_path),
            "--model_dir",
            str(model_dir),
            "--device",
            "cpu",
            "--output_path",
            str(output_path),
        ]
    )

    result = cli.run_from_args(args)

    assert result == str(output_path)
    assert captured["init"]["device"] == "cpu"
    assert captured["infer"]["spk_audio_prompt"] == str(voice_path)
    assert captured["infer"]["speaker_references"] == [str(voice_ref_path)]
    assert captured["infer"]["emo_audio_prompt"] == str(emotion_ref_path)
    assert captured["infer"]["emotion_references"] == []
    assert captured["infer"]["speaker_fusion_mode"] == "default"
    assert captured["infer"]["emotion_fusion_mode"] == "default"
