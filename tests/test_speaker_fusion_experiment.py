import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.fusion import coerce_fusion_recipe, recipe_cache_token
from tools import speaker_fusion_experiment as sfe


def build_sample_slice(tmp_path: Path) -> Path:
    sample_slice = tmp_path / "ten_batch_slice.jsonl"
    rows = [
        {
            "dataset": "cmu_arctic",
            "speaker_id": "speaker_a0",
            "utterance_id": "utt_a0",
            "text": "hello world",
            "audio_path": str(tmp_path / "audio" / "speaker_a0.wav"),
            "source_split": "a0",
            "batch_id": "batch-00",
        },
        {
            "dataset": "cmu_arctic",
            "speaker_id": "speaker_b0",
            "utterance_id": "utt_b0",
            "text": "unused second transcript",
            "audio_path": str(tmp_path / "audio" / "speaker_b0.wav"),
            "source_split": "b0",
            "batch_id": "batch-00",
        },
        {
            "dataset": "librispeech",
            "speaker_id": "speaker_a1",
            "utterance_id": "utt_a1",
            "text": "another prompt",
            "audio_path": str(tmp_path / "audio" / "speaker_a1.wav"),
            "source_split": "dev-clean",
            "batch_id": "batch-01",
        },
        {
            "dataset": "librispeech",
            "speaker_id": "speaker_b1",
            "utterance_id": "utt_b1",
            "text": "unused paired transcript",
            "audio_path": str(tmp_path / "audio" / "speaker_b1.wav"),
            "source_split": "dev-clean",
            "batch_id": "batch-01",
        },
    ]
    sfe.write_jsonl(sample_slice, rows)
    return sample_slice


def test_default_branch_profiles_are_exhaustive():
    profiles = sfe.default_branch_profiles()
    assert len(profiles) == 13
    names = {profile["name"] for profile in profiles}
    assert "shared" in names
    assert "all_B_only" in names
    assert "ref_mel_B_only" in names
    assert "emotion_B_only" in names


def test_recipe_cache_token_changes_with_fusion_fields():
    recipe_a = coerce_fusion_recipe(
        {
            "references": [
                {"path": "a.wav", "weight": 0.2},
                {"path": "b.wav", "weight": 0.8},
            ],
            "enabled_levels": ["style"],
        }
    )
    recipe_b = coerce_fusion_recipe(
        {
            "references": [
                {"path": "a.wav", "weight": 0.8},
                {"path": "b.wav", "weight": 0.2},
            ],
            "enabled_levels": ["style"],
        }
    )
    token_a = recipe_cache_token(recipe_a, "style", {"field": "style"})
    assert token_a == recipe_cache_token(recipe_a, "style", {"field": "style"})
    assert token_a != recipe_cache_token(recipe_b, "style", {"field": "style"})


def test_manifest_generation_is_reproducible(tmp_path: Path):
    sample_slice = build_sample_slice(tmp_path)
    output_dir = tmp_path / "artifacts"
    branch_profiles = sfe.default_branch_profiles()[:2]

    manifest_path = sfe.build_experiment_manifest(
        sample_slice_path=sample_slice,
        output_dir=output_dir,
        weights=[0.5],
        anchor_modes=["A"],
        exploratory=False,
        branch_profiles=branch_profiles,
        include_baselines=True,
    )
    manifest_first = manifest_path.read_text(encoding="utf-8")

    manifest_path = sfe.build_experiment_manifest(
        sample_slice_path=sample_slice,
        output_dir=output_dir,
        weights=[0.5],
        anchor_modes=["A"],
        exploratory=False,
        branch_profiles=branch_profiles,
        include_baselines=True,
    )
    manifest_second = manifest_path.read_text(encoding="utf-8")

    assert manifest_first == manifest_second
    rows = sfe.read_jsonl(manifest_path)
    assert len(rows) == 2 * (2 + 31 * 2)
    assert any(row["case_kind"] == "baseline_a" for row in rows)
    assert any(row["case_kind"] == "baseline_b" for row in rows)
    assert any(row["case_kind"] == "fusion" for row in rows)


def test_csv_manifest_round_trip_uses_json_literals(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    row = {
        "case_id": "case-01",
        "case_kind": "fusion",
        "scheme_id": "scheme-01",
        "profile": "shared",
        "anchor_mode": "symmetric",
        "role_assignments": {"speaker": "shared", "emotion": "A_only"},
        "batch_id": "batch-00",
        "dataset": "cmu_arctic",
        "text": "hello world",
        "spk_audio_prompt": "a.wav",
        "emo_audio_prompt": "a.wav",
        "speaker_a": {"audio_path": "a.wav"},
        "speaker_b": {"audio_path": "b.wav"},
        "fusion_recipe": {"references": [{"path": "a.wav", "weight": 0.5}]},
        "output_path": "out.wav",
        "metadata_output_path": "meta.json",
        "recipe_name": "style",
        "experimental_name": "default",
        "supported_levels": ["style"],
        "experimental_levels": ["vc_target"],
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(row.keys())
        writer = sfe.csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(sfe.serialize_manifest_row(row))

    loaded = sfe.read_manifest(manifest_path)
    assert loaded == [row]


def test_run_manifest_resumes_completed_cases(tmp_path: Path, monkeypatch):
    skipped_output = tmp_path / "outputs" / "done.wav"
    skipped_metadata = tmp_path / "metadata" / "done.json"
    skipped_output.parent.mkdir(parents=True, exist_ok=True)
    skipped_metadata.parent.mkdir(parents=True, exist_ok=True)
    skipped_output.write_text("done", encoding="utf-8")
    skipped_metadata.write_text("{}", encoding="utf-8")

    pending_output = tmp_path / "outputs" / "pending.wav"
    pending_metadata = tmp_path / "metadata" / "pending.json"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {
            "case_id": "case-skip",
            "scheme_id": "scheme-skip",
            "spk_audio_prompt": "a.wav",
            "emo_audio_prompt": "a.wav",
            "text": "hello",
            "fusion_recipe": {"references": [{"path": "a.wav", "weight": 1.0}], "enabled_levels": []},
            "output_path": str(skipped_output),
            "metadata_output_path": str(skipped_metadata),
        },
        {
            "case_id": "case-run",
            "scheme_id": "scheme-run",
            "spk_audio_prompt": "a.wav",
            "emo_audio_prompt": "a.wav",
            "text": "hello",
            "fusion_recipe": {"references": [{"path": "a.wav", "weight": 1.0}], "enabled_levels": []},
            "output_path": str(pending_output),
            "metadata_output_path": str(pending_metadata),
        },
    ]
    sfe.write_jsonl(manifest_path, manifest_rows)

    class FakeIndexTTS2:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def infer(self, **kwargs):
            Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kwargs["metadata_output_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kwargs["output_path"]).write_text("generated", encoding="utf-8")
            metadata = {
                "timings": {"total_time": 1.0, "rtf": 0.5},
                "segments": [{"index": 0}],
            }
            Path(kwargs["metadata_output_path"]).write_text(json.dumps(metadata), encoding="utf-8")
            return {"output_path": kwargs["output_path"], "metadata": metadata}

    monkeypatch.setattr(sfe, "_require_tts", lambda: FakeIndexTTS2)
    results_path = sfe.run_manifest(manifest_path, cfg_path="cfg.yaml", model_dir="checkpoints")
    results = sfe.read_jsonl(results_path)
    statuses = {row["case_id"]: row["status"] for row in results}

    assert statuses == {"case-skip": "skipped", "case-run": "done"}
    assert pending_output.exists()
    assert pending_metadata.exists()
