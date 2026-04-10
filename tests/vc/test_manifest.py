"""
tests/vc/test_manifest.py

TDD tests for indextts/vc_train/manifest.py (task 4.2a).

Tests:
  - JSONL schema validation (required fields present)
  - Speaker-level train/val/test split (no overlap)
  - Relative path resolution
  - manifest generation scanning a synthetic directory
"""
import json
import os
import tempfile
import wave
import struct

import pytest

from indextts.vc_train.manifest import (
    VCManifestEntry,
    validate_entry,
    generate_manifest,
    split_manifest_by_speaker,
    resolve_paths,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fake_wav(path: str, duration_s: float = 1.0, sr: int = 22050) -> None:
    """Create a minimal valid WAV file at path."""
    n_samples = int(sr * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


@pytest.fixture
def fake_dataset_dir(tmp_path):
    """
    Build a fake AISHELL-3-like directory tree:
      <root>/
        SSB0001/
          SSB0001_001.wav
          SSB0001_002.wav
        SSB0002/
          SSB0002_001.wav
        SSB0003/
          SSB0003_001.wav
          SSB0003_002.wav
          SSB0003_003.wav
    """
    speakers = {
        "SSB0001": ["SSB0001_001.wav", "SSB0001_002.wav"],
        "SSB0002": ["SSB0002_001.wav"],
        "SSB0003": ["SSB0003_001.wav", "SSB0003_002.wav", "SSB0003_003.wav"],
    }
    for spk, files in speakers.items():
        spk_dir = tmp_path / spk
        spk_dir.mkdir()
        for fname in files:
            _make_fake_wav(str(spk_dir / fname), duration_s=0.5)
    return tmp_path


# ---------------------------------------------------------------------------
# Schema / entry validation tests
# ---------------------------------------------------------------------------

class TestManifestEntry:
    def test_required_fields_present(self):
        entry = VCManifestEntry(
            audio_path="data/vc/SSB0001/SSB0001_001.wav",
            speaker_id="SSB0001",
            language="zh",
            duration_s=3.5,
            text=None,
        )
        assert entry.audio_path == "data/vc/SSB0001/SSB0001_001.wav"
        assert entry.speaker_id == "SSB0001"
        assert entry.language == "zh"
        assert entry.duration_s == 3.5
        assert entry.text is None

    def test_to_dict_has_required_keys(self):
        entry = VCManifestEntry(
            audio_path="a/b.wav",
            speaker_id="SPK001",
            language="en",
            duration_s=2.0,
            text="hello",
        )
        d = entry.to_dict()
        for key in ("audio_path", "speaker_id", "language", "duration_s"):
            assert key in d, f"Missing key: {key}"

    def test_to_json_roundtrip(self):
        entry = VCManifestEntry(
            audio_path="foo/bar.wav",
            speaker_id="S01",
            language="zh",
            duration_s=1.5,
            text=None,
        )
        line = json.dumps(entry.to_dict())
        parsed = json.loads(line)
        assert parsed["speaker_id"] == "S01"
        assert parsed["duration_s"] == 1.5


class TestValidateEntry:
    def test_valid_entry_passes(self, tmp_path):
        wav_path = str(tmp_path / "a.wav")
        _make_fake_wav(wav_path)
        entry = VCManifestEntry(wav_path, "S1", "zh", 0.5, None)
        # validate_entry checks audio file exists; should not raise
        validate_entry(entry, base_dir=str(tmp_path))

    def test_missing_file_raises(self, tmp_path):
        entry = VCManifestEntry("nonexistent.wav", "S1", "zh", 0.5, None)
        with pytest.raises((FileNotFoundError, ValueError)):
            validate_entry(entry, base_dir=str(tmp_path))

    def test_empty_speaker_id_raises(self, tmp_path):
        wav_path = str(tmp_path / "b.wav")
        _make_fake_wav(wav_path)
        entry = VCManifestEntry(wav_path, "", "zh", 0.5, None)
        with pytest.raises(ValueError, match="speaker_id"):
            validate_entry(entry, base_dir=str(tmp_path))

    def test_invalid_language_raises(self, tmp_path):
        wav_path = str(tmp_path / "c.wav")
        _make_fake_wav(wav_path)
        entry = VCManifestEntry(wav_path, "S1", "xx_invalid_99", 0.5, None)
        with pytest.raises(ValueError, match="language"):
            validate_entry(entry, base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Manifest generation tests
# ---------------------------------------------------------------------------

class TestGenerateManifest:
    def test_generates_entries_for_all_wav_files(self, fake_dataset_dir):
        entries = generate_manifest(
            root_dir=str(fake_dataset_dir),
            language="zh",
        )
        # SSB0001 (2) + SSB0002 (1) + SSB0003 (3) = 6
        assert len(entries) == 6

    def test_all_entries_have_speaker_id(self, fake_dataset_dir):
        entries = generate_manifest(str(fake_dataset_dir), language="zh")
        for e in entries:
            assert e.speaker_id, "speaker_id must not be empty"

    def test_speaker_ids_match_directory_names(self, fake_dataset_dir):
        entries = generate_manifest(str(fake_dataset_dir), language="zh")
        expected_speakers = {"SSB0001", "SSB0002", "SSB0003"}
        found_speakers = {e.speaker_id for e in entries}
        assert found_speakers == expected_speakers

    def test_duration_s_positive(self, fake_dataset_dir):
        entries = generate_manifest(str(fake_dataset_dir), language="zh")
        for e in entries:
            assert e.duration_s > 0, "duration_s must be positive"

    def test_relative_paths_used(self, fake_dataset_dir):
        """audio_path in entries should be relative to root_dir."""
        entries = generate_manifest(
            str(fake_dataset_dir), language="zh", use_relative_paths=True
        )
        for e in entries:
            assert not os.path.isabs(e.audio_path), (
                f"Expected relative path, got: {e.audio_path}"
            )

    def test_jsonl_output_roundtrip(self, fake_dataset_dir, tmp_path):
        """generate_manifest → write JSONL → read back → same entries."""
        entries = generate_manifest(str(fake_dataset_dir), language="zh")
        output_path = str(tmp_path / "manifest.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e.to_dict()) + "\n")
        # Read back
        loaded = []
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                loaded.append(json.loads(line))
        assert len(loaded) == len(entries)
        # Verify fields
        for rec in loaded:
            assert "audio_path" in rec
            assert "speaker_id" in rec
            assert "language" in rec
            assert "duration_s" in rec


# ---------------------------------------------------------------------------
# Speaker-level split tests
# ---------------------------------------------------------------------------

class TestSplitManifest:
    def _make_entries(self):
        """Create 10 speakers, 3 utterances each = 30 entries."""
        entries = []
        for spk_idx in range(10):
            spk = f"SPK{spk_idx:04d}"
            for utt_idx in range(3):
                entries.append(
                    VCManifestEntry(
                        audio_path=f"{spk}/utt{utt_idx}.wav",
                        speaker_id=spk,
                        language="zh",
                        duration_s=2.0,
                        text=None,
                    )
                )
        return entries

    def test_split_no_speaker_overlap(self):
        entries = self._make_entries()
        train, val, test = split_manifest_by_speaker(entries, train=0.8, val=0.1, test=0.1)
        train_spks = {e.speaker_id for e in train}
        val_spks = {e.speaker_id for e in val}
        test_spks = {e.speaker_id for e in test}
        assert train_spks.isdisjoint(val_spks), "train and val speakers must not overlap"
        assert train_spks.isdisjoint(test_spks), "train and test speakers must not overlap"
        assert val_spks.isdisjoint(test_spks), "val and test speakers must not overlap"

    def test_split_covers_all_entries(self):
        entries = self._make_entries()
        train, val, test = split_manifest_by_speaker(entries, train=0.8, val=0.1, test=0.1)
        assert len(train) + len(val) + len(test) == len(entries)

    def test_split_5pct_held_out(self):
        """Default split: 90% train, 5% val, 5% test speakers."""
        entries = self._make_entries()  # 10 speakers
        train, val, test = split_manifest_by_speaker(entries, train=0.90, val=0.05, test=0.05)
        # 10 speakers → at least 1 goes to val, at least 1 to test
        val_spks = {e.speaker_id for e in val}
        test_spks = {e.speaker_id for e in test}
        assert len(val_spks) >= 1
        assert len(test_spks) >= 1

    def test_split_deterministic_with_seed(self):
        entries = self._make_entries()
        train_a, val_a, test_a = split_manifest_by_speaker(entries, train=0.8, val=0.1, test=0.1, seed=42)
        train_b, val_b, test_b = split_manifest_by_speaker(entries, train=0.8, val=0.1, test=0.1, seed=42)
        assert [e.audio_path for e in train_a] == [e.audio_path for e in train_b]
        assert [e.audio_path for e in val_a] == [e.audio_path for e in val_b]


# ---------------------------------------------------------------------------
# Relative path resolution tests
# ---------------------------------------------------------------------------

class TestResolvePaths:
    def test_resolves_relative_to_base(self, tmp_path):
        wav_path = str(tmp_path / "SPK001" / "utt1.wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        _make_fake_wav(wav_path)

        entries = [
            VCManifestEntry(
                audio_path="SPK001/utt1.wav",
                speaker_id="SPK001",
                language="zh",
                duration_s=0.5,
                text=None,
            )
        ]
        resolved = resolve_paths(entries, base_dir=str(tmp_path))
        assert os.path.isabs(resolved[0].audio_path)
        assert os.path.exists(resolved[0].audio_path)

    def test_absolute_paths_unchanged(self, tmp_path):
        wav_path = str(tmp_path / "utt1.wav")
        _make_fake_wav(wav_path)
        entries = [
            VCManifestEntry(
                audio_path=wav_path,
                speaker_id="S1",
                language="zh",
                duration_s=0.5,
                text=None,
            )
        ]
        resolved = resolve_paths(entries, base_dir=str(tmp_path))
        assert resolved[0].audio_path == wav_path
