"""
indextts/vc_train/manifest.py

JSONL manifest generation and management for VC training data.

Key features:
  - Scans dataset directories (AISHELL-3, LibriTTS layout) to produce JSONL manifest
  - Speaker-level train/val/test split (no utterance-level leakage)
  - Relative and absolute path support
  - Entry validation (file existence, required fields)

JSONL entry schema (per line):
  {
    "audio_path": "SSB0001/SSB0001_001.wav",   # relative to manifest base_dir
    "speaker_id": "SSB0001",
    "language": "zh",
    "duration_s": 3.14,
    "text": null                               # optional transcript
  }

CLI usage:
  python -m indextts.vc_train.manifest \\
    --root_dir data/raw/aishell3/wav \\
    --language zh \\
    --output data/vc/aishell3_manifest.jsonl \\
    --split 0.90 0.05 0.05
"""
from __future__ import annotations

import json
import math
import os
import random
import wave
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Entry dataclass
# ---------------------------------------------------------------------------

VALID_LANGUAGES = {"zh", "en", "ja", "ko", "fr", "de", "es", "pt", "ru"}


@dataclass
class VCManifestEntry:
    """A single row in the VC training manifest.

    Args:
        audio_path: Path to WAV file (relative to manifest base_dir, or absolute).
        speaker_id: Speaker identifier string. Must not be empty.
        language: BCP-47 language code (e.g. "zh", "en").
        duration_s: Audio duration in seconds (positive float).
        text: Optional transcript string. None if not available.
    """

    audio_path: str
    speaker_id: str
    language: str
    duration_s: float
    text: Optional[str]

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "speaker_id": self.speaker_id,
            "language": self.language,
            "duration_s": self.duration_s,
            "text": self.text,
        }

    @staticmethod
    def from_dict(d: dict) -> "VCManifestEntry":
        return VCManifestEntry(
            audio_path=d["audio_path"],
            speaker_id=d["speaker_id"],
            language=d["language"],
            duration_s=float(d["duration_s"]),
            text=d.get("text"),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_entry(entry: VCManifestEntry, base_dir: Optional[str] = None) -> None:
    """Validate a manifest entry.

    Args:
        entry: The manifest entry to validate.
        base_dir: If provided, resolve relative audio_path against this directory.

    Raises:
        ValueError: If speaker_id is empty or language is not recognised.
        FileNotFoundError: If the audio file does not exist.
    """
    if not entry.speaker_id:
        raise ValueError(f"speaker_id must not be empty; got: {entry.speaker_id!r}")

    if entry.language not in VALID_LANGUAGES:
        raise ValueError(
            f"language {entry.language!r} is not in the recognised set {VALID_LANGUAGES}. "
            "Add it to VALID_LANGUAGES in manifest.py if needed."
        )

    # Resolve path
    path = entry.audio_path
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Audio file not found: {path!r}. "
            "Check that audio_path is correct and base_dir is set properly."
        )


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def _get_wav_duration(path: str) -> float:
    """Return WAV file duration in seconds using Python standard library."""
    with wave.open(path, "r") as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        return n_frames / float(framerate)


def _get_speaker_from_path(audio_path: str, root_dir: str) -> str:
    """Derive speaker_id from directory structure relative to root_dir.

    Expects: root_dir/<speaker_id>/<utterance>.wav  (AISHELL-3 / LibriTTS layout)
    The speaker_id is the first path component under root_dir.

    Args:
        audio_path: Absolute path to the audio file.
        root_dir: Root directory of the dataset.

    Returns:
        Speaker ID string.
    """
    rel = os.path.relpath(audio_path, root_dir)
    parts = rel.replace("\\", "/").split("/")
    return parts[0]


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------


def generate_manifest(
    root_dir: str,
    language: str,
    use_relative_paths: bool = True,
    min_duration_s: float = 0.1,
    max_duration_s: float = 30.0,
) -> List[VCManifestEntry]:
    """Scan a dataset directory and generate a list of VCManifestEntry objects.

    Directory layout expected:
      root_dir/
        <speaker_id>/
          <utterance>.wav
          ...
        ...

    Args:
        root_dir: Root directory to scan. Each immediate subdirectory is treated
                  as a speaker.
        language: Language code to assign to all entries (e.g. "zh").
        use_relative_paths: If True, audio_path is relative to root_dir.
                            If False, audio_path is absolute.
        min_duration_s: Skip files shorter than this (default 0.1 s).
        max_duration_s: Skip files longer than this (default 30.0 s).

    Returns:
        Sorted list of VCManifestEntry (sorted by audio_path for determinism).
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"root_dir does not exist or is not a directory: {root_dir!r}")

    if language not in VALID_LANGUAGES:
        raise ValueError(
            f"language {language!r} is not in the recognised set {VALID_LANGUAGES}."
        )

    entries: List[VCManifestEntry] = []

    for speaker_id in sorted(os.listdir(root_dir)):
        speaker_dir = os.path.join(root_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue  # skip non-directory files at top level

        for fname in sorted(os.listdir(speaker_dir)):
            if not fname.lower().endswith(".wav"):
                continue

            abs_path = os.path.join(speaker_dir, fname)
            try:
                duration_s = _get_wav_duration(abs_path)
            except Exception:
                # Skip unreadable files
                continue

            if duration_s < min_duration_s or duration_s > max_duration_s:
                continue

            if use_relative_paths:
                audio_path = os.path.relpath(abs_path, root_dir).replace("\\", "/")
            else:
                audio_path = abs_path

            entries.append(
                VCManifestEntry(
                    audio_path=audio_path,
                    speaker_id=speaker_id,
                    language=language,
                    duration_s=duration_s,
                    text=None,
                )
            )

    return entries


# ---------------------------------------------------------------------------
# Speaker-level split
# ---------------------------------------------------------------------------


def split_manifest_by_speaker(
    entries: List[VCManifestEntry],
    train: float = 0.90,
    val: float = 0.05,
    test: float = 0.05,
    seed: int = 42,
) -> Tuple[List[VCManifestEntry], List[VCManifestEntry], List[VCManifestEntry]]:
    """Split entries into train / val / test by speaker ID (no leakage).

    All utterances of a speaker end up in the same split.

    Args:
        entries: Full list of manifest entries.
        train: Fraction of speakers assigned to training set.
        val: Fraction of speakers assigned to validation set.
        test: Fraction of speakers assigned to test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_entries, val_entries, test_entries).

    Note:
        Due to integer rounding, at least 1 speaker is assigned to each non-zero
        fraction split, even if the computed count rounds to 0.
    """
    assert abs(train + val + test - 1.0) < 1e-6 or (train + val + test) <= 1.0, (
        "train + val + test must be <= 1.0"
    )

    # Collect unique speakers in sorted order (deterministic)
    all_speakers = sorted({e.speaker_id for e in entries})
    n_speakers = len(all_speakers)

    rng = random.Random(seed)
    speakers_shuffled = list(all_speakers)
    rng.shuffle(speakers_shuffled)

    # Compute integer speaker counts
    n_test = max(1, math.floor(n_speakers * test)) if test > 0 else 0
    n_val = max(1, math.floor(n_speakers * val)) if val > 0 else 0
    n_train = n_speakers - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Not enough speakers ({n_speakers}) to satisfy the requested split ratios. "
            "Reduce val/test fractions."
        )

    train_spks = set(speakers_shuffled[:n_train])
    val_spks = set(speakers_shuffled[n_train : n_train + n_val])
    test_spks = set(speakers_shuffled[n_train + n_val :])

    train_entries = [e for e in entries if e.speaker_id in train_spks]
    val_entries = [e for e in entries if e.speaker_id in val_spks]
    test_entries = [e for e in entries if e.speaker_id in test_spks]

    return train_entries, val_entries, test_entries


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_paths(
    entries: List[VCManifestEntry],
    base_dir: str,
) -> List[VCManifestEntry]:
    """Return new entries with audio_path resolved to absolute paths.

    If audio_path is already absolute, it is returned unchanged.
    If audio_path is relative, it is joined with base_dir.

    Args:
        entries: List of manifest entries with possibly relative paths.
        base_dir: Base directory to resolve relative paths against.

    Returns:
        New list of VCManifestEntry objects with absolute audio_path.
    """
    resolved = []
    for e in entries:
        if os.path.isabs(e.audio_path):
            resolved.append(e)
        else:
            abs_path = os.path.abspath(os.path.join(base_dir, e.audio_path))
            resolved.append(
                VCManifestEntry(
                    audio_path=abs_path,
                    speaker_id=e.speaker_id,
                    language=e.language,
                    duration_s=e.duration_s,
                    text=e.text,
                )
            )
    return resolved


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_jsonl(entries: List[VCManifestEntry], output_path: str) -> None:
    """Write entries to a JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(manifest_path: str) -> List[VCManifestEntry]:
    """Read entries from a JSONL manifest file."""
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(VCManifestEntry.from_dict(json.loads(line)))
    return entries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate VC training manifest from a dataset directory."
    )
    parser.add_argument("--root_dir", required=True, help="Dataset root directory")
    parser.add_argument("--language", required=True, help="Language code (e.g. zh, en)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.90, 0.05, 0.05],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Speaker-level split ratios (default: 0.90 0.05 0.05)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Store absolute paths instead of relative",
    )
    args = parser.parse_args()

    print(f"Scanning {args.root_dir!r} ...")
    entries = generate_manifest(
        args.root_dir,
        language=args.language,
        use_relative_paths=not args.absolute_paths,
    )
    print(f"Found {len(entries)} audio files")

    train_e, val_e, test_e = split_manifest_by_speaker(
        entries,
        train=args.split[0],
        val=args.split[1],
        test=args.split[2],
        seed=args.seed,
    )
    print(
        f"Split: train={len(train_e)}, val={len(val_e)}, test={len(test_e)} "
        f"({len({e.speaker_id for e in train_e})} / "
        f"{len({e.speaker_id for e in val_e})} / "
        f"{len({e.speaker_id for e in test_e})} speakers)"
    )

    # Write one output file per split
    base, ext = os.path.splitext(args.output)
    write_jsonl(train_e, f"{base}_train{ext}")
    write_jsonl(val_e, f"{base}_val{ext}")
    write_jsonl(test_e, f"{base}_test{ext}")
    print(f"Manifests written to: {base}_train{ext}, _val, _test")


if __name__ == "__main__":
    main()
