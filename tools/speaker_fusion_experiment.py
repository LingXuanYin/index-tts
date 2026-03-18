import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.fusion import EXPERIMENTAL_FUSION_LEVELS, SUPPORTED_FUSION_LEVELS

if TYPE_CHECKING:
    import torch


CMU_ARCTIC_SPEAKERS = (
    "aew",
    "ahw",
    "aup",
    "awb",
    "axb",
    "bdl",
    "clb",
    "eey",
    "fem",
    "gka",
    "jmk",
    "ksp",
    "ljm",
    "lnh",
    "rms",
    "rxr",
    "slp",
    "slt",
)
BRANCH_NAMES = ("speaker", "prompt", "style", "ref_mel", "emotion")
BRANCH_ROLE_OPTIONS = ("shared", "A_only", "B_only")
MANIFEST_LITERAL_FIELDS = (
    "role_assignments",
    "speaker_a",
    "speaker_b",
    "fusion_recipe",
    "supported_levels",
    "experimental_levels",
)


def _require_torch_modules():
    try:
        import torch
        import torchaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch dependencies are not installed. Run `uv sync --python 3.10` and then "
            "invoke this tool with `uv run python tools/speaker_fusion_experiment.py ...`."
        ) from exc
    return torch, torchaudio


def _require_tts():
    try:
        from indextts.infer_v2 import IndexTTS2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IndexTTS2 runtime dependencies are not installed. Run `uv sync --python 3.10` and "
            "use `uv run python tools/speaker_fusion_experiment.py ...`."
        ) from exc
    return IndexTTS2


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    speaker_id: str
    utterance_id: str
    text: str
    audio_path: str
    source_split: str
    batch_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "speaker_id": self.speaker_id,
            "utterance_id": self.utterance_id,
            "text": self.text,
            "audio_path": self.audio_path,
            "source_split": self.source_split,
            "batch_id": self.batch_id,
        }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def serialize_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    serialized = dict(row)
    for field_name in MANIFEST_LITERAL_FIELDS:
        serialized[field_name] = json.dumps(row[field_name], ensure_ascii=False, sort_keys=True)
    return serialized


def deserialize_manifest_row(row: Dict[str, str]) -> Dict[str, Any]:
    restored: Dict[str, Any] = dict(row)
    for field_name in MANIFEST_LITERAL_FIELDS:
        value = row.get(field_name)
        restored[field_name] = json.loads(value) if value else {}
    return restored


def iter_manifest(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield deserialize_manifest_row(row)
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    return list(iter_manifest(path))


def ensure_audio_saved(path: Path, waveform: Any, sample_rate: int) -> str:
    _, torchaudio = _require_torch_modules()
    path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate)
    return str(path)


def sample_cmu_arctic(
    root: Path,
    output_dir: Path,
    max_batches: int,
    seed: int,
) -> List[SampleRecord]:
    _, torchaudio = _require_torch_modules()
    (root / "cmu_arctic").mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    samples: List[SampleRecord] = []
    speaker_pool = list(CMU_ARCTIC_SPEAKERS)
    random.shuffle(speaker_pool)
    selected = speaker_pool[: max_batches * 2]
    batch_index = 0
    for idx in range(0, len(selected), 2):
        speaker_ids = selected[idx:idx + 2]
        if len(speaker_ids) < 2 or batch_index >= max_batches:
            break
        batch_id = f"cmu-arctic-{batch_index:02d}"
        for speaker_id in speaker_ids:
            dataset = torchaudio.datasets.CMUARCTIC(
                root=str(root / "cmu_arctic"),
                url=speaker_id,
                download=True,
            )
            item_index = random.randint(0, len(dataset) - 1)
            waveform, sample_rate, transcript, utterance_id = dataset[item_index]
            audio_path = ensure_audio_saved(
                output_dir / "audio" / batch_id / f"{speaker_id}_{utterance_id}.wav",
                waveform,
                sample_rate,
            )
            samples.append(
                SampleRecord(
                    dataset="cmu_arctic",
                    speaker_id=speaker_id,
                    utterance_id=str(utterance_id),
                    text=transcript,
                    audio_path=audio_path,
                    source_split=speaker_id,
                    batch_id=batch_id,
                )
            )
        batch_index += 1
    return samples


def sample_librispeech(
    root: Path,
    output_dir: Path,
    max_batches: int,
    seed: int,
) -> List[SampleRecord]:
    _, torchaudio = _require_torch_modules()
    (root / "librispeech").mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=str(root / "librispeech"),
        url="dev-clean",
        download=True,
    )
    per_speaker: Dict[str, List[int]] = {}
    for idx in range(len(dataset)):
        _, _, transcript, speaker_id, _, utterance_id = dataset[idx]
        key = str(speaker_id)
        per_speaker.setdefault(key, []).append(idx)
    speaker_ids = list(per_speaker)
    random.shuffle(speaker_ids)
    samples: List[SampleRecord] = []
    batch_index = 0
    for idx in range(0, len(speaker_ids), 2):
        pair = speaker_ids[idx:idx + 2]
        if len(pair) < 2 or batch_index >= max_batches:
            break
        batch_id = f"librispeech-{batch_index:02d}"
        for speaker_id in pair:
            item_index = random.choice(per_speaker[speaker_id])
            waveform, sample_rate, transcript, speaker_val, _, utterance_id = dataset[item_index]
            audio_path = ensure_audio_saved(
                output_dir / "audio" / batch_id / f"{speaker_id}_{utterance_id}.wav",
                waveform,
                sample_rate,
            )
            samples.append(
                SampleRecord(
                    dataset="librispeech",
                    speaker_id=str(speaker_val),
                    utterance_id=str(utterance_id),
                    text=transcript,
                    audio_path=audio_path,
                    source_split="dev-clean",
                    batch_id=batch_id,
                )
            )
        batch_index += 1
    return samples


def build_open_source_sample_slice(
    root: Path,
    output_dir: Path,
    num_batches: int = 10,
    seed: int = 42,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[SampleRecord] = []
    cmu_target = min(num_batches // 2, 5)
    libri_target = num_batches - cmu_target
    rows.extend(sample_cmu_arctic(root, output_dir, cmu_target, seed))
    rows.extend(sample_librispeech(root, output_dir, libri_target, seed + 1))
    manifest_path = output_dir / "ten_batch_slice.jsonl"
    write_jsonl(manifest_path, (row.to_dict() for row in rows))
    summary_path = output_dir / "ten_batch_slice.summary.json"
    summary = {
        "seed": seed,
        "requested_batches": num_batches,
        "actual_batches": len({row.batch_id for row in rows}),
        "datasets": {
            "cmu_arctic": len([row for row in rows if row.dataset == "cmu_arctic"]),
            "librispeech": len([row for row in rows if row.dataset == "librispeech"]),
        },
        "manifest_path": str(manifest_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest_path


def recipe_subsets(levels: Sequence[str]) -> Iterator[Tuple[str, ...]]:
    for size in range(1, len(levels) + 1):
        yield from itertools.combinations(levels, size)


def build_branch_profile(name: str, role_assignments: Dict[str, str]) -> Dict[str, Any]:
    normalized_roles = {branch_name: role_assignments.get(branch_name, "shared") for branch_name in BRANCH_NAMES}
    branch_configs: Dict[str, Dict[str, Any]] = {}
    for branch_name, selector in normalized_roles.items():
        if selector == "shared":
            continue
        slot = "A" if selector == "A_only" else "B"
        branch_configs[branch_name] = {
            "references": [{"slot": slot, "weight": 1.0}],
            "anchor_mode": slot,
        }
    return {
        "name": name,
        "branch_configs": branch_configs,
        "role_assignments": normalized_roles,
    }


def default_branch_profiles() -> List[Dict[str, Any]]:
    # Keep role coverage broad enough to probe branch sensitivity, but avoid the
    # 3^4 Cartesian blow-up that dominates the experiment matrix.
    return [
        build_branch_profile("shared", {}),
        build_branch_profile("all_A_only", {branch_name: "A_only" for branch_name in BRANCH_NAMES}),
        build_branch_profile("all_B_only", {branch_name: "B_only" for branch_name in BRANCH_NAMES}),
        build_branch_profile("speaker_A_only", {"speaker": "A_only"}),
        build_branch_profile("speaker_B_only", {"speaker": "B_only"}),
        build_branch_profile("prompt_A_only", {"prompt": "A_only"}),
        build_branch_profile("prompt_B_only", {"prompt": "B_only"}),
        build_branch_profile("style_A_only", {"style": "A_only"}),
        build_branch_profile("style_B_only", {"style": "B_only"}),
        build_branch_profile("ref_mel_A_only", {"ref_mel": "A_only"}),
        build_branch_profile("ref_mel_B_only", {"ref_mel": "B_only"}),
        build_branch_profile("emotion_A_only", {"emotion": "A_only"}),
        build_branch_profile("emotion_B_only", {"emotion": "B_only"}),
    ]


def baseline_recipe_templates(exploratory: bool) -> List[Dict[str, Any]]:
    templates = [
        {
            "case_kind": "baseline_a",
            "profile": "baseline-speaker-a",
            "references": [{"slot": "A", "weight": 1.0}],
            "enabled_levels": [],
            "experimental_levels": [],
            "branch_configs": {},
            "anchor_mode": "A",
            "diffusion_steps": 25,
            "inference_cfg_rate": 0.7,
        },
        {
            "case_kind": "baseline_b",
            "profile": "baseline-speaker-b",
            "references": [{"slot": "B", "weight": 1.0}],
            "enabled_levels": [],
            "experimental_levels": [],
            "branch_configs": {},
            "anchor_mode": "B",
            "diffusion_steps": 25,
            "inference_cfg_rate": 0.7,
        },
    ]
    if exploratory:
        templates.append(
            {
                "case_kind": "baseline_waveform_mix",
                "profile": "baseline-waveform-mix",
                "references": [{"slot": "A", "weight": 0.5}, {"slot": "B", "weight": 0.5}],
                "enabled_levels": [],
                "experimental_levels": ["waveform"],
                "branch_configs": {},
                "anchor_mode": "symmetric",
                "diffusion_steps": 25,
                "inference_cfg_rate": 0.7,
            }
        )
    return templates


def build_recipe_domain(
    weights: Sequence[float],
    anchor_modes: Sequence[str],
    exploratory: bool = False,
    branch_profiles: Optional[List[Dict[str, Any]]] = None,
    include_baselines: bool = True,
) -> List[Dict[str, Any]]:
    return list(
        iter_recipe_domain(
            weights,
            anchor_modes,
            exploratory=exploratory,
            branch_profiles=branch_profiles,
            include_baselines=include_baselines,
        )
    )


def iter_recipe_domain(
    weights: Sequence[float],
    anchor_modes: Sequence[str],
    exploratory: bool = False,
    branch_profiles: Optional[List[Dict[str, Any]]] = None,
    include_baselines: bool = True,
) -> Iterator[Dict[str, Any]]:
    branch_profiles = branch_profiles or default_branch_profiles()
    supported_subsets = list(recipe_subsets(SUPPORTED_FUSION_LEVELS))
    experimental_subsets = [()] + list(recipe_subsets(EXPERIMENTAL_FUSION_LEVELS)) if exploratory else [()]
    if include_baselines:
        yield from baseline_recipe_templates(exploratory)

    for supported in supported_subsets:
        for experimental in experimental_subsets:
            for alpha in weights:
                for anchor_mode in anchor_modes:
                    for branch_profile in branch_profiles:
                        branch_configs = {}
                        for branch_name, config in branch_profile["branch_configs"].items():
                            branch_configs[branch_name] = {
                                "references": [
                                    {
                                        "slot": reference["slot"],
                                        "weight": float(reference.get("weight", 1.0)),
                                    }
                                    for reference in config.get("references", [])
                                ],
                                "anchor_mode": config.get("anchor_mode", anchor_mode),
                                "operator": config.get("operator", "weighted_sum"),
                                "levels": list(config.get("levels", [])),
                            }
                        for config in branch_configs.values():
                            config.setdefault("anchor_mode", anchor_mode)
                        yield {
                            "case_kind": "fusion",
                            "profile": branch_profile["name"],
                            "role_assignments": branch_profile.get("role_assignments", {}),
                            "references": [
                                {"slot": "A", "weight": round(alpha, 4)},
                                {"slot": "B", "weight": round(1.0 - alpha, 4)},
                            ],
                            "enabled_levels": list(supported),
                            "experimental_levels": list(experimental),
                            "branch_configs": branch_configs,
                            "anchor_mode": anchor_mode,
                            "diffusion_steps": 25,
                            "inference_cfg_rate": 0.7,
                        }


def pair_batches(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["batch_id"], []).append(row)
    for batch_id in grouped:
        grouped[batch_id] = sorted(grouped[batch_id], key=lambda item: (item["speaker_id"], item["utterance_id"]))
    return grouped


def fill_recipe_slots(recipe_template: Dict[str, Any], batch_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = list(batch_rows)
    speaker_a = ordered[0]
    speaker_b = ordered[1] if len(ordered) > 1 else ordered[0]
    slot_map = {"A": speaker_a, "B": speaker_b}
    recipe = {
        "references": [
            {
                "slot": reference["slot"],
                "weight": float(reference.get("weight", 1.0)),
                "name": reference.get("name"),
                "metadata": dict(reference.get("metadata", {})),
            }
            for reference in recipe_template.get("references", [])
        ],
        "enabled_levels": list(recipe_template.get("enabled_levels", [])),
        "experimental_levels": list(recipe_template.get("experimental_levels", [])),
        "branch_configs": {},
        "diffusion_steps": recipe_template.get("diffusion_steps", 25),
        "inference_cfg_rate": recipe_template.get("inference_cfg_rate", 0.7),
    }
    for branch_name, branch_config in recipe_template.get("branch_configs", {}).items():
        recipe["branch_configs"][branch_name] = {
            "anchor_mode": branch_config.get("anchor_mode", "symmetric"),
            "operator": branch_config.get("operator", "weighted_sum"),
            "levels": list(branch_config.get("levels", [])),
            "references": [
                {
                    "slot": reference["slot"],
                    "weight": float(reference.get("weight", 1.0)),
                    "name": reference.get("name"),
                    "metadata": dict(reference.get("metadata", {})),
                }
                for reference in branch_config.get("references", [])
            ],
        }
    for ref in recipe["references"]:
        mapped = slot_map[ref["slot"]]
        ref["path"] = mapped["audio_path"]
        ref["name"] = mapped["speaker_id"]
        ref["metadata"] = {
            "dataset": mapped["dataset"],
            "batch_id": mapped["batch_id"],
            "utterance_id": mapped["utterance_id"],
        }
        del ref["slot"]
    for branch_config in recipe.get("branch_configs", {}).values():
        branch_refs = branch_config.get("references") or []
        for ref in branch_refs:
            mapped = slot_map[ref["slot"]]
            ref["path"] = mapped["audio_path"]
            ref["name"] = mapped["speaker_id"]
            ref["metadata"] = {
                "dataset": mapped["dataset"],
                "batch_id": mapped["batch_id"],
                "utterance_id": mapped["utterance_id"],
            }
            del ref["slot"]
    return recipe


def compact_speaker_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "speaker_id": row["speaker_id"],
        "utterance_id": row["utterance_id"],
        "audio_path": row["audio_path"],
        "batch_id": row["batch_id"],
    }


def compact_fusion_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "references": [
            {
                "path": reference.get("path"),
                "name": reference.get("name"),
                "weight": reference.get("weight"),
                "metadata": reference.get("metadata", {}),
            }
            for reference in recipe.get("references", [])
        ],
        "enabled_levels": list(recipe.get("enabled_levels", [])),
        "experimental_levels": list(recipe.get("experimental_levels", [])),
        "branch_configs": {},
        "diffusion_steps": recipe.get("diffusion_steps", 25),
        "inference_cfg_rate": recipe.get("inference_cfg_rate", 0.7),
    }
    for branch_name, branch_config in sorted(recipe.get("branch_configs", {}).items()):
        compact["branch_configs"][branch_name] = {
            "anchor_mode": branch_config.get("anchor_mode", "symmetric"),
            "operator": branch_config.get("operator", "weighted_sum"),
            "levels": list(branch_config.get("levels", [])),
            "references": [
                {
                    "path": reference.get("path"),
                    "name": reference.get("name"),
                    "weight": reference.get("weight"),
                    "metadata": reference.get("metadata", {}),
                }
                for reference in branch_config.get("references", [])
            ],
        }
    return compact


def scheme_identity(case_kind: str, recipe: Dict[str, Any], profile: str, anchor_mode: str) -> str:
    compact_recipe = {
        "enabled_levels": list(recipe.get("enabled_levels", [])),
        "experimental_levels": list(recipe.get("experimental_levels", [])),
        "references": [
            {
                "path": reference.get("path"),
                "name": reference.get("name"),
                "weight": reference.get("weight"),
            }
            for reference in recipe.get("references", [])
        ],
        "branch_configs": {
            branch_name: {
                "anchor_mode": branch_config.get("anchor_mode"),
                "levels": list(branch_config.get("levels", [])),
                "references": [
                    {
                        "path": reference.get("path"),
                        "name": reference.get("name"),
                        "weight": reference.get("weight"),
                    }
                    for reference in branch_config.get("references", [])
                ],
            }
            for branch_name, branch_config in sorted(recipe.get("branch_configs", {}).items())
        },
    }
    payload = {
        "case_kind": case_kind,
        "profile": profile,
        "anchor_mode": anchor_mode,
        "recipe": compact_recipe,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def build_experiment_manifest(
    sample_slice_path: Path,
    output_dir: Path,
    weights: Sequence[float],
    anchor_modes: Sequence[str],
    exploratory: bool = False,
    branch_profiles: Optional[List[Dict[str, Any]]] = None,
    include_baselines: bool = True,
) -> Path:
    rows = read_jsonl(sample_slice_path)
    batches = pair_batches(rows)
    branch_profile_count = len(branch_profiles) if branch_profiles is not None else len(default_branch_profiles())
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_batch_count = len([batch_rows for batch_rows in batches.values() if len(batch_rows) >= 2])
    supported_subsets = 2 ** len(SUPPORTED_FUSION_LEVELS) - 1
    experimental_subsets = 1 + (2 ** len(EXPERIMENTAL_FUSION_LEVELS) - 1) if exploratory else 1
    baseline_count = len(baseline_recipe_templates(exploratory)) if include_baselines else 0
    per_batch_case_count = baseline_count + (
        supported_subsets
        * experimental_subsets
        * len(weights)
        * len(anchor_modes)
        * branch_profile_count
    )
    use_csv_manifest = valid_batch_count * per_batch_case_count > 100000
    manifest_path = output_dir / (
        "full_combination_manifest.csv" if use_csv_manifest else "full_combination_manifest.jsonl"
    )
    case_count = 0
    batch_count = 0
    scheme_ids = set()
    if use_csv_manifest:
        csv_fieldnames = [
            "case_id",
            "case_kind",
            "scheme_id",
            "profile",
            "anchor_mode",
            "role_assignments",
            "batch_id",
            "dataset",
            "text",
            "spk_audio_prompt",
            "emo_audio_prompt",
            "speaker_a",
            "speaker_b",
            "fusion_recipe",
            "output_path",
            "metadata_output_path",
            "recipe_name",
            "experimental_name",
            "supported_levels",
            "experimental_levels",
        ]
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
            writer.writeheader()
            for batch_id, batch_rows in sorted(batches.items()):
                if len(batch_rows) < 2:
                    continue
                batch_count += 1
                speaker_a = batch_rows[0]
                speaker_b = batch_rows[1]
                text = speaker_a["text"]
                for recipe_index, recipe_template in enumerate(
                    iter_recipe_domain(
                        weights,
                        anchor_modes,
                        exploratory=exploratory,
                        branch_profiles=branch_profiles,
                        include_baselines=include_baselines,
                    )
                ):
                    recipe = fill_recipe_slots(recipe_template, batch_rows)
                    recipe_name = "-".join(recipe["enabled_levels"]) or "none"
                    experimental_name = "-".join(recipe["experimental_levels"]) or "default"
                    case_kind = recipe_template["case_kind"]
                    compact_recipe = compact_fusion_recipe(recipe)
                    scheme_id = scheme_identity(
                        case_kind=case_kind,
                        recipe=compact_recipe,
                        profile=recipe_template["profile"],
                        anchor_mode=recipe_template["anchor_mode"],
                    )
                    case_id = f"{batch_id}-r{recipe_index:05d}"
                    manifest_row = {
                        "case_id": case_id,
                        "case_kind": case_kind,
                        "scheme_id": scheme_id,
                        "profile": recipe_template["profile"],
                        "anchor_mode": recipe_template["anchor_mode"],
                        "role_assignments": recipe_template.get("role_assignments", {}),
                        "batch_id": batch_id,
                        "dataset": speaker_a["dataset"],
                        "text": text,
                        "spk_audio_prompt": speaker_a["audio_path"],
                        "emo_audio_prompt": speaker_a["audio_path"],
                        "speaker_a": compact_speaker_record(speaker_a),
                        "speaker_b": compact_speaker_record(speaker_b),
                        "fusion_recipe": compact_recipe,
                        "output_path": str(output_dir / "outputs" / f"{case_id}.wav"),
                        "metadata_output_path": str(output_dir / "metadata" / f"{case_id}.json"),
                        "recipe_name": recipe_name,
                        "experimental_name": experimental_name,
                        "supported_levels": compact_recipe["enabled_levels"],
                        "experimental_levels": compact_recipe["experimental_levels"],
                    }
                    writer.writerow(serialize_manifest_row(manifest_row))
                    case_count += 1
                    scheme_ids.add(scheme_id)
    else:
        with manifest_path.open("w", encoding="utf-8") as handle:
            for batch_id, batch_rows in sorted(batches.items()):
                if len(batch_rows) < 2:
                    continue
                batch_count += 1
                speaker_a = batch_rows[0]
                speaker_b = batch_rows[1]
                text = speaker_a["text"]
                for recipe_index, recipe_template in enumerate(
                    iter_recipe_domain(
                        weights,
                        anchor_modes,
                        exploratory=exploratory,
                        branch_profiles=branch_profiles,
                        include_baselines=include_baselines,
                    )
                ):
                    recipe = fill_recipe_slots(recipe_template, batch_rows)
                    recipe_name = "-".join(recipe["enabled_levels"]) or "none"
                    experimental_name = "-".join(recipe["experimental_levels"]) or "default"
                    case_kind = recipe_template["case_kind"]
                    compact_recipe = compact_fusion_recipe(recipe)
                    scheme_id = scheme_identity(
                        case_kind=case_kind,
                        recipe=compact_recipe,
                        profile=recipe_template["profile"],
                        anchor_mode=recipe_template["anchor_mode"],
                    )
                    case_id = f"{batch_id}-r{recipe_index:05d}"
                    manifest_row = {
                        "case_id": case_id,
                        "case_kind": case_kind,
                        "scheme_id": scheme_id,
                        "profile": recipe_template["profile"],
                        "anchor_mode": recipe_template["anchor_mode"],
                        "role_assignments": recipe_template.get("role_assignments", {}),
                        "batch_id": batch_id,
                        "dataset": speaker_a["dataset"],
                        "text": text,
                        "spk_audio_prompt": speaker_a["audio_path"],
                        "emo_audio_prompt": speaker_a["audio_path"],
                        "speaker_a": compact_speaker_record(speaker_a),
                        "speaker_b": compact_speaker_record(speaker_b),
                        "fusion_recipe": compact_recipe,
                        "output_path": str(output_dir / "outputs" / f"{case_id}.wav"),
                        "metadata_output_path": str(output_dir / "metadata" / f"{case_id}.json"),
                        "recipe_name": recipe_name,
                        "experimental_name": experimental_name,
                        "supported_levels": compact_recipe["enabled_levels"],
                        "experimental_levels": compact_recipe["experimental_levels"],
                    }
                    handle.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
                    case_count += 1
                    scheme_ids.add(scheme_id)
    summary_path = output_dir / "full_combination_manifest.summary.json"
    summary = {
        "sample_slice_path": str(sample_slice_path),
        "manifest_path": str(manifest_path),
        "batch_count": batch_count,
        "case_count": case_count,
        "scheme_count": len(scheme_ids),
        "exploratory": exploratory,
        "weights": list(weights),
        "anchor_modes": list(anchor_modes),
        "branch_profile_count": branch_profile_count,
        "include_baselines": include_baselines,
        "manifest_format": "csv" if use_csv_manifest else "jsonl",
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest_path


def cosine_similarity(a: "torch.Tensor", b: "torch.Tensor") -> float:
    torch, _ = _require_torch_modules()
    a = a.float().reshape(1, -1)
    b = b.float().reshape(1, -1)
    return float(torch.nn.functional.cosine_similarity(a, b).item())


def pooled_semantic_embedding(tts: Any, audio_path: str) -> Any:
    audio, _ = tts._load_and_cut_audio(audio_path, 15, False, sr=16000)
    inputs = tts.extract_features(audio, sampling_rate=16000, return_tensors="pt")
    embedding = tts.get_emb(inputs["input_features"].to(tts.device), inputs["attention_mask"].to(tts.device))
    return embedding.mean(dim=1).squeeze(0).detach().cpu()


def style_embedding(tts: Any, audio_path: str) -> Any:
    bundle = tts._get_reference_conditioning(audio_path, verbose=False, cache_key=f"style::{audio_path}")
    return bundle["style"].squeeze(0).detach().cpu()


def load_metadata(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_target_weights(row: Dict[str, Any]) -> Tuple[float, float]:
    speaker_a_path = row["speaker_a"]["audio_path"]
    speaker_b_path = row["speaker_b"]["audio_path"]
    weight_a = 0.0
    weight_b = 0.0
    for reference in row["fusion_recipe"]["references"]:
        if reference["path"] == speaker_a_path:
            weight_a += float(reference.get("weight", 0.0))
        elif reference["path"] == speaker_b_path:
            weight_b += float(reference.get("weight", 0.0))
    total = weight_a + weight_b
    if total <= 0:
        return 1.0, 0.0
    return weight_a / total, weight_b / total


def generation_health(duration_seconds: float, silence_ratio: float, metadata: Dict[str, Any]) -> float:
    score = 1.0
    if duration_seconds <= 0.2:
        score -= 0.35
    if silence_ratio >= 0.98:
        score -= 0.45
    elif silence_ratio >= 0.9:
        score -= 0.25
    if not metadata.get("segments"):
        score -= 0.15
    timings = metadata.get("timings", {})
    if timings.get("rtf") is None:
        score -= 0.05
    return max(0.0, score)


def score_case(
    tts: Any,
    row: Dict[str, Any],
    style_cache: Dict[str, Any],
    semantic_cache: Dict[str, Any],
    batch_baselines: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    _, torchaudio = _require_torch_modules()
    output_path = row["output_path"]
    speaker_a_path = row["speaker_a"]["audio_path"]
    speaker_b_path = row["speaker_b"]["audio_path"]

    def cached_style(path: str) -> Any:
        if path not in style_cache:
            style_cache[path] = style_embedding(tts, path)
        return style_cache[path]

    def cached_semantic(path: str) -> Any:
        if path not in semantic_cache:
            semantic_cache[path] = pooled_semantic_embedding(tts, path)
        return semantic_cache[path]

    generated_style = cached_style(output_path)
    style_a = cached_style(speaker_a_path)
    style_b = cached_style(speaker_b_path)
    weight_a, weight_b = resolve_target_weights(row)
    fused_target = style_a * weight_a + style_b * weight_b
    generated_semantic = cached_semantic(output_path)
    prompt_semantic = cached_semantic(speaker_a_path)
    semantic_targets = []
    baseline_paths = batch_baselines.get(row["batch_id"], {})
    for baseline_key in ("baseline_a", "baseline_b"):
        baseline_path = baseline_paths.get(baseline_key)
        if baseline_path and os.path.exists(baseline_path):
            semantic_targets.append(cached_semantic(baseline_path))
    semantic_stability = (
        statistics.mean(cosine_similarity(generated_semantic, target) for target in semantic_targets)
        if semantic_targets
        else None
    )

    waveform, sample_rate = torchaudio.load(output_path)
    duration_seconds = waveform.shape[-1] / sample_rate
    silence_ratio = float((waveform.abs() < 1e-4).float().mean().item())
    metadata = load_metadata(row.get("metadata_output_path"))
    timings = metadata.get("timings", {})
    return {
        "case_id": row["case_id"],
        "case_kind": row["case_kind"],
        "scheme_id": row["scheme_id"],
        "speaker_similarity_a": cosine_similarity(generated_style, style_a),
        "speaker_similarity_b": cosine_similarity(generated_style, style_b),
        "fused_target_similarity": cosine_similarity(generated_style, fused_target),
        "semantic_prompt_similarity": cosine_similarity(generated_semantic, prompt_semantic),
        "semantic_stability": semantic_stability,
        "duration_seconds": duration_seconds,
        "silence_ratio": silence_ratio,
        "total_time_seconds": timings.get("total_time"),
        "rtf": timings.get("rtf"),
        "generation_health": generation_health(duration_seconds, silence_ratio, metadata),
        "metadata_segments": len(metadata.get("segments", [])),
        "experimental_case": bool(row["experimental_levels"]),
    }


def aggregate_scores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scheme_id"], []).append(row)

    report: List[Dict[str, Any]] = []
    for scheme_id, values in sorted(grouped.items()):
        first = values[0]
        speaker_similarity_a_mean = statistics.mean(value["speaker_similarity_a"] for value in values)
        speaker_similarity_b_mean = statistics.mean(value["speaker_similarity_b"] for value in values)
        fused_target_similarity_mean = statistics.mean(value["fused_target_similarity"] for value in values)
        semantic_prompt_similarity_mean = statistics.mean(value["semantic_prompt_similarity"] for value in values)
        semantic_stability_values = [value["semantic_stability"] for value in values if value["semantic_stability"] is not None]
        semantic_stability_mean = statistics.mean(semantic_stability_values) if semantic_stability_values else None
        silence_ratio_mean = statistics.mean(value["silence_ratio"] for value in values)
        generation_health_mean = statistics.mean(value["generation_health"] for value in values)
        duration_seconds_mean = statistics.mean(value["duration_seconds"] for value in values)
        total_time_values = [value["total_time_seconds"] for value in values if value["total_time_seconds"] is not None]
        rtf_values = [value["rtf"] for value in values if value["rtf"] is not None]
        total_time_seconds_mean = statistics.mean(total_time_values) if total_time_values else None
        rtf_mean = statistics.mean(rtf_values) if rtf_values else None
        blend_balance_mean = 1.0 - abs(speaker_similarity_a_mean - speaker_similarity_b_mean)
        speed_score = 1.0 / (1.0 + (rtf_mean if rtf_mean is not None else 1.0))
        stability_component = semantic_stability_mean if semantic_stability_mean is not None else semantic_prompt_similarity_mean
        overall_score = (
            0.4 * fused_target_similarity_mean
            + 0.2 * blend_balance_mean
            + 0.2 * stability_component
            + 0.1 * generation_health_mean
            + 0.1 * speed_score
        )
        report.append(
            {
                "scheme_id": scheme_id,
                "case_kind": first["case_kind"],
                "profile": first["profile"],
                "anchor_mode": first["anchor_mode"],
                "recipe_name": first["recipe_name"],
                "experimental_name": first["experimental_name"],
                "supported_levels": ",".join(first["supported_levels"]),
                "experimental_levels": ",".join(first["experimental_levels"]),
                "count": len(values),
                "speaker_similarity_a_mean": speaker_similarity_a_mean,
                "speaker_similarity_b_mean": speaker_similarity_b_mean,
                "blend_balance_mean": blend_balance_mean,
                "fused_target_similarity_mean": fused_target_similarity_mean,
                "semantic_prompt_similarity_mean": semantic_prompt_similarity_mean,
                "semantic_stability_mean": semantic_stability_mean,
                "duration_seconds_mean": duration_seconds_mean,
                "silence_ratio_mean": silence_ratio_mean,
                "generation_health_mean": generation_health_mean,
                "total_time_seconds_mean": total_time_seconds_mean,
                "rtf_mean": rtf_mean,
                "overall_score": overall_score,
                "experimental_case": bool(first["experimental_levels"]),
            }
        )

    report.sort(key=lambda row: row["overall_score"], reverse=True)
    for rank, row in enumerate(report, start=1):
        row["rank"] = rank
    return report


def select_recommendations(report_rows: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    supported_fusion = [
        row
        for row in report_rows
        if row["case_kind"] == "fusion" and not row["experimental_case"]
    ]
    exploratory_fusion = [
        row
        for row in report_rows
        if row["case_kind"] == "fusion" and row["experimental_case"]
    ]
    return {
        "default_recommendation": supported_fusion[0]["scheme_id"] if supported_fusion else None,
        "fallback_recommendation": supported_fusion[1]["scheme_id"] if len(supported_fusion) > 1 else None,
        "top_experimental": exploratory_fusion[0]["scheme_id"] if exploratory_fusion else None,
    }


def export_listening_review(
    scored_rows: List[Dict[str, Any]],
    recommendations: Dict[str, Optional[str]],
    output_dir: Path,
) -> Path:
    selected_scheme_ids = [
        scheme_id
        for scheme_id in (
            recommendations["default_recommendation"],
            recommendations["fallback_recommendation"],
            recommendations["top_experimental"],
        )
        if scheme_id
    ]
    listening_rows = [row for row in scored_rows if row["scheme_id"] in selected_scheme_ids]
    listening_rows.sort(key=lambda row: (row["scheme_id"], row["batch_id"], row["case_id"]))
    listening_path = output_dir / "listening_review.jsonl"
    write_jsonl(listening_path, listening_rows)
    return listening_path


def run_manifest(
    manifest_path: Path,
    cfg_path: str,
    model_dir: str,
    limit: Optional[int] = None,
    force: bool = False,
) -> Path:
    IndexTTS2 = _require_tts()
    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )
    results_path = manifest_path.with_name("run_results.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with results_path.open("w", encoding="utf-8") as handle:
        for row in iter_manifest(manifest_path):
            if limit is not None and processed >= limit:
                break
            processed += 1
            output_path = row["output_path"]
            metadata_output_path = row["metadata_output_path"]
            if not force and os.path.exists(output_path) and os.path.exists(metadata_output_path):
                result = {
                    "case_id": row["case_id"],
                    "scheme_id": row["scheme_id"],
                    "status": "skipped",
                    "output_path": output_path,
                    "metadata_output_path": metadata_output_path,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue

            print(f"[{processed}] {row['case_id']}")
            output = tts.infer(
                spk_audio_prompt=row["spk_audio_prompt"],
                text=row["text"],
                output_path=output_path,
                emo_audio_prompt=row["emo_audio_prompt"],
                fusion_recipe=row["fusion_recipe"],
                metadata_output_path=metadata_output_path,
                return_metadata=True,
            )
            result = {
                "case_id": row["case_id"],
                "scheme_id": row["scheme_id"],
                "status": "done",
                "output_path": output_path,
                "metadata_output_path": metadata_output_path,
                "inference_output": output,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return results_path


def score_manifest(
    manifest_path: Path,
    cfg_path: str,
    model_dir: str,
) -> Tuple[Path, Path, Path, Path]:
    IndexTTS2 = _require_tts()
    rows = read_manifest(manifest_path)
    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )
    style_cache: Dict[str, Any] = {}
    semantic_cache: Dict[str, Any] = {}
    batch_baselines: Dict[str, Dict[str, Optional[str]]] = {}
    for row in rows:
        if row["case_kind"] not in {"baseline_a", "baseline_b"}:
            continue
        batch_baselines.setdefault(row["batch_id"], {"baseline_a": None, "baseline_b": None})
        if row["case_kind"] == "baseline_a":
            batch_baselines[row["batch_id"]]["baseline_a"] = row["output_path"]
        elif row["case_kind"] == "baseline_b":
            batch_baselines[row["batch_id"]]["baseline_b"] = row["output_path"]

    scored_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not os.path.exists(row["output_path"]):
            continue
        metrics = score_case(tts, row, style_cache, semantic_cache, batch_baselines)
        scored = dict(row)
        scored.update(metrics)
        scored_rows.append(scored)
    scores_path = manifest_path.with_name("scores.jsonl")
    write_jsonl(scores_path, scored_rows)

    report_rows = aggregate_scores(scored_rows)
    report_path = manifest_path.with_name("ranked_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(report_rows[0].keys()) if report_rows else ["rank"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    recommendations = select_recommendations(report_rows)
    summary_path = manifest_path.with_name("ranked_summary.json")
    summary = {
        "manifest_path": str(manifest_path),
        "scores_path": str(scores_path),
        "report_path": str(report_path),
        "recommendations": recommendations,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    listening_path = export_listening_review(scored_rows, recommendations, manifest_path.parent)
    return scores_path, report_path, summary_path, listening_path


def parse_branch_profiles(raw: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None
    with open(raw, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker fusion experiment utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--root", default="data/open_source")
    sample_parser.add_argument("--output-dir", default="artifacts/speaker_fusion")
    sample_parser.add_argument("--num-batches", type=int, default=10)
    sample_parser.add_argument("--seed", type=int, default=42)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--sample-slice", required=True)
    manifest_parser.add_argument("--output-dir", default="artifacts/speaker_fusion")
    manifest_parser.add_argument("--weights", default="0.2,0.35,0.5,0.65,0.8")
    manifest_parser.add_argument("--anchor-modes", default="A,B,symmetric")
    manifest_parser.add_argument("--exploratory", action="store_true")
    manifest_parser.add_argument("--branch-profiles")
    manifest_parser.add_argument("--skip-baselines", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--cfg-path", default="checkpoints/config.yaml")
    run_parser.add_argument("--model-dir", default="checkpoints")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--force", action="store_true")

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--manifest", required=True)
    score_parser.add_argument("--cfg-path", default="checkpoints/config.yaml")
    score_parser.add_argument("--model-dir", default="checkpoints")

    args = parser.parse_args()

    if args.command == "sample":
        manifest = build_open_source_sample_slice(
            root=Path(args.root),
            output_dir=Path(args.output_dir),
            num_batches=args.num_batches,
            seed=args.seed,
        )
        print(manifest)
        return

    if args.command == "manifest":
        weights = [float(value) for value in args.weights.split(",") if value]
        anchor_modes = [value.strip() for value in args.anchor_modes.split(",") if value.strip()]
        manifest = build_experiment_manifest(
            sample_slice_path=Path(args.sample_slice),
            output_dir=Path(args.output_dir),
            weights=weights,
            anchor_modes=anchor_modes,
            exploratory=args.exploratory,
            branch_profiles=parse_branch_profiles(args.branch_profiles),
            include_baselines=not args.skip_baselines,
        )
        print(manifest)
        return

    if args.command == "run":
        results = run_manifest(
            manifest_path=Path(args.manifest),
            cfg_path=args.cfg_path,
            model_dir=args.model_dir,
            limit=args.limit,
            force=args.force,
        )
        print(results)
        return

    if args.command == "score":
        scores_path, report_path, summary_path, listening_path = score_manifest(
            manifest_path=Path(args.manifest),
            cfg_path=args.cfg_path,
            model_dir=args.model_dir,
        )
        print(scores_path)
        print(report_path)
        print(summary_path)
        print(listening_path)


if __name__ == "__main__":
    main()
