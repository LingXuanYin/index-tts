import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_FUSION_LEVELS = (
    "spk_cond_emb",
    "speech_conditioning_latent",
    "prompt_condition",
    "style",
    "ref_mel",
)

EXPERIMENTAL_FUSION_LEVELS = (
    "waveform",
    "cat_condition",
    "vc_target",
)

BRANCH_NAMES = (
    "speaker",
    "prompt",
    "style",
    "ref_mel",
    "emotion",
)

SUPPORTED_SPEAKER_FUSION_MODES = (
    "default",
    "fallback",
)

SUPPORTED_EMOTION_FUSION_MODES = (
    "default",
    "fallback",
)

DEFAULT_TIMBRE_FUSION_LEVELS = (
    "spk_cond_emb",
    "speech_conditioning_latent",
)

FALLBACK_TIMBRE_FUSION_LEVELS = (
    "speech_conditioning_latent",
)

TIMBRE_FUSION_PRESETS = {
    "default": {
        "enabled_levels": DEFAULT_TIMBRE_FUSION_LEVELS,
        "anchor_mode": "symmetric",
    },
    "fallback": {
        "enabled_levels": FALLBACK_TIMBRE_FUSION_LEVELS,
        "anchor_mode": "symmetric",
    },
}

EMOTION_FUSION_PRESETS = {
    "default": {
        "anchor_mode": "A",
        "operator": "weighted_sum",
    },
    "fallback": {
        "anchor_mode": "symmetric",
        "operator": "weighted_sum",
    },
}


@dataclass(frozen=True)
class FusionReference:
    path: str
    weight: float = 1.0
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "weight": float(self.weight),
            "name": self.name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FusionBranchConfig:
    references: Tuple[FusionReference, ...] = ()
    levels: Tuple[str, ...] = ()
    anchor_mode: str = "symmetric"
    operator: str = "weighted_sum"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "references": [ref.to_dict() for ref in self.references],
            "levels": list(self.levels),
            "anchor_mode": self.anchor_mode,
            "operator": self.operator,
        }


@dataclass(frozen=True)
class FusionRecipe:
    references: Tuple[FusionReference, ...]
    enabled_levels: Tuple[str, ...]
    branch_configs: Dict[str, FusionBranchConfig] = field(default_factory=dict)
    experimental_levels: Tuple[str, ...] = ()
    diffusion_steps: int = 25
    inference_cfg_rate: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_enabled(self, level: str) -> bool:
        return level in self.enabled_levels or level in self.experimental_levels

    def branch(self, name: str) -> FusionBranchConfig:
        return self.branch_configs.get(name, FusionBranchConfig())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "references": [ref.to_dict() for ref in self.references],
            "enabled_levels": list(self.enabled_levels),
            "branch_configs": {
                key: value.to_dict() for key, value in sorted(self.branch_configs.items())
            },
            "experimental_levels": list(self.experimental_levels),
            "diffusion_steps": int(self.diffusion_steps),
            "inference_cfg_rate": float(self.inference_cfg_rate),
            "metadata": dict(self.metadata),
        }


def _coerce_reference(item: Any) -> FusionReference:
    if isinstance(item, FusionReference):
        return item
    if isinstance(item, str):
        return FusionReference(path=item)
    if isinstance(item, dict):
        if "path" not in item:
            raise ValueError("Fusion reference dict must include `path`.")
        return FusionReference(
            path=str(item["path"]),
            weight=float(item.get("weight", 1.0)),
            name=item.get("name"),
            metadata=dict(item.get("metadata", {})),
        )
    raise TypeError(f"Unsupported fusion reference type: {type(item)!r}")


def _coerce_references(items: Optional[Iterable[Any]]) -> Tuple[FusionReference, ...]:
    if items is None:
        return ()
    refs = tuple(_coerce_reference(item) for item in items)
    if not refs:
        raise ValueError("Fusion reference list must not be empty.")
    return refs


def _normalize_levels(
    levels: Optional[Iterable[str]],
    allowed: Tuple[str, ...],
    field_name: str,
) -> Tuple[str, ...]:
    if levels is None:
        return ()
    ordered: List[str] = []
    seen = set()
    for level in levels:
        if level not in allowed:
            raise ValueError(f"Unknown {field_name} level: {level}")
        if level not in seen:
            ordered.append(level)
            seen.add(level)
    return tuple(ordered)


def _coerce_branch_config(
    name: str,
    config: Any,
    default_refs: Tuple[FusionReference, ...],
) -> FusionBranchConfig:
    if isinstance(config, FusionBranchConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError(f"Unsupported branch config type for {name}: {type(config)!r}")
    refs = _coerce_references(config.get("references")) or default_refs
    levels = _normalize_levels(
        config.get("levels"),
        SUPPORTED_FUSION_LEVELS + EXPERIMENTAL_FUSION_LEVELS,
        f"{name} branch",
    )
    return FusionBranchConfig(
        references=refs,
        levels=levels,
        anchor_mode=str(config.get("anchor_mode", "symmetric")),
        operator=str(config.get("operator", "weighted_sum")),
    )


def coerce_fusion_recipe(recipe: Optional[Any]) -> Optional[FusionRecipe]:
    if recipe is None:
        return None
    if isinstance(recipe, FusionRecipe):
        return recipe
    if not isinstance(recipe, dict):
        raise TypeError(f"Unsupported fusion recipe type: {type(recipe)!r}")

    refs = _coerce_references(recipe.get("references"))
    enabled_levels = _normalize_levels(
        recipe.get("enabled_levels"),
        SUPPORTED_FUSION_LEVELS,
        "supported fusion",
    )
    experimental_levels = _normalize_levels(
        recipe.get("experimental_levels"),
        EXPERIMENTAL_FUSION_LEVELS,
        "experimental fusion",
    )
    if len(refs) < 2 and (enabled_levels or experimental_levels):
        raise ValueError("Fusion recipes must include at least two references.")

    raw_branches = recipe.get("branch_configs", {}) or {}
    branch_configs = {
        name: _coerce_branch_config(name, raw_branches[name], refs)
        for name in raw_branches
    }
    for name in branch_configs:
        if name not in BRANCH_NAMES:
            raise ValueError(f"Unknown fusion branch: {name}")

    return FusionRecipe(
        references=refs,
        enabled_levels=enabled_levels,
        branch_configs=branch_configs,
        experimental_levels=experimental_levels,
        diffusion_steps=int(recipe.get("diffusion_steps", 25)),
        inference_cfg_rate=float(recipe.get("inference_cfg_rate", 0.7)),
        metadata=dict(recipe.get("metadata", {})),
    )


def normalize_weights(references: Tuple[FusionReference, ...]) -> List[float]:
    weights = [max(0.0, float(ref.weight)) for ref in references]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(references)] * len(references)
    return [weight / total for weight in weights]


def branch_references(recipe: Optional[FusionRecipe], branch_name: str) -> Tuple[FusionReference, ...]:
    if recipe is None:
        return ()
    branch = recipe.branch(branch_name)
    return branch.references or recipe.references


def branch_anchor_mode(recipe: Optional[FusionRecipe], branch_name: str) -> str:
    if recipe is None:
        return "symmetric"
    branch = recipe.branch(branch_name)
    return branch.anchor_mode


def branch_operator(recipe: Optional[FusionRecipe], branch_name: str) -> str:
    if recipe is None:
        return "weighted_sum"
    branch = recipe.branch(branch_name)
    return branch.operator


def recipe_metadata(recipe: Optional[FusionRecipe]) -> Dict[str, Any]:
    if recipe is None:
        return {}
    return recipe.to_dict()


def recipe_cache_token(
    recipe: Optional[FusionRecipe],
    branch_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "branch": branch_name,
        "recipe": recipe_metadata(recipe),
        "extra": extra or {},
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _coerce_reference_paths(items: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if items is None:
        return ()
    if isinstance(items, (str, FusionReference, dict)):
        items = [items]

    paths: List[str] = []
    for item in items:
        if isinstance(item, FusionReference):
            path = item.path
        elif isinstance(item, dict):
            path = item.get("path")
        else:
            path = item
        if path is None:
            continue
        path_text = str(path).strip()
        if path_text:
            paths.append(path_text)
    return tuple(paths)


def combine_reference_paths(primary_path: Optional[Any], references: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    ordered: List[str] = []
    seen = set()
    primary_paths = _coerce_reference_paths(primary_path)
    extra_paths = _coerce_reference_paths(references)
    for path in primary_paths + extra_paths:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return tuple(ordered)


def _build_weighted_references(
    paths: Tuple[str, ...],
    weights: Optional[Sequence[float]],
) -> Tuple[FusionReference, ...]:
    if not paths:
        return ()
    if weights is not None and len(weights) != len(paths):
        raise ValueError(
            f"Reference weights length ({len(weights)}) must match reference count ({len(paths)})."
        )
    refs: List[FusionReference] = []
    for idx, path in enumerate(paths):
        refs.append(FusionReference(path=path, weight=1.0 if weights is None else float(weights[idx])))
    return tuple(refs)


def _normalize_fusion_mode(mode: Optional[str], allowed: Tuple[str, ...], field_name: str) -> str:
    normalized = (mode or "default").strip().lower()
    if normalized not in allowed:
        allowed_text = ", ".join(allowed)
        raise ValueError(f"Unknown {field_name}: {mode!r}. Allowed values: {allowed_text}.")
    return normalized


def build_rollout_fusion_recipe(
    speaker_references: Optional[Iterable[Any]] = None,
    speaker_reference_weights: Optional[Sequence[float]] = None,
    speaker_fusion_mode: str = "default",
    emotion_references: Optional[Iterable[Any]] = None,
    emotion_reference_weights: Optional[Sequence[float]] = None,
    emotion_fusion_mode: str = "default",
    diffusion_steps: int = 25,
    inference_cfg_rate: float = 0.7,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[FusionRecipe]:
    speaker_paths = _coerce_reference_paths(speaker_references)
    emotion_paths = _coerce_reference_paths(emotion_references)
    use_speaker_fusion = len(speaker_paths) >= 2
    use_emotion_fusion = len(emotion_paths) >= 2
    if not use_speaker_fusion and not use_emotion_fusion:
        return None

    speaker_mode = _normalize_fusion_mode(
        speaker_fusion_mode,
        SUPPORTED_SPEAKER_FUSION_MODES,
        "speaker_fusion_mode",
    )
    emotion_mode = _normalize_fusion_mode(
        emotion_fusion_mode,
        SUPPORTED_EMOTION_FUSION_MODES,
        "emotion_fusion_mode",
    )

    branch_configs: Dict[str, FusionBranchConfig] = {}

    if use_speaker_fusion:
        speaker_refs = _build_weighted_references(speaker_paths, speaker_reference_weights)
        branch_configs["speaker"] = FusionBranchConfig(
            references=speaker_refs,
            levels=TIMBRE_FUSION_PRESETS[speaker_mode]["enabled_levels"],
            anchor_mode=TIMBRE_FUSION_PRESETS[speaker_mode]["anchor_mode"],
            operator="weighted_sum",
        )

    if use_emotion_fusion:
        emotion_refs = _build_weighted_references(emotion_paths, emotion_reference_weights)
        branch_configs["emotion"] = FusionBranchConfig(
            references=emotion_refs,
            levels=(),
            anchor_mode=EMOTION_FUSION_PRESETS[emotion_mode]["anchor_mode"],
            operator=EMOTION_FUSION_PRESETS[emotion_mode]["operator"],
        )

    recipe_metadata_dict = dict(metadata or {})
    recipe_metadata_dict.setdefault("rollout", {})
    recipe_metadata_dict["rollout"].update(
        {
            "preset_source": "rollout",
            "speaker_fusion_mode": speaker_mode if use_speaker_fusion else None,
            "emotion_fusion_mode": emotion_mode if use_emotion_fusion else None,
            "speaker_reference_count": len(speaker_paths),
            "emotion_reference_count": len(emotion_paths),
            "timbre_levels": list(branch_configs["speaker"].levels) if "speaker" in branch_configs else [],
        }
    )

    return FusionRecipe(
        references=(),
        enabled_levels=(),
        branch_configs=branch_configs,
        diffusion_steps=int(diffusion_steps),
        inference_cfg_rate=float(inference_cfg_rate),
        metadata=recipe_metadata_dict,
    )
