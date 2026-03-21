from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indextts.fusion import build_rollout_fusion_recipe
from indextts.infer_v2 import IndexTTS2


def test_build_rollout_fusion_recipe_uses_selected_timbre_default():
    recipe = build_rollout_fusion_recipe(
        speaker_references=["a.wav", "b.wav"],
        speaker_fusion_mode="default",
    )

    assert recipe is not None
    assert recipe.enabled_levels == ()
    assert recipe.references == ()
    speaker_branch = recipe.branch_configs["speaker"]
    assert [ref.path for ref in speaker_branch.references] == ["a.wav", "b.wav"]
    assert [ref.weight for ref in speaker_branch.references] == [1.0, 1.0]
    assert speaker_branch.levels == ("spk_cond_emb", "speech_conditioning_latent")
    assert recipe.metadata["rollout"]["speaker_fusion_mode"] == "default"
    assert recipe.metadata["rollout"]["timbre_levels"] == ["spk_cond_emb", "speech_conditioning_latent"]
    assert set(recipe.branch_configs) == {"speaker"}


def test_build_rollout_fusion_recipe_supports_emotion_only_default():
    recipe = build_rollout_fusion_recipe(
        emotion_references=["emo_a.wav", "emo_b.wav"],
        emotion_fusion_mode="default",
    )

    assert recipe is not None
    assert recipe.enabled_levels == ()
    assert recipe.references == ()
    emotion_branch = recipe.branch_configs["emotion"]
    assert [ref.path for ref in emotion_branch.references] == ["emo_a.wav", "emo_b.wav"]
    assert emotion_branch.anchor_mode == "A"
    assert emotion_branch.operator == "weighted_sum"
    assert recipe.metadata["rollout"]["emotion_fusion_mode"] == "default"


def test_build_rollout_fusion_recipe_supports_fallback_presets():
    recipe = build_rollout_fusion_recipe(
        speaker_references=["speaker_a.wav", "speaker_b.wav"],
        speaker_fusion_mode="fallback",
        emotion_references=["emo_a.wav", "emo_b.wav"],
        emotion_fusion_mode="fallback",
    )

    assert recipe is not None
    assert recipe.enabled_levels == ()
    assert recipe.references == ()
    assert recipe.branch_configs["speaker"].levels == ("speech_conditioning_latent",)
    assert [ref.path for ref in recipe.branch_configs["speaker"].references] == ["speaker_a.wav", "speaker_b.wav"]
    assert recipe.branch_configs["emotion"].anchor_mode == "symmetric"
    assert recipe.metadata["rollout"]["speaker_fusion_mode"] == "fallback"
    assert recipe.metadata["rollout"]["emotion_fusion_mode"] == "fallback"


def test_resolve_requested_fusion_recipe_keeps_raw_recipe():
    tts = IndexTTS2.__new__(IndexTTS2)
    raw_recipe = {
        "references": [
            {"path": "raw_a.wav", "weight": 0.3},
            {"path": "raw_b.wav", "weight": 0.7},
        ],
        "enabled_levels": ["style"],
    }

    spk_prompt, emo_prompt, speaker_refs, emotion_refs, recipe = tts._resolve_requested_fusion_recipe(
        "speaker.wav",
        "emotion.wav",
        raw_recipe,
        ["other.wav"],
        None,
        "default",
        ["emotion_b.wav"],
        None,
        "fallback",
        25,
        0.7,
    )

    assert spk_prompt == "speaker.wav"
    assert emo_prompt == "emotion.wav"
    assert speaker_refs == ["other.wav"]
    assert emotion_refs == ["emotion_b.wav"]
    assert [ref.path for ref in recipe.references] == ["raw_a.wav", "raw_b.wav"]
    assert recipe.enabled_levels == ("style",)


def test_resolve_requested_fusion_recipe_uses_list_shorthand_with_rollout_defaults():
    tts = IndexTTS2.__new__(IndexTTS2)

    spk_prompt, emo_prompt, speaker_refs, emotion_refs, recipe = tts._resolve_requested_fusion_recipe(
        ["speaker_a.wav", "speaker_b.wav"],
        None,
        None,
        None,
        None,
        "default",
        None,
        None,
        "default",
        31,
        0.9,
    )

    assert spk_prompt == "speaker_a.wav"
    assert emo_prompt is None
    assert speaker_refs == ["speaker_a.wav", "speaker_b.wav"]
    assert emotion_refs is None
    assert recipe is not None
    assert recipe.enabled_levels == ()
    speaker_branch = recipe.branch_configs["speaker"]
    assert speaker_branch.levels == ("spk_cond_emb", "speech_conditioning_latent")
    assert [ref.path for ref in speaker_branch.references] == ["speaker_a.wav", "speaker_b.wav"]
    assert recipe.diffusion_steps == 31
    assert recipe.inference_cfg_rate == 0.9
    assert recipe.metadata["speaker_prompt_list_shorthand"] is True
