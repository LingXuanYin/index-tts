## Context

The experiment-oriented fusion work for IndexTTS v2 is already present in [`indextts/infer_v2.py`](../../../indextts/infer_v2.py), [`indextts/fusion.py`](../../../indextts/fusion.py), and the related scoring/handoff documents. The project has also already converged on supported defaults:

- timbre default: `spk_cond_emb + speech_conditioning_latent`
- timbre fallback: `speech_conditioning_latent`
- emotion default: `emotion_tensor_anchor_a`
- emotion fallback: `emotion_tensor_sym`

However, the repository still exposes this mostly as low-level `fusion_recipe` control plus experiment tooling. Public surfaces are inconsistent:

- Python runtime accepts `fusion_recipe`, but only provides a shorthand for `spk_audio_prompt` as a list and that shorthand enables every supported level instead of the selected default.
- WebUI exposes only a single timbre reference and a single emotion reference.
- CLI still targets the older runtime path and does not provide IndexTTS v2 multi-reference controls.
- README does not document the supported rollout behavior.

This rollout crosses runtime, interface, docs, and test boundaries. It also needs strong recovery artifacts because the user explicitly requires branch-scoped, document-backed progress that survives compaction or interruption.

## Goals / Non-Goals

**Goals:**
- Expose stable multi-reference timbre and emotion fusion as formal project behavior.
- Encode the selected supported defaults and supported fallbacks as named presets.
- Keep timbre and emotion multi-reference inputs separated at the public API boundary.
- Preserve legacy single-reference inference behavior when no multi-reference input is provided.
- Update Python runtime, WebUI, CLI, README, and tests together.
- Make rollout progress recoverable from repository documents.

**Non-Goals:**
- Expanding the public default path to unstable `ref_mel`-based fusion.
- Replacing `fusion_recipe` as the expert escape hatch.
- Exposing the full experiment matrix directly in the normal UI.
- Adding retraining, new models, or new external inference dependencies.

## Decisions

### 1. Add rollout presets instead of making raw branch configuration the primary public contract

The runtime will expose first-class convenience inputs for multi-reference timbre and emotion fusion and will internally build a `FusionRecipe` from supported presets.

Planned public shape:

- `speaker_references`
- `speaker_reference_weights`
- `speaker_fusion_mode`
- `emotion_references`
- `emotion_reference_weights`
- `emotion_fusion_mode`
- `fusion_recipe` remains supported for advanced callers

Rationale:
- The experiment has already selected defaults, so normal users should not need to author branch-level tensor settings.
- Presets let the project guarantee stable supported behavior while keeping `fusion_recipe` for expert workflows.
- This keeps timbre and emotion variables explicitly separated, which matches the experiment design and the user requirement.

Alternatives considered:
- Keep only `fusion_recipe`.
  Rejected because rollout would still depend on internal knowledge and remain too easy to misconfigure.
- Hide `fusion_recipe` entirely.
  Rejected because experiment and expert workflows still need it.

### 2. Encode timbre and emotion defaults as separate preset builders

The runtime will build the final `FusionRecipe` by composing a timbre preset and an emotion preset rather than using one monolithic preset identifier.

Planned supported presets:

- timbre:
  - `default` -> `spk_cond_emb + speech_conditioning_latent`, symmetric anchor, symmetric weights by default
  - `fallback` -> `speech_conditioning_latent`, symmetric anchor, symmetric weights by default
- emotion:
  - `default` -> anchor `A`, weighted sum, symmetric weights by default
  - `fallback` -> anchor `symmetric`, weighted sum, symmetric weights by default

Rationale:
- Timbre and emotion were evaluated separately and must remain independently controllable.
- The implementation already supports branch-specific configuration, so composition is lower risk than a second recipe system.

Alternatives considered:
- One combined preset string such as `recommended`.
  Rejected because it would couple timbre and emotion choices and hide the separated-variable requirement.

### 3. Keep the existing single-reference path as the zero-configuration default

If no multi-reference timbre or emotion inputs are supplied, inference will continue to behave as it does today.

Rationale:
- Backward compatibility is a hard rollout requirement.
- Existing code paths, examples, and user scripts must not break.

Alternatives considered:
- Always route every call through preset construction.
  Rejected because it adds unnecessary surface area to legacy calls and risks subtle regressions.

### 4. Productize supported presets across all current user-facing surfaces

The rollout will update all surfaces that currently expose inference behavior:

- runtime Python API
- `webui.py`
- `indextts/cli.py`
- `README.md`

WebUI will expose stable multi-reference controls directly. CLI will target `IndexTTS2` and expose repeatable reference flags plus preset selectors. README will document preset-driven usage first, with `fusion_recipe` documented as advanced usage.

Rationale:
- Leaving one surface behind would create inconsistent user expectations and support burden.
- The current CLI is explicitly outdated, so rollout is the right time to align it with the v2 runtime.

Alternatives considered:
- Update Python API only.
  Rejected because the user explicitly asked that existing features be updated together.

### 5. Validate with lightweight unit seams plus one end-to-end smoke path

Validation will be split into:

- unit tests for preset building, recipe composition, and backward compatibility
- surface tests for CLI/runtime invocation behavior
- one end-to-end smoke path in `.venv` using the current project environment

Rationale:
- Full real-model regression is expensive and brittle for every test, but rollout still needs at least one real execution path.
- The repository already has seams for fake runtime coverage in `tests/test_speaker_fusion_experiment.py`, which can be extended or mirrored.

Alternatives considered:
- Rely on unit tests only.
  Rejected because interface integration is part of the rollout scope.
- Make every test real-model.
  Rejected because test cost and environment sensitivity would be too high.

## Risks / Trade-offs

- [Preset API becomes ambiguous against `fusion_recipe`] -> Define precedence explicitly: raw `fusion_recipe` wins, preset args only build a recipe when `fusion_recipe` is absent.
- [WebUI multi-file UX becomes fragile on Windows] -> Prefer a simple primary file plus explicit extra-reference path input or multi-file input with documented constraints; test the parsing path carefully.
- [Legacy scripts depend on `spk_audio_prompt` as a list] -> Keep that shorthand working, but normalize it through the new preset builder and document it as compatibility behavior.
- [Rollout accidentally promotes unstable branches] -> Restrict supported presets to experiment-backed stable levels and leave unstable paths behind `fusion_recipe` or experimental flows.
- [CLI rewrite introduces behavior drift from WebUI/runtime] -> Make CLI call the same runtime convenience arguments instead of inventing a separate recipe parser.

## Migration Plan

1. Add preset-building helpers and argument normalization in `indextts/fusion.py`.
2. Extend `IndexTTS2.infer()` and `infer_generator()` to accept rollout-facing convenience inputs and compose recipes from presets.
3. Upgrade CLI to target `IndexTTS2` and reuse the new preset arguments.
4. Update WebUI to expose multi-reference timbre and emotion controls using the supported presets.
5. Add unit and end-to-end coverage in `.venv`.
6. Update README and rollout recovery docs.
7. Roll back by disabling preset construction and falling back to current single-reference behavior while leaving `fusion_recipe` support intact.

## Open Questions

- Should the first WebUI iteration prefer multi-file upload, path-list text input, or both for extra references?
- Should public custom weights be first-rollout scope, or should the supported rollout stay preset-only beyond equal weighting?
- Does CLI need raw JSON `fusion_recipe` passthrough in the first rollout, or is preset-driven support enough for now?
