## Context

The completed no-training fusion experiment already selected stable rollout defaults for `IndexTTS2`:

- timbre default: `spk_cond_emb + speech_conditioning_latent`
- timbre fallback: `speech_conditioning_latent`
- emotion default: `emotion_tensor_anchor_a`
- emotion fallback: `emotion_tensor_sym`

The current repository already contains the low-level fusion runtime in `indextts/fusion.py` and the supporting tensor override path in `indextts/infer_v2.py`, but the public project surfaces still behave as if fusion were an internal or experiment-only feature. In practice, the gap is now productization rather than model research:

- Python callers still have to understand raw `fusion_recipe` details to use the selected defaults explicitly.
- `webui.py` still presents single-reference speaker flow.
- `indextts/cli.py` still targets the older `IndexTTS` entrypoint and has no formal `IndexTTS2` multi-reference path.
- project docs describe single-reference usage, not the supported default multi-reference workflow.
- the existing tests focus on experiment tooling, not the rollout-facing API and interfaces.

This rollout must keep timbre and emotion variables separated, preserve current single-reference behavior, and avoid promoting unstable `ref_mel` routes into the supported default path. The local development convention for this work is repo-root `.venv`; Windows-oriented developer guidance should be documented, but the current validation environment remains the checked-out workspace.

Current execution snapshot before implementation:

- active mode: `parallel`
- branch: `multiref-fusion-rollout`
- blockers: no hard blocker; interface ergonomics still need to be fixed in design
- next intended step: finish rollout artifacts, then implement runtime, interface, docs, and validation changes

## Goals / Non-Goals

**Goals:**

- add rollout-facing convenience inputs for multi-timbre and multi-emotion reference fusion without requiring users to hand-author a full `fusion_recipe`
- keep raw `fusion_recipe` support for advanced users and for backwards-compatible access to existing fusion internals
- encode the selected default and fallback presets directly in reusable helper builders
- update current project surfaces that already expose inference behavior: Python API, CLI, WebUI, README, and Chinese docs
- add unit and smoke-level end-to-end validation for single-reference compatibility and the new multi-reference presets
- keep recovery state in tracked documents so future compacted sessions can resume safely

**Non-Goals:**

- re-running the fusion search or changing the experimentally selected defaults during this rollout
- promoting `ref_mel`-based supported defaults into the formal supported path
- introducing new model dependencies, retraining, or external serving infrastructure
- exposing the full experiment matrix through the first-rollout CLI or WebUI

## Decisions

### 1. Add convenience preset builders on top of `fusion_recipe`

The runtime will keep `fusion_recipe` as the expert-level override, but rollout-facing APIs will add higher-level parameters that build the supported recipe automatically from the selected defaults or fallback presets.

The first formal surface will expose separate convenience concepts:

- timbre multi-reference inputs
- emotion multi-reference inputs
- preset selectors for each axis

Rationale:

- users need a supported path that reflects experiment conclusions directly
- existing fusion internals are already expressive enough and should not be duplicated
- keeping the builder layer thin reduces regression risk and preserves advanced access

Alternatives considered:

- expose only raw `fusion_recipe`
  rejected because the rollout goal is to remove experiment knowledge from ordinary usage
- remove `fusion_recipe` and support only presets
  rejected because it would regress advanced use and internal tooling flexibility

### 2. Keep timbre and emotion axes separate in both API shape and recipe construction

The convenience API will not accept one undifferentiated multi-reference list. Instead it will distinguish:

- speaker/timbre references for timbre fusion
- emotion references for emotion fusion

If only timbre references are provided, the emotion path remains legacy-compatible unless explicitly overridden. If only emotion references are provided, the speaker path remains legacy-compatible.

Rationale:

- the experiment design explicitly separated variables
- the default timbre and emotion schemes differ in levels, anchor policy, and fallback behavior
- separate inputs make tests and docs easier to reason about

Alternatives considered:

- one merged reference list with branch auto-routing
  rejected because it hides the variable boundary the experiment depended on

### 3. Encode the selected defaults in `indextts.fusion` as named supported presets

The selected rollout schemes will be encoded as preset builder helpers in `indextts.fusion` so all interfaces can reuse the same source of truth. The preset layer will cover:

- timbre default and fallback
- emotion default and fallback
- construction of a combined supported recipe when both axes are present

Rationale:

- centralizing the preset logic prevents WebUI, CLI, and Python API drift
- the experiment conclusions already define the supported recipes precisely enough
- tests can target one builder layer instead of duplicating recipe JSON assembly in many places

Alternatives considered:

- embed preset JSON inline in each interface
  rejected because it would fragment behavior and make review harder

### 4. Roll out multi-reference support across current interfaces, but constrain the first UI/CLI surface

The WebUI and CLI will expose supported multi-reference usage through simple inputs, but the first rollout will prefer bounded ergonomics over full arbitrary recipe editing. Advanced `fusion_recipe` usage remains a Python-level path.

For the WebUI, the safest first rollout is additive controls that preserve the existing single-reference flow and allow optional multi-reference entry without replacing the current layout. For the CLI, the first rollout should prefer repeated reference arguments and preset flags rather than raw JSON input.

Rationale:

- public interfaces need a stable supported path, not a thin shell over internal JSON structures
- the WebUI must remain usable for current users
- repeated arguments are more robust than shell-escaped JSON on Windows

Alternatives considered:

- raw JSON recipe input in CLI and WebUI
  rejected for first rollout because it is fragile and exposes internal complexity
- WebUI-only rollout with no CLI update
  rejected because the user explicitly required current functionality surfaces to stay aligned

### 5. Add two validation layers: pure unit tests and one runtime smoke path

Validation will be split into:

- unit tests for preset builders, compatibility rules, and interface argument normalization
- an end-to-end smoke test path for `IndexTTS2` multi-reference execution using the project virtual environment and local assets

The smoke path should remain small and targeted. It does not need to be a full model-quality benchmark, but it must prove the supported path executes and records metadata.

Rationale:

- unit tests catch construction and compatibility regressions cheaply
- a smoke path is necessary because the rollout crosses runtime, interfaces, and metadata output

Alternatives considered:

- unit tests only
  rejected because the runtime path is too cross-cutting to trust without at least one integration path
- full experimental rerun as rollout validation
  rejected because the experiment phase is already complete and this would slow productization unnecessarily

## Risks / Trade-offs

- [Convenience inputs accidentally change legacy single-reference behavior] -> keep `fusion_recipe=None` semantics intact and add explicit compatibility tests for unchanged single-reference calls.
- [Preset builders drift from experiment conclusions] -> centralize preset construction in `indextts.fusion` and assert the selected levels, anchors, and weights in tests.
- [WebUI multi-reference inputs make the UI confusing or brittle] -> add new controls as optional advanced inputs while leaving the current single-reference interaction intact.
- [CLI design becomes hostile for Windows shells] -> prefer repeated path arguments and preset flags over JSON blobs.
- [End-to-end tests become flaky because they depend on full model assets] -> keep the smoke test small, use existing local assets/checkpoints, and separate smoke execution from fast unit coverage.
- [Untracked recovery state causes future compacted sessions to lose context] -> keep active mode, assumptions, blockers, and next step in tracked rollout documents and OpenSpec artifacts.

## Migration Plan

1. Add preset builders and argument normalization for supported timbre and emotion multi-reference rollout behavior.
2. Update `IndexTTS2` inference entrypoints to accept convenience inputs and translate them into the centralized preset builders.
3. Update CLI and WebUI to expose the supported multi-reference path while preserving current single-reference use.
4. Update README and Chinese documentation with supported examples and local development notes for the rollout.
5. Add unit tests and an end-to-end smoke workflow in `.venv`.
6. Run validation, capture review findings, and push the rollout branch.
7. Roll back by disabling the convenience layer and falling back to legacy single-reference usage; raw `fusion_recipe` support remains available for controlled debugging if needed.

## Open Questions

- Should the first rollout CLI expose the fallback preset selectors directly, or only the recommended default path unless a flag is set?
- For the WebUI, is a small fixed number of optional extra audio inputs preferable to a free-form text path list for the first supported release?
- Should the end-to-end smoke test live in `tests/` as an opt-in marker or as a dedicated root-level validation script invoked from the virtual environment?
