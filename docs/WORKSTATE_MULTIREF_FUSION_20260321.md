# Multi-Reference Fusion Rollout Workstate

Last updated: 2026-03-21

## Current Objective

Productize the selected no-training fusion scheme for IndexTTS v2 so the repository formally supports:

- multi-timbre reference fusion
- multi-emotion reference fusion
- backward-compatible single-reference usage
- aligned runtime, UI, CLI, docs, and tests

## Recovery Anchor

Reload this document first after any compact or interruption, then continue from the referenced OpenSpec change and team charter.

- Team charter: `docs/TEAM_MULTIREF_FUSION_ROLLOUT.md`
- OpenSpec change: `openspec/changes/rollout-multiref-fusion-indextts-v2/`
- Prior experiment handoff: `docs/SPEAKER_FUSION_HANDOFF_20260319.md`
- Active branch: `multiref-fusion-rollout`
- Local Python: `.venv/bin/python`

## Active Mode

- Workflow mode: `parallel`
- Lead path:
  - document first
  - implement after rollout artifacts are complete
  - require design, development, and review participation in each iteration

## Confirmed Experiment Conclusions

### Timbre

- Supported default:
  - levels: `spk_cond_emb + speech_conditioning_latent`
  - weights: `0.5 / 0.5`
  - anchor: `symmetric`
- Supported fallback:
  - levels: `speech_conditioning_latent`

### Emotion

- Supported default:
  - recipe: `emotion_tensor_anchor_a`
  - weights: `0.5 / 0.5`
  - anchor: `A`
- Supported fallback:
  - recipe: `emotion_tensor_sym`

### Explicit Exclusion

- `ref_mel` and `style + ref_mel` remain excluded from default rollout.
- Rationale:
  - failures reproduced on CPU
  - instability is structural on short-text cases, not a transient GPU issue

## Current Assumptions

- Keep `fusion_recipe` as the expert interface, but add first-class rollout-facing convenience parameters.
- Keep timbre and emotion multi-reference variables separated in both API shape and tests.
- Update all current user-facing surfaces that already expose inference behavior:
  - Python runtime API
  - WebUI
  - CLI
  - README and related docs
- Use repo-root `.venv` for validation.
- Provide Windows-oriented local development guidance in repository docs, but implement and validate from the current environment.

## Current Blockers

- No hard technical blocker is active.
- Remaining ambiguity to resolve during design artifact writing:
  - whether public custom weights are first-rollout scope or only preset selection
  - which WebUI input pattern is safest for Windows users: repeatable file inputs or newline-delimited file paths
  - whether CLI exposes raw JSON recipe input in the first rollout or keeps presets only

## Pre-Coding Next Step

1. Finalize and commit team orchestration and workstate documents.
2. Write and commit rollout OpenSpec proposal, design, specs, and tasks.
3. Start implementation only after those artifacts are committed.

## Implementation Direction

- Runtime:
  - add preset builders for recommended and fallback timbre/emotion fusion
  - expose convenience arguments without breaking legacy calls
- Interfaces:
  - promote multi-reference timbre and emotion into formal WebUI and CLI inputs
- Validation:
  - add unit coverage for preset construction and backward compatibility
  - add one end-to-end smoke path under `.venv`

## Review Gate

Before merge or handoff, confirm:

- single-reference behavior still works
- multi-timbre and multi-emotion paths both work independently
- defaults match experiment conclusions
- tests pass
- documents can restore context without hidden chat state
