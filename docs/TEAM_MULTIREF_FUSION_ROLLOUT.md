# Multi-Reference Fusion Rollout Team Charter

Last updated: 2026-03-21

## Objective

Roll out the experimentally selected no-training fusion scheme for IndexTTS v2 into the main project surfaces so the project supports:

- multi-timbre reference fusion
- multi-emotion reference fusion
- backward-compatible single-reference inference
- synchronized updates across runtime API, WebUI, CLI, docs, unit tests, and end-to-end validation

## Active Mode

- Mode: `parallel`
- Lead agent: `Codex`
- Change branch: `multiref-fusion-rollout`
- OpenSpec change: `rollout-multiref-fusion-indextts-v2`
- Python entrypoint for local work: `.venv/bin/python`

## Team Roles

### Design

- Owner: design workstream
- Responsibility:
  - define rollout-level user-facing behavior
  - convert experiment conclusions into supported defaults
  - define public input shapes for multi-reference timbre and emotion control
  - keep backward compatibility explicit
  - define documentation and recovery artifacts required before coding
- Primary write scope:
  - OpenSpec proposal and design artifacts
  - rollout-facing product docs

### Development

- Owner: development workstream
- Responsibility:
  - implement runtime and interface changes
  - keep existing single-reference behavior intact
  - update WebUI, CLI, and helper utilities
  - add and maintain unit and end-to-end tests
- Primary write scope:
  - `indextts/`
  - `webui.py`
  - `tools/`
  - `tests/`

### Review

- Owner: review workstream
- Responsibility:
  - challenge assumptions before merge
  - identify regressions, unstable defaults, and API ambiguity
  - verify that design, implementation, and tests stay aligned
  - gate iteration completion on concrete findings or an explicit no-finding result
- Primary write scope:
  - review notes
  - validation reports

## Shared Mental Model

- Stop condition:
  - multi-reference timbre and emotion fusion are both exposed through formal project interfaces
  - recommended defaults reflect experiment results
  - tests pass in `.venv`
  - at least one end-to-end validation path runs successfully
  - all rollout decisions are recoverable from repository documents
- Non-goals for this rollout:
  - retraining models
  - promoting unstable `ref_mel`-based recipes to default behavior
  - changing the selected experimental recommendation into the supported default path

## Selected Defaults From Experiments

- Timbre default:
  - levels: `spk_cond_emb + speech_conditioning_latent`
  - weights: `0.5 / 0.5`
  - anchor: `symmetric`
- Timbre fallback:
  - levels: `speech_conditioning_latent`
- Emotion default:
  - recipe: `emotion_tensor_anchor_a`
  - weights: `0.5 / 0.5`
  - anchor: `A`
- Emotion fallback:
  - recipe: `emotion_tensor_sym`

## Coordination Rules

- The lead agent owns final writes for shared artifacts and final integration.
- Each iteration cycle must include all three roles:
  - design produces or updates intended behavior
  - development implements against that behavior
  - review signs off or returns findings
- Handoffs must include:
  - target files
  - assumptions
  - blockers
  - expected verification
- No workstream may revert unrelated user changes.

## Compact Recovery Protocol

If the session is compacted or interrupted, reload context in this order:

1. `docs/TEAM_MULTIREF_FUSION_ROLLOUT.md`
2. `docs/WORKSTATE_MULTIREF_FUSION_20260321.md`
3. `openspec/changes/rollout-multiref-fusion-indextts-v2/`
4. `docs/SPEAKER_FUSION_HANDOFF_20260319.md`

Restore:

- active mode
- branch
- selected defaults
- current assumptions
- blockers
- next intended step

## Iteration Gate

An iteration is complete only when all three roles have participated and the lead agent has recorded:

- design decision or unchanged design confirmation
- implementation delta
- review findings or explicit no-finding result
- test/validation outcome
