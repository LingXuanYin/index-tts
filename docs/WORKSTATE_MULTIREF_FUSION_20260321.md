# Multi-Reference Fusion Rollout Workstate

Last updated: 2026-03-21 (rollout implementation complete)

## Current Objective

Productize the selected no-training fusion scheme for IndexTTS v2 so the repository formally supports:

- multi-timbre reference fusion
- multi-emotion reference fusion
- backward-compatible single-reference usage
- aligned runtime, UI, CLI, docs, and tests

## Recovery Anchor

Reload this document first after any compact or interruption, then continue from the referenced OpenSpec change and team charter.

- Team charter: `TEAM_MULTIREF_FUSION_ROLLOUT.md`
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
- Remaining non-blocking follow-up:
  - the first WebUI rollout uses newline-delimited local file paths for extra references; this is documented and intentionally chosen over raw JSON editing
  - CLI remains preset-based for the supported rollout path; advanced custom recipes stay Python-only

## Current Status

- Rollout-facing preset builders are implemented in `indextts/fusion.py`.
- `IndexTTS2.infer()` and `infer_generator()` accept supported multi-reference speaker and emotion arguments while preserving explicit `fusion_recipe` precedence.
- CLI now runs `IndexTTS2` and exposes supported multi-reference flags.
- WebUI exposes supported multi-reference timbre and emotion controls.
- README, Chinese docs, and a root-level Windows local development note are updated.
- Rollout unit and interface-smoke tests pass in `.venv`.

## Implementation Direction

- Runtime:
  - add preset builders for recommended and fallback timbre/emotion fusion
  - expose convenience arguments without breaking legacy calls
- Interfaces:
  - promote multi-reference timbre and emotion into formal WebUI and CLI inputs
- Validation:
  - add unit coverage for preset construction and backward compatibility
  - add one end-to-end smoke path under `.venv`

## Validation Result

Executed with `.venv/bin/python`:

```bash
.venv/bin/python -m pytest tests/test_fusion_rollout.py tests/test_cli_v2.py tests/test_infer_v2_rollout_smoke.py tests/test_speaker_fusion_experiment.py tests/test_flow_matching.py -q
```

Result:

- `13 passed`

CPU-only smoke validation also completed under the resource cap:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=96 MKL_NUM_THREADS=96 OPENBLAS_NUM_THREADS=96 NUMEXPR_NUM_THREADS=96 \
VECLIB_MAXIMUM_THREADS=96 RAYON_NUM_THREADS=96 \
taskset -c 0-95 prlimit --as=51539607552 -- \
.venv/bin/python -m indextts.cli "This is a CPU multi reference rollout smoke test." \
  --voice data/open_source/cmu_arctic/ARCTIC/cmu_us_lnh_arctic/wav/arctic_a0457.wav \
  --voice-ref data/open_source/cmu_arctic/ARCTIC/cmu_us_lnh_arctic/wav/arctic_a0526.wav \
  --emotion data/open_source/cmu_arctic/ARCTIC/cmu_us_lnh_arctic/wav/arctic_b0285.wav \
  --emotion-ref data/open_source/cmu_arctic/ARCTIC/cmu_us_lnh_arctic/wav/arctic_b0412.wav \
  --speaker-fusion-mode default \
  --emotion-fusion-mode default \
  --device cpu \
  --output_path artifacts/rollout_smoke/cli_multiref_smoke_cpu.wav \
  --force
```

CPU smoke outcome:

- generated `artifacts/rollout_smoke/cli_multiref_smoke_cpu.wav`
- total inference time: `59.50s`
- generated audio length: `4.11s`
- RTF: `14.4780`

## Review Gate

Before merge or handoff, confirm:

- single-reference behavior still works
- multi-timbre and multi-emotion paths both work independently
- defaults match experiment conclusions
- tests pass
- documents can restore context without hidden chat state

Current gate result:

- all checks satisfied for this rollout iteration
