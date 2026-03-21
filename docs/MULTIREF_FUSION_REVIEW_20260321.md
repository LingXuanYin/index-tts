# Multi-Reference Fusion Rollout Iteration Review

Date: 2026-03-21

## Design

- locked the supported rollout defaults to the experiment-selected schemes only
- kept timbre and emotion variables separated in runtime inputs and tests
- kept `fusion_recipe` as the expert override path
- chose preset-based CLI and newline-delimited local path input for WebUI instead of raw JSON editing

## Development

- centralized supported rollout recipe building in `indextts/fusion.py`
- extended `IndexTTS2` rollout-facing inference inputs in `indextts/infer_v2.py`
- upgraded `indextts/cli.py` to the `IndexTTS2` path with supported multi-reference flags
- added WebUI controls for supported multi-reference timbre and emotion references
- updated English and Chinese documentation plus Windows local development guidance
- added rollout-focused automated tests

## Review

Review focus:

- backward compatibility of the single-reference path
- consistency between experiment conclusions and supported defaults
- avoidance of accidental `ref_mel` default promotion
- interface clarity for CLI and WebUI
- minimum automated validation for rollout entrypoints

Review result:

- no blocking findings
- supported defaults remain aligned with the completed experiments
- rollout builder now scopes timbre fusion to the `speaker` branch instead of leaking merged references into prompt/style/ref_mel defaults
- CLI and WebUI expose supported presets without exposing raw internal recipe editing

## Validation Evidence

Command:

```bash
.venv/bin/python -m pytest tests/test_fusion_rollout.py tests/test_cli_v2.py tests/test_speaker_fusion_experiment.py tests/test_flow_matching.py -q
```

Outcome:

- `12 passed`
