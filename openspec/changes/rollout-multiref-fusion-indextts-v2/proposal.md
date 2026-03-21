## Why

The no-training fusion experiment for IndexTTS v2 is complete and the default schemes have converged, but the project still exposes fusion mainly as an experimental runtime hook. We need to promote the selected timbre and emotion fusion strategies into formal project surfaces now so users can use multi-reference synthesis without hand-authoring raw recipes or relying on internal experiment knowledge.

## What Changes

- Add rollout-facing multi-reference convenience inputs for IndexTTS v2 inference while preserving the existing single-reference path.
- Encode the selected supported defaults for timbre and emotion fusion, plus explicit fallback presets, without promoting unstable `ref_mel` routes into the default behavior.
- Upgrade current user-facing surfaces so Python API, WebUI, CLI, and README all support and document multi-timbre and multi-emotion reference fusion consistently.
- Add validation coverage for preset construction, backward compatibility, and end-to-end multi-reference execution.
- Add rollout recovery artifacts so future compacted sessions can restore current assumptions, blockers, and next steps from repository documents.

## Capabilities

### New Capabilities
- `multiref-fusion-runtime`: Formal runtime support for recommended and fallback multi-reference timbre and emotion fusion presets in IndexTTS v2.
- `multiref-fusion-interfaces`: Consistent multi-reference fusion exposure across Python API, WebUI, CLI, and rollout documentation.
- `multiref-fusion-validation`: Verification coverage for backward compatibility, preset correctness, and end-to-end multi-reference synthesis.

### Modified Capabilities

## Impact

- Affected code: `indextts/fusion.py`, `indextts/infer_v2.py`, `indextts/cli.py`, `webui.py`, `tests/`, `README.md`, and rollout documentation under `docs/` and `openspec/`.
- Affected APIs: `IndexTTS2.infer()` and `infer_generator()` public inputs, CLI flags, and WebUI controls for timbre and emotion references.
- Dependencies: no new model dependency is intended; rollout must reuse the existing inference stack and local `.venv`.
- Systems: runtime inference, user-facing interaction surfaces, documentation/recovery flow, and validation workflows.
