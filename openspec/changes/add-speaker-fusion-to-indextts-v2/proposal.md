## Why

IndexTTS2 already separates several speaker-related conditioning tensors during inference, but the public v2 path only accepts a single speaker reference and does not expose a controlled way to blend multiple speakers without retraining. We need a no-training speaker fusion workflow now so the project can support mixed timbre generation and systematically identify the best fusion recipe by running full-combination experiments on an initial ten-batch sample drawn from open-source datasets.

## What Changes

- Add a speaker fusion inference path for `indextts/infer_v2.py` that can blend multiple reference speakers without retraining or introducing new models.
- Decouple the current single `spk_audio_prompt` path into explicit fusion-ready conditioning branches for speaker semantic conditioning, prompt conditioning, global style, reference mel, and emotion reference conditioning.
- Expose override hooks and configuration for fusion-related tensors so experiments can cover all supported fusion levels instead of only a single heuristic.
- Add an experiment harness that enumerates full-combination fusion layer combinations, weights, experimental layers, and a representative branch-role profile set, then saves outputs and metadata for comparison.
- Add an initial evaluation dataset definition that samples ten experiment batches from open-source speech datasets for the first large comparison run.
- Add evaluation and ranking outputs that score each experiment setting against blend-target speaker similarity, generation stability, and operational cost, then recommend a default scheme for implementation.
- Harden cache behavior so multi-reference and weighted fusion runs do not reuse stale cached tensors.

## Capabilities

### New Capabilities
- `speaker-fusion-inference`: Run IndexTTS2 with multi-reference speaker fusion across all supported conditioning branches without retraining or adding new models.
- `speaker-fusion-experiments`: Generate full-combination speaker fusion experiments that cover layer choices, branch combinations, weights, anchor/reference assignments, and the first ten-batch open-source sample run.
- `speaker-fusion-evaluation`: Produce comparable metrics and ranked summaries so the project can choose a default speaker fusion strategy from the experiment matrix.

### Modified Capabilities

## Impact

- Affected code: `indextts/infer_v2.py`, `indextts/gpt/model_v2.py`, `indextts/s2mel/modules/commons.py`, `indextts/s2mel/modules/flow_matching.py`, and any new experiment utilities under `tools/` or `tests/`.
- Affected APIs: Python inference API for `IndexTTS2`, possible CLI or batch experiment entrypoints, and cache key behavior for prompt-conditioned inference.
- Dependencies: no new model dependencies are intended; reuse existing `SeamlessM4TFeatureExtractor`, semantic codec, CAMPPlus, GPT, CFM, and BigVGAN components.
- Systems: inference pipeline, experiment workflow, evaluation outputs, and artifact/report storage for fusion runs.
