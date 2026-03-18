## Context

`IndexTTS2` currently exposes a single `spk_audio_prompt` input in [`indextts/infer_v2.py`](K:/workspace/index-tts/indextts/infer_v2.py). Internally, that one reference audio is expanded into several distinct speaker-related conditions:

- `spk_cond_emb`: W2V-BERT hidden-state sequence used by GPT speaker conditioning
- `speech_conditioning_latent`: compressed GPT conditioning latent derived from `spk_cond_emb`
- `prompt_condition`: semantic prompt condition derived from `S_ref`
- `style`: CAMPPlus global style vector
- `ref_mel`: reference mel prompt used by the CFM decoder

The v2 inference path does not use a speaker-conditioned BigVGAN. Speaker identity is folded into the generated mel before the vocoder stage, so speaker fusion has to happen upstream. The project constraint is strict: speaker fusion must be achieved at inference time only, without retraining and without introducing extra models.

The user also wants full combination coverage rather than a staged pruning strategy for the first pass. That means the design must support both production-safe fusion paths and high-risk exploratory paths, while keeping cache behavior correct and making experiment outputs comparable on an initial ten-batch sample drawn from open-source datasets. The current direction keeps full coverage over fusion layers, weights, and anchor modes, but limits branch-role variation to a representative profile set so the matrix does not explode on branch permutations alone.

## Goals / Non-Goals

**Goals:**
- Support multi-reference speaker fusion in IndexTTS2 without retraining or new model dependencies.
- Expose all meaningful fusion levels that already exist in the current architecture.
- Separate production-safe fusion levels from experimental high-risk levels.
- Add an experiment harness that can cover the full Cartesian combination of configured fusion layers, operators, weights, and anchor modes with reproducible metadata.
- Rank candidate fusion strategies with objective metrics available from the current stack and with structured listening-review inputs.
- Keep the existing single-speaker inference path intact when fusion is not requested.
- Define an initial ten-batch evaluation slice sampled from open-source datasets before any broader rollout.

**Non-Goals:**
- Replacing the v2 vocoder with the older speaker-conditioned BigVGAN path.
- Training new speaker encoders, fusion adapters, or routing models.
- Solving generalized voice conversion beyond the current IndexTTS2 inference stack.
- Fully automating perceptual quality judgment without any listening review.

## Decisions

### 1. Treat fusion as explicit tensor overrides, not as implicit audio mixing

Speaker fusion will be implemented by exposing override-ready conditioning tensors in `IndexTTS2` rather than by relying on waveform-level mixing. The core API will separate reference roles so the inference path can independently control:

- speaker semantic branch: `spk_cond_emb` or `speech_conditioning_latent`
- prompt semantic branch: `prompt_condition`
- decoder style branch: `style`
- decoder prompt mel branch: `ref_mel`

Rationale:
- These tensors already exist in the v2 path and correspond to real conditioning boundaries in the current architecture.
- Tensor-level fusion is deterministic, inspectable, and compatible with cache reuse.
- Waveform mixing is still useful as an exploratory baseline, but it is too noisy to be the primary implementation strategy.

Alternatives considered:
- Only mix waveform references before feature extraction.
  Rejected because it entangles speaker, prosody, noise, and content.
- Only expose one “fusion_alpha” on top of the existing single-reference API.
  Rejected because different branches encode different speaker cues and need independent experiments.

### 2. Split fusion levels into supported and experimental tiers

The implementation will explicitly classify fusion levels:

- Supported tier:
  - `spk_cond_emb`
  - `speech_conditioning_latent`
  - `prompt_condition`
  - `style`
  - `ref_mel`
- Experimental tier:
  - waveform or pre-feature reference mixing
  - `cat_condition`
  - `vc_target`

Rationale:
- The supported tier maps to stable, pre-existing boundaries in the inference graph.
- The experimental tier is still “fusion-capable” and should be covered because the user asked for broad coverage, but it should not contaminate the default production path.

Alternatives considered:
- Excluding high-risk layers completely.
  Rejected because it would leave the experiment space incomplete.
- Treating every layer equally in the default implementation.
  Rejected because high-risk layers can destabilize synthesis and debugging.

### 3. Use role-based multi-reference inputs instead of a single fused prompt

The fusion API will distinguish reference roles rather than assuming one list of speakers must be fused identically for every branch. The design will support:

- a shared multi-reference list for simple cases
- per-branch overrides for `speaker`, `prompt`, `style`, `ref_mel`, and `emotion`
- anchor assignment when alignment-sensitive branches require a dominant timing reference

Rationale:
- `style` is a global vector and tolerates symmetric weighted fusion.
- `prompt_condition` and `ref_mel` are sequence-conditioned and often need one anchor reference for alignment.
- `spk_cond_emb` and `speech_conditioning_latent` can be fused symmetrically or asymmetrically depending on experiment mode.
- `emo_cond_emb` is also a sequence-conditioned hidden-state tensor and can reuse the same interpolation and anchor rules as the speaker semantic branch without adding new models.

Alternatives considered:
- Force every branch to use the same weight vector and same anchor.
  Rejected because it hides branch-specific optimal settings.

### 4. Standardize fusion operators per layer instead of using one generic operator everywhere

Fusion operators will be chosen per layer:

- global vectors (`style`): weighted linear interpolation
- latent token sequences (`speech_conditioning_latent`): weighted token-wise interpolation
- frame sequences (`spk_cond_emb`, `prompt_condition`, `ref_mel`): align-to-anchor, then weighted interpolation
- exploratory post-conditions (`cat_condition`, `vc_target`): shape-safe interpolation behind explicit experimental flags

Rationale:
- Sequence tensors have alignment constraints that global vectors do not.
- A single generic “average tensors” rule would create silent mismatches and poor results.

Alternatives considered:
- Pool everything to one global vector before fusion.
  Rejected because it would throw away prompt and temporal detail that likely matters for timbre preservation.

### 5. Make full-combination coverage the default experiment strategy

The initial experiment workflow will use full-combination enumeration over the configured experiment domain, with a bounded dataset slice rather than staged candidate pruning:

- fixed fusion domain:
  - supported tier: `spk_cond_emb`, `speech_conditioning_latent`, `prompt_condition`, `style`, `ref_mel`
  - experimental tier: waveform mixing, `cat_condition`, `vc_target`
  - first-pass configured run: enable both supported and experimental tier combinations
- fixed parameter domain:
  - configured weights such as `{0.2, 0.35, 0.5, 0.65, 0.8}`
  - anchor modes `{A, B, symmetric}` where applicable
  - a representative branch-role profile set for speaker, prompt, style, reference mel, and emotion
- fixed baseline set:
  - speaker A only
  - speaker B only
  - waveform-mix baseline
  - self-style baseline
- fixed initial dataset slice:
  - ten batches sampled from one or more open-source speech datasets
  - each batch records source dataset, speaker ids, prompt ids, text ids, and split identity

Rationale:
- The user explicitly asked for full-combination testing first, not early pruning.
- Bounding the first pass by ten sampled batches gives wide recipe coverage without hiding any fusion layer combinations.
- Reducing branch-role coverage to a curated representative profile set keeps the first pass focused on branch sensitivity instead of spending most cases on near-duplicate role permutations.
- Emotion-reference fusion should follow the same multi-reference mechanics as speaker-reference fusion, but it is modeled as an additional branch-role dimension rather than a separate layer-combination axis to avoid doubling the matrix size for every supported-tier subset.
- This keeps the first decision grounded in data instead of manual pre-selection.

Alternatives considered:
- One fixed hand-picked combination.
  Rejected because it does not answer the user’s “cover everything” request.
- Stage-wise pruning before any full pass.
  Rejected because it would bias the result space before the first broad measurement.

### 6. Evaluation must use existing project models plus structured listening review

The evaluation pipeline will avoid new model dependencies and reuse components already present in the stack:

- speaker similarity:
  - CAMPPlus embedding cosine against speaker A
  - CAMPPlus embedding cosine against speaker B
  - cosine against the target fused embedding
- semantic stability:
  - W2V-BERT semantic feature similarity across generated samples for the same text
  - generation failure and truncation counters
- decoder stability:
  - NaN, silence, overlong, or severe collapse detection
- runtime cost:
  - total inference time and per-stage timing
- human review:
  - listening sheet over a fixed prompt subset for naturalness, blend plausibility, and artifact severity

Rationale:
- These metrics are available without changing the model stack.
- Purely automatic ranking is not trustworthy for timbre blend quality.
- The first comparison set must be produced on the same ten-batch open-source sample slice so scheme rankings are comparable.

Alternatives considered:
- Adding ASR or new speaker verification models.
  Rejected for now because the change explicitly avoids new model dependencies.

### 7. Cache keys must include fusion recipe identity

Current caches are path-based and assume one prompt per branch. Multi-reference fusion will extend cache identity to include:

- normalized reference file identity
- branch role
- fusion level
- fusion weights
- anchor mode
- preprocessing options affecting tensor content

Rationale:
- Without richer cache keys, fusion experiments will reuse stale tensors and produce invalid comparisons.

Alternatives considered:
- Disable all caches during fusion.
  Rejected because broad experiment coverage would become unnecessarily slow.

## Risks / Trade-offs

- [Sequence alignment drift in `spk_cond_emb`, `prompt_condition`, or `ref_mel`] → Use anchor-based alignment, record the alignment mode in metadata, and compare against vector-only baselines.
- [High-risk layers produce unstable or muddy audio] → Keep `cat_condition`, `vc_target`, and waveform mixing under an explicit experimental flag and exclude them from default recommendations unless they clearly win.
- [Cache contamination across experiment runs] → Expand cache keys and add validation logs for cache hits and misses.
- [Objective metrics prefer one speaker too strongly] → Rank with multi-objective scoring and require listening review before promoting a default fusion scheme.
- [Experiment matrix becomes too large] → Use staged pruning, fixed prompt subsets, and resumable experiment manifests.
- [Experiment matrix becomes too large] → Keep the recipe domain exhaustive but bound the first pass to ten sampled open-source batches, support resumable manifests, and shard execution by batch.
- [Production API becomes difficult to use] → Keep a simple top-level API that accepts a default fusion recipe while exposing advanced branch overrides only for expert use.

## Migration Plan

1. Add fusion-ready internal extraction helpers so single-reference inference and fused inference share the same tensor-building code path.
2. Add supported-tier branch overrides and cache-key expansion while preserving current behavior when no fusion recipe is provided.
3. Add experiment manifest generation, full-combination batched execution, metadata capture, and result storage.
4. Add open-source dataset sampling and generate the first ten-batch evaluation slice.
5. Add evaluation scripts and ranked report generation for the ten-batch full-combination run.
6. Promote one default supported-tier fusion recipe after reviewing the ten-batch results, while keeping experimental-tier layers opt-in.
7. Roll back by disabling the fusion recipe path and falling back to the existing single-reference inference path if instability is detected.

## Open Questions

- Should the first production default use symmetric speaker fusion or anchored fusion with a dominant speaker and a secondary speaker?
- How aggressive should sequence alignment be for `spk_cond_emb` and `prompt_condition` in the first implementation: trim-to-min, interpolate-to-anchor, or attention-based alignment without training?
- Should experimental-tier results be stored with the same report format as supported-tier runs, or separated to avoid accidental promotion?
- Which open-source datasets should seed the first ten-batch slice while keeping language, gender, and recording-condition diversity balanced enough for fusion comparisons?
