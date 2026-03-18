## ADDED Requirements

### Requirement: Multi-reference speaker fusion SHALL be available at supported conditioning levels
The system SHALL allow IndexTTS2 inference to blend multiple speaker references without retraining or adding new models. The supported conditioning levels SHALL include `spk_cond_emb`, `speech_conditioning_latent`, `prompt_condition`, `style`, and `ref_mel`.

#### Scenario: Run fusion with supported-tier branch selection
- **WHEN** a fusion request specifies two or more reference speakers and selects one or more supported conditioning levels
- **THEN** the system produces speaker-conditioned tensors for each selected level and applies the configured fusion operator before synthesis

#### Scenario: Use different branch roles in one request
- **WHEN** a fusion request assigns different reference-role settings to speaker semantic, prompt semantic, style, reference mel, or emotion branches
- **THEN** the system honors those branch-specific settings in a single synthesis call

#### Scenario: Fuse multiple emotion references without new models
- **WHEN** a fusion request provides two or more emotion references for the emotion branch
- **THEN** the system derives `emo_cond_emb` from those references with the configured anchor and weighting rules and reuses the existing emotion-conditioning path without retraining or adding extra models

### Requirement: Experimental fusion levels SHALL be isolated from the default path
The system SHALL classify waveform mixing, `cat_condition`, and `vc_target` fusion as experimental and SHALL require explicit opt-in before using them.

#### Scenario: Experimental level is not requested
- **WHEN** a synthesis request omits the experimental flag
- **THEN** the system excludes experimental fusion levels from tensor construction and synthesis

#### Scenario: Experimental level is requested
- **WHEN** a synthesis request explicitly enables experimental fusion levels
- **THEN** the system includes those levels in the run metadata and applies them only to that request

### Requirement: Single-speaker inference SHALL remain backward compatible
The system SHALL preserve current IndexTTS2 behavior when no fusion recipe is provided.

#### Scenario: Legacy inference call without fusion
- **WHEN** a caller uses the existing single-speaker arguments without any fusion configuration
- **THEN** the system follows the existing v2 inference path and produces a single-speaker output

### Requirement: Fusion-aware caches SHALL not reuse stale tensors
The system SHALL key cached speaker-related tensors by fusion-relevant identity, including reference inputs, branch role, fusion level, weights, and anchor mode.

#### Scenario: Two runs use different fusion recipes
- **WHEN** two synthesis requests differ in any fusion-relevant field
- **THEN** the system treats them as distinct cache identities and does not reuse stale fused tensors across runs

#### Scenario: Two runs reuse the same fusion recipe
- **WHEN** two synthesis requests use the same normalized fusion recipe
- **THEN** the system may reuse cached tensors for matching branches
