## ADDED Requirements

### Requirement: Speaker fusion evaluation SHALL report separate speaker-target and blend-target scores
The system SHALL evaluate speaker fusion results with separate scores for similarity to speaker A, similarity to speaker B, and similarity to the configured fused target representation, instead of collapsing them into a single speaker score.

#### Scenario: Evaluate a fused sample
- **WHEN** the system evaluates a generated sample from a speaker fusion case
- **THEN** it records similarity to each source speaker and to the target fused speaker representation as separate metrics

### Requirement: Evaluation SHALL reuse the current model stack for automatic scoring
The system SHALL use existing project models and features for automatic evaluation, including CAMPPlus-based speaker similarity, semantic stability signals, generation-failure counters, and runtime measurements.

#### Scenario: Run automatic scoring for one experiment case
- **WHEN** a generated sample is processed by the evaluation workflow
- **THEN** the system records speaker similarity metrics, semantic stability metrics, runtime metrics, and generation health indicators for that case

### Requirement: The evaluation workflow SHALL rank candidate fusion schemes
The system SHALL produce a ranked summary of fusion schemes and identify at least one default recommendation and one fallback recommendation.

#### Scenario: Rank a completed experiment set
- **WHEN** all experiment cases in a comparison set have been scored
- **THEN** the system aggregates case-level metrics by fusion scheme and outputs a ranked summary with recommended schemes

### Requirement: The first ranking pass SHALL be computed on the ten-batch open-source sample run
The system SHALL compute the first default and fallback recommendations from the complete full-combination results of the ten-batch open-source sample slice.

#### Scenario: Produce initial recommendations
- **WHEN** the ten-batch open-source full-combination run completes
- **THEN** the system derives the initial recommended fusion scheme and fallback scheme from that run's aggregate results

### Requirement: Listening review inputs SHALL be generated for final selection
The system SHALL generate a structured review set for manual listening on a fixed prompt subset before the default production scheme is finalized.

#### Scenario: Prepare listening review assets
- **WHEN** a comparison set reaches the review stage
- **THEN** the system exports the selected audio samples and their scheme metadata in a format that supports side-by-side listening review
