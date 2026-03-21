## ADDED Requirements

### Requirement: Rollout preset construction SHALL be covered by automated tests
The system SHALL include automated tests for the supported multi-reference timbre and emotion rollout presets.

#### Scenario: Validate supported preset construction
- **WHEN** the automated unit suite runs
- **THEN** it verifies the selected supported defaults and fallbacks for timbre and emotion, including levels, anchors, and compatibility behavior

### Requirement: Backward compatibility SHALL be covered by automated tests
The system SHALL include automated tests proving that the legacy single-reference `IndexTTS2` path still works when rollout-facing multi-reference inputs are absent.

#### Scenario: Validate legacy call compatibility
- **WHEN** the automated unit suite runs against the rollout runtime surface
- **THEN** it verifies that legacy single-reference calls remain valid and do not require rollout-specific arguments

### Requirement: Supported multi-reference execution SHALL have an end-to-end smoke path
The system SHALL provide an end-to-end smoke validation path for supported multi-reference synthesis in the project virtual environment.

#### Scenario: Run the multi-reference smoke path
- **WHEN** the rollout validation workflow executes the smoke path from `.venv`
- **THEN** it proves that supported multi-reference timbre and emotion synthesis can run end to end and emit output metadata
