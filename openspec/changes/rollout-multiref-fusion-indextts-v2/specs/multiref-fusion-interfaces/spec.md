## ADDED Requirements

### Requirement: Python inference entrypoints SHALL expose supported multi-reference usage
The system SHALL expose rollout-facing multi-reference timbre and emotion usage through the public `IndexTTS2` inference entrypoints.

#### Scenario: Use supported timbre and emotion presets from Python
- **WHEN** a Python caller invokes `IndexTTS2.infer()` or `infer_generator()` with rollout-facing multi-reference timbre and emotion inputs
- **THEN** the system accepts those inputs and executes synthesis without requiring the caller to build raw recipe JSON manually

### Requirement: The CLI SHALL expose supported IndexTTS2 multi-reference fusion
The system SHALL provide a supported CLI path for IndexTTS2 multi-reference timbre and emotion usage.

#### Scenario: Run multi-reference synthesis from CLI
- **WHEN** a user invokes the CLI with supported multi-reference timbre and or emotion reference arguments
- **THEN** the CLI executes the IndexTTS2 path and forwards the corresponding supported rollout preset into inference

### Requirement: The WebUI SHALL expose optional multi-reference timbre and emotion controls
The system SHALL expose supported multi-reference timbre and emotion controls in the WebUI while keeping the current single-reference flow usable.

#### Scenario: Use WebUI without multi-reference inputs
- **WHEN** a WebUI user keeps the new multi-reference controls empty
- **THEN** the WebUI continues to synthesize with the existing single-reference behavior

#### Scenario: Use WebUI with supported multi-reference inputs
- **WHEN** a WebUI user provides supported extra timbre and or emotion references
- **THEN** the WebUI synthesizes through the supported rollout preset path

### Requirement: Project documentation SHALL describe the supported rollout path
The system SHALL document the supported multi-reference timbre and emotion workflow in the main project documentation.

#### Scenario: Read the supported multi-reference docs
- **WHEN** a user reads the README or corresponding localized documentation
- **THEN** the user can find the supported default multi-reference workflow, fallback behavior, compatibility notes, and local development expectations
