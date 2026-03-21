## ADDED Requirements

### Requirement: WebUI SHALL expose supported multi-reference timbre and emotion controls
The system SHALL expose the supported multi-reference rollout behavior in the WebUI without requiring users to author raw tensor recipes.

#### Scenario: Configure multi-reference timbre from WebUI
- **WHEN** a user supplies a primary timbre reference and one or more additional timbre references in the WebUI
- **THEN** the WebUI passes the selected timbre preset and the full timbre reference list into the IndexTTS v2 runtime

#### Scenario: Configure multi-reference emotion from WebUI
- **WHEN** a user selects emotion-reference mode and supplies one or more additional emotion references in the WebUI
- **THEN** the WebUI passes the selected emotion preset and the full emotion reference list into the IndexTTS v2 runtime

### Requirement: CLI SHALL expose supported multi-reference timbre and emotion controls
The system SHALL provide a CLI path for IndexTTS v2 that supports multi-reference timbre and emotion fusion through rollout-facing flags.

#### Scenario: Use CLI with multiple timbre references
- **WHEN** a user invokes the CLI with a primary timbre reference and one or more additional timbre-reference flags
- **THEN** the CLI runs IndexTTS v2 with the selected timbre fusion preset

#### Scenario: Use CLI with multiple emotion references
- **WHEN** a user invokes the CLI with one or more emotion-reference flags
- **THEN** the CLI runs IndexTTS v2 with the selected emotion fusion preset while preserving the requested timbre configuration

### Requirement: Project documentation SHALL describe supported multi-reference rollout behavior
The system SHALL document the supported multi-reference timbre and emotion workflows in the repository documentation.

#### Scenario: Read supported rollout docs
- **WHEN** a user reads the main project documentation for IndexTTS v2 inference
- **THEN** the documentation includes supported multi-reference usage for Python, WebUI, and CLI together with the selected recommended presets

#### Scenario: Read advanced recipe docs
- **WHEN** a user needs expert-level tensor control beyond the supported presets
- **THEN** the documentation distinguishes advanced `fusion_recipe` usage from the supported rollout path
