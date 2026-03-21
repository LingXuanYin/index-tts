## ADDED Requirements

### Requirement: Rollout-facing multi-reference presets SHALL be available for timbre and emotion
The system SHALL provide formal rollout-facing multi-reference support for `IndexTTS2` timbre and emotion fusion without requiring callers to author a full raw `fusion_recipe`.

#### Scenario: Build the supported timbre default
- **WHEN** a caller requests supported multi-timbre fusion through the rollout-facing API
- **THEN** the system constructs a supported recipe that uses `spk_cond_emb + speech_conditioning_latent` with symmetric `0.5 / 0.5` weighting by default

#### Scenario: Build the supported emotion default
- **WHEN** a caller requests supported multi-emotion fusion through the rollout-facing API
- **THEN** the system constructs a supported recipe that uses the `emotion_tensor_anchor_a` behavior with `0.5 / 0.5` weighting and `A` anchoring by default

### Requirement: Timbre and emotion rollout inputs SHALL remain separated
The system SHALL model timbre multi-reference control and emotion multi-reference control as separate axes in the rollout-facing interface.

#### Scenario: Timbre-only rollout request
- **WHEN** a caller provides multi-reference timbre inputs but no multi-reference emotion inputs
- **THEN** the system applies the timbre rollout preset without changing the legacy emotion path

#### Scenario: Emotion-only rollout request
- **WHEN** a caller provides multi-reference emotion inputs but no multi-reference timbre inputs
- **THEN** the system applies the emotion rollout preset without changing the legacy timbre path

### Requirement: Advanced `fusion_recipe` support SHALL remain available
The system SHALL preserve explicit `fusion_recipe` support for advanced users and internal tooling.

#### Scenario: Expert override request
- **WHEN** a caller passes an explicit `fusion_recipe`
- **THEN** the system honors that explicit recipe instead of replacing it with a rollout preset

### Requirement: Single-reference inference SHALL remain backward compatible
The system SHALL preserve current single-reference `IndexTTS2` behavior when rollout-facing multi-reference inputs are not used.

#### Scenario: Legacy single-reference call
- **WHEN** a caller uses the existing single-reference arguments and does not request rollout-facing multi-reference behavior
- **THEN** the system follows the current single-reference inference path and returns a valid output without requiring new arguments

### Requirement: Unsupported unstable defaults SHALL not be promoted
The system SHALL not promote `ref_mel`-based unstable routes into the supported rollout defaults.

#### Scenario: Supported rollout preset selection
- **WHEN** the system constructs the supported rollout default or fallback presets
- **THEN** those presets exclude `ref_mel` and `style + ref_mel` from the supported default path
