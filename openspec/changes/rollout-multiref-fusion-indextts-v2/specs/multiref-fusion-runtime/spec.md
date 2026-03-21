## ADDED Requirements

### Requirement: IndexTTS2 SHALL expose supported multi-reference timbre presets
The system SHALL allow callers to request multi-reference timbre fusion through rollout-facing convenience inputs instead of requiring a raw `fusion_recipe`. The supported timbre presets SHALL encode the experiment-selected default and fallback schemes.

#### Scenario: Use recommended timbre preset
- **WHEN** a caller provides two or more speaker references and selects the supported default timbre fusion mode
- **THEN** the system builds a fusion recipe that enables `spk_cond_emb` and `speech_conditioning_latent` with symmetric weighting and symmetric anchoring

#### Scenario: Use fallback timbre preset
- **WHEN** a caller provides two or more speaker references and selects the supported fallback timbre fusion mode
- **THEN** the system builds a fusion recipe that enables only `speech_conditioning_latent` with symmetric weighting and symmetric anchoring

### Requirement: IndexTTS2 SHALL expose supported multi-reference emotion presets
The system SHALL allow callers to request multi-reference emotion fusion through rollout-facing convenience inputs that are independent from timbre inputs. The supported emotion presets SHALL encode the experiment-selected default and fallback schemes.

#### Scenario: Use recommended emotion preset
- **WHEN** a caller provides two or more emotion references and selects the supported default emotion fusion mode
- **THEN** the system builds an emotion-branch fusion configuration that uses weighted sum with anchor mode `A`

#### Scenario: Use fallback emotion preset
- **WHEN** a caller provides two or more emotion references and selects the supported fallback emotion fusion mode
- **THEN** the system builds an emotion-branch fusion configuration that uses weighted sum with anchor mode `symmetric`

### Requirement: Timbre and emotion multi-reference controls SHALL remain separate
The system SHALL keep timbre-reference fusion and emotion-reference fusion as separate public controls so callers can vary them independently.

#### Scenario: Multi-reference timbre without multi-reference emotion
- **WHEN** a caller provides multiple speaker references and no additional emotion references
- **THEN** the system applies the requested timbre preset while preserving the existing emotion selection behavior

#### Scenario: Multi-reference emotion without multi-reference timbre
- **WHEN** a caller provides multiple emotion references and only one speaker reference
- **THEN** the system applies the requested emotion preset without requiring timbre fusion

### Requirement: Raw fusion recipes SHALL override rollout presets
The system SHALL preserve `fusion_recipe` as the expert interface and SHALL not overwrite it with rollout preset construction.

#### Scenario: Raw recipe provided alongside rollout-facing preset inputs
- **WHEN** a caller provides a non-null `fusion_recipe` together with multi-reference preset arguments
- **THEN** the system uses the provided `fusion_recipe` as the effective recipe

### Requirement: Legacy single-reference inference SHALL remain backward compatible
The system SHALL preserve current IndexTTS v2 behavior when no multi-reference timbre or emotion inputs are provided.

#### Scenario: Legacy inference call without rollout inputs
- **WHEN** a caller uses the existing single-reference arguments only
- **THEN** the system follows the current single-reference inference path and produces output without requiring preset construction
