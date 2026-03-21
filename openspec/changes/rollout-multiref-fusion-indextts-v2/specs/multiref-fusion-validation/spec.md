## ADDED Requirements

### Requirement: Rollout validation SHALL cover preset construction and backward compatibility
The system SHALL include automated validation for supported timbre and emotion preset construction and for unchanged legacy single-reference behavior.

#### Scenario: Validate supported preset construction
- **WHEN** automated tests execute against the rollout helper layer
- **THEN** they verify the effective recipe for supported timbre and emotion defaults and fallbacks

#### Scenario: Validate backward compatibility
- **WHEN** automated tests execute against the runtime API without any rollout-facing multi-reference inputs
- **THEN** they verify that legacy single-reference inference behavior remains available

### Requirement: Rollout validation SHALL cover interface-level multi-reference invocation
The system SHALL include automated coverage for at least one user-facing invocation surface that exercises multi-reference timbre and emotion arguments through the supported rollout path.

#### Scenario: Validate interface invocation
- **WHEN** automated tests execute against a supported invocation surface
- **THEN** they verify that multi-reference timbre and emotion arguments reach the runtime with the expected preset-driven behavior

### Requirement: Rollout validation SHALL include one end-to-end smoke path
The system SHALL run at least one end-to-end validation path in the project virtual environment before the rollout is considered complete.

#### Scenario: Run end-to-end smoke validation
- **WHEN** the rollout validation step executes in `.venv`
- **THEN** it runs one end-to-end synthesis smoke path and records whether the supported multi-reference flow executes successfully
