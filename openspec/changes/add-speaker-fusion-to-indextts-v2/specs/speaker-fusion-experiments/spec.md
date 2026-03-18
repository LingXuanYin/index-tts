## ADDED Requirements

### Requirement: The experiment workflow SHALL enumerate the full configured fusion combination space
The system SHALL provide an experiment workflow that enumerates the full configured combination space across supported and explicitly enabled experimental fusion levels, weights, anchor modes, and branch-role assignments, so the project can compare complete speaker fusion coverage instead of a pruned candidate subset.

#### Scenario: Generate full-combination manifest
- **WHEN** an experiment run starts with a configured fusion domain
- **THEN** the system creates manifest entries for every valid combination in that domain instead of pruning combinations before the first pass

### Requirement: Experimental fusion levels SHALL be coverable without polluting the default experiment set
The system SHALL allow waveform mixing, `cat_condition`, and `vc_target` experiments to be included only through an explicit opt-in mode.

#### Scenario: Default experiment run
- **WHEN** an experiment run uses the default configuration
- **THEN** the system covers supported fusion levels and excludes experimental levels

#### Scenario: Exploratory experiment run
- **WHEN** an experiment run enables exploratory coverage
- **THEN** the system adds experimental fusion levels to the manifest and labels their outputs as experimental

### Requirement: The first large comparison run SHALL use a ten-batch open-source sample slice
The system SHALL support generating an initial evaluation slice composed of ten batches sampled from one or more open-source speech datasets, and each batch SHALL preserve dataset provenance and batch identity in the manifest.

#### Scenario: Build the first ten-batch slice
- **WHEN** the initial comparison dataset is prepared
- **THEN** the system samples ten batches from open-source datasets and records dataset source, batch id, speaker ids, prompt ids, and text ids for each batch

### Requirement: Experiment outputs SHALL be reproducible and resumable
The system SHALL write a manifest that records input references, text, layer selections, weights, anchor modes, and output paths for every experiment case.

#### Scenario: Resume an interrupted experiment batch
- **WHEN** an experiment batch is interrupted after some cases complete
- **THEN** the system can resume from the manifest and skip completed cases with matching metadata

#### Scenario: Compare results across runs
- **WHEN** a reviewer inspects two experiment outputs
- **THEN** the stored metadata is sufficient to identify the exact fusion recipe used for each case
