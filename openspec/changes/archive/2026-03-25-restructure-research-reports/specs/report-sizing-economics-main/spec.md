# report-sizing-economics-main Specification

## Purpose
TBD - created by archiving change restructure-research-reports. Update Purpose after archive.
## Requirements
### Requirement: Serve as the canonical source for sizing and economics
The sizing & economics main report SHALL be the canonical source for all quantitative economic content, including CapEx, OpEx, TCO, unit economics, tariffs, and sizing tables.

#### Scenario: Numbers live in one canonical place
- **WHEN** a numeric table is required anywhere in the pack
- **THEN** the table exists in this report and other documents only reference it by title

### Requirement: Keep methodology non-canonical in the economics report
The sizing & economics main report SHALL NOT be the canonical source of the implementation operating model and delivery methodology.

#### Scenario: Methodology is referenced but not duplicated
- **WHEN** methodology content is needed for context
- **THEN** the report points to the methodology main report and does not duplicate full methodology blocks

### Requirement: Provide scenario-based sizing and cost factors
The report SHALL provide scenario-based sizing and a cost-factor tree that supports decision-making across cloud RU, on-prem, and hybrid deployment options.

#### Scenario: Reader can map a workload to a scenario
- **WHEN** a reader provides workload assumptions (DAU/requests/context/SLO)
- **THEN** the report maps these assumptions to conservative/base/enterprise scenarios and associated cost drivers

### Requirement: Provide a concise related-documents guide
The sizing & economics main report SHALL include a concise section pointing to appendices for IP/code transfer implications, CMW existing work, and security/observability factors that affect OpEx and risk.

#### Scenario: Related documents are discoverable from the economics report
- **WHEN** a reader needs non-economic deep dives behind cost drivers
- **THEN** the report lists the relevant appendix titles to consult

