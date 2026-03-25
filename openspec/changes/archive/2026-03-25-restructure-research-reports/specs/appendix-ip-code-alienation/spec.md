# appendix-ip-code-alienation Specification

## Purpose
TBD - created by archiving change restructure-research-reports. Update Purpose after archive.
## Requirements
### Requirement: Define the IP/code alienation and knowledge transfer package
Appendix B SHALL define the IP/code alienation (KT/IP) package, including the deliverables, acceptance criteria, and handover checklist for transferring the solution to a client.

#### Scenario: Client handover package is explicit
- **WHEN** a delivery model requires transfer to the client
- **THEN** the appendix defines what is transferred (code, docs, configs, models, data exports) and how acceptance is verified

### Requirement: Cover licensing and third-party constraints as transfer inputs
Appendix B SHALL include requirements for tracking model/software licenses and third-party constraints that affect transfer and ongoing use.

#### Scenario: License constraints are captured for transfer
- **WHEN** a model or dependency has non-permissive terms
- **THEN** the appendix specifies that the license terms and constraints are captured as part of the transfer package

### Requirement: Avoid duplicating economic tables
Appendix B SHALL NOT duplicate detailed CapEx/OpEx/TCO tables and SHALL reference the economics main report for quantitative impacts.

#### Scenario: Quantitative economics stay canonical elsewhere
- **WHEN** the appendix discusses cost implications of transfer choices
- **THEN** it references the sizing & economics report as the canonical source for numbers

