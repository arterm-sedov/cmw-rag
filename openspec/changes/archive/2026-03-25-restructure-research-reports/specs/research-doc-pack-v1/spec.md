# research-doc-pack-v1 Specification

## Purpose
TBD - created by archiving change restructure-research-reports. Update Purpose after archive.
## Requirements
### Requirement: Preserve original reports as immutable ground truth
The system SHALL keep the original research reports unchanged and treat them as immutable ground truth for completeness checks.

#### Scenario: Existing report remains unchanged
- **WHEN** the new document pack is produced
- **THEN** the original reports remain present and unmodified

### Requirement: Produce a six-document pack
The system SHALL produce exactly six new research documents: two main documents and four appendices labeled A, B, C, and D.

#### Scenario: Pack contains six new documents
- **WHEN** the document pack is finalized
- **THEN** exactly six new documents exist as the new deliverables set

### Requirement: Enforce single canonical owner for each topic
The system SHALL define a single canonical “topic owner” document for each major topic to minimize duplication.

#### Scenario: Duplicate content is avoided by canonical ownership
- **WHEN** a topic appears in multiple places during drafting
- **THEN** one document is marked canonical for the full content and others contain only a short summary plus a link

### Requirement: Maintain traceability from originals to new pack
The system SHALL maintain a traceability map that assigns every substantial section from the original reports to exactly one location in the new pack.

#### Scenario: Every original section has a destination
- **WHEN** the traceability map is reviewed
- **THEN** each original heading/section has a destination document and destination section

### Requirement: Limit cross-references to human-readable document titles
The system SHALL cross-reference other documents only by human-readable document titles and SHALL avoid internal repository paths inside the new reports, except where explicitly required for verifiability.

#### Scenario: Cross-reference uses title, not repo paths
- **WHEN** a document references another document in the pack
- **THEN** the reference uses the document title (and optionally Appendix label), not a repository file path

