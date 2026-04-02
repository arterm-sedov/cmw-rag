# research-vitrine-index Specification

## Purpose
TBD - created by archiving change restructure-research-reports. Update Purpose after archive.
## Requirements
### Requirement: Provide a user-friendly navigation view for the pack
The vitrine document SHALL provide a user-friendly navigation section mapping common stakeholder questions to the specific document(s) to request and read.

#### Scenario: Reader finds the right document by question
- **WHEN** a reader has a question (e.g., “budget”, “security”, “IP transfer”, “what exists today”)
- **THEN** the vitrine lists the recommended document(s) for that question

### Requirement: Provide a canonical registry of documents in the pack
The vitrine document SHALL include a registry of all documents in the pack with titles, intended audience, and one-paragraph purpose statements.

#### Scenario: Registry enumerates the full pack
- **WHEN** the vitrine is reviewed
- **THEN** it lists the two main documents and Appendices A–D with their purpose and audience

### Requirement: Provide a canonical registry of used sources
The vitrine document SHALL include a canonical registry of sources that were used across the pack.

#### Scenario: Used sources are listed once canonically
- **WHEN** a source is used in any document in the pack
- **THEN** it appears in the vitrine’s used-sources registry

### Requirement: Provide additional sources cataloged by theme
The vitrine document SHALL include an additional-sources section containing relevant sources from the task list that were not used directly in the pack, categorized by theme, without reliability scoring and without “used/not used” status labels.

#### Scenario: Additional sources are categorized without scoring
- **WHEN** additional sources are presented
- **THEN** they are grouped by themes (e.g., governance, FinOps, security, observability, legal/RU) and do not include reliability ratings

### Requirement: Provide per-document “related documents” micro-guide
Each document in the pack SHALL include a short “related documents” micro-guide indicating which other document to consult for adjacent questions, kept concise.

#### Scenario: Document contains a short related-documents section
- **WHEN** any document in the pack is opened
- **THEN** it includes a brief section that points to other documents by title for adjacent topics

