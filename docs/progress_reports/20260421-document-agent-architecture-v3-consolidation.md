# Document Agent Architecture Report — v3 Consolidation

**Date:** 2026-04-21
**Status:** Completed

## Summary

Consolidated the document agent architecture report from v2 (243 lines, significant redundancy) to v3 (217 lines, zero redundancy) following a three-persona consilium (systems architect, AI architect, technical writer) and deep research into best practices from C4 Model, ArchiMate, AWS Well-Architected, LangChain/CrewAI/AutoGen.

## Consilium Analysis

### Problems Identified (v2)

1. **Architecture stated 4×**: layers list, component table, diagram, algorithm — each restated what things ARE rather than adding new views per C4 progressive disclosure
2. **LLM-orchestrator boundary broken**: diagram showed LLM directly calling external services (`E -.-> G`), contradicting the note that LLM has no network access
3. **Timing 3×**: Résюме (line 22), Infrastructure table (line 93), tip (line 243)
4. **Infrastructure 2×**: Infrastructure table and "Что требуется на стороне агента"
5. **External systems 2×**: Component table row and dedicated section
6. **Async model 2×**: Résюме and dedicated section
6. **LLM limitations orphaned**: Critical content buried as a note under Infrastructure

### Research-Based Consolidation Strategy

Per C4 Model (progressive disclosure), ArchiMate (viewpoint mechanism), and TOGAF (building blocks):

| Principle | Application |
|:--|:--|
| Define once, reference everywhere | Component catalog = single source of truth; all other views reference it |
| Each view tells ONE question | What exists? → Catalog. How they interact? → Diagram. What's the process? → Pipeline. How is it operated? → Operational |
| LLM-orchestrator boundary (LangChain/AutoGen/CrewAI) | Explicit "Decides vs Executes" table; diagram arrows through orchestrator only |
| Timing in business-value location | Timing appears only in Résюме (Результат) and Operational Characteristics table |

## Changes Applied

### Structural Consolidation

| v2 Structure | v3 Structure | Change |
|:--|:--|:--|
| 4 separate sections (layers, table, diagram, algorithm) | Unified "Архитектура" section with component model + diagram + pipeline | 4→1 views, each answering different question |
| "Слой интеллектуальной обработки" implying LLM has tools | "LLM" as decision-making component, "Оркестратор" as execution component | Accurate per LangChain/AutoGen pattern |
| LLM note under Infrastructure | Promoted to "Граница ответственности: LLM и оркестратор" in Architecture | Architectural concept, not infrastructure footnote |
| 3× timing references | 2×: Résюме + Operational Characteristics | Bùsiness value first, operational reference second |
| "Инфраструктура" table + "Что требуется на стороне агента" section | Single "Операционные характеристики" table with all operational data | DRY |
| "Интеграция с внешними системами" standalone section | Integrated into Architecture under "Внешние интеграции" | External systems are architectural, not a separate concern |
| Diagram: LLM → External Services directly | Diagram: LLM → ToolCall → Orchestrator → External Services | Accurately reflects orchestrator pattern |

### Content Improvements

- **Responsibility table**: explicit "Decides vs Executes" boundary (per AutoGen FunctionCall/FunctionExecutionResult pattern)
- **Diagram redesign**: arrows go through orchestrator, not LLM; added tools node (calculations, datetime); split external services into ГНС/1С/Консультант+ vs web search
- **Diagram numbering**: steps 1–10 clearly showing orchestrator-mediated tool calls
- **Pipeline steps**: match diagram numbering, explicitly note ToolCall pattern
- **Typo fix**: "Арххитектура" → "Архитектура"
- **Version**: v2 → v3

## Files Modified

- `docs/research/executive-research-technology-transfer/report-pack/20260420-document-agent-architecture-ru.md`

## Sources (Research)

- C4 Model (c4model.com) — progressive disclosure, define once reference everywhere
- ArchiMate 3.2 (The Open Group) — viewpoint mechanism, realization relationships
- AWS Well-Architected Framework + ML Lens — lens-based decomposition, operational vs logical views
- LangChain/LangGraph — declarative graph, agent boundaries
- CrewAI — hierarchical orchestration, function_calling_llm separation
- AutoGen — UserProxyAgent pattern, FunctionCall/FunctionExecutionResult types