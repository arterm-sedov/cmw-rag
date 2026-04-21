# Research Report: Best Practices for Documenting AI Agent Architectures

**Date:** 2026-04-21  
**Mode:** Standard  
**Sources:** 12+ distinct sources across systems architecture, AI agent frameworks, and technical writing domains

---

## Executive Summary

The gold standard for documenting AI agent architectures emerges from a convergence of three disciplines: **layered architecture modeling** (C4, ArchiMate), **agent orchestration patterns** (LangChain/LangGraph, CrewAI, AutoGen), and **viewpoint-based documentation** (ArchiMate viewpoints, AWS Well-Architected). Best-in-class references share one principle: **each view tells one story to one audience**, and cross-view consistency is maintained through a shared metamodel rather than repetition. The LLM-orchestrator tool-call pattern — where the LLM decides but the orchestrator executes — is documented via **sequence diagrams with explicit responsibility boundaries**, not via prose repetition. Timing/SLA information belongs in operational views (deployment, monitoring), not in logical architecture views.

---

## 1. Systems Architecture Perspective

### How Top References Document Agentic Architectures

#### C4 Model: Hierarchical Abstraction Without Redundancy

Simon Brown's C4 model provides the canonical pattern for avoiding "same idea in multiple views" [1]. The key insight is **progressive disclosure through levels of zoom**:

1. **System Context** — shows the system in its environment, stakeholder-oriented
2. **Container** — shows the internal "containers" (deployable units), developer-oriented
3. **Component** — shows internal components within a container, implementer-oriented
4. **Code** — class/module level, rarely needed

C4 explicitly states: "You don't need to use all 4 levels of diagram; only those that add value" [1]. The system context and container diagrams are sufficient for most teams. The principle: **each level adds NEW detail that doesn't exist at the level above**. You never repeat what a higher level already showed — you decompose it.

The C4 notation page reinforces: "Every element should have a short description, to provide an 'at a glance' view of key responsibilities" and "Every container and component should have a technology explicitly specified" [2]. This is a key differentiator from free-form diagrams — **C4 mandates metadata per element per level**.

**Key technique:** C4's supplementary diagrams (Dynamic, Deployment) address specific concerns (runtime behavior, infrastructure) that the static structure diagrams don't cover. They are NOT redundant with the core diagrams — they project the same model through a different lens.

#### ArchiMate: Viewpoint Mechanism for Eliminating Redundancy

ArchiMate's approach to the "same idea in multiple views" problem is its **viewpoint mechanism** [3]. Each viewpoint deliberately restricts what concepts and relationships are visible:

- A view "is defined as a part of an architecture description that addresses a set of related concerns and is tailored for specific stakeholders"
- Viewpoints define "specific conditions like concepts, analysis techniques, models, and visualizations — a viewpoint, from which the model should be perceived"
- "By reducing the 'view' by setting the right conditions and intentionally limiting the perspective, it is easier to solve specific problems"

ArchiMate's integration across layers (Business → Application → Technology) uses **realization relationships** — elements in lower layers "realize" elements in higher layers. This means you document a business service once, and the application component that realizes it is linked via a typed relationship, not duplicated prose [3].

**Key technique:** ArchiMate uses a **3×3 matrix** (3 layers × 3 aspects: active structure, behavior, passive structure). The same metamodel entities appear in multiple layers, but each layer specializes them. This is the "single source of truth" pattern — the metamodel is defined once, instantiated per layer.

#### AWS Well-Architected: Lens-Based Decomposition

AWS Well-Architected Framework uses a **lens pattern** [4]. The base framework covers 6 pillars (Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimization, Sustainability). Domain-specific concerns are addressed through **separate lenses** that extend, not duplicate, the base:

- The Machine Learning Lens "complements and builds upon the Well-Architected Framework to address the fundamental differences between traditional application workloads and machine learning workloads" [5]
- Each lens defines its own design principles and questions, mapped to the same 6 pillars
- The ML Lens explicitly distinguishes itself from the Generative AI Lens — same pillars, different concerns

**Key technique:** AWS avoids redundancy by **extending a base model** per domain rather than creating parallel documentation. Each lens references the base and adds only domain-specific content.

### Consolidating 4 Views into One Coherent Representation

Based on these references, the recommended consolidation strategy is:

| Original View | C4 Equivalent | ArchiMate Equivalent | What It Captures | Where to Put It |
|---|---|---|---|---|
| Layer Diagram | System Context | Layered View | Scope, boundary, external actors | Section 1: System Context |
| Component Table | Container Diagram + notation keys | Active/Passive Structure View | Components, responsibilities, technologies | Section 2: Component Catalog (structured data, not prose) |
| Data Flow Diagram | Dynamic Diagram | Behavior View | Runtime message flow, tool calls | Section 3: Interaction Patterns |
| Algorithm/Process | (Supplementary) | Process/Service Realization View | Step-by-step orchestration logic | Section 4: Process Descriptions (cross-referencing Section 3) |

**The golden rule from all references:** Define each concept ONCE in its canonical location, then reference it from other views. Never copy-paste descriptions across sections.

---

## 2. AI Agent Patterns Perspective

### How the Orchestrator/Agent Pattern Is Documented

#### LangChain/LangGraph: Declarative Graph with Explicit Boundaries

LangChain's current architecture separates concerns clearly [6]:

- **LangChain agents** are built on top of **LangGraph** — "our low-level agent orchestration framework and runtime"
- LangChain provides "a prebuilt agent architecture" while LangGraph handles "deterministic and agentic workflows"
- The agent abstraction includes `model`, `tools`, and `system_prompt` — the LLM receives tool definitions but **the framework executes them**

The key documentation pattern: LangChain **never shows the LLM calling external services directly**. The architecture is:

```
User → Agent (orchestration layer) → LLM (decision) → Agent (execution) → Tool (external service)
```

This is documented via:
1. A conceptual diagram showing the agent as the boundary
2. Code that shows `tools=[get_weather]` registered WITH the agent, not the LLM directly
3. The `function_calling_llm` parameter in CrewAI explicitly separates "which LLM makes tool-call decisions" from "which LLM does reasoning" [7]

#### CrewAI: Hierarchical Orchestration with Separated Concerns

CrewAI documents the orchestrator pattern through its **hierarchical process** attribute [7]:

- `Process.sequential`: Tasks flow agent-to-agent in order
- `Process.hierarchical`: "A manager agent coordinates the crew, delegating tasks and validating outcomes before proceeding"
- The `manager_llm` or `manager_agent` attribute **mandates** an orchestrator distinct from worker agents
- `function_calling_llm` at the crew level can override agent-level LLMs for tool calls — making the boundary explicit

CrewAI also documents **execution control** parameters separately from reasoning:
- `max_rpm`: Rate limiting (orchestration concern)
- `max_execution_time`: Timeout (orchestration concern)
- `max_iter`: Iteration limit (reasoning concern)

**This separation is the key documentation pattern**: orchestration concerns (when, how many, timeout) are documented separately from reasoning concerns (what to decide, how to plan).

#### AutoGen: Conversational Agent Pattern with UserProxy

AutoGen (Microsoft) uses the **ConversableAgent/UserProxyAgent** pattern [8]:

- `AssistantAgent`: The LLM-powered reasoning agent
- `UserProxyAgent`: The execution proxy that "acts as the user" — it executes code, calls tools, and manages human-in-the-loop interactions
- The UserProxyAgent **never makes LLM calls itself** — it's purely an orchestrator

This pattern is documented via:
1. Clear class diagrams showing the separation
2. Conversation flows showing `FunctionCall` → `FunctionExecutionResult` as explicit message types
3. The architecture diagram showing bidirectional chat between agents with typed messages

**AutoGen's key documentation insight:** The separation is visible in the code structure itself — `FunctionCall` and `FunctionExecutionResult` are distinct types, making the boundary between "decision" and "execution" explicit in the type system.

### How to Clearly Show the LLM-Orchestrator Tool-Call Pattern

Based on analysis of all three frameworks, the recommended documentation approach is:

1. **Use a sequence diagram** (C4 Dynamic Diagram or UML sequence) showing the flow:
   ```
   User → Orchestrator → LLM (returns FunctionCall)
   Orchestrator executes FunctionCall → External Service
   Orchestrator → LLM (returns FunctionExecutionResult)
   LLM → Orchestrator (final response)
   ```
   
   The key: the LLM never has an arrow pointing to the External Service. Only the Orchestrator does.

2. **Use a responsibility table** (not prose) making boundaries explicit:
   | Component | Decides | Executes |
   |---|---|---|
   | LLM | Which tool to call, with what parameters | Nothing external |
   | Orchestrator | Which step comes next, error handling, retries | All tool calls, API requests, DB queries |
   | Tools | Nothing | Specific domain operations |

3. **Use typed interfaces** in the component catalog:
   - LLM output type: `ToolCall | TextResponse`
   - Orchestrator interface: `execute(call: ToolCall) → ToolResult`
   - This mirrors AutoGen's `FunctionCall`/`FunctionExecutionResult` type separation

4. **Name the boundary explicitly**: Call it the "Tool Execution Boundary" or "Orchestration Layer" — not just "the system" or "the agent". LangChain calls it "LangGraph runtime", CrewAI calls it "Crew process", AutoGen calls it "UserProxyAgent". Your system should have its own name.

---

## 3. Technical Writing Perspective

### How to Handle the "Same Idea in Multiple Views" Problem

#### ArchiMate's Solution: Viewpoints as Deliberate Restrictions

ArchiMate's viewpoint mechanism [3] solves redundancy by making each view a **projection** of the same underlying model, not a separate copy:

- The underlying model is the **single source of truth**
- Each viewpoint filters concepts and relationships relevant to specific stakeholders
- Elements that appear in multiple views are **referenced, not duplicated**
- Realization relationships (`«realizes»`) link elements across layers without repeating descriptions

**Application to AI architecture docs:** Define your component metamodel ONCE (in a component catalog/table), then every diagram, algorithm description, and data flow references entries from that catalog. The diagram shows relationships; the table shows properties.

#### C4 Model's Solution: Progressive Decomposition

C4's principle: "Every diagram should have a title describing the diagram type and scope" and "Every diagram should have a key/legend" [2]. The title and scope make it clear what level you're at, so you never accidentally repeat content from another level.

The key technique: **each diagram level only shows what INSIDE the container from the level above looks like when decomposed**. You never re-explain what's already explained at the higher level.

**Application:** Your layer diagram establishes scope and boundary. Your component table lists what's inside each layer. Your data flow shows how they interact at runtime. Your algorithm shows the step-by-step process. Each builds on the previous — none repeats it.

#### TOGAF's Solution: Architecture Repository with Building Blocks

TOGAF (companion to ArchiMate, both from The Open Group) uses **Architecture Building Blocks (ABBs)** and **Solution Building Blocks (SBBs)** stored in an Architecture Repository. Each building block is defined once and referenced by multiple views. Views are generated from the repository, not authored independently.

**Application:** Your component table IS your building block repository. Every other view references it.

#### DRY Principle Applied to Architecture Documentation

The DRY (Don't Repeat Yourself) principle from software engineering applies directly:

1. **Define Once, Reference Everywhere:** Each component, relationship, and concept is defined in exactly one canonical location (the component catalog).
2. **Use Cross-References, Not Copy-Paste:** When a concept appears in another view, reference it: "see Component Catalog → Orchestrator" rather than re-describing it.
3. **Separate Concerns into Orthogonal Views:** 
   - Structural view = what components exist and their responsibilities
   - Behavioral view = how they interact at runtime
   - Operational view = how they're deployed and monitored (including SLAs/timing)
4. **UseTyped References:** In ArchiMate, elements in one layer "realize" elements in another layer via typed relationships. In your documentation, use hyperlinks or section references rather than restating.

### Concrete Recommendation: Consolidating 4 Views

The following structure eliminates redundancy while preserving clarity for each audience:

```
1. Architecture Overview (System Context)
   - Purpose, scope, external actors
   - ONE diagram: system context showing boundaries
   - Cross-references to Component Catalog for details

2. Component Catalog (Single Source of Truth)
   - Structured TABLE: Component | Layer | Responsibility | Technology | Interfaces
   - Each row is defined ONCE, referenced everywhere else
   - Include: Component name, type, responsibility, technology, 
     key interfaces (input/output types), SLA requirements

3. Interaction Patterns (Dynamic/Behavioral View)
   - Sequence diagrams showing tool-call boundary
   - Data flow descriptions CROSS-REFERENCING Component Catalog entries
   - Explicit typed interfaces at each arrow
   - ONE place for runtime behavior

4. Process Descriptions (Algorithm View)
   - Step-by-step algorithms CROSS-REFERENCING Interaction Patterns
   - Pseudocode or numbered steps
   - Reference Component Catalog for "who does what"
   - Reference Interaction Patterns for "how they communicate"
```

**What NOT to do:**
- Don't put component descriptions in the data flow section (reference the catalog)
- Don't put SLA/timing information in the component catalog (put it in operational/deployment views where business value lives)
- Don't describe the tool-call boundary in prose multiple times (define it once with a diagram + responsibility table)
- Don't repeat what a component does if it's already in the catalog

---

## 4. Handling Timing/SLA Information

### Where Business Value Lives

AWS Well-Architected separates **design decisions** from **operational concerns** [4]. The base framework handles architectural decisions; lenses add domain-specific operational guidance. ML Lens adds model performance metrics (accuracy, drift, retraining) without duplicating base framework content.

**Recommendation:** Timing and SLA information belongs in **operational views**, not logical architecture views:

| Type of Timing Info | Where It Belongs | Why |
|---|---|---|
| Component response time SLAs | Component Catalog (as a property) | Part of the component's contract |
| End-to-end pipeline latency | Operational/Deployment View | Business-level concern, not architectural |
| Retry/timeout configuration | Interaction Patterns (as annotations) | Runtime behavior, not structure |
| Model inference latency | Performance Efficiency section | AWS WA-style pillar, not in architecture |
| Deployment SLAs | Deployment View | Infrastructure concern, not logical design |

The principle: **Put timing information where the business stakeholder who needs it will look for it.** If the CTO cares about 99.9% availability, that's an operational/deployment concern. If the developer needs to know the orchestrator timeout is 30s, that's an interface contract in the component catalog.

---

## 5. Summary of Key Insights

### Patterns from Best-in-Class References

| Pattern | Source | Application |
|---|---|---|
| Progressive disclosure (zoom levels) | C4 Model | Layer → Component → Sequence → Algorithm, each adding new detail |
| Viewpoint mechanism | ArchiMate | Each view filters for specific stakeholder concerns |
| Realization relationships | ArchiMate | Lower layers realize higher-layer services; typed links, not copy |
| Lens extension | AWS Well-Architected | Domain-specific concerns extend base model |
| Separated orchestration | CrewAI, AutoGen, LangChain | Orchestrator and LLM are distinct components with typed interfaces |
| Declarative graph | LangGraph | Agent behavior defined as state graph, not imperative code |
| Function call type separation | AutoGen | `FunctionCall` vs `FunctionExecutionResult` as distinct types |
| Single source of truth | TOGAF, ArchiMate, C4 | Component catalog defined once, referenced everywhere |
| DRY for docs | Software engineering | Define once, reference everywhere, separate concerns orthogonally |

### Specific Techniques for the LLM-Orchestrator Boundary

1. **Sequence diagram** showing arrows only between Orchestrator and External Services (never LLM → External Service)
2. **Responsibility table** with explicit "Decides" vs "Executes" columns
3. **Typed interfaces** in component catalog (LLM outputs `ToolCall`; Orchestrator input: `ToolCall`, output: `ToolResult`)
4. **Named boundary** ("Tool Execution Boundary" or "Orchestration Layer")
5. **Separation of orchestration concerns** (rate limiting, retries, timeouts) from reasoning concerns (what to call, what parameters)

### Handling Redundancy Across Views

1. **Component Catalog = single source of truth** for what each component IS
2. **Interaction Patterns = behavioral view** showing how they COMMUNICATE (not what they are)
3. **Process Descriptions = algorithmic view** showing STEP-BY-STEP logic (not what components are or how they communicate)
4. **Cross-references** instead of copy-paste (hyperlinks, section references, "see Catalog entry X")
5. **Each view answers ONE question**: What exists? How do they interact? What's the process? How is it operated?

---

## Bibliography

[1] Simon Brown, "The C4 Model for Visualising Software Architecture," c4model.com, 2024. https://c4model.com/

[2] Simon Brown, "C4 Model — Notation," c4model.com, 2024. https://c4model.com/diagrams/notation

[3] Wikipedia, "ArchiMate," last edited March 2026. https://en.wikipedia.org/wiki/ArchiMate; The Open Group, "ArchiMate 3.2 Specification," October 2022.

[4] Amazon Web Services, "AWS Well-Architected Framework," November 2024. https://docs.aws.amazon.com/wellarchitected/latest/framework/welcome.html

[5] Amazon Web Services, "Machine Learning Lens — AWS Well-Architected Framework," November 2025. https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html

[6] LangChain, "LangChain Overview," 2026. https://python.langchain.com/docs/concepts/architecture/ ; LangChain, "Agents," 2026. https://python.langchain.com/docs/concepts/agents/

[7] CrewAI, "CrewAI Agents Documentation," 2026. https://docs.crewai.com/concepts/agents ; CrewAI, "CrewAI Crews Documentation," 2026. https://docs.crewai.com/concepts/crews

[8] Microsoft, "AutoGen v0.4 Quickstart," 2026. https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html ; Microsoft, "AutoGen v0.2 Getting Started," 2025. https://microsoft.github.io/autogen/0.2/docs/Getting-Started

[9] Simon Brown, "C4 Model — Diagrams," c4model.com, 2024. https://c4model.com/diagrams

[10] The Open Group, "TOGAF Standard, Version 9.2 — Architecture Repository," 2018. https://pubs.opengroup.org/architecture/togaf9-doc/arch/

[11] Gerben Wierda, "Mastering ArchiMate Edition 3.1," R&A, 2021. ISBN 978-9083143415.

[12] ArchiMate tool, "C4 Model, Architecture Viewpoint and Archi 4.7," 2020. https://www.archimatetool.com/blog/2020/04/18/c4-model-architecture-viewpoint-and-archi-4-7/

---

## Methodology Appendix

**Research approach:** Standard mode (6 phases). Parallel retrieval from 12+ sources across systems architecture (C4, ArchiMate, TOGAF, AWS Well-Architected), AI agent frameworks (LangChain/LangGraph, CrewAI, AutoGen), and technical writing practices.

**Source diversity:** Mixed academic (Wikipedia/ArchiMate spec), industry (AWS, Microsoft, LangChain), and practitioner (C4 Model blog, CrewAI docs).

**Quality notes:** 
- C4 and ArchiMate are mature standards with broad adoption
- LangChain, CrewAI, and AutoGen documentation represents current best practices in the rapidly-evolving agent framework space
- AWS Well-Architected represents enterprise-grade operational documentation patterns
- Some LangChain/LangGraph pages had redirect issues; information was gathered from stable URLs