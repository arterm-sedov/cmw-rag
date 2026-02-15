# Support Resolution Plan (SRP) Tool - Implementation Plan

**Status:** APPROVED  
**Created:** 2026-02-14  
**Updated:** 2026-02-15  
**Based on:** Analysis of existing SGR architecture, IT support best practices, and ticket resolution workflows

---

## 1. Executive Summary

### Purpose
Create a second SGR-style tool that generates a **step-by-step resolution plan for human support engineers** to resolve user issues based on the conversation context. Unlike SGR (which analyzes the incoming request), this tool operates at **final answer generation** to provide actionable guidance for support staff.

### Key Characteristics
- **Trigger:** Forced tool call during final answer generation (not at request intake)
- **Output:** Structured resolution plan with numbered steps
- **TOC:** Dynamic table of contents prepended to final answer (in agent context) - only if headings exist
- **Display:** Answer message = [TOC] + Answer + [Plan section] (single assistant message)
- **Bubble:** "📝 Generating support engineer plan..." UI indicator (like SGR bubble, hidden upon completion)
- **Storage:** String field in downstream structured output (for CRM/ticketing systems)
- **Audience:** Human support engineers (not the AI agent itself)
- **Multi-turn:** Plan visible in context for multi-turn conversations (unlike SGR which is hidden)

### Relationship to Existing SGR
```
User Request
    │
    ▼
┌─────────────────┐
│ Guardian Check  │ (if enabled)
└─────────────────┘
    │
    ▼
┌─────────────────────────┐
│ SGR Tool (analyse_)    │ ← Analyzes incoming request
│ - Intent understanding │
│ - Topic identification │
│ - Search strategy      │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Agent processes request │
│ - Search KB             │
│ - Generate answer       │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Post-Answer Processing               │
│ 1. Show "📝 Generating plan..."     │
│ 2. FORCE SRP tool call              │
│ 3. Hide SRP bubble on completion    │
│ 4. Extract headers from answer      │
│ 5. Build TOC (after SRP result)     │
│ 6. Assemble: [TOC] + Answer + [Plan]│
│ 7. Append Sources at end            │ ← Best practice: sources at end
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Final Message Structure │
│ [AI Disclaimer]  (UI)   │ ← UI-only, separate message
│ [[TOC] + Answer + [Plan]]│ ← In agent context
│ [Sources]               │ ← At very end, after Plan
└─────────────────────────┘
```

---

## 2. Core Architectural Decisions

### Decision 1: Forced Call at Final Answer Generation
**Status:** ✅ CONFIRMED

**Rationale:** The resolution plan should be generated AFTER the agent has processed the request and formulated an answer. This ensures:
1. The plan is based on the actual solution provided, not speculation
2. The plan can reference specific KB articles and solutions used
3. The plan summarizes what was done/communicated for the human engineer

**Implementation:**
```python
# In the handler, after agent completes answer generation
# Force call to SRP tool before emitting final response
srp_model = llm.bind_tools(
    [generate_resolution_plan_tool],
    tool_choice={"type": "function", "function": {"name": "generate_resolution_plan"}},
)

srp_response = await srp_model.ainvoke(messages)
tool_call = srp_response.tool_calls[0]
resolution_plan = tool_call["args"]
```

### Decision 2: Clean Injection Pattern (Same as SGR)
**Status:** ✅ CONFIRMED

**Rationale:** Mirror the successful SGR injection pattern - remove tool trace from model context, inject synthetic message.

**Implementation:**
```python
# 1. Execute SRP tool (returns structured plan)
srp_result = await execute_srp_tool(messages, final_answer)
resolution_plan = json.loads(srp_result)

# 2. Store for downstream
agent_context.resolution_plan = resolution_plan

# 3. Remove tool trace from context
# (skip adding tool call and result to messages)

# 4. Render plan as markdown H1 section
plan_section = render_resolution_plan(resolution_plan)

# 5. Append to final answer (single message)
final_response = f"{final_answer}\n\n---\n\n{plan_section}"
```

### Decision 3: TOC Conditional on Headings + Configurable
**Status:** ✅ CONFIRMED

**Rationale:** 
- **No headings = no TOC** - Avoid empty TOC for short/simple answers
- **Configurable** - Separate flags for SRP and TOC enable independent A/B testing
- **LLM context** - When TOC exists, model understands document structure
- **Disclaimer separation** - AI disclaimer stays as metadata (UI-only)

**Implementation:**
```markdown
## Оглавление:                          ← Only if headings exist AND toc_enabled=true
- [Настройка LDAP-коннектора](#anchor1)  ← Only answer headings
- [Проверка подключения](#anchor2)      ← Only answer headings
- [План решения...](#plan)              ← Only if SRP succeeds

---

[Answer content with headers...]

---

# План решения для инженера поддержки   ← Only if SRP succeeds
[Plan sections...]
```

### Decision 4: Downstream Storage - Flag + Metadata Only
**Status:** ✅ CONFIRMED - UPDATED

**Rationale:** 
- Full plan is already embedded in the `answer` text (TOC + Answer + Plan section)
- Downstream systems get a boolean flag + key metadata for filtering/searching
- Avoids data duplication and keeps downstream payload lean

```python
# Structured output for downstream (Option B - Flag + Metadata)
class AgentOutput(BaseModel):
    answer: str  # Contains complete message: TOC + Answer + Plan
    srp_generated: bool  # Boolean flag indicating plan was generated
    srp_error: str | None  # Error details if SRP failed
    metadata: dict  # Contains srp_outcome, srp_priority, srp_doc_count, etc.
```

### Decision 5: Two-Stage Generation (Answer → Plan)
**Status:** ✅ CONFIRMED

**Rationale:** Separating answer generation from plan generation is superior to one-stage:

1. **Answer First** = Customer-facing response (what user sees)
2. **SRP Second** = Engineer action strategy (what support does)
3. **SRP sees actual answer** - Can reference specific solutions, KB articles cited, tone used
4. **Different audiences** - User wants help, engineer wants actionable steps
5. **Higher quality** - SRP analyzes completed answer, not speculating at start

**Why not one-stage?**
- Model would generate both blindly without seeing what it actually said
- Plan might reference solutions not actually provided
- No ability to cite specific answer content
- Confuses customer-facing vs. internal action concerns

### Decision 6: Multi-Turn Conversation Analysis
**Status:** ✅ CONFIRMED

**Rationale:** SRP should analyze and refine across conversation turns:

- **Full context analysis** - SRP receives complete conversation history in `messages`
- **Cumulative planning** - Each SRP call builds upon previous context
- **Refinement over turns** - Plan updates based on conversation progression
- **Track resolution state** - What was tried, what worked, what's next

**Example multi-turn flow:**
```
Turn 1: User reports LDAP issue
→ SRP generates initial troubleshooting plan

Turn 2: User says "tried steps 1-3, still failing"
→ SRP sees previous plan + new context
→ Updates plan with advanced troubleshooting steps

Turn 3: User confirms fix worked
→ SRP marks outcome as "resolved"
→ Plan reflects resolution path taken
```

---

## 3. Message Structure

### Final Assistant Message Structure

**Best Practice: Sources at End**
Following documentation and RAG best practices, sources/references are appended at the very end of the response. This:
- Follows normal documentation conventions
- Separates answer content from evidence
- Provides transparency without cluttering main response
- Makes sources easy to reference and verify

**Scenario A: With Headings + SRP Success**
```
┌──────────────────────────────────────────┐
│ AI Disclaimer (UI-only, separate msg)   │ ← Not in agent context
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ ## Оглавление:                          │
│ - [Header 1](#anchor1)                  │
│ - [Header 2](#anchor2)                  │
│ - [План решения...](#plan-anchor)       │
│                                          │
│ ---                                      │
│                                          │
│ [Answer content...]                     │
│ - H1, H2 sections                       │
│ - Minimal inline citations or none       │
│                                          │
│ ---                                      │
│                                          │
│ # План решения...                       │
│ [Plan sections...]                      │
│                                          │
│ ---                                      │
│                                          │
│ **Источники / Sources:**               │ ← At very end
│ 1. https://kb.comindware.ru/article/... │
│ 2. https://kb.comindware.ru/article/... │
└──────────────────────────────────────────┘
```

**Scenario B: No Headings + SRP Success**
```
┌──────────────────────────────────────────┐
│ AI Disclaimer (UI-only, separate msg)   │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ [Answer content...]                     │ ← No TOC (no headings)
│ - Plain text answer                     │
│                                          │
│ ---                                      │
│                                          │
│ # План решения...                       │
│ [Plan sections...]                      │
│                                          │
│ ---                                      │
│                                          │
│ **Источники / Sources:**               │ ← At very end
│ 1. https://kb.comindware.ru/article/... │
└──────────────────────────────────────────┘
```

**Scenario C: Headings + SRP Disabled/Failed**
```
┌──────────────────────────────────────────┐
│ AI Disclaimer (UI-only, separate msg)   │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ ## Оглавление:                          │
│ - [Header 1](#anchor1)                  │
│ - [Header 2](#anchor2)                  │
│                                          │
│ ---                                      │
│                                          │
│ [Answer content...]                     │
│                                          │
│ ---                                      │
│                                          │
│ **Источники / Sources:**               │ ← At very end
│ 1. https://kb.comindware.ru/article/... │
└──────────────────────────────────────────┘
```

**Scenario D: No Headings + SRP Disabled**
```
┌──────────────────────────────────────────┐
│ AI Disclaimer (UI-only, separate msg)   │
└──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────┐
│ [Answer content...]                     │ ← No TOC, no plan
│ - Plain text answer                     │
│                                          │
│ ---                                      │
│                                          │
│ **Источники / Sources:**               │ ← At very end
│ 1. https://kb.comindware.ru/article/... │
└──────────────────────────────────────────┘
```

---

## 4. Table of Contents (TOC) Design

### TOC Architecture
- **Location:** Prepend to final answer message (in agent context)
- **Source:** Dynamically extracted from answer headers (H1-H6)
- **Timing:** Generated **ONCE** after SRP result is known
- **Plan link:** Only included if SRP succeeds
- **Format:** Markdown list with anchor links
- **Condition:** Only generated if:
  1. `toc_enabled=true` (config flag), AND
  2. Answer has at least one H1-H6 heading

### Correct Flow: TOC Generated Once After SRP Result

```python
# After streaming completes
final_answer = accumulated_answer

# 1. CALL SRP (if enabled)
resolution_plan = None
if settings.srp_enabled:
    try:
        # Build SRP context:
        # 1. Base system prompt + appended sources list (already compiled in context)
        # 2. Appendage: instruction to call generate_resolution_plan tool
        # 3. Full conversation history (with full article texts for current turn)
        from rag_engine.llm.prompts import get_system_prompt
        
        srp_system_prompt = get_system_prompt()
        
        # Append sources list to system prompt
        if agent_context.sources_compiled:
            srp_system_prompt += "\n\n" + agent_context.sources_compiled
        
        # Append SRP instruction
        srp_system_prompt += "\n\n" + """After answering the user's question, analyze the conversation and call the generate_resolution_plan tool to create a support engineer resolution plan. The plan should summarize the issue and recommend next steps if human intervention is needed."""
        
        # Build messages for SRP LLM call
        srp_messages = [
            {"role": "system", "content": srp_system_prompt},
        ]
        # Add conversation history (includes current turn with full article texts)
        srp_messages.extend(messages)
        
        srp_model = llm.bind_tools(
            [generate_resolution_plan_tool],
            tool_choice={"type": "function", "function": {"name": "generate_resolution_plan"}},
        )
        srp_response = await srp_model.ainvoke(srp_messages)
        tool_call = srp_response.tool_calls[0]
        resolution_plan = tool_call["args"]
        agent_context.resolution_plan = resolution_plan
    except Exception as exc:
        logger.error("SRP tool failed: %s", exc)
        agent_context.resolution_plan_error = str(exc)
        # resolution_plan stays None

# 2. EXTRACT HEADERS FROM ANSWER ==========
answer_headers = extract_markdown_headers(final_answer)

# 3. BUILD TOC (ONCE, conditionally) ==========
# Conditions: toc_enabled=true AND (has headers OR SRP succeeded)
has_content_for_toc = len(answer_headers) > 0 or resolution_plan is not None
toc = ""
if settings.toc_enabled and has_content_for_toc:
    # Only include plan link if SRP actually succeeded
    toc = build_toc(answer_headers, include_plan=(resolution_plan is not None))
    if toc:
        toc = toc + "\n\n---\n\n"  # Add separator after TOC

# 4. RENDER PLAN ==========
plan_markdown = render_resolution_plan_markdown(resolution_plan) if resolution_plan else ""
if plan_markdown:
    plan_markdown = "\n\n---\n\n" + plan_markdown  # Add separator before plan

# 5. ASSEMBLE FINAL MESSAGE
complete_content = f"{toc}{final_answer}{plan_markdown}"
# Results in:
# - TOC + separator + Answer + separator + Plan (if all present)
# - TOC + separator + Answer (if no plan)
# - Answer + separator + Plan (if no headings but SRP success)
# - Answer (if no headings and no SRP)
```

### Lightweight Header Extractor (No External Deps)
```python
def extract_markdown_headers(text: str) -> list[tuple[int, str, str]]:
    """
    Extract H1-H6 headers from markdown text.
    
    Returns: List of (level, text, anchor) tuples
    
    Handles:
    - Standard headers: # Header
    - Headers with inline code: ## `code` example
    - Headers with links: ### [Link](url)
    - Escaped characters
    - Skips headers inside code blocks
    """
    headers = []
    in_code_block = False
    code_block_fence = None
    
    for line in text.split('\n'):
        stripped = line.lstrip()
        
        # Track code blocks
        if stripped.startswith('```') or stripped.startswith('~~~'):
            fence = stripped[:3]
            if not in_code_block:
                in_code_block = True
                code_block_fence = fence
            elif stripped.startswith(code_block_fence):
                in_code_block = False
                code_block_fence = None
            continue
        
        if in_code_block:
            continue
        
        # Match header pattern
        match = re.match(r'^(#{1,6})\s+(.+?)\s*$', stripped)
        if match:
            level = len(match.group(1))
            header_text = match.group(2).strip()
            
            # Clean inline formatting
            clean_text = re.sub(r'`([^`]+)`', r'\1', header_text)
            clean_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_text)
            clean_text = clean_text.strip()
            
            # Generate anchor (slugify)
            anchor = clean_text.lower()
            anchor = re.sub(r'[^\w\s-]', '', anchor)
            anchor = re.sub(r'[\s]+', '-', anchor)
            anchor = anchor.strip('-')[:50]
            
            if anchor:
                headers.append((level, clean_text, anchor))
    
    return headers
```

### TOC Builder
```python
def build_toc(answer_headers: list[tuple[int, str, str]], 
              include_plan: bool = True) -> str:
    """Build TOC markdown from answer headers and optional plan section."""
    lines = [get_text("srp_toc_title") + ":", ""]
    
    for level, text, anchor in answer_headers:
        indent = "  " * (level - 1)
        lines.append(f"{indent}- [{text}](#{anchor})")
    
    if include_plan:
        plan_title = get_text("srp_section_title")
        plan_anchor = "plan-resheniya-inzhenera"
        lines.append(f"- [{plan_title}](#{plan_anchor})")
    
    return "\n".join(lines)
```

### SRP Bubble Helper (stream_helpers.py)

```python
def yield_srp_planning_started() -> dict:
    """Yield metadata message for SRP (Support Resolution Plan) generation phase.
    
    This indicates the agent is generating the resolution plan after answer completion.
    Hidden/removed when SRP completes (like SGR bubble).
    
    Returns:
        Gradio message dict with metadata for SRP planning indicator.
        Uses i18n for title and content.
        Includes status="pending" to show native Gradio spinner.
    """
    from rag_engine.api.i18n import get_text
    
    title = get_text("srp_planning_title")
    content = get_text("srp_planning_content")
    
    return {
        "role": "assistant",
        "content": content,
        "metadata": {
            "title": title,
            "ui_type": "srp_planning",
            "status": "pending",
            "id": short_uid(),
        },
    }
```

### TOC Scenarios

| Scenario | toc_enabled | Has Headers | SRP Result | Plan Applicable | TOC Generated | Plan Link | Plan Section |
|----------|-------------|-------------|------------|-----------------|---------------|-----------|--------------|
| A | Yes | Yes | ✅ Generated | ✅ Yes | Yes | Yes | Yes |
| B | Yes | No | ✅ Generated | ✅ Yes | Yes | Yes | Yes |
| C | Yes | Yes | ✅ Generated | ❌ No | Yes | No | No |
| D | Yes | No | ✅ Generated | ❌ No | No | No | No |
| E | Yes | Yes | ❌ Failed | N/A | Yes | No | No |
| F | Yes | No | ❌ Failed | N/A | No | No | No |
| G | No | Yes | ✅ Generated | ✅ Yes | No | No | Yes |
| H | No | Any | Disabled | N/A | No | No | No |

**Key Points:**
- TOC generated **only once**, after SRP result known
- TOC requires `toc_enabled=true` AND (has headers OR plan_applicable=True)
- No plan link in TOC if plan_applicable=False or SRP failed
- No TOC at all if `toc_enabled=false`
- Plan section only rendered when plan_applicable=True

---

## 5. Template System Design

### Resolution Plan Template Structure

```markdown
# {srp_section_title}

## {srp_issue_summary}
{issue_summary}

## {srp_steps_completed}
1. {step_1}
2. {step_2}
3. {step_n}

## {srp_next_steps}
1. {next_step_1}
2. {next_step_2}

## {srp_result}
{outcome_status}

## {srp_doc_references}
- {doc_link_1}
- {doc_link_2}

## {srp_additional_notes}
{additional_notes}
```

### SRP Prompt for prompts.py

Add to `rag_engine/llm/prompts.py`:

```python
# Support Resolution Plan prompt (prepended to messages when calling SRP tool)
SRP_PROMPT = """Проанализируй результаты поиска в базе знаний по запросу пользователя и составь план решения для специалиста поддержки."""
```

---

## 6. Pydantic Schema Design

Note: Use English descriptions for token efficiency. Include note for LLM to fill actual text values in Russian (human-facing content), while enum values remain in English.

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class ResolutionOutcome(str, Enum):
    """Resolution status for support engineer plan."""
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    ESCALATION_REQUIRED = "escalation_required"
    USER_FOLLOWUP_NEEDED = "user_followup_needed"
    NOT_APPLICABLE = "not_applicable"  # For simple lookups - no plan needed


class ResolutionPlanResult(BaseModel):
    """Support Resolution Plan - generates actionable plan for human engineers.
    
    Called at final answer generation to provide structured guidance.
    The LLM decides if a plan is actually needed via engineer_intervention_needed field.
    
    NOTE: Fill text fields in Russian for human readability.
    Enum values remain in English for token efficiency.
    NOTE: Documentation references are NOT included here - they are already 
    embedded in the answer text with proper citations and ranking.
    """
    
    # FIELD 1: Engineer Intervention Needed (Decision Gate)
    engineer_intervention_needed: bool = Field(
        ...,
        description=(
            "Is support engineer intervention or escalation needed for this issue? "
            "Set to FALSE for: version queries, simple how-tos with complete KB answers, "
            "factual lookups that are fully resolved, self-service queries. "
            "Set to TRUE for: errors, bugs, configuration issues, incomplete solutions, "
            "troubleshooting required, or any issue requiring human investigation/action. "
            "This field is REQUIRED - the LLM must explicitly decide."
        ),
    )
    
    # FIELD 2: Issue Summary (optional - only when engineer_intervention_needed=True)
    issue_summary: str = Field(
        default="",
        max_length=500,
        description=(
            "Brief summary of the user's issue in 2-3 sentences. "
            "Write in Russian for the support engineer. "
            "Only meaningful when engineer_intervention_needed=True."
        ),
    )
    
    # FIELD 3: Steps Completed by System (optional - only when engineer_intervention_needed=True)
    steps_completed: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "List of steps already taken by the system. "
            "Include: KB search, documentation analysis, solutions provided. "
            "Write in Russian, be concise and informative. "
            "Only meaningful when engineer_intervention_needed=True."
        ),
    )
    
    # FIELD 4: Recommended Next Steps (optional - only when engineer_intervention_needed=True)
    next_steps: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Recommended next steps for the support engineer. "
            "What does the human need to do after this response? "
            "Examples: 'Check user permissions', 'Update documentation', 'Create dev ticket'. "
            "Write in Russian. "
            "Only meaningful when engineer_intervention_needed=True."
        ),
    )
    
    # FIELD 5: Outcome Status (enum in English)
    # Use NOT_APPLICABLE when engineer_intervention_needed=False
    outcome: ResolutionOutcome | None = Field(
        default=None,
        description=(
            "Resolution status based on the answer provided. "
            "Use English enum value: "
            "'resolved': Fully resolved; "
            "'partially_resolved': Partially resolved, additional actions needed; "
            "'escalation_required': Requires escalation; "
            "'user_followup_needed': User follow-up required; "
            "'not_applicable': Plan not needed (use when engineer_intervention_needed=False)."
        ),
    )
    
    # FIELD 6: Additional Notes (optional)
    additional_notes: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Additional notes for the engineer. "
            "Warnings, important context, caveats. "
            "Write in Russian."
        ),
    )
    
    # FIELD 7: Priority (optional)
    priority: str | None = Field(
        default=None,
        description=(
            "Ticket priority (if applicable): "
            "low / medium / high / critical"
        ),
    )
    
    # FIELD 8: Priority (optional)
    priority: str | None = Field(
        default=None,
        description=(
            "Ticket priority (if applicable): "
            "low / medium / high / critical"
        ),
    )
```

---

## 7. Handler Architecture

### Complete Flow with SRP Integration

```python
async def agent_chat_handler_with_srp(user_message, gradio_history, messages):
    """
    Complete handler flow with SRP integration:
    1. Guardian check (if enabled)
    2. SGR analysis (request understanding)
    3. Agent processes request with tools
    4. Generate final answer
    5. Show SRP planning bubble (if enabled)
    6. FORCE SRP tool call (if enabled)
    7. Hide SRP bubble on completion
    8. Extract headers from answer
    9. Build TOC (ONCE, conditionally)
    10. Assemble: [TOC] + Answer + [Plan]
    11. Store in downstream output
    """
    
    # ========== 0. GUARDIAN CHECK ==========
    if settings.guard_enabled:
        moderation_result = await guard_client.classify(user_message)
        # ... (existing SGR integration)
    
    # ========== 1. SGR ANALYSIS (Request Understanding) ==========
    sgr_plan = await execute_sgr_tool(messages)
    agent_context.sgr_plan = sgr_plan
    
    # ========== 2. AGENT PROCESSES REQUEST ==========
    agent = create_react_agent(
        model=llm,
        tools=available_tools,
        prompt=agent_system_prompt,
    )
    
    final_answer = ""
    async for stream_mode, chunk in agent.astream(
        {"messages": messages},
        config={...}
    ):
        final_answer += chunk.content
    
    # ========== 3. SHOW SRP PLANNING BUBBLE & EXECUTE ==========
    resolution_plan = None
    if settings.srp_enabled:
        try:
            # Show SRP planning bubble (like SGR)
            from rag_engine.api.stream_helpers import yield_srp_planning_started
            gradio_history.append(yield_srp_planning_started())
            yield list(gradio_history)
            
            # Build SRP context:
            # 1. Base system prompt + appended sources list
            # 2. Appendage: instruction to call generate_resolution_plan tool
            # 3. Full conversation history
            from rag_engine.llm.prompts import get_system_prompt
            
            srp_system_prompt = get_system_prompt()
            if agent_context.sources_compiled:
                srp_system_prompt += "\n\n" + agent_context.sources_compiled
            srp_system_prompt += "\n\n" + """After answering the user's question, analyze the conversation and call the generate_resolution_plan tool to create a support engineer resolution plan. The plan should summarize the issue and recommend next steps if human intervention is needed."""
            
            srp_messages = [{"role": "system", "content": srp_system_prompt}]
            srp_messages.extend(messages)
            
            srp_model = llm.bind_tools(
                [generate_resolution_plan_tool],
                tool_choice={"type": "function", "function": {"name": "generate_resolution_plan"}},
            )
            srp_response = await srp_model.ainvoke(srp_messages)
            tool_call = srp_response.tool_calls[0]
            resolution_plan = tool_call["args"]
            agent_context.resolution_plan = resolution_plan
            
            # Hide SRP bubble on completion
            from rag_engine.api.stream_helpers import update_message_status_in_history
            update_message_status_in_history(gradio_history, "srp_planning", "done")
            
        except Exception as exc:
            logger.error("SRP tool failed: %s", exc)
            agent_context.resolution_plan_error = str(exc)
            # Hide bubble even on failure
            from rag_engine.api.stream_helpers import update_message_status_in_history
            update_message_status_in_history(gradio_history, "srp_planning", "done")
            # resolution_plan stays None
    
    # ========== 4. EXTRACT HEADERS FROM ANSWER ==========
    answer_headers = extract_markdown_headers(final_answer)
    
    # ========== 5. BUILD TOC (ONCE, conditionally) ==========
    toc = ""
    if settings.toc_enabled:
        # TOC only if there are headers OR plan section
        has_content_for_toc = len(answer_headers) > 0 or resolution_plan is not None
        if has_content_for_toc:
            toc_content = build_toc(answer_headers, include_plan=(resolution_plan is not None and resolution_plan.get("engineer_intervention_needed", False)))
            if toc_content:
                toc = toc_content + "\n\n---\n\n"
    
    # ========== 6. RENDER PLAN ==========
    # Only render plan section if engineer_intervention_needed=True
    plan_section = ""
    if resolution_plan and resolution_plan.get("engineer_intervention_needed", False):
        plan_markdown = render_resolution_plan_markdown(resolution_plan)
        plan_section = "\n\n---\n\n" + plan_markdown
    
    # ========== 6.5 APPEND PRE-COMPILED SOURCES AT END ==========
    # Sources are pre-compiled elsewhere in the RAG pipeline (synthetic message)
    # Just append them at the very end after everything else
    # This happens regardless of whether SRP was called
    
    sources_section = "\n\n" + (agent_context.sources_compiled or "")
    
    # ========== 7. ASSEMBLE FINAL MESSAGE ==========
    # Structure: [TOC] + Answer + [Plan] (if applicable) + [Sources at end]
    complete_content = f"{toc}{final_answer}{plan_section}{sources_section}"
    
    # UI: Disclaimer (UI-only) + Complete Message (in context)
    gradio_history.append({
        "role": "assistant",
        "content": complete_content,
        "metadata": {
            "ui_type": "final_answer_with_plan",
            "resolution_plan": resolution_plan,
        }
    })
    
    yield gradio_history
    
    # ========== 8. PREPARE DOWNSTREAM OUTPUT ==========
    # Note: Full plan is in answer text, downstream gets status enum + metadata only
    
    # Determine SRP status
    if not settings.srp_enabled:
        srp_status = "disabled"
    elif agent_context.resolution_plan_error:
        srp_status = "failed"
    elif resolution_plan and not resolution_plan.get("engineer_intervention_needed", False):
        srp_status = "not_applicable"
    elif resolution_plan:
        srp_status = "generated"
    else:
        srp_status = "disabled"
    
    downstream_output = {
        "answer": final_answer,  # Contains complete message: TOC + Answer + Plan (or just Answer)
        "srp_status": srp_status,  # Enum: disabled|generated|not_applicable|failed
        "srp_error": getattr(agent_context, "resolution_plan_error", None),
        "metadata": {
            "srp_engineer_intervention_needed": resolution_plan.get("engineer_intervention_needed") if resolution_plan else None,
            "srp_outcome": resolution_plan.get("outcome") if resolution_plan else None,
            "srp_priority": resolution_plan.get("priority") if resolution_plan else None,
            "toc_generated": bool(toc),
            "heading_count": len(answer_headers),
            "srp_enabled": settings.srp_enabled,
            "toc_enabled": settings.toc_enabled,
        }
    }
    
    return downstream_output


def render_resolution_plan_markdown(plan: dict) -> str:
    """Render resolution plan as markdown string."""
    from rag_engine.api.i18n import get_text
    
    outcome_key = f"srp_outcome_{plan.get('outcome', 'unknown')}"
    outcome_text = get_text(outcome_key)
    
    doc_links = ""
    if plan.get("doc_references"):
        doc_links = "\n".join(f"- {ref}" for ref in plan["doc_references"])
    else:
        doc_links = get_text("srp_no_doc_references")
    
    notes = plan.get("additional_notes") or get_text("srp_no_additional_notes")
    
    return f"""# {get_text('srp_section_title')}

## {get_text('srp_issue_summary')}
{plan.get('issue_summary', '')}

## {get_text('srp_steps_completed')}
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(plan.get('steps_completed', [])))}

## {get_text('srp_next_steps')}
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(plan.get('next_steps', [])))}

## {get_text('srp_result')}
{outcome_text}

## {get_text('srp_additional_notes')}
{notes}"""
```

---

## 8. Three-Output Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SRP TOOL (Pure Function)                  │
│  Returns: Structured JSON with resolution plan              │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    HANDLER (Orchestrator)                    │
│  1. Agent generates final answer                             │
│  2. Show SRP bubble → Execute SRP tool (if enabled)         │
│  3. Hide SRP bubble on completion                           │
│  4. Extract headers from answer                             │
│  5. Store structured data → agent_context.resolution_plan   │
│  6. SKIP adding tool call/result to messages                │
│  7. Build TOC (ONCE, conditionally)                         │
│  8. Render plan as markdown                                 │
│  9. Assemble: [TOC] + Answer + [Plan]                       │
│  10. Prepare downstream output                              │
└─────────────────────────────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌─────────────┐
│   UI     │  │  Agent   │  │ Downstream  │
│ Display  │  │ Context  │  │  (CRM)      │
├──────────┤  ├──────────┤  ├─────────────┤
│Disclaimer│  │          │  │             │
│(UI-only) │  │          │  │             │
│ SRP      │  │          │  │             │
│ Bubble   │  │          │  │             │
│(hidden)  │  │          │  │             │
│ [TOC]    │  │ [TOC]    │  │             │
│ Answer   │  │ Answer   │  │  answer     │
│ [Plan]   │  │ [Plan]   │  │  + plan     │
│          │  │          │  │  + error    │
│          │  │          │  │  + toc_meta │
└──────────┘  └──────────┘  └─────────────┘
```

### Output Details

| Output Channel      | Content                                    | Visibility       |
| ------------------- | ------------------------------------------ | ---------------- |
| Disclaimer (UI)    | AI disclaimer                              | UI decoration    |
| SRP Bubble (UI)    | "📝 Generating support engineer plan..."  | UI decoration (hidden on completion) |
| Context (Model)     | [TOC] + Answer + [Plan]                    | Model sees it   |
| UI (User)          | [TOC] + Answer + [Plan] (same as context) | User sees it    |
| Downstream (CRM)   | Answer + Plan + error + toc metadata      | System sees it  |

**Note:** 
- Disclaimer remains as separate UI-only message (unchanged from current behavior)
- SRP bubble is temporary UI decoration, hidden when SRP completes
- Brackets [] indicate conditional content

---

## 9. Data Visibility Matrix

| Data                    | Disclaimer (UI) | SRP Bubble (UI) | Context (Model) | UI (User) | Downstream (CRM) |
| ---------------------- | --------------- | --------------- | --------------- | ---------- | ----------------- |
| SRP Bubble indicator  | ❌ No           | ✅ Yes (temp)   | ❌ No           | ❌ No      | ❌ No             |
| TOC (if generated)    | ❌ No           | ❌ No           | ✅ Yes          | ✅ Yes     | ❌ No             |
| Complete Answer Text  | ❌ No           | ❌ No           | ✅ Yes          | ✅ Yes     | ✅ Yes (in answer field) |
| SRP Plan Content      | ❌ No           | ❌ No           | ✅ Yes          | ✅ Yes     | ❌ No (in answer) |
| SRP Generated Flag    | ❌ No           | ❌ No           | ❌ No           | ❌ No      | ✅ Yes            |
| SRP Outcome           | ❌ No           | ❌ No           | ✅ Yes          | ✅ Yes     | ✅ Yes (metadata) |
| SRP Priority          | ❌ No           | ❌ No           | ✅ Yes          | ✅ Yes     | ✅ Yes (metadata) |
| SRP Doc Count         | ❌ No           | ❌ No           | ❌ No           | ❌ No      | ✅ Yes (metadata) |
| SRP Error             | ❌ No           | ❌ No           | ❌ No           | ❌ No      | ✅ Yes            |
| TOC Generated Flag    | ❌ No           | ❌ No           | ❌ No           | ❌ No      | ✅ Yes (metadata) |
| Heading Count         | ❌ No           | ❌ No           | ❌ No           | ❌ No      | ✅ Yes (metadata) |
| Tool call trace       | ❌ No           | ❌ No           | ❌ No           | ❌ No      | ❌ No             |

---

## 10. Configuration Options

```python
# Settings in rag_engine/config/settings.py

# SRP Enabled (opt-in, controlled via SRP_ENABLED env var)
srp_enabled: bool = Field(description="Enable Support Resolution Plan tool")

# TOC Enabled (separate flag for A/B testing, controlled via TOC_ENABLED env var)
toc_enabled: bool = Field(description="Enable dynamic Table of Contents generation")
```

`.env`:
```
# SRP and TOC configuration (opt-in)
SRP_ENABLED=false          # Enable Support Resolution Plan generation
TOC_ENABLED=false          # Enable dynamic Table of Contents
```

**Benefits of separate flags:**
- **Independent testing** - Can enable TOC without SRP (and vice versa)
- **A/B experiments** - Easy to toggle features independently
- **Gradual rollout** - Enable TOC first, then SRP, or both
- **Performance testing** - Measure impact of each feature separately

---

## 11. i18n Keys

```python
# English
ENGLISH = {
    # SRP Bubble UI
    "srp_planning_title": "📝 Generating support engineer plan",
    "srp_planning_content": "Analyzing conversation and building resolution steps...",
    # SRP Content
    "srp_toc_title": "Table of Contents",
    "srp_section_title": "Support Engineer Resolution Plan",
    "srp_issue_summary": "Issue Summary",
    "srp_steps_completed": "Steps Completed",
    "srp_next_steps": "Recommended Next Steps",
    "srp_result": "Result",
    "srp_additional_notes": "Additional Notes",
    "srp_outcome_resolved": "Resolved",
    "srp_outcome_partially_resolved": "Partially Resolved",
    "srp_outcome_escalation_required": "Escalation Required",
    "srp_outcome_user_followup_needed": "User Follow-up Needed",
    "srp_outcome_unknown": "Unknown",
    "srp_no_additional_notes": "No additional notes",
}

# Russian
RUSSIAN = {
    # SRP Bubble UI
    "srp_planning_title": "📝 Формирую план для инженера поддержки",
    "srp_planning_content": "Анализирую диалог и создаю план решения...",
    # SRP Content
    "srp_toc_title": "Оглавление",
    "srp_section_title": "План решения для инженера поддержки",
    "srp_issue_summary": "Краткое описание проблемы",
    "srp_steps_completed": "Выполненные шаги",
    "srp_next_steps": "Рекомендуемые следующие шаги",
    "srp_result": "Результат",
    "srp_additional_notes": "Примечания",
    "srp_outcome_resolved": "Решено",
    "srp_outcome_partially_resolved": "Частично решено",
    "srp_outcome_escalation_required": "Требуется эскалация",
    "srp_outcome_user_followup_needed": "Требуется уточнение у пользователя",
    "srp_outcome_unknown": "Неизвестно",
    "srp_no_additional_notes": "Дополнительных примечаний нет",
}
```

---

## 12. Implementation Phases

### Phase 1: Core SRP Tool

- [ ] Add `SRP_PROMPT` to `rag_engine/llm/prompts.py`
- [ ] Create `rag_engine/tools/generate_resolution_plan.py`
- [ ] Define Pydantic schema `ResolutionPlanResult` in `rag_engine/llm/schemas.py`
- [ ] Implement tool function with schema binding
- [ ] Add tool to tools registry (`rag_engine/tools/__init__.py`)

### Phase 2: Template & Rendering

- [ ] Implement `extract_markdown_headers()` function (lightweight, no deps)
- [ ] Implement `build_toc()` function
- [ ] Implement `render_resolution_plan_markdown()` function
- [ ] Implement `yield_srp_planning_started()` helper in `rag_engine/api/stream_helpers.py`
- [ ] Add all i18n keys to `rag_engine/api/i18n.py` (including bubble text)
- [ ] Test template rendering

### Phase 3: Handler Integration

- [ ] Add `SRP_ENABLED` setting to `rag_engine/config/settings.py`
- [ ] Add `TOC_ENABLED` setting to `rag_engine/config/settings.py`
- [ ] Modify `agent_chat_handler` in `rag_engine/api/app.py`:
  - Stream/generate answer
  - Show SRP planning bubble (if srp_enabled)
  - Force SRP call (if enabled)
  - Hide SRP bubble on completion (success or failure)
  - Extract headers from answer
  - Build TOC (ONCE, conditionally based on settings + content)
  - Assemble final message: [TOC] + Answer + [Plan]
- [ ] Store resolution plan in downstream output
- [ ] Handle SRP errors gracefully (clean TOC or no TOC, log error)

### Phase 4: Testing

- [ ] Unit tests for header extraction (various edge cases)
- [ ] Unit tests for TOC builder (with/without headings, with/without plan)
- [ ] Unit tests for schema validation
- [ ] Unit tests for template rendering
- [ ] Unit tests for SRP bubble helper
- [ ] Integration test: full flow with SRP + TOC
- [ ] Verify SRP bubble appears when srp_enabled=true
- [ ] Verify SRP bubble hidden on completion (success)
- [ ] Verify SRP bubble hidden on failure
- [ ] Verify TOC IS in agent context (when generated)
- [ ] Verify TOC NOT generated when no headings
- [ ] Verify TOC NOT generated when toc_enabled=false
- [ ] Verify tool trace NOT in message history
- [ ] Verify plan IS in final answer (when SRP succeeds)
- [ ] Verify downstream output contains plan string
- [ ] Verify TOC generated only once
- [ ] Verify error handling (SRP disabled, SRP fails)

---

## 13. Error Handling

### SRP Disabled
- Skip SRP call entirely
- TOC generated only if `toc_enabled=true` AND has headers
- No plan section
- Normal answer flow with optional TOC

### SRP Fails
- TOC generated only if `toc_enabled=true` AND (has headers OR plan would exist)
  - Since SRP failed, no plan section
  - So TOC only if has headers
- No plan section in message
- Error logged:
  - In debug pane (visible to engineers)
  - In `agent_context.resolution_plan_error`
  - In downstream output
- User sees [TOC] + Answer or just Answer (no error message in UI)

### TOC Disabled
- Skip TOC generation entirely
- Plan section still generated if SRP enabled and succeeds
- Answer flows normally

### Downstream Error Reporting
```python
downstream_output = {
    "answer": final_answer,  # Contains complete message: TOC + Answer + Plan
    "srp_status": srp_status,  # Enum: disabled|generated|not_applicable|failed
    "srp_error": error_message or None,  # Only if failed
    "metadata": {
        "srp_engineer_intervention_needed": resolution_plan.get("engineer_intervention_needed") if resolution_plan else None,
        "srp_outcome": resolution_plan.get("outcome") if resolution_plan else None,
        "srp_priority": resolution_plan.get("priority") if resolution_plan else None,
        "toc_generated": bool(toc),
        "heading_count": len(answer_headers),
        "srp_enabled": settings.srp_enabled,
        "toc_enabled": settings.toc_enabled,
    }
}
```

---

## 14. Comparison: SGR vs SRP

| Aspect                    | SGR Tool                        | SRP Tool                        |
| ------------------------- | ------------------------------- | ------------------------------- |
| **Purpose**              | Analyze incoming request        | Generate resolution plan        |
| **Trigger**               | Start of turn (request intake) | End of turn (answer generation)|
| **Audience**              | AI Agent (internal)            | Human support engineer          |
| **Output**                | Structured JSON                | Markdown + JSON                 |
| **Display**               | Injected as reasoning          | [TOC] + Answer + [Plan]        |
| **TOC**                   | N/A                            | Conditional (configurable)     |
| **Storage**               | agent_context.sgr_plan         | agent_context.resolution_plan  |
| **Schema Fields**         | topic, intent, confidence     | issue_summary, steps, next    |
| **SRP Config**            | Always enabled                 | Opt-in (SRP_ENABLED)           |
| **TOC Config**            | N/A                            | Opt-in (TOC_ENABLED)           |

---

## 15. Success Metrics

### Quantitative
- **Plan Completeness:** All required fields populated
- **Next Steps Actionability:** Human engineer can follow steps
- **Doc Reference Accuracy:** Links are valid and relevant
- **Downstream Integration:** Plan string correctly stored
- **TOC Accuracy:** Headers correctly extracted and linked
- **TOC Generation:** Generated exactly once per response (when applicable)
- **Conditional Logic:** TOC skipped when no headings
- **A/B Testing:** Separate flags allow independent feature testing
- **Context Integration:** TOC visible to LLM in multi-turn conversations

### Qualitative
- **Clarity:** Plan is easy to understand
- **Actionability:** Engineer knows what to do next
- **Consistency:** Plan aligns with actual answer provided
- **Navigation:** TOC provides useful navigation within long answers
- **Clean UI:** No empty TOC for simple/short answers

---

## 16. References

### From Web Research
1. IT Support Ticket Resolution Checklist - ChecklistGuro
2. Support Ticket Response Template - Suptask
3. Handling Customer Support Tickets Process - Trainual
4. Issue Resolution Process Flowchart - MockFlow

### Codebase References
1. `rag_engine/tools/analyse_user_request.py` - Existing SGR tool
2. `rag_engine/api/app.py` - Agent chat handler
3. `rag_engine/api/i18n.py` - Internationalization system
4. `.opencode/plans/sgr_synthetic_assistant_enhancement_plan.md` - SGR architecture

---

## 17. Sources Placement Best Practice

### Research Findings

Based on industry research and documentation standards:

1. **CustomGPT Documentation**: "After the bot's response – Displays citations automatically at the end of the response. Useful when transparency is important."

2. **Best Practice Pattern**: Sources at end follows normal documentation conventions:
   - Clean separation between answer and evidence
   - Transparency without cluttering main response
   - Follows academic and technical documentation standards
   - Easy for users to reference and verify later

3. **Benefits for Support Engineers**:
   - Plan section focused on action items (no source duplication)
   - Sources consolidated at end for verification
   - Clear distinction between "what to do" and "what was referenced"

### Implementation

**Key Principle:** Sources are pre-compiled elsewhere in the RAG pipeline (synthetic message) and simply appended at the very end. No extraction needed.

```python
# After assembling: TOC + Answer + [Plan if applicable]
# Append pre-compiled sources list at the very end

sources_section = "\n\n" + (agent_context.sources_compiled or "")
complete_content = f"{toc}{final_answer}{plan_section}{sources_section}"
```

This happens **regardless of whether SRP was called or not**.

### Final Message Structure (All Scenarios)

```
[TOC - if enabled and has content]
[Answer]
[SRP Plan Section - if engineer_intervention_needed=True]
---
Sources section (pre-compiled, always at very end)
```

**Three scenarios:**
1. **SRP disabled** → Answer + Sources
2. **SRP enabled, engineer_intervention_needed=False** → Answer + Sources  
3. **SRP enabled, engineer_intervention_needed=True** → Answer + Plan + Sources

Sources always appear at the very end, after any SRP plan section.

---

**Status:** ✅ APPROVED - Ready for Implementation

**Key Decisions:**
- ✅ TOC prepended to final answer (in agent context) - only when applicable
- ✅ No TOC generated if answer has no headings
- ✅ Disclaimer remains as separate UI-only message (unchanged)
- ✅ Single message structure: [TOC] + Answer + [Plan] + [Pre-compiled Sources at end]
- ✅ TOC generated **ONCE**, after SRP result known
- ✅ No plan link in TOC if SRP fails or plan not applicable (clean TOC)
- ✅ **Separate config flags:** `SRP_ENABLED=false`, `TOC_ENABLED=false` (both opt-in)
- ✅ **SRP Bubble:** "📝 Generating support engineer plan..." UI indicator (like SGR, hidden on completion)
- ✅ **Downstream Output:** Status enum + metadata only (full plan is in answer text)
- ✅ **Two-stage approach:** Answer generation first, then SRP analysis (superior to one-stage)
- ✅ **Engineer Intervention Decision:** LLM decides via `engineer_intervention_needed` field (required)
- ✅ **doc_references removed from SRP:** Sources already in answer text, pre-compiled separately
- ✅ **Sources at End:** Pre-compiled sources list appended at very end (synthetic message)
- ✅ **Sources always at end:** Regardless of SRP disabled/enabled/needed
- ✅ Lightweight header extraction (no external deps)
- ✅ A/B testing support via independent feature flags

**Next Steps:**
1. Implement Phase 1: Core SRP tool
2. Implement Phase 2: Template & rendering  
3. Implement Phase 3: Handler integration
4. Implement Phase 4: Testing
