# Support Resolution Plan (SRP) Tool - Implementation Plan

**Status:** DRAFT  
**Created:** 2026-02-14  
**Based on:** Analysis of existing SGR architecture, IT support best practices, and ticket resolution workflows

---

## 1. Executive Summary

### Purpose
Create a second SGR-style tool that generates a **step-by-step resolution plan for human support engineers** to resolve user issues based on the conversation context. Unlike SGR (which analyzes the incoming request), this tool operates at **final answer generation** to provide actionable guidance for support staff.

### Key Characteristics
- **Trigger:** Forced tool call during final answer generation (not at request intake)
- **Output:** Structured resolution plan with numbered steps
- **Display:** Appended as H1 section in final answer (single assistant message)
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
┌─────────────────────────┐
│ SRP Tool (generate_)   │ ← Generates resolution plan
│ - Issue summarization   │
│ - Step-by-step guide   │
│ - Verification steps    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Final Answer + Plan    │
│ - Answer to user       │
│ - Resolution plan for   │
│   support engineer      │
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

# 5. Append to final answer or emit as separate message
final_response = f"{final_answer}\n\n---\n\n{plan_section}"
```

### Decision 3: Synthetic Injection (Same as SGR)
**Status:** ✅ CONFIRMED - ALWAYS ON

**Rationale:** SRP follows the exact same synthetic injection pattern as SGR:
- Forced tool call produces structured output
- Extract tool arguments → render as markdown
- Inject as assistant message in conversation context
- Display in UI as H1 section
- Store in downstream output

The difference from SGR: Content visibility
- **SGR**: Hidden from user, used for model continuation
- **SRP**: Visible to user (plan for support engineer), also in context for multi-turn

**Implementation: Append as H1 Section**
```markdown
[Final Answer to User]

---

# План решения для инженера поддержки

## Краткое описание проблемы
[Summary of the issue]

## Выполненные шаги
1. [Step 1 - what was done]
2. [Step 2 - what was done]
...

## Рекомендуемые следующие шаги
1. [Next step for engineer]
2. [Next step for engineer]
...

## Результат
[Outcome status]

## Ссылки на документацию
- [Doc reference]

## Примечания
[Additional notes]
```

### Decision 4: Downstream Storage as String
**Status:** ✅ CONFIRMED

**Rationale:** The plan must be stored as a string field for integration with CRM/ticketing systems.

```python
# Structured output for downstream
class AgentOutput(BaseModel):
    answer: str
    resolution_plan: str  # Markdown string for CRM/ticketing
    metadata: dict
```

---

## 3. Template System Design

### Resolution Plan Template Structure

```markdown
# План решения для инженера поддержки

## Краткое описание проблемы
{issue_summary}

## Выполненные шаги
1. {step_1}
2. {step_2}
3. {step_n}

## Рекомендуемые следующие шаги
{next_steps}

## Результат
{outcome_status}

## Ссылки на документацию
- {doc_link_1}
- {doc_link_2}

## Примечания
{additional_notes}
```

### Template Variables

```python
TEMPLATE_VARIABLES = {
    "issue_summary": "Brief summary of the user's issue (2-3 sentences)",
    "steps_completed": "List of steps already taken by the system",
    "next_steps": "Recommended next steps for human engineer",
    "outcome_status": "Resolved / Partially Resolved / Escalation Required / User Follow-up Needed",
    "doc_references": "Links to KB articles used",
    "additional_notes": "Any warnings, caveats, or context",
}
```

### SRP Prompt for prompts.py

Add to `rag_engine/llm/prompts.py`:

```python
# Support Resolution Plan prompt (prepended to messages when calling SRP tool)
SRP_PROMPT = """Проанализируй результаты поиска в базе знаний по запросу пользователя и составь план решения для специалиста поддержки."""
```

---

## 4. Pydantic Schema Design

Note: Use English descriptions for token efficiency. Include note for LLM to fill actual text values in Russian (human-facing content), while enum values remain in English.

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class ResolutionOutcome(str, Enum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    ESCALATION_REQUIRED = "escalation_required"
    USER_FOLLOWUP_NEEDED = "user_followup_needed"

class ResolutionPlanResult(BaseModel):
    """Support Resolution Plan - generates actionable plan for human engineers.
    
    Called at final answer generation to provide:
    1. Issue summary for context
    2. Steps already taken by the system
    3. Recommended next steps for human engineer
    4. Outcome status
    5. Documentation references
    
    NOTE: Fill text fields in Russian for human readability.
    Enum values remain in English for token efficiency.
    """
    
    # FIELD 1: Issue Summary
    issue_summary: str = Field(
        ...,
        max_length=500,
        description=(
            "Brief summary of the user's issue in 2-3 sentences. "
            "Write in Russian for the support engineer."
        ),
    )
    
    # FIELD 2: Steps Completed by System
    steps_completed: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "List of steps already taken by the system. "
            "Include: KB search, documentation analysis, solutions provided. "
            "Write in Russian, be concise and informative."
        ),
    )
    
    # FIELD 3: Recommended Next Steps
    next_steps: list[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "Recommended next steps for the support engineer. "
            "What does the human need to do after this response? "
            "Examples: 'Check user permissions', 'Update documentation', 'Create dev ticket'. "
            "Write in Russian."
        ),
    )
    
    # FIELD 4: Outcome Status (enum in English)
    outcome: ResolutionOutcome = Field(
        ...,
        description=(
            "Resolution status based on the answer provided. "
            "Use English enum value: "
            "'resolved': Fully resolved; "
            "'partially_resolved': Partially resolved, additional actions needed; "
            "'escalation_required': Requires escalation; "
            "'user_followup_needed': User follow-up required."
        ),
    )
    
    # FIELD 5: Documentation References
    doc_references: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Documentation or KB article references used. "
            "Full URLs or article IDs."
        ),
    )
    
    # FIELD 6: Additional Notes
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
```

---

## 5. Handler Architecture

### Complete Flow with SRP Integration

```python
async def agent_chat_handler_with_srp(user_message, gradio_history, messages):
    """
    Complete handler flow with SRP integration:
    1. Guardian check (if enabled)
    2. SGR analysis (request understanding)
    3. Agent processes request with tools
    4. Generate final answer
    5. FORCE SRP tool call (resolution plan)
    6. Clean injection (remove tool trace)
    7. Append plan to final answer
    8. Store in downstream output
    """
    
    # ========== 0. GUARDIAN CHECK ==========
    if settings.guard_enabled:
        moderation_result = await guard_client.classify(user_message)
        # ... (existing SGR integration)
    
    # ========== 1. SGR ANALYSIS (Request Understanding) ==========
    # Existing SGR flow: analyze_user_request tool
    sgr_plan = await execute_sgr_tool(messages)
    agent_context.sgr_plan = sgr_plan
    
    # ========== 2. AGENT PROCESSES REQUEST ==========
    # Normal agent flow with tools (search_kb, etc.)
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
        # Accumulate final answer
        final_answer += chunk.content
    
    # ========== 3. FORCE SRP TOOL CALL ==========
    # Build context for SRP: conversation + final answer
    srp_context = build_srp_context(messages, final_answer, sgr_plan)
    
    srp_model = llm.bind_tools(
        [generate_resolution_plan_tool],
        tool_choice={"type": "function", "function": {"name": "generate_resolution_plan"}},
    )
    
    srp_response = await srp_model.ainvoke(srp_context)
    tool_call = srp_response.tool_calls[0]
    resolution_plan = tool_call["args"]
    
    # ========== 4. STORE FOR DOWNSTREAM ==========
    agent_context.resolution_plan = resolution_plan
    
    # ========== 5. CLEAN INJECTION ==========
    # Remove tool trace from context (do NOT add to messages)
    # Skip: messages.append({"role": "assistant", "content": None, "tool_calls": [...]})
    # Skip: messages.append({"role": "tool", "content": srp_result})
    
    # ========== 6. RENDER PLAN ==========
    # Always append as H1 section to final answer (single message)
    plan_markdown = render_resolution_plan_markdown(resolution_plan)
    final_response = f"{final_answer}\n\n---\n\n{plan_markdown}"
    
    gradio_history.append({
        "role": "assistant",
        "content": final_response,
        "metadata": {
            "ui_type": "final_answer_with_plan",
            "resolution_plan": resolution_plan
        }
    })
    
    yield gradio_history
    
    # ========== 7. PREPARE DOWNSTREAM OUTPUT ==========
    downstream_output = {
        "answer": final_answer,
        "resolution_plan": plan_markdown,  # String for CRM
        "sgr_plan": agent_context.sgr_plan,
        "metadata": {
            "outcome": resolution_plan["outcome"],
            "priority": resolution_plan.get("priority"),
            "doc_references": resolution_plan["doc_references"],
        }
    }
    
    return downstream_output


def build_srp_context(messages, final_answer, sgr_plan):
    """Build context for SRP tool call.
    
    Includes:
    - Recent conversation history
    - Final answer generated by agent
    - SGR analysis (for context)
    
    Note: Follow the pattern from prompts.py - add SRP_PROMPT suffix and prepend to messages.
    """
    
    # Build user context with final answer and SGR plan
    user_context = f"""Контекст запроса:
- Тема: {sgr_plan.get('topic', 'N/A')}
- Намерение пользователя: {sgr_plan.get('user_intent', 'N/A')}

Ответ агента:
{final_answer}

Составь план решения для инженера поддержки."""

    # Prepend SRP prompt suffix to existing messages (like other tool prompts in prompts.py)
    from rag_engine.llm.prompts import SRP_PROMPT
    
    context_messages = [{"role": "system", "content": SRP_PROMPT}] + messages + [{"role": "user", "content": user_context}]
    
    return context_messages


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
    
    return f"""# План решения для инженера поддержки

## Краткое описание проблемы
{plan.get('issue_summary', '')}

## Выполненные шаги
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(plan.get('steps_completed', [])))}

## Рекомендуемые следующие шаги
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(plan.get('next_steps', [])))}

## Результат
{outcome_text}

## Ссылки на документацию
{doc_links}

## Примечания
{notes}"""
```

---

## 6. Three-Output Architecture (Extended)

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
│  2. Execute SRP tool (forced call)                          │
│  3. Store structured data → agent_context.resolution_plan   │
│  4. SKIP adding tool call/result to messages                │
│  5. Render plan as markdown                                 │
│  6. Append/Inject to final answer                           │
│  7. Prepare downstream output                               │
└─────────────────────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌──────────┐   ┌─────────────┐
│Answer   │   │Plan      │   │Structured   │
│to User  │   │Section   │   │Output       │
│(UI)     │   │(UI)      │   │(Downstream) │
└─────────┘   └──────────┘   └─────────────┘
```

### Output Details

Note: SRP plan is visible in ALL THREE outputs (unlike SGR which is hidden from user).

| Output Channel      | Content                                    | Visibility       |
| ------------------- | ------------------------------------------ | ---------------- |
| Context (Model)     | Injected assistant message with plan       | Model sees it   |
| UI (User)          | H1 section appended to answer             | User sees it    |
| Downstream (CRM)   | Markdown string in structured output       | System sees it  |

---

## 7. Data Visibility Matrix

Note: SRP plan is visible in ALL outputs (Context, UI, Downstream).

| Data                    | Context (Model) | UI (User) | Downstream (CRM) |
| ---------------------- | --------------- | ---------- | ----------------- |
| Issue summary          | ✅ Yes          | ✅ Yes     | ✅ Yes            |
| Steps completed        | ✅ Yes          | ✅ Yes     | ✅ Yes            |
| Next steps             | ✅ Yes          | ✅ Yes     | ✅ Yes            |
| Outcome status         | ✅ Yes          | ✅ Yes     | ✅ Yes            |
| Doc references         | ✅ Yes          | ✅ Yes     | ✅ Yes            |
| Additional notes       | ✅ Yes          | ✅ Yes     | ✅ Yes            |
| Tool call trace       | ❌ No           | ❌ No      | ❌ No             |

---

## 8. Configuration Options

```python
# Settings in rag_engine/config/settings.py

# SRP Enabled
srp_enabled: bool = True
```

---

## 9. Implementation Phases

### Phase 1: Core SRP Tool

- [ ] Add `SRP_PROMPT` to `rag_engine/llm/prompts.py` (like other tool prompts)
- [ ] Create `rag_engine/tools/generate_resolution_plan.py`
- [ ] Define Pydantic schema `ResolutionPlanResult`
- [ ] Implement tool function with schema binding
- [ ] Add tool to tools registry

### Phase 2: Template & Rendering

- [ ] Define markdown template for resolution plan
- [ ] Implement `render_resolution_plan_markdown()` function
- [ ] Add i18n keys to `rag_engine/api/i18n.py`:
  - `srp_outcome_resolved`
  - `srp_outcome_partially_resolved`
  - `srp_outcome_escalation_required`
  - `srp_outcome_user_followup_needed`
  - `srp_outcome_unknown`
  - `srp_no_doc_references`
  - `srp_no_additional_notes`
- [ ] Test template rendering

### Phase 3: Handler Integration

- [ ] Modify `agent_chat_handler` to force SRP call after answer generation
- [ ] Implement clean injection (remove tool trace)
- [ ] Append plan as H1 section to final answer (single message)
- [ ] Store in downstream output

### Phase 4: Testing

- [ ] Unit tests for schema validation
- [ ] Unit tests for template rendering
- [ ] Integration test: full flow with SRP
- [ ] Verify tool trace NOT in message history
- [ ] Verify plan IS in final answer
- [ ] Verify downstream output contains plan string

---

## 10. Comparison: SGR vs SRP

| Aspect                    | SGR Tool                        | SRP Tool                        |
| ------------------------- | ------------------------------- | ------------------------------- |
| **Purpose**              | Analyze incoming request        | Generate resolution plan        |
| **Trigger**               | Start of turn (request intake) | End of turn (answer generation)|
| **Audience**              | AI Agent (internal)            | Human support engineer          |
| **Output**                | Structured JSON                | Markdown + JSON                 |
| **Display**               | Injected as reasoning          | Appended to final answer       |
| **Storage**               | agent_context.sgr_plan         | agent_context.resolution_plan  |
| **Schema Fields**         | topic, intent, confidence     | issue_summary, steps, next    |

---

## 11. Success Metrics

### Quantitative
- **Plan Completeness:** All required fields populated
- **Next Steps Actionability:** Human engineer can follow steps
- **Doc Reference Accuracy:** Links are valid and relevant
- **Downstream Integration:** Plan string correctly stored

### Qualitative
- **Clarity:** Plan is easy to understand
- **Actionability:** Engineer knows what to do next
- **Consistency:** Plan aligns with actual answer provided

---

## 12. Open Questions

1. **Should plan be visible to user?** 
   - Current decision: Yes, as H1 section or separate message
   - Rationale: Provides transparency on resolution process

2. **What if SRP fails?**
   - Fallback: Skip plan generation, emit answer only
   - Log error for debugging

3. **Should we include SLA recommendations?**
   - Future enhancement, not in initial version

---

## 13. References

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

**Next Steps:**
1. Review and approve plan
2. Implement Phase 1: Core SRP tool
3. Implement Phase 2: Template & rendering
4. Implement Phase 3: Handler integration
5. Test and validate
