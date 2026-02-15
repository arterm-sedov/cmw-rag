# SGR Formatted Tool Result Enhancement Plan

**Status:** DRAFT - REVISED ARCHITECTURE (FORMATTED TOOL RESULT)
**Created:** 2026-02-13
**Updated:** 2026-02-15 (Changed from Synthetic Assistant Injection to Formatted Tool Result)
**Based on:** Analysis of current SGR implementation, novelty research, and early termination issue identified

---

## 1. Executive Summary

### Current State
- SGR tool (`analyse_user_request`) extracts structured plan via forced tool call
- Plan stored in `AgentContext.sgr_plan` for external use (UI, metrics)
- Model receives raw JSON via standard `role: "tool"` message
- Guardian already integrated and runs before SGR
- UI already displays `user_intent` - but model doesn't leverage it as "its own reasoning"

### Current Flow (BEFORE Enhancement)
```
User Request
    │
    ▼
┌─────────────────┐
│ Guardian Check  │ (if enabled)
│ - enforce/report│
└─────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Forced SGR Tool Call    │
│ Returns: Raw JSON       │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Tool Call in History    │
│ + role: "tool" result   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Agent continues with    │
│ raw JSON in context     │
└─────────────────────────┘
```

### Proposed Enhancement (FORMATTED TOOL RESULT)
**Format SGR tool result** as human-readable markdown instead of raw JSON. Model sees actionable instructions in the tool result—maintains natural ReAct flow without early termination risk.

```
User Request
    │
    ▼
┌─────────────────┐
│ Guardian Check  │ (if enabled)
│ - enforce/report│
└─────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Forced SGR Tool Call    │
│ Returns: Formatted      │
│ markdown analysis       │ → Stored in agent_context.sgr_plan (JSON)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Tool Call in History    │
│ + role: "tool" result  │ (formatted, not raw JSON)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Agent continues with    │
│ formatted analysis      │
│ in context             │
└─────────────────────────┘
```

**Key Advantages:** 
- No early termination risk (model continues naturally after tool result)
- Maintains existing ReAct flow (no architectural changes)
- Minimal implementation effort (change only tool return format)
- Structured JSON preserved for downstream (UI, metrics, debugging)
- Model receives actionable directives, not confusing raw data

---

## 2. Core Architectural Decisions

### Decision 1: Single Combined Assistant Message
**Status:** ✅ CONFIRMED

**Rationale:** Two consecutive assistant messages (analysis + response) create ambiguity. The model might treat the response as already answering the user, causing premature turn termination.

**Solution:** Combine analysis and response into ONE assistant message in the agent context:

```python
# Single combined message
synthetic_message = f"""## Request Analysis
- **Topic**: {topic}
- **Intent**: {user_intent}
...

## Response

{response}"""

messages.append({
    "role": "assistant",
    "content": synthetic_message  # Analysis + Response combined
})
```

**UI Display:** Only the `## Response` section is shown to the user (preceded by user_intent), not the full analysis.

### Decision 2: Unbind SGR Tool After First Call
**Status:** ✅ CONFIRMED

**Rationale:** No plan stacking per turn. SGR should analyze once at the start, then the agent proceeds with that plan. Mid-turn re-planning causes instability and confusion.

**Implementation:**
```python
# After SGR execution, remove from available tools
available_tools = [t for t in all_tools if t.name != "analyse_user_request"]

# Create agent WITHOUT SGR tool for remainder of turn
agent = create_react_agent(
    model=llm,
    tools=available_tools,  # SGR tool excluded
    ...
)
```

**Turn Flow:**
1. User sends message
2. Guardian check (if enabled)
3. **FORCED** SGR tool call (exactly once)
4. SGR tool **removed** from available tools
5. Agent continues with remaining tools (search_kb, answer_user, etc.)
6. Next user message → SGR tool re-enabled for new turn

### Decision 3: Format Tool Result (NOT Remove It)
**Status:** ✅ CONFIRMED (REVISED)

**Rationale:** Tool result (Observation) is a core part of ReAct pattern. Removing it causes early termination issues. Instead, we format the tool result to be human-readable and actionable—model continues naturally after processing it.

**Implementation:**
```python
# 1. Execute SGR tool (returns formatted string instead of JSON)
sgr_result = await execute_sgr_tool(messages)  # Returns formatted markdown

# 2. Parse for storage (extract JSON fields)
sgr_plan = extract_plan_from_result(sgr_result)  # Parse back to dict

# 3. Store structured data for downstream (JSON)
agent_context.sgr_plan = sgr_plan

# 4. Tool call + formatted result stay in message history
messages.append({
    "role": "assistant",
    "content": None,
    "tool_calls": [...]
})

messages.append({
    "role": "tool",
    "content": sgr_result,  # Formatted markdown (not raw JSON)
    "tool_call_id": call_id
})
```

**Result:** Model sees formatted, actionable analysis in tool result and continues naturally via ReAct cycle.

### Decision 4: Directive Template Language
**Status:** ✅ CONFIRMED

**Rationale:** Tool result must instruct the model what to do next—not pretend to be the model's own thinking. Use imperative language like "Call retrieve_context tool with:" not "I will search the knowledge base".

**Implementation:**
```python
# BAD (confusing - pretends to be LLM):
"I will search the knowledge base using these queries..."

# GOOD (directive - instructs LLM):
"Call retrieve_context tool with these queries:
- query1
- query2"

# Model processes this as: "I need to call this tool next"
```

---

## 3. Three-Output Architecture (FORMATTED TOOL RESULT)

Based on the architectural decisions, we maintain a **three-output architecture** with **formatted tool result** (NOT removed):

```
┌─────────────────────────────────────────────────────────────┐
│                    SGR TOOL (Pure Function)                  │
│  Returns: Formatted markdown for model + JSON for downstream │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    HANDLER (Orchestrator)                    │
│  1. Guardian check → add classification to system message   │
│  2. Execute SGR tool internally                              │
│  3. Store structured data → agent_context.sgr_plan (JSON)  │
│  4. Return formatted string from tool (for model)           │
│  5. Tool call + formatted result in messages (ReAct flow)   │
│  6. Unbind SGR tool for remainder of turn                  │
│  7. Route based on action                                   │
└─────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Output 1:       │  │ Output 2:       │  │ Output 3:       │
│ Context         │  │ UI Response     │  │ Structured      │
│ Tool Result     │  │ Synthetics      │  │ Metadata        │
│ (Agent Context) │  │ (User Display)  │  │ (Downstream)    │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ - Formatted     │  │ - Response only │  │ - agent_context │
│   analysis      │  │ - No reasoning  │  │   .sgr_plan     │
│ - Actionable    │  │ - Preceded by   │  │ - Guardian      │
│ - Tool call in  │  │   user_intent   │  │   result        │
│   history       │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Three Outputs Explained:**

1. **Context Tool Result** (Output 1) → Model's view
   - Formatted markdown with directive instructions
   - Stays in message history as standard ReAct tool result
   - Model processes and continues naturally

2. **UI Response Synthetics** (Output 2) → User's view
   - Only the response text (extracted from structured data)
   - Preceded by user_intent display
    - No internal reasoning shown (spam_score, clarification_questions_to_ask, etc.)
   - Rendered via Gradio history

3. **Structured Metadata** (Output 3) → System/downstream
   - Raw JSON from SGR tool (stored in `agent_context.sgr_plan`)
   - Guardian classification results
   - Available for: metrics, analytics, debugging, future tools
   - Not visible to model or user

**Key Differences from Original Plan:**
- ✅ Tool result STAYS in context (formatted, not raw JSON)
- ✅ No early termination risk (ReAct flow preserved)
- ✅ Directive template language (not pretending to be LLM)
- ✅ Minimal implementation changes
- ✅ All SGR fields preserved in JSON for downstream

---

## 4. Template System Design (FORMATTED TOOL RESULT)

### Template Structure (Directive Tool Result)

All templates produce a **single tool result** formatted as directive instructions for the LLM.
Key principles:
- **Directive language**: "Call retrieve_context tool with:" not "I will search..."
- **Bullet lists** for queries (easier to parse than comma-separated)
- **Actionable next steps**: Clear instructions on what to do
- **No persona confusion**: Tool result instructs, not pretends to be LLM

**Templating Engine:** Python built-in `str.format()` - no external dependencies (e.g., Jinja2) required.

#### Template 1: Normal (High Confidence, Safe)

**Tool Result (for Model Context):**
```markdown
**Request Analysis**
Intent: {user_intent}
Category: {category}
Confidence: {intent_confidence_pct}
<Other SGR fields where relevant>

**Required next steps**
1. Call retrieve_context tool with these queries:
   - {knowledge_base_search_queries_bullet_list}
   - [+ add any additional relevant queries if needed]
2. Review results and provide answer based on retrieved documentation
```

**Example:**
```markdown
**Request Analysis**
Intent: Пользователь хочет настроить SSO для интеграции с корпоративной системой аутентификации
Category: Помощь в настройке
Confidence: 92%

**Required next steps**
1. Call retrieve_context tool with these queries:
   - настройка SSO Comindware
   - аутентификация Active Directory
   - интеграция LDAP
   - [+ add any additional relevant queries if needed]
2. Review results and provide step-by-step configuration guide
```

**UI Display:**
```
Как я понял ваш запрос:

{user_intent}

{response}
```

#### Template 2: Clarify (Low Confidence)

**Tool Result (for Model Context):**
```markdown
**Request Analysis**
Intent: {user_intent}
Category: {category}
Confidence: {intent_confidence_pct} (низкая)

**Required next steps**
1. Ask the user these clarification questions:
   - {clarification_questions_to_ask_bullet_list}
   - [+ add any additional questions if needed]
2. Wait for user response before proceeding
```

**Example:**
```markdown
**Request Analysis**
Intent: Пользователь сообщает о проблеме в процессе, но конкретные детали неясны
Category: Устранение неполадок
Confidence: 42% (низкая)

**Required next steps**
1. Ask the user these clarification questions:
   - Какой именно процесс не работает? (название или ID)
   - Что конкретно происходит: ошибка, зависание, или процесс не запускается?
   - Когда проблема началась?
   - [+ add any additional questions if needed]
2. Wait for user response before proceeding
```

**UI Display:**
```
Как я понял ваш запрос:

{user_intent}

{first_uncertainty_from_list}
```

#### Template 3: Block (Spam/Off-topic)

**Tool Result (for Model Context):**
```markdown
**Request Analysis**
Assessment: Запрос не относится к платформе Comindware
Reason: {spam_reason}
Spam Score: {spam_score}

**Required next steps**
Do not process this request. Use this refusal message:

"{refusal}"
```

**UI Display:**
```
{refusal}
```

#### Template 4: Guardian Blocked (Safety Concern)

**Tool Result (for Model Context):**
```markdown
**Request Analysis**
Assessment: Запрос заблокирован системой безопасности
Reason: {guard_reason}

**Required next steps**
Do not process this request. Use this refusal message:

"{refusal}"
```

**UI Display:**
```
{refusal}
```

### Template Variables

```python
TEMPLATE_VARIABLES = {
    # From SGR schema
    "topic": "Inferred topic from request",
    "user_intent": "Parsed user intent from SGR",
    "category": "Request classification",
    "spam_score": "0.0-1.0",
    "spam_reason": "Explanation if spam",
    "intent_confidence": "0.0-1.0",
    "intent_confidence_pct": "0-100% format for display",
    "clarification_questions_to_ask": "List of clarification questions to ask user",
    "clarification_questions_to_ask_bullet_list": "Formatted as markdown bullets",
    "knowledge_base_search_queries": "List of search queries for knowledge base",
    "knowledge_base_search_queries_bullet_list": "Formatted as markdown bullets (no quotes)",
    "action_plan": "List of steps (for reference in normal template)",
    "guard_reason": "Guardian block reason",
    "action": "Enum: normal, clarify, block, guardian_block",
    
    # From i18n (resolved via get_text())
    "refusal": "i18n resolved refusal text (sgr_spam_refusal or sgr_guardian_refusal)",
}
```

**Note:** 
- `{refusal}` is resolved via `get_text("sgr_spam_refusal")` or `get_text("sgr_guardian_refusal")` in the render functions
- All other dynamic values come directly from the SGR schema

**Key Design:**
- `action` enum values map to template functions (internal routing)
- `{refusal}` resolved via `get_text()` for feature-proof i18n support
- UI response texts use i18n keys resolved via existing `get_text()` function
- Guardian categories come from prompt context (preceding guardian call), not SGR tool
- Queries in bullets WITHOUT quotes (model parses easier)

---

## 5. Updated Schema Design

### Enhanced Pydantic Schema with Reasoning Descriptions

```python
from enum import Enum
from pydantic import BaseModel, Field

class SGRAction(str, Enum):
    """Routing actions for request handling."""
    NORMAL = "normal"           # Proceed with normal assistance
    CLARIFY = "clarify"         # Need clarification from user
    BLOCK = "block"             # Spam/off-topic
    GUARDIAN_BLOCK = "guardian_block"  # Safety concern from guardian

class SGRPlanResult(BaseModel):
    """Schema-Guided Reasoning (SGR) for Comindware Platform support requests.
    
    Sequential reasoning like a human would think:
    1. Understand what user wants
    2. Identify topic and category
    3. Assess confidence in understanding
    4. Check if request is relevant/safe
    5. Plan search strategy
    6. Decide action
    7. Prepare clarification if needed
    
    Note: Guardian assessment comes from preceding guardian call (in prompt context),
    not generated by this tool.
    """

    # STEP 1: Intent Understanding
    user_intent: str = Field(
        ...,
        max_length=300,
        description=(
            "REASONING STEP 1 - Intent Understanding: "
            "What does the user actually want to achieve? "
            "Think beyond keywords: What is their underlying goal? "
            "What business problem are they trying to solve? "
            "Write 1-2 clear sentences in Russian, as if explaining to a colleague."
        ),
    )
    
    # STEP 2: Topic Identification
    topic: str = Field(
        ...,
        max_length=100,
        description=(
            "REASONING STEP 2 - Topic Identification: "
            "What is this request about? "
            "Example: 'Настройка SSO', 'Создание процесса', 'Интеграция с API'. "
            "Keep it concise (2-5 words). "
            "Write in Russian."
        ),
    )
    
    # STEP 3: Category Classification
    category: str = Field(
        ...,
        max_length=50,
        description=(
            "REASONING STEP 3 - Category Classification: "
            "What type of request is this? "
            "'Помощь в настройке', 'Устранение неполадок', 'Запрос функции', 'Общий вопрос'. "
            "Choose the most appropriate category. "
            "Write in Russian."
        ),
    )
    
    # STEP 4: Confidence Assessment
    intent_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "REASONING STEP 4 - Confidence Assessment: "
            "How confident are you in understanding what the user wants? "
            "Think: Is the request clear? Do you understand the context? "
            "0.0-0.4: Very unclear, major uncertainties; "
            "0.5-0.7: Somewhat clear but some gaps; "
            "0.8-1.0: Clear and well-understood."
        ),
    )
    
    # STEP 5: Clarification Questions to Ask
    clarification_questions_to_ask: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "REASONING STEP 5 - Clarification Questions to Ask: "
            "If intent_confidence < 0.7, what specific questions would help you understand better? "
            "Write in Russian, be polite and specific. "
            "These questions will be shown to the user to get clarification. "
            "Empty list if intent_confidence >= 0.7."
        ),
    )
    
    # STEP 6: Validity Assessment
    spam_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "REASONING STEP 6 - Validity Assessment: "
            "Is this request appropriate for Comindware Platform support? "
            "0.0-0.2: Clearly relevant; "
            "0.3-0.5: Ambiguous or partially related; "
            "0.6-0.8: Likely irrelevant; "
            "0.9-1.0: Obviously spam or malicious."
        ),
    )
    
    # STEP 7: Spam Justification
    spam_reason: str = Field(
        ...,
        max_length=150,
        description=(
            "REASONING STEP 7 - Spam Justification: "
            "Briefly explain your spam_score in 10-20 words. "
            "Write in Russian."
        ),
    )
    
    # STEP 8: Knowledge Base Search Queries
    knowledge_base_search_queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "REASONING STEP 8 - Knowledge Base Search Queries: "
            "What specific terms should be used to search the knowledge base? "
            "Include: feature names, technical terms, error messages, relevant keywords. "
            "Write in Russian, avoid duplicates. "
            "These queries will be used to retrieve relevant documentation."
        ),
    )
    
    # STEP 9: Execution Plan
    action_plan: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "REASONING STEP 9 - Execution Plan: "
            "How will you answer this request? "
            "Steps: search docs → evaluate → synthesize answer OR ask clarification. "
            "Write in Russian as actionable instructions to yourself."
        ),
    )
    
    # STEP 10: Routing Decision
    action: SGRAction = Field(
        ...,
        description=(
            "REASONING STEP 10 - Routing Decision: "
            "Based on all previous reasoning, what action to take? "
            "'normal': spam_score < 0.7 AND confidence >= 0.6; "
            "'clarify': confidence < 0.6 AND spam_score < 0.7; "
            "'block': spam_score >= 0.7; "
            "'guardian_block': safety issues from guardian (in prompt context)."
        ),
    )
```

---

## 6. Handler Architecture (FORMATTED TOOL RESULT)

### Complete Handler Flow

**Key Difference from Original Plan:**
- NO removal of tool trace from context
- ONLY change: tool return format (formatted string instead of raw JSON)
- Minimal code changes required

```python
from rag_engine.api.i18n import get_text

async def agent_chat_handler_with_sgr(user_message, gradio_history, messages):
    """
    Handler with formatted SGR tool result pattern:
    0. Guardian check (if enabled)
    1. A/B test flag check (lean feature flag)
    2. Forced SGR tool call (forced tool_choice)
    3. Store structured data (JSON for downstream)
    4. Build response text (for UI)
    5. Tool result automatically formatted by tool return value
    6. Route based on action
    7. Unbind SGR tool for remainder of turn
    8. Create agent and continue
    """
    
    # ========== 0. GUARDIAN CHECK ==========
    if settings.guard_enabled:
        moderation_result = await guard_client.classify(user_message)
        guard_mode = getattr(settings, "guard_mode", "enforce")
        should_block = guard_client.should_block(moderation_result)
        
        if should_block and guard_mode == "enforce":
            refusal_msg = get_text("guardian_refusal_unsafe")
            gradio_history.append({
                "role": "assistant",
                "content": refusal_msg,
                "metadata": {"ui_type": "guardian_block"}
            })
            
            agent_context.guardian_result = moderation_result
            agent_context.blocked = True
            agent_context.block_reason = "guardian_unsafe"
            
            yield gradio_history
            return
    else:
        moderation_result = None
    
    # ========== 1. A/B TEST FLAG ==========
    # Toggle between formatted tool result and raw JSON
    use_formatted_result = getattr(settings, "sgr_formatted_tool_result_enabled", True)
    
    # ========== 2. FORCED SGR TOOL CALL ==========
    sgr_system_prompt = build_sgr_system_prompt(moderation_result)
    
    sgr_llm = LLMManager(...)._chat_model()
    sgr_model = sgr_llm.bind_tools(
        [analyse_user_request_tool],
        tool_choice={"type": "function", "function": {"name": "analyse_user_request"}},
    )
    
    sgr_response = await sgr_model.ainvoke([sgr_system_prompt] + messages)
    
    # Extract plan from tool call arguments
    tool_call = sgr_response.tool_calls[0]
    call_id = tool_call["id"]
    sgr_plan = tool_call["args"]  # Already validated by schema
    
    # ========== 3. STORE STRUCTURED DATA (JSON) ==========
    # Save complete JSON for downstream (UI, metrics, integrations)
    agent_context.sgr_plan = sgr_plan
    agent_context.guardian_result = moderation_result
    
    # ========== 4. BUILD RESPONSE TEXT (for UI) ==========
    if sgr_plan["action"] == "normal":
        response_text = get_text("sgr_normal_response", user_intent=sgr_plan["user_intent"])
    elif sgr_plan["action"] == "clarify":
        # For UI, we can either show the first uncertainty or a generic message
        clarification = sgr_plan.get("clarification_questions_to_ask", [""])[0] if sgr_plan.get("clarification_questions_to_ask") else ""
        response_text = get_text("sgr_clarify_response", 
            user_intent=sgr_plan["user_intent"], 
            clarification_question=clarification)
    elif sgr_plan["action"] == "block":
        response_text = get_text("sgr_spam_response")
    elif sgr_plan["action"] == "guardian_block":
        response_text = get_text("sgr_guardian_response")
    else:
        response_text = ""
    
    # ========== 5. TOOL RESULT ALREADY FORMATTED ==========
    # The tool itself returns formatted string (see Tool Implementation below)
    # Standard ReAct flow: tool call + tool result stay in messages
    # No need to manually inject - LangChain handles this automatically
    
    # ========== 6. BUILD UI MESSAGE ==========
    user_intent_prefix = get_text("user_intent_prefix")
    ui_message = f"**{user_intent_prefix}**\n\n{sgr_plan['user_intent']}\n\n{response_text}"
    gradio_history.append({
        "role": "assistant",
        "content": ui_message,
        "metadata": {
            "ui_type": "sgr_response",
            "sgr_action": sgr_plan["action"],
            "user_intent": sgr_plan["user_intent"]
        }
    })
    
    yield gradio_history
    
    # ========== 7. ROUTE BASED ON ACTION ==========
    if sgr_plan["action"] in ["block", "guardian_block"]:
        return
    
    elif sgr_plan["action"] == "clarify":
        return
    
    else:  # action == "normal"
        # ========== 8. UNBIND SGR TOOL ==========
        available_tools = [t for t in all_tools if t.name != "analyse_user_request"]
        
        # ========== 9. CREATE AGENT WITHOUT SGR ==========
        agent = create_react_agent(
            model=llm,
            tools=available_tools,
            prompt=agent_system_prompt,
        )
        
        # ========== 10. CONTINUE WITH AGENT FLOW ==========
        # Model sees: user → wrapped message → tool call → formatted tool result → continues
        async for stream_mode, chunk in agent.astream(
            {"messages": messages},
            config={...}
        ):
            pass
```

### Tool Implementation (Key Change)

```python
@tool("analyse_user_request", args_schema=SGRPlanResult)
async def analyse_user_request(
    user_intent: str,
    topic: str,
    category: str,
    intent_confidence: float,
    clarification_questions_to_ask: list[str],
    spam_score: float,
    spam_reason: str,
    knowledge_base_search_queries: list[str],
    action_plan: list[str],
    action: SGRAction,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> str:
    """Analyze the user request and produce the resolution plan."""
    
    # Build structured data for downstream (JSON)
    plan = {
        "user_intent": user_intent,
        "topic": topic,
        "category": category,
        "intent_confidence": intent_confidence,
        "clarification_questions_to_ask": clarification_questions_to_ask,  # These ARE the clarification questions
        "spam_score": spam_score,
        "spam_reason": spam_reason,
        "knowledge_base_search_queries": knowledge_base_search_queries,
        "action_plan": action_plan,
        "action": action,
    }
    
    # Store JSON for downstream
    if runtime and runtime.context:
        runtime.context.sgr_plan = plan
    
    # Return FORMATTED string for model context (NEW)
    return render_sgr_template(action, plan)
```

### Helper Functions

```python
def render_sgr_template(action: str, plan: dict) -> str:
    """Render SGR plan as directive tool result."""
    
    if action == "normal":
        return _render_normal_template(plan)
    elif action == "clarify":
        return _render_clarify_template(plan)
    elif action == "block":
        return _render_block_template(plan)
    elif action == "guardian_block":
        return _render_guardian_template(plan)
    return ""


def _render_normal_template(plan: dict) -> str:
    """Render normal (proceed) template with directive language."""
    queries = "\n   - ".join(plan["knowledge_base_search_queries"])
    return f"""**Request Analysis**
Intent: {plan['user_intent']}
Category: {plan['category']}
Confidence: {plan['intent_confidence'] * 100:.0f}%

**Required next steps**
1. Call retrieve_context tool with these queries:
   - {queries}
2. Review results and provide answer based on retrieved documentation"""


def _render_clarify_template(plan: dict) -> str:
    """Render clarification needed template - clarification_questions_to_ask merged into required next steps."""
    questions = "\n   - ".join(plan.get("clarification_questions_to_ask", []))
    return f"""**Request Analysis**
Intent: {plan['user_intent']}
Category: {plan['category']}
Confidence: {plan['intent_confidence'] * 100:.0f}% (низкая)

**Required next steps**
1. Ask the user these clarification questions:
   - {questions}
   - [+ add any additional questions if needed]
2. Wait for user response before proceeding"""


def _render_block_template(plan: dict) -> str:
    """Render spam/block template."""
    refusal = get_text("sgr_spam_refusal")
    return f"""**Request Analysis**
Assessment: Запрос не относится к платформе Comindware
Reason: {plan['spam_reason']}
Spam Score: {plan['spam_score']}

**Required next steps**
Do not process this request. Use this refusal message:

"{refusal}"""


def _render_guardian_template(plan: dict) -> str:
    """Render guardian block template."""
    refusal = get_text("sgr_guardian_refusal")
    return f"""**Request Analysis**
Assessment: Запрос заблокирован системой безопасности

**Required next steps**
Do not process this request. Use this refusal message:

"{refusal}"""
```

### Key Implementation Details

#### Helper Functions

```python
def build_sgr_system_prompt(moderation_result: dict | None) -> SystemMessage:
    """Build SGR system prompt with guardian context if available."""
    base_prompt = get_sgr_system_prompt_base()
    
    if moderation_result:
        guardian_context = f"""
<guardian_assessment>
Safety Assessment from Guardian Model:
- Risk Level: {moderation_result.get('safety_level', 'Safe')}
- Categories: {moderation_result.get('categories', [])}
- Decision Guidance:
  * If level='Unsafe': action SHOULD be 'guardian_block'
  * If level='Controversial': CONSIDER 'guardian_block' or 'clarify'
  * If level='Safe': Use normal SGR routing (spam_score + confidence)
</guardian_assessment>
"""
        content = base_prompt + "\n\n" + guardian_context
    else:
        content = base_prompt
    
    return SystemMessage(content=content)


def render_sgr_template(
    template_name: str,
    sgr_plan: dict,
    moderation_result: dict | None,
    response_text: str,
) -> str:
    """Render SGR template with all variables.
    
    Args:
        template_name: Which template to use (normal, clarify, block, guardian_block)
        sgr_plan: Structured plan from SGR tool
        moderation_result: Guardian classification result
        response_text: Pre-constructed response text (used in both template and UI)
    """
    
    # Get template
    template = SGR_TEMPLATES[template_name]
    
    # Prepare variables - all from SGR directly, no inference needed
    variables = {
        "topic": sgr_plan.get("topic"),           # Direct from SGR schema
        "user_intent": sgr_plan["user_intent"],   # Direct from SGR schema
        "category": sgr_plan.get("category"),     # Direct from SGR schema
        "spam_score": sgr_plan["spam_score"],
        "spam_reason": sgr_plan["spam_reason"],
        "intent_confidence": sgr_plan["intent_confidence"],
        # Format lists internally
        "clarification_questions_to_ask": _format_list(sgr_plan.get("clarification_questions_to_ask", [])),
        "knowledge_base_search_queries": _format_list(sgr_plan["knowledge_base_search_queries"]),
        "action_plan": _format_numbered_list(sgr_plan.get("action_plan", [])),
        "action": sgr_plan["action"],
        "guard_categories": moderation_result.get("categories", []) if moderation_result else [],
        # Response - passed in, used in both template and UI (no duplicate logic)
        "response": response_text,
    }
    
    return template.format(**variables)


def _format_list(items: list) -> str:
    """Format list as markdown bullets."""
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def _format_numbered_list(items: list) -> str:
    """Format list as numbered markdown."""
    if not items:
        return ""
    return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
```

---

## 7. Guardian Integration (ALREADY IMPLEMENTED)

### Current State (Already Merged)

Guardian implementation is **already in the repository** at:
- `rag_engine/core/guard_client.py` - Main guard client
- `rag_engine/core/vllm_guard_adapter.py` - VLLM adapter
- Integrated in `rag_engine/api/app.py` - Agent chat handler

### Guardian Flow (Current Implementation)

```python
# From rag_engine/api/app.py (lines 935-1018)

if not settings.guard_enabled:
    logger.info("Guardian is disabled, skipping moderation")
else:
    moderation_result = await guard_client.classify(message)

# Handle based on guard mode
guard_mode = getattr(settings, "guard_mode", "enforce")
should_block = guard_client.should_block(moderation_result) if moderation_result else False

if should_block and guard_mode == "enforce":
    # Block unsafe content immediately
    error_message = f"❌ **Сообщение заблокировано...**"
    ...

# Build moderation context for SGR
if moderation_result:
    safety_level = moderation_result.get("safety_level", "Safe")
    categories = moderation_result.get("categories", [])
    # Inject as system message
    system_msg = {
        "role": "system",
        "content": f"{moderation_context}\n\nИспользуйте эти данные модерации...",
    }
    messages = [system_msg] + messages
```

### Guardian Settings (Already Implemented)

```python
# From rag_engine/config/settings.py (lines 171-194)

guard_enabled: bool
guard_mode: str           # "enforce" or "report"
guard_provider_type: str  # "mosec" or "vllm"

# MOSEC provider settings
guard_mosec_url: str
guard_mosec_port: int
guard_mosec_path: str

# VLLM provider settings
guard_vllm_url: str
guard_vllm_model: str

guard_timeout: float
guard_max_retries: int
```

### Updated Flow with Guardian + Formatted Tool Result

```
User Request
    │
    ▼
┌──────────────────────────────────┐
│ Guardian Check                   │
│ (Already implemented)            │
│ - Returns: level + categories    │
└──────────────────────────────────┘
    │
    ├─► Level = "Unsafe" + ENFORCE mode
    │   └─► Block immediately, no SGR
    │       - Refusal message to UI
    │       - Structured metadata saved
    │
    ├─► Level = "Unsafe" + REPORT mode
    │   └─► Continue to SGR with guardian context
    │       - SGR may route to guardian_block
    │
    ├─► Level = "Controversial"
    │   └─► Continue to SGR with guardian context
    │       - SGR considers block or clarify
    │
    └─► Level = "Safe"
        └─► Continue to SGR (no guardian context needed)
                │
                ▼
        ┌──────────────────────┐
        │ Forced SGR Call      │
        │ Returns: Formatted    │
        │ markdown (NOT JSON)  │
        └──────────────────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Store JSON in        │
        │ agent_context.sgr_plan│
        │ (for downstream)     │
        └──────────────────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Tool Call + Result   │
        │ Stay in Messages     │
        │ (Standard ReAct)     │
        └──────────────────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Unbind SGR Tool      │
        │ (one call per turn)  │
        └──────────────────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Agent Continues      │
        │ (Natural ReAct flow) │
        └──────────────────────┘
```

### Guardian + SGR Integration Points

1. **Guardian runs FIRST** (already implemented)
2. **Guardian classification is GLUED to system message** before SGR call (already implemented)
   - Guardian result added via `build_sgr_system_prompt()` 
   - Provides safety context without being part of SGR tool schema
   - Single system message pattern (best practice)
3. **SGR uses guardian data** for routing decisions (action enum)
4. **SGR schema does NOT generate** guard_categories (already correct - comes from prompt context)
5. **Structured output includes both** guardian and SGR data (enhanced in this plan)

---

## 8. Data Visibility Matrix

### What Each Component Sees (Three Outputs)

| Data                   | Context Tool Result (Model) | UI Synthetics (User) | Structured Metadata (System) |
| ---------------------- | -------------------------- | -------------------- | ---------------------------- |
| user_intent            | ✅ Yes (in analysis)        | ✅ Yes                | ✅ Yes                        |
| topic                  | ✅ Yes                      | ❌ No                 | ✅ Yes                        |
| category               | ✅ Yes                      | ❌ No                 | ✅ Yes                        |
| spam_score             | ✅ Yes (if relevant)        | ❌ No                 | ✅ Yes                        |
| spam_reason            | ✅ Yes (if block)           | ❌ No                 | ✅ Yes                        |
| intent_confidence      | ✅ Yes                      | ❌ No                 | ✅ Yes                        |
| clarification_questions_to_ask | ✅ Yes (as questions)       | ❌ No                 | ✅ Yes                        |
| knowledge_base_search_queries | ✅ Yes (as bullet list)     | ❌ No                 | ✅ Yes                        |
| action_plan            | ✅ Yes (referenced)         | ❌ No                 | ✅ Yes                        |
| action                 | ✅ Yes (determines template)| ❌ No                 | ✅ Yes                        |
| guard_categories       | ✅ Yes (in analysis)        | ❌ No                 | ✅ Yes                        |
| Tool call trace        | ✅ Yes (standard ReAct)     | ❌ No                 | ✅ Yes (in logs)              |
| Formatted tool result  | ✅ Yes                      | ❌ No                 | ✅ Yes                        |
| Raw JSON result        | ❌ No                       | ❌ No                 | ✅ Yes (sgr_plan)             |

**Note:** No separate `clarification_question` field - `clarification_questions_to_ask` list serves as both the questions to ask (in tool result) and is stored in JSON for downstream.

**Key Principle:** Three outputs with clean separation:
- **Context Tool Result**: Formatted analysis with directive instructions (stays in ReAct flow)
- **UI Synthetics**: Response section only (user-facing, no internal details)
- **Structured Metadata**: Raw JSON + Guardian data (downstream processing)

---

## 9. Implementation Phases (FORMATTED TOOL RESULT)

### Phase 1: Formatted Tool Result Implementation

**Complexity: LOW - Minimal changes required**

- [ ] **Update SGRPlanResult schema** with enhanced field descriptions (REASONING STEP N pattern)
- [ ] Add new fields: `action` enum, `intent_confidence`, `clarification_questions_to_ask`, `topic`, `category`
- [ ] **Remove from schema**: `ask_for_clarification` (replaced by action), `template_hint`
- [ ] Define template functions for each action type:
  - `_render_normal_template(plan)` 
  - `_render_clarify_template(plan)`
  - `_render_block_template(plan)`
  - `_render_guardian_template(plan)`
- [ ] **Add i18n keys** to `rag_engine/api/i18n.py`:
  - `sgr_normal_response`
  - `sgr_clarify_response`
  - `sgr_spam_response`
  - `sgr_spam_refusal`  # Full refusal text for block template
  - `sgr_guardian_response`
  - `sgr_guardian_refusal`  # Full refusal text for guardian template
  - `user_intent_prefix`
- [ ] **Modify analyse_user_request tool** to return formatted string:
  ```python
  # Instead of: return json.dumps(plan)
  return render_sgr_template(plan["action"], plan)
  ```
- [ ] **Verify handler** - No changes needed! Tool result automatically in messages
- [ ] Verify SGR tool unbound after first call (already implemented)
- [ ] Update UI emission to show only response section (already implemented)

### Phase 2: Guardian Integration (ALREADY DONE - Verify Alignment)

- [x] Guardian already merged from `cmw-rag-guard-test` branch
- [x] `GuardClient` class in `rag_engine/core/guard_client.py`
- [x] Settings in `rag_engine/config/settings.py`
- [ ] **Verify integration** with formatted tool result
- [ ] **Test** ENFORCE mode: Guardian blocks → no SGR called
- [ ] **Test** REPORT mode: Guardian runs → SGR gets context → may route to guardian_block
- [ ] **Test** Safe flow: Guardian passes → SGR executes → formatted tool result

### Phase 3: Testing & Validation

- [ ] Unit tests for template rendering functions
- [ ] Integration test: Full flow with formatted tool result
- [ ] Verify tool result IS in message history (formatted, not JSON)
- [ ] Verify model continues after tool result (no early termination)
- [ ] Verify SGR tool unbound after first call
- [ ] A/B test: Formatted tool result vs. raw JSON
- [ ] Measure: response quality, token efficiency
- [ ] Guardian integration tests (all modes)

### Phase 4: Documentation & Rollout

- [ ] Update architecture documentation
- [ ] Document formatted tool result pattern for team
- [ ] Add inline code comments explaining template rendering

---

## 10. Open Questions

### Resolved ✅

1. **Early termination risk:** ✅ Avoided by keeping tool result in ReAct flow
2. **Synthetic injection issue:** ✅ Replaced with formatted tool result
3. **Template language:** ✅ Directive/instructional, not pretending to be LLM
4. **Implementation complexity:** ✅ Minimal - change only tool return format

### Confirmed Decisions ✅

5. **Template format:** 
   - [x] Bullet lists for queries (no quotes) - easier for model to parse
   - [x] "Required next steps" section with numbered actions
   - [x] "Call [tool] with:" pattern for tool invocation

6. **Implementation approach:**
   - [x] Tool returns formatted string (not JSON)
   - [x] JSON stored separately in agent_context.sgr_plan for downstream
   - [x] Standard ReAct flow preserved (tool call + result in messages)

7. **A/B Testing (Lean):**
   - Simple boolean flag in settings: `SGR_FORMATTED_TOOL_RESULT_ENABLED = True/False`
   - **When disabled (False):** Current behavior - tool returns raw JSON
   - **When enabled (True):** Formatted tool result - human-readable markdown
   - Metrics: Compare response quality, token usage, latency between modes
   - Rollout: Toggle in config, no complex percentage system (add later if needed)

8. **Different templates for different cases:**
   - [x] `normal` - proceed with retrieval and answer
   - [x] `clarify` - ask user for clarification
   - [x] `block` - refuse as off-topic/spam
   - [x] `guardian_block` - refuse as unsafe

---

## 11. Success Metrics

### Quantitative

- **Plan-to-Action Consistency:** BLEU/ROUGE between plan and final answer
- **Early Termination Rate:** Verify model continues after formatted tool result
- **Token Efficiency:** Compare context usage (formatted vs. raw JSON)
- **Response Latency:** Measure impact of template rendering (minimal expected)
- **User Satisfaction:** Blind test comparing responses
- **Error Rate:** Fewer "apology loops" or confusion in agent responses

### Qualitative

- **Response Quality:** Does model follow directive instructions in tool result?
- **Coherence:** Natural flow in multi-turn conversations (ReAct preserved)?
- **Debugging:** Easy to trace with formatted tool result?
- **Clarity:** Model correctly interprets "Call retrieve_context tool" instructions?

---

## 12. DMN: Decision Model and Notation

Decision tables for SGR routing, Guardian enforcement, and tool binding logic.

### DT-001: Guardian Enforcement Decision

Determines whether to block request before SGR processing based on Guardian safety assessment.

| guardian_level | guard_mode | Decision | Action                               |
| -------------- | ---------- | -------- | ------------------------------------ |
| Unsafe         | enforce    | Block    | Immediate refusal, no SGR call       |
| Unsafe         | report     | Continue | Proceed to SGR with guardian context |
| Controversial  | enforce    | Continue | Proceed to SGR with guardian context |
| Controversial  | report     | Continue | Proceed to SGR with guardian context |
| Safe           | enforce    | Continue | Proceed to SGR normally              |
| Safe           | report     | Continue | Proceed to SGR normally              |

**Inputs:**
- `guardian_level`: Enum [Unsafe, Controversial, Safe]
- `guard_mode`: Enum [enforce, report]

**Output:**
- `Decision`: Block | Continue
- `Action`: Refusal message | Proceed to SGR

---

### DT-002: SGR Routing Decision

Determines action and template based on SGR analysis results and Guardian context.

| spam_score | intent_confidence | guardian_level         | action         | template       | Flow                        |
| ---------- | ----------------- | ---------------------- | -------------- | -------------- | --------------------------- |
| < 0.7      | >= 0.6            | Safe/Controversial/N/A | normal         | normal         | Continue with agent tools   |
| < 0.7      | < 0.6             | Safe/Controversial/N/A | clarify        | clarify        | Wait for user clarification |
| >= 0.7     | Any               | Safe/Controversial/N/A | block          | block          | Stop, show refusal          |
| Any        | Any               | Unsafe (via context)   | guardian_block | guardian_block | Stop, show safety refusal   |

**Inputs:**
- `spam_score`: Float [0.0-1.0]
- `intent_confidence`: Float [0.0-1.0]
- `guardian_level`: Enum [Safe, Controversial, Unsafe, N/A]

**Output:**
- `action`: Enum [normal, clarify, block, guardian_block]
- `template`: String (matches action)
- `Flow`: Continue | Wait | Stop

**Hit Policy:** First match (priority order: guardian_block > block > clarify > normal)

---

### DT-003: SGR Tool Binding Decision

Determines whether SGR tool is available based on turn state.

| turn_state           | sgr_tool_available | Action                                         |
| -------------------  | ----------------- | ---------------------------------------------- |
| Start of turn       | Yes               | Bind SGR tool, force call                     |
| After SGR executed  | No                | Unbind SGR tool, continue with remaining tools |

**Note:** SGR executes exactly once per turn. No counter needed - tool is bound at turn start, unbound after execution.

**Inputs:**
- `turn_state`: Enum [start_of_turn, after_sgr_executed]

**Output:**
- `sgr_tool_available`: Boolean
- `Action`: Bind | Unbind | Force call

---

### DT-004: Output Routing Decision

Determines what gets emitted to each output channel.

| Output Channel               | Formatted Analysis | Response Section | Tool Trace | Structured Data |
| ---------------------------- | ------------------ | ---------------- | ---------- | --------------- |
| Context Tool Result (Model) | Yes (directive)   | No               | Yes        | No              |
| UI Synthetics (User)         | No                | Yes              | No         | No              |
| Structured Metadata (System)| Yes (JSON)        | Yes (JSON)       | Yes        | Yes             |

**Note:** 
- Tool trace stays in context (standard ReAct flow)
- Tool result is formatted markdown (not raw JSON)
- Raw JSON only in agent_context.sgr_plan for downstream

---

## 13. References

### From Web Research
1. Chen, Z., & Yao, B. (2024). Pseudo-Conversation Injection for LLM Goal Hijacking.
2. Meng, W., et al. (2025). Dialogue Injection Attack.
3. Instructor Library - https://python.useinstructor.com/
4. sgr-agent-core - https://github.com/vamplabAI/sgr-agent-core
5. Anthropic. (2024). Building Effective Agents.
6. 12-Factor Agents. https://github.com/humanlayer/12-factor-agents

### Codebase References
7. `rag_engine/tools/analyse_user_request.py` - Current SGR tool
8. `rag_engine/core/guard_client.py` - Guardian implementation
9. `rag_engine/api/app.py` - Agent chat handler
10. `rag_engine/api/i18n.py` - Internationalization system

---

**Next Steps:**
1. Review and approve revised architecture (Formatted Tool Result approach)
2. Implement Phase 1: Formatted tool result implementation
3. Verify Phase 2: Guardian alignment (already implemented)
4. Begin testing Phase 3
5. Document formatted tool result pattern for team
