# SGR Synthetic Assistant Message Enhancement Plan

**Status:** DRAFT - REVISED ARCHITECTURE
**Created:** 2026-02-13
**Updated:** 2026-02-14 (Single-Message Injection, Tool Trace Removal, Guardian Already Merged)
**Based on:** Analysis of current SGR implementation, novelty research, and architectural decisions

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

### Proposed Enhancement (CLEAN INJECTION PATTERN)
**Inject single synthetic assistant message** that replaces the tool call trace entirely. Model sees only its "own" reasoning—no tool artifacts, no raw JSON.

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
│ Forced SGR Tool Call    │ (INTERNAL ONLY)
│ Returns: Raw JSON       │ → Stored in agent_context.sgr_plan
└─────────────────────────┘
    │
    ▼ (CLEANUP)
┌─────────────────────────┐
│ REMOVE from context:    │
│ - Tool call message     │
│ - role: "tool" result   │
└─────────────────────────┘
    │
    ▼ (INJECTION)
┌─────────────────────────┐
│ ADD to context:         │
│ Single synthetic        │
│ assistant message       │
│ (analysis + response)   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Agent continues from    │
│ "its own" reasoning     │
└─────────────────────────┘
```

**Key Innovation:** 
- Model believes it performed the analysis
- No cognitive dissonance between tool output and generation
- Clean state: tool trace completely removed from visible history
- Structured data preserved for downstream (UI, metrics, debugging)

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

{i18n_response_text}"""

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

### Decision 3: Clean Tool Call Trace from Model Context
**Status:** ✅ CONFIRMED

**Rationale:** The tool call + JSON result must be completely removed from the message history visible to the model. The model should see only the synthetic reasoning as its "own" thought process.

**Implementation:**
```python
# 1. Execute SGR tool (returns JSON)
sgr_result = await execute_sgr_tool(messages)
sgr_plan = json.loads(sgr_result)

# 2. Store structured data for downstream
agent_context.sgr_plan = sgr_plan

# 3. DO NOT add tool call or result to message history
# (skip: messages.append({"role": "assistant", "content": None, "tool_calls": [...]}))
# (skip: messages.append({"role": "tool", "content": sgr_result}))

# 4. Render synthetic message (replaces tool trace entirely)
synthetic_message = render_synthetic_template(sgr_plan)

# 5. Inject single synthetic message
messages.append({
    "role": "assistant",
    "content": synthetic_message,
    "metadata": {"sgr_injected": True}  # For tracking/debugging
})
```

**Result:** Model context contains clean narrative flow—user message → synthetic reasoning → agent continues.

---

## 3. Three-Output Architecture (Revised)

Based on the architectural decisions, we maintain a **three-output architecture** with **tool trace removal**:

```
┌─────────────────────────────────────────────────────────────┐
│                    SGR TOOL (Pure Function)                  │
│  Returns: Structured JSON with all metadata                 │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    HANDLER (Orchestrator)                    │
│  1. Guardian check → add classification to system message   │
│  2. Execute SGR tool internally                              │
│  3. Store structured data → agent_context.sgr_plan          │
│  4. SKIP adding tool call/result to messages                │
│  5. Render single synthetic message (analysis + response)   │
│  6. Inject synthetic message to agent context               │
│  7. Unbind SGR tool for remainder of turn                   │
│  8. Route based on action                                   │
└─────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Output 1:       │  │ Output 2:       │  │ Output 3:       │
│ Context         │  │ UI Response     │  │ Structured      │
│ Synthetics      │  │ Synthetics      │  │ Metadata        │
│ (Agent Context) │  │ (User Display)  │  │ (Downstream)    │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ - Full message  │  │ - Response only │  │ - agent_context │
│ - Analysis      │  │ - No reasoning  │  │   .sgr_plan     │
│ - Response      │  │ - Preceded by   │  │ - Guardian      │
│ - No tool trace │  │   user_intent   │  │   result        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Three Outputs Explained:**

1. **Context Synthetics** (Output 1) → Model's view
   - Single assistant message with `## Analysis` + `## Response` sections
   - Injected to message history as model's "own reasoning"
   - Tool trace completely removed
   - Model continues from this point

2. **UI Response Synthetics** (Output 2) → User's view
   - Only the `## Response` section (extracted)
   - Preceded by user_intent display
   - No internal reasoning shown (spam_score, uncertainties, etc.)
   - Rendered via Gradio history

3. **Structured Metadata** (Output 3) → System/downstream
   - Raw JSON from SGR tool (stored in `agent_context.sgr_plan`)
   - Guardian classification results
   - Available for: metrics, analytics, debugging, future tools
   - Not visible to model or user

**Key Changes from Original Plan:**
- ✅ Single assistant message (analysis + response combined)
- ✅ Tool call and JSON result NOT added to message history
- ✅ SGR tool unbind after first call per turn
- ✅ Clean injection: model sees only its "own" reasoning
- ✅ Guardian classification glued to system message pre-SGR

---

## 4. Template System Design (Updated)

### Template Structure (Single Message with Two Sections)

All templates now produce a **single message** with two sections:
1. `## Analysis` - Model's "internal reasoning" (hidden from UI)
2. `## Response` - User-facing text (shown in UI)

**Templating Engine:** Python built-in `str.format()` - no external dependencies (e.g., Jinja2) required.

#### Template 1: Normal (High Confidence, Safe)

```markdown
## Analysis
**Topic**: {topic}
**Intent**: {user_intent}
**Category**: {category}
**Validity**: Legitimate support request [spam_score: {spam_score}]
**Confidence**: High ({intent_confidence})
**Subqueries**: {subqueries}
**Action Plan**:
{action_plan}

## Response
{i18n_sgr_normal_response}
```

**UI Display:**
```
How I understood your request:

{user_intent}

{i18n_sgr_normal_response}
```

#### Template 2: Clarify (Low Confidence)

```markdown
## Analysis
**Topic**: {topic}
**Intent**: {user_intent} (not completely understood)
**Category**: {category}
**Validity**: Request needs clarification [spam_score: {spam_score}]
**Confidence**: Low ({intent_confidence})
**Uncertainties**:
{uncertainties}
**Subqueries**: {subqueries}

## Response
{i18n_sgr_clarify_intro}

{clarification_question}

{i18n_sgr_clarify_outro}
```

**UI Display:**
```
How I understood your request:

{user_intent}

{i18n_sgr_clarify_intro}

{clarification_question}

{i18n_sgr_clarify_outro}
```

#### Template 3: Block (Spam/Off-topic)

```markdown
## Analysis
**Assessment**: Off-topic or spam request
**Validity**: Request unrelated to Comindware Platform [spam_score: {spam_score}]
**Reason**: {spam_reason}
**Action**: block

## Response
{i18n_sgr_spam_response}
```

**UI Display:**
```
How I understood your request:

{user_intent}

{i18n_sgr_spam_response}
```

#### Template 4: Guardian Blocked (Safety Concern)

```markdown
## Analysis
**Assessment**: Request blocked by safety policy
**Validity**: Potentially harmful [guard_categories: {guard_categories}]
**Category**: Unsafe request
**Action**: guardian_block

## Response
{i18n_sgr_guardian_response}
```

**UI Display:**
```
{i18n_sgr_guardian_response}
```

### Template Variables

```python
TEMPLATE_VARIABLES = {
    "topic": "Inferred topic from request",
    "user_intent": "Parsed user intent from SGR",
    "category": "Request classification",
    "spam_score": "0.0-1.0",
    "spam_reason": "Explanation if spam",
    "intent_confidence": "0.0-1.0",
    "uncertainties": "List if confidence low",
    "subqueries": "Comma-separated list of search queries (formatted in render function)",
    "action_plan": "List of steps (formatted as numbered list in render function)",
    "guard_categories": "From guardian result (prompt context)",
    "action": "Enum: normal, clarify, block, guardian_block",
    "clarification_question": "Generated question if clarify",
    "i18n_*": "Resolved via get_text() from rag_engine.api.i18n"
}
```

**Note:** Formatting (comma-separated, numbered lists) happens inside `render_sgr_template()`, not passed as separate variables.

**Key Design:**
- `action` enum values ARE the template names (normal, clarify, block, guardian_block)
- UI response texts use i18n keys resolved via existing `get_text()` function
- Guardian categories come from prompt context (preceding guardian call), not SGR tool

---

## 5. Updated Schema Design

### Enhanced Pydantic Schema with Reasoning Descriptions

```python
from enum import Enum
from pydantic import BaseModel, Field

class SGRAction(str, Enum):
    """Action names ARE template names - no mapping needed."""
    NORMAL = "normal"           # Proceed with normal assistance
    CLARIFY = "clarify"         # Need clarification from user
    BLOCK = "block"             # Spam/off-topic
    GUARDIAN_BLOCK = "guardian_block"  # Safety concern from guardian

class SGRPlanResult(BaseModel):
    """Schema-Guided Reasoning (SGR) for Comindware Platform support requests.
    
    This schema enforces structured reasoning by requiring the model to:
    1. Analyze request validity (spam assessment)
    2. Extract and articulate user intent
    3. Plan search strategy (subqueries)
    4. Determine confidence and routing action
    
    Each field description guides the model through a specific reasoning step.
    
    Note: Guardian assessment comes from preceding guardian call (in prompt context),
    not generated by this tool.
    """

    # STEP 1: Validity Assessment
    spam_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "REASONING STEP 1 - Validity Assessment: "
            "Evaluate if this request is appropriate for Comindware Platform support. "
            "Think: Is this about Comindware configuration, troubleshooting, or features? "
            "Scoring guide: "
            "0.0-0.2: Clearly relevant to Comindware; "
            "0.3-0.5: Ambiguous or partially related; "
            "0.6-0.8: Likely irrelevant (general IT, unrelated software); "
            "0.9-1.0: Obviously spam (ads, gibberish, malicious). "
            "Provide your calculated score based on this analysis."
        ),
    )
    
    spam_reason: str = Field(
        ...,
        max_length=150,
        description=(
            "REASONING STEP 1b - Justification: "
            "Explain your spam_score classification in 10-20 words. "
            "If score < 0.5: explain why it's relevant; "
            "If score >= 0.5: explain what's wrong with the request. "
            "Write in Russian."
        ),
    )
    
    # STEP 2: Intent Extraction
    user_intent: str = Field(
        ...,
        max_length=300,
        description=(
            "REASONING STEP 2 - Intent Understanding: "
            "Synthesize what the user actually wants to achieve. "
            "Think beyond keywords: What is their underlying goal? "
            "What business problem are they trying to solve? "
            "Write 1-2 clear sentences in Russian, as if explaining to a colleague."
        ),
    )
    
    # STEP 3: Search Strategy
    subqueries: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "REASONING STEP 3 - Search Strategy: "
            "Break down the user intent into 1-10 specific search queries. "
            "Think: What specific terms would find relevant documentation? "
            "Include: feature names, technical terms, error messages, synonyms. "
            "Each query should be focused and specific. "
            "Write in Russian, avoid duplicates and semantically close queries."
        ),
    )
    
    action_plan: list[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "REASONING STEP 4 - Execution Plan: "
            "Plan concrete steps to answer this request. "
            "Think: What information do I need? In what order? "
            "Consider: search docs → evaluate results → search more OR ask clarification → synthesize answer. "
            "List up to 10 steps in Russian, as actionable instructions to yourself."
        ),
    )
    
    # STEP 4: Confidence Assessment
    intent_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "REASONING STEP 5 - Confidence Assessment: "
            "How confident are you in your understanding of the user intent? "
            "Think: Is the request clear? Do I understand the context? Are there ambiguities? "
            "0.0-0.4: Very unclear, major uncertainties; "
            "0.5-0.7: Somewhat clear but some gaps; "
            "0.8-1.0: Clear and well-understood. "
            "This affects whether you'll ask for clarification."
        ),
    )
    
    uncertainties: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "REASONING STEP 5b - Uncertainty Analysis: "
            "If intent_confidence < 0.7, list specific uncertainties. "
            "Think: What don't I understand? What information is missing? "
            "List up to 5 specific questions or gaps in Russian. "
            "Empty list if confidence >= 0.7."
        ),
    )
    
    # STEP 5: Routing Decision (Action = Template Name)
    action: SGRAction = Field(
        ...,
        description=(
            "REASONING STEP 6 - Routing Decision (ALSO Template Selection): "
            "Based on ALL previous reasoning steps, decide how to proceed. "
            "NOTE: This value IS the template name - choose carefully: "
            "'normal': spam_score < 0.7 AND intent_confidence >= 0.6 AND no safety issues from guardian; "
            "'clarify': intent_confidence < 0.6 AND spam_score < 0.7; "
            "'block': spam_score >= 0.7; "
            "'guardian_block': safety issues detected (from prompt context). "
            "Consider guardian assessment provided in prompt context."
        ),
    )
    
    # STEP 6: Clarification (if needed)
    clarification_question: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "REASONING STEP 7 - Clarification Strategy: "
            "If action='clarify', formulate a specific helpful question. "
            "Think: What information would most help me understand this request? "
            "Ask about: missing context, ambiguous terms, specific use case, etc. "
            "Write in Russian, be polite and specific. "
            "None if action != 'clarify'."
        ),
    )
```

---

## 6. Updated Handler Architecture (Tool Trace Removal)

### Complete Handler Flow

```python
from rag_engine.api.i18n import get_text

async def agent_chat_handler_with_sgr_injection(user_message, gradio_history, messages):
    """
    Handler with clean SGR injection pattern:
    1. Guardian check (if enabled)
    2. Forced SGR tool call (internal only)
    3. Store structured data, skip adding tool trace to messages
    4. Inject single synthetic message
    5. Unbind SGR tool for remainder of turn
    """
    
    # ========== 0. GUARDIAN CHECK ==========
    if settings.guard_enabled:
        moderation_result = await guard_client.classify(user_message)
        guard_mode = getattr(settings, "guard_mode", "enforce")
        should_block = guard_client.should_block(moderation_result)
        
        if should_block and guard_mode == "enforce":
            # Block immediately - no SGR processing
            refusal_msg = get_text("guardian_refusal_unsafe")
            gradio_history.append({
                "role": "assistant",
                "content": refusal_msg,
                "metadata": {"ui_type": "guardian_block"}
            })
            
            # Store metadata for downstream
            agent_context.guardian_result = moderation_result
            agent_context.blocked = True
            agent_context.block_reason = "guardian_unsafe"
            
            yield gradio_history
            return
    else:
        moderation_result = None
    
    # ========== 1. FORCED SGR TOOL CALL ==========
    # Prepare SGR prompt with guardian context (if available)
    # Guardian classification is GLUED to the system message BEFORE SGR call
    # This provides safety context without being part of the tool schema
    sgr_system_prompt = build_sgr_system_prompt(moderation_result)
    
    # Execute SGR tool (forced tool_choice)
    sgr_llm = LLMManager(...)._chat_model()
    sgr_model = sgr_llm.bind_tools(
        [analyse_user_request_tool],
        tool_choice={"type": "function", "function": {"name": "analyse_user_request"}},
    )
    
    sgr_response = await sgr_model.ainvoke([sgr_system_prompt] + messages)
    
    # Extract plan from tool call arguments
    tool_call = sgr_response.tool_calls[0]
    sgr_plan = tool_call["args"]  # Already validated by schema
    
    # ========== 2. STORE STRUCTURED DATA ==========
    # Save to agent_context for downstream processing
    agent_context.sgr_plan = sgr_plan
    agent_context.guardian_result = moderation_result
    
    # ========== 3. SKIP TOOL TRACE IN HISTORY ==========
    # CRITICAL: Do NOT add tool call or tool result to messages
    # The model should NOT see: assistant message with tool_calls
    # The model should NOT see: tool message with JSON result
    # 
    # This is the CLEAN INJECTION pattern - remove tool artifacts entirely
    
    # ========== 4. RENDER SINGLE SYNTHETIC MESSAGE ==========
    template_name = sgr_plan["action"]  # "normal", "clarify", "block", "guardian_block"
    
    # Render full template (analysis + response sections)
    synthetic_content = render_sgr_template(template_name, sgr_plan, moderation_result)
    
    # ========== 5. INJECT SINGLE ASSISTANT MESSAGE ==========
    messages.append({
        "role": "assistant",
        "content": synthetic_content,
        "metadata": {
            "sgr_injected": True,
            "sgr_action": sgr_plan["action"],
            "sgr_turn": current_turn_id
        }
    })
    
    # ========== 6. BUILD UI RESPONSE ==========
    user_intent_prefix = get_text("user_intent_prefix")
    
    # Extract response section for UI
    response_text = extract_response_section(synthetic_content)
    
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
        # Blocking actions - stop here, no further processing
        return
    
    elif sgr_plan["action"] == "clarify":
        # Clarification needed - wait for user response
        # Agent state is ready for next user message
        return
    
    else:  # action == "normal"
        # ========== 8. UNBIND SGR TOOL ==========
        # Remove SGR tool from available tools for this turn
        available_tools = [t for t in all_tools if t.name != "analyse_user_request"]
        
        # ========== 9. CREATE AGENT WITHOUT SGR ==========
        agent = create_react_agent(
            model=llm,
            tools=available_tools,  # SGR tool NOT included
            prompt=agent_system_prompt,
            # ... other config
        )
        
        # ========== 10. CONTINUE WITH NORMAL AGENT FLOW ==========
        # Model sees: user message → synthetic analysis (as its own) → continues
        async for stream_mode, chunk in agent.astream(
            {"messages": messages},
            config={...}
        ):
            # Handle streaming, tool calls, final answer
            # ... (existing streaming logic)
            pass
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
    moderation_result: dict | None
) -> str:
    """Render SGR template with all variables."""
    
    # Get template
    template = SGR_TEMPLATES[template_name]
    
    # Prepare variables - format lists inside, not as separate variables
    variables = {
        "topic": infer_topic(sgr_plan["user_intent"]),
        "user_intent": sgr_plan["user_intent"],
        "category": infer_category(sgr_plan["user_intent"]),
        "spam_score": sgr_plan["spam_score"],
        "spam_reason": sgr_plan["spam_reason"],
        "intent_confidence": sgr_plan["intent_confidence"],
        # Format lists internally
        "uncertainties": _format_list(sgr_plan.get("uncertainties", [])),
        "subqueries": ", ".join(sgr_plan["subqueries"]),
        "action_plan": _format_numbered_list(sgr_plan.get("action_plan", [])),
        "action": sgr_plan["action"],
        "clarification_question": sgr_plan.get("clarification_question"),
        "guard_categories": moderation_result.get("categories", []) if moderation_result else [],
        # i18n texts
        "i18n_sgr_normal_response": get_text("sgr_normal_response", user_intent=sgr_plan["user_intent"]),
        "i18n_sgr_clarify_intro": get_text("sgr_clarify_intro", user_intent=sgr_plan["user_intent"]),
        "i18n_sgr_clarify_outro": get_text("sgr_clarify_outro"),
        "i18n_sgr_spam_response": get_text("sgr_spam_response"),
        "i18n_sgr_guardian_response": get_text("sgr_guardian_response"),
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


def extract_response_section(synthetic_content: str) -> str:
    """Extract the ## Response section from synthetic message for UI."""
    parts = synthetic_content.split("## Response")
    if len(parts) > 1:
        return parts[1].strip()
    return synthetic_content  # Fallback: return full content
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

### Updated Flow with Guardian + Clean SGR Injection

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
        │ (INTERNAL ONLY)      │
        └──────────────────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Remove Tool Trace    │
        │ - No tool message    │
        │ - No JSON in history │
        └──────────────────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Inject Synthetic     │
        │ Single assistant msg │
        │ (analysis+response)  │
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
        │ With clean context   │
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

| Data | Context Synthetics (Model) | UI Synthetics (User) | Structured Metadata (System) |
|------|---------------------------|---------------------|------------------------------|
| user_intent | ✅ Yes | ✅ Yes | ✅ Yes |
| topic | ✅ Yes | ❌ No | ✅ Yes |
| category | ✅ Yes | ❌ No | ✅ Yes |
| spam_score | ✅ Yes | ❌ No | ✅ Yes |
| spam_reason | ✅ Yes | ❌ No | ✅ Yes |
| intent_confidence | ✅ Yes | ❌ No | ✅ Yes |
| uncertainties | ✅ Yes | ❌ No | ✅ Yes |
| subqueries | ✅ Yes | ❌ No | ✅ Yes |
| action_plan | ✅ Yes | ❌ No | ✅ Yes |
| action | ✅ Yes | ❌ No | ✅ Yes |
| clarification_question | ✅ Yes | ✅ Yes (if clarify) | ✅ Yes |
| guard_categories | ✅ Yes (in analysis) | ❌ No | ✅ Yes |
| Response section | ✅ Yes | ✅ Yes | ✅ Yes |
| Tool call trace | ❌ **NO** | ❌ No | ✅ Yes (in logs) |
| Raw JSON result | ❌ **NO** | ❌ No | ✅ Yes (sgr_plan) |

**Key Principle:** Three outputs with clean separation:
- **Context Synthetics**: Full message with analysis (model's "own" reasoning)
- **UI Synthetics**: Response section only (user-facing, no internal details)
- **Structured Metadata**: Raw JSON + Guardian data (downstream processing)

---

## 9. Implementation Phases (Updated)

### Phase 1: Core Clean Injection System

- [ ] **Update SGRPlanResult schema** with enhanced field descriptions (REASONING STEP N pattern)
- [ ] Add new fields: `action` enum, `intent_confidence`, `uncertainties`
- [ ] **Remove from schema**: `ask_for_clarification` (replaced by action), `template_hint`
- [ ] Define template catalog (normal, clarify, block, guardian_block - single message format)
- [ ] **Add i18n keys** to `rag_engine/api/i18n.py`:
  - `sgr_normal_response`
  - `sgr_clarify_intro`, `sgr_clarify_outro`
  - `sgr_spam_response`
  - `sgr_guardian_response`
  - `user_intent_prefix`
- [ ] Implement `render_sgr_template()` function
- [ ] Implement `extract_response_section()` function
- [ ] Implement helper functions for template rendering:
  - `infer_topic(user_intent: str) -> str` - Infer topic from user intent
  - `infer_category(user_intent: str) -> str` - Infer category from user intent
  - `format_bullets(items: list) -> str` - Format list as markdown bullets
  - `format_numbered(items: list) -> str` - Format list as numbered markdown
- [ ] **Modify handler flow**:
  - Execute SGR tool internally
  - Store result in `agent_context.sgr_plan`
  - **SKIP adding tool call/result to messages**
  - Render and inject single synthetic message
  - **Unbind SGR tool** after first call
- [ ] Create agent WITHOUT SGR tool for remainder of turn
- [ ] Update UI emission to show only response section

### Phase 2: Guardian Integration (ALREADY DONE - Verify Alignment)

- [x] Guardian already merged from `cmw-rag-guard-test` branch
- [x] `GuardClient` class in `rag_engine/core/guard_client.py`
- [x] Settings in `rag_engine/config/settings.py`
- [ ] **Verify integration** with clean SGR injection
- [ ] **Update** `build_sgr_system_prompt()` to use existing guardian client
- [ ] **Test** ENFORCE mode: Guardian blocks → no SGR called
- [ ] **Test** REPORT mode: Guardian runs → SGR gets context → may route to guardian_block
- [ ] **Test** Safe flow: Guardian passes → SGR executes → clean injection

### Phase 3: Testing & Validation

- [ ] Unit tests for template rendering
- [ ] Unit tests for `extract_response_section()`
- [ ] Integration test: Full flow with clean injection
- [ ] Verify tool trace is NOT in message history
- [ ] Verify synthetic message IS in message history
- [ ] Verify SGR tool unbound after first call
- [ ] A/B test: Clean injection vs. standard tool result
- [ ] Measure: plan-to-answer consistency, token efficiency
- [ ] Guardian integration tests (all modes)

### Phase 4: Documentation & Rollout

- [ ] Update architecture documentation
- [ ] Document clean injection pattern for team
- [ ] Add inline code comments explaining tool trace removal
- [ ] Consider provisional patent application for clean injection pattern
- [ ] Gradual rollout with feature flag

---

## 10. Open Questions

### Resolved ✅

1. **Single vs dual assistant messages:** ✅ Single combined message
2. **Mid-turn SGR calls:** ✅ Unbind after first call per turn
3. **Tool trace in context:** ✅ Remove entirely, clean injection only
4. **Guardian integration:** ✅ Already implemented, align with SGR

### Remaining ❓

5. **Template language:** 
   - [x] Python `.format()` (simple, chosen)
   - [ ] Jinja2 (more powerful, conditionals)
   
6. **Template storage:**
   - [x] Hardcoded in Python (current)
   - [ ] External YAML/JSON config
   - [ ] Database (admin-editable)

7. **Response section extraction:**
   - [x] String split on "## Response" (simple)
   - [ ] Regex-based extraction (more robust)
   - [ ] Structured return from template (separate analysis/response)

8. **A/B Testing:**
   - [ ] Feature flag: `SGR_CLEAN_INJECTION_ENABLED`
   - [ ] Compare: Clean injection vs. standard tool result
   - [ ] Gradual rollout percentage

---

## 11. Success Metrics

### Quantitative

- **Plan-to-Action Consistency:** BLEU/ROUGE between plan and final answer
- **Tool Trace Visibility:** Verify via logs that tool calls NOT in model context
- **Token Efficiency:** Compare context usage (clean injection vs. tool trace)
- **Response Latency:** Measure any impact of template rendering
- **User Satisfaction:** Blind test comparing responses
- **Error Rate:** Fewer "apology loops" or confusion in agent responses

### Qualitative

- **Response Quality:** Does model follow plan better with clean injection?
- **Coherence:** More natural flow in multi-turn conversations?
- **Debugging:** Easier or harder to trace with tool trace removed?
- **Transparency:** Are we comfortable with hiding tool usage from model?

---

## 12. DMN: Decision Model and Notation

Decision tables for SGR routing, Guardian enforcement, and tool binding logic.

### DT-001: Guardian Enforcement Decision

Determines whether to block request before SGR processing based on Guardian safety assessment.

| guardian_level | guard_mode | Decision | Action |
|----------------|------------|----------|--------|
| Unsafe | enforce | Block | Immediate refusal, no SGR call |
| Unsafe | report | Continue | Proceed to SGR with guardian context |
| Controversial | enforce | Continue | Proceed to SGR with guardian context |
| Controversial | report | Continue | Proceed to SGR with guardian context |
| Safe | enforce | Continue | Proceed to SGR normally |
| Safe | report | Continue | Proceed to SGR normally |

**Inputs:**
- `guardian_level`: Enum [Unsafe, Controversial, Safe]
- `guard_mode`: Enum [enforce, report]

**Output:**
- `Decision`: Block | Continue
- `Action`: Refusal message | Proceed to SGR

---

### DT-002: SGR Routing Decision

Determines action and template based on SGR analysis results and Guardian context.

| spam_score | intent_confidence | guardian_level | action | template | Flow |
|------------|-------------------|----------------|--------|----------|------|
| < 0.7 | >= 0.6 | Safe/Controversial/N/A | normal | normal | Continue with agent tools |
| < 0.7 | < 0.6 | Safe/Controversial/N/A | clarify | clarify | Wait for user clarification |
| >= 0.7 | Any | Safe/Controversial/N/A | block | block | Stop, show refusal |
| Any | Any | Unsafe (via context) | guardian_block | guardian_block | Stop, show safety refusal |

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

Determines whether SGR tool is available for calling based on turn state.

| turn_state | sgr_call_count | sgr_tool_available | Action |
|------------|----------------|---------------------|--------|
| New user message | 0 | Yes | Bind SGR tool, force call |
| New user message | > 0 | Yes | Reset counter, bind SGR tool, force call |
| After SGR execution | 1 | No | Unbind SGR tool, continue with remaining tools |
| Mid-turn | 1 | No | SGR unavailable (prevent re-planning) |
| Next user message | 0 | Yes | Re-bind SGR tool for new turn |

**Inputs:**
- `turn_state`: Enum [new_user_msg, after_sgr, mid_turn]
- `sgr_call_count`: Integer [0, 1, >1]

**Output:**
- `sgr_tool_available`: Boolean
- `Action`: Bind | Unbind | Force call

---

### DT-004: Output Routing Decision

Determines what gets emitted to each output channel.

| Output Channel | Analysis Section | Response Section | Tool Trace | Structured Data |
|----------------|------------------|------------------|------------|-----------------|
| Context Synthetics (Model) | Yes | Yes | No | No |
| UI Synthetics (User) | No | Yes | No | No |
| Structured Metadata (System) | Yes | Yes | Yes | Yes |

**Note:** Tool trace includes raw JSON result and tool call metadata. Only stored in `agent_context.sgr_plan` for downstream processing, not visible to model or user.

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
1. Review and approve revised architecture
2. Implement Phase 1: Clean injection with single message
3. Verify Phase 2: Guardian alignment (already implemented)
4. Begin testing Phase 3
5. Document clean injection pattern for team
