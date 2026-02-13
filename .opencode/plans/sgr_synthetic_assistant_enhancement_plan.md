# SGR Synthetic Assistant Message Enhancement Plan

**Status:** DRAFT  
**Created:** 2026-02-13  
**Updated:** 2026-02-13 (Handler Rendering & Three-Output Architecture)  
**Based on:** Analysis of current SGR implementation, novelty research, and conversation about synthetic message injection patterns

---

## 1. Executive Summary

### Current State
- SGR tool (`analyse_user_request`) extracts structured plan
- Plan stored in `AgentContext.sgr_plan` for external use (UI, metrics)
- Model receives raw JSON via standard `role: "tool"` message
- UI already displays `user_intent` - but model doesn't leverage it as "its own reasoning"

### Proposed Enhancement
Inject **synthetic assistant message** (natural language + structured sections) into model context as `role: "assistant"`, while keeping structured data for external processing.

**Key Innovation:** Model believes it performed the analysis, eliminating cognitive dissonance between raw tool output and creative generation phase.

### Three-Output Architecture (Simplified)
Based on deep research into industry best practices (LangChain, Anthropic, 12-Factor Agents), we adopt a **clean separation** where:
1. **Tool** returns pure structured data (`action` IS template name - no mapping)
2. **Handler** renders templates and orchestrates flow
3. **Guardian assessment** passed via prompt context (not LLM-generated)
4. **Three outputs** are generated: synthetic analysis (context only), synthetic response (context + UI), structured metadata (downstream)

**Key Simplification:** `action` enum values (normal, clarify, block, guardian_block) ARE the template names - no separate `template_hint` field needed.

---

## 2. Novelty Verification (Web Research Summary)

### Confirmed: Approach is NOVEL

**What Exists (Standard):**
- Forced Tool Calling - Industry standard (OpenAI, Anthropic APIs)
- Schema validation via Pydantic - Instructor, LangChain standard
- Structured output extraction - Widely used

**What Does NOT Exist (Our Innovation):**
- ✅ Synthetic assistant message injection as **benign architectural pattern**
- ✅ Tool-as-cognitive-filter preprocessing
- ✅ Transparent history rewrite for self-consistency
- ✅ Hybrid structured/natural language reasoning guidance

**Prior Art Found:**
1. **Attack papers** (Pseudo-Conversation Injection, Dialogue Injection Attack) - Prove technical feasibility but for malicious manipulation
2. **sgr-agent-core** - Uses classic SGR with standard tool results, NO synthetic injection
3. **Instructor library** - Returns validated objects to code, NO message injection
4. **Self-correction papers** (CRITIC, ReAct refinements) - Explicit feedback loops, NOT transparent injection

### Web Research: Assistant vs Reasoning Field

**Key Finding:** No definitive industry standard for "reasoning" vs "assistant" role injection found. Current API patterns:

- **OpenAI API:** `role: "assistant"` for model messages, no separate "reasoning" role
- **Chain-of-thought models** (o1, etc.) - Internal reasoning not exposed in API
- **Best practice emerging:** Assistant messages can contain structured sections (markdown) for complex reasoning

**Recommendation:** Use `role: "assistant"` with structured markdown sections - this is:
- More natural for model continuation
- Compatible with all major APIs
- Allows hybrid structure (sections + natural language)
- Tested pattern in production systems

---

## 3. Three-Output Architecture

### Overview

Based on deep research of industry best practices (Anthropic's "Building Effective Agents", LangChain patterns, 12-Factor Agents principles), we adopt a **clean three-output architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                    SGR TOOL (Pure Function)                  │
│  Returns: Structured JSON with all metadata                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    HANDLER (Orchestrator)                    │
│  Renders templates, decides routing, manages outputs        │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Output 1:       │  │ Output 2:       │  │ Output 3:       │
│ Synthetic       │  │ Synthetic       │  │ Structured      │
│ Analysis        │  │ Response        │  │ Metadata        │
│ (Agent Context) │  │ (Context + UI)  │  │ (Downstream)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Output 1: Synthetic Analysis → Agent Context ONLY

**Purpose:** Give model "its own" structured reasoning to continue from

**Content:** Full template with all sections including `## Internal Assessment`

**Recipients:**
- ✅ Agent context (model sees this)
- ❌ NOT shown in UI

**Example:**
```markdown
## Request Analysis
- **Topic**: Comindware Platform configuration
- **Intent**: Set up Single Sign-On (SSO) integration
- **Validity**: Legitimate support request [spam_score: 0.1]
- **Confidence**: High (0.92)

## Internal Assessment
- Category: Technical configuration
- Action: proceed

## Approach
1. Retrieve SSO documentation
2. Search for SAML guides
3. Provide step-by-step instructions

Proceeding with knowledge base search...
```

### Output 2: Synthetic Response → BOTH Context and UI

**Purpose:** Immediate user-facing response for blocking/clarification cases

**Content:** `## Response` section from template

**Recipients:**
- ✅ Agent context (model sees this as its own response)
- ✅ UI (user sees this immediately)

**When Generated:** All templates now include a response section

**Template-Specific Responses:**

| Template | Response Content |
|----------|------------------|
| **normal** | "I'll help you with [user intent]..." |
| **uncertain** | "Please clarify the following..." |
| **spam** | "Can't proceed, request unrelated..." |
| **guardian** | "Can't process this request..." |

### Output 3: Structured Metadata → Downstream Processing

**Purpose:** Analytics, metrics, debugging, future tool inputs

**Content:** Complete SGR plan with all fields

**Recipients:**
- ✅ `agent_context.sgr_plan` for structured end-of-turn output
- ✅ Logging, metrics, A/B testing
- ✅ Future downstream tools

**Fields Included:**
```python
{
  "spam_score": 0.1,
  "spam_reason": "...",
  "user_intent": "...",
  "intent_confidence": 0.92,
  "subqueries": [...],
  "action_plan": [...],
  "uncertainties": [...],
  "action": "normal",           # Enum: normal, clarify, block, guardian_block
  "clarification_question": "..."  # When action=clarify
}
```

**Note:** `guard_categories` comes from preceding guardian call and is passed in the prompt context, not generated by the SGR tool.

### Why Handler Rendering? (Deep Research Summary)

Based on research of LangChain, Anthropic, 12-Factor Agents, and Microsoft Semantic Kernel:

| Aspect | Tool Rendering | Handler Rendering |
|--------|---------------|-------------------|
| **Separation of Concerns** | ❌ Tool does too much | ✅ Tool = data, Handler = presentation |
| **Reusability** | ❌ Locked to one template | ✅ Same tool, different presentations |
| **Testability** | ❌ Hard to test variations | ✅ Test templates independently |
| **Flexibility** | ❌ Must modify tool to change UI | ✅ Swap templates without touching tool |
| **A/B Testing** | ❌ Tool changes needed | ✅ Just change handler config |
| **Debugging** | ❌ Can't see raw data easily | ✅ Raw data always available |
| **Future-proof** | ❌ Template logic in tool | ✅ Easy to add new templates |

**Industry Consensus:**
- **Anthropic:** "Simple, composable patterns... separate data extraction from orchestration"
- **12-Factor Agents:** "Tools should be stateless and portable... orchestration logic belongs in the workflow layer"
- **LangChain:** Tools return structured data, handlers decide presentation

---

## 4. Template System Design

### Why Templates Over Creative Generation

| Aspect | Templates | Creative Generation |
|--------|-----------|---------------------|
| Determinism | ✅ Same input = same output | ❌ Variable |
| Testability | ✅ Unit testable | ❌ Hard to test |
| Maintainability | ✅ Edit templates easily | ❌ Retrain/regenerate |
| Intent Clarity | ✅ Clear purpose per template | ❌ May drift |
| Token Efficiency | ✅ Optimized | ❌ Verbose |

**Decision:** Templates for SGR reasoning (deterministic), model handles creative answer generation.

### Template Catalog (All Include ## Response Section)

**Important:** UI response texts use i18n keys for internationalization, following the existing pattern in `rag_engine/api/i18n.py`.

#### 1. Normal Template (High Confidence, Safe)

```markdown
## Request Analysis
- **Topic**: {topic}
- **Intent**: {user_intent}
- **Category**: {category}
- **Context**: {context}
- **Validity**: Legitimate support request [spam_score: {spam_score}]
- **Confidence**: High ({intent_confidence})
- **Action**: {action}

## Approach

Proceed with this plan to answer the user request:

{action_plan_numbered}

## Response

{i18n_sgr_normal_response}
```

**UI Output:**
- User Intent (already shown)
- Response: i18n text from `sgr_normal_response` key

**i18n Keys:**
```python
"sgr_normal_response": "I'll help you with {user_intent}. Let me search our knowledge base for the most relevant information."
"sgr_normal_response": "Я помогу вам с {user_intent}. Позвольте мне найти наиболее релевантную информацию в базе знаний."
```

#### 2. Uncertain Template (Low Confidence)

```markdown
## Request Analysis
- **Topic**: {topic}
- **Intent**: {user_intent} (not completely understood)
- **Context**: {context}
- **Validity**: Request needs clarification [spam_score: {spam_score}]
- **Confidence**: Low ({intent_confidence})
- **Uncertainties**:
    {uncertainties_bullets}
- **Action**: {action}

## Response

{i18n_sgr_clarify_intro}

{clarification_question}

{i18n_sgr_clarify_outro}
```

**UI Output:**
- User Intent (already shown)
- Response: i18n text + generated clarification question

**i18n Keys:**
```python
"sgr_clarify_intro": "I want to make sure I understand your request correctly. You mentioned {user_intent}, but I need some clarification:"
"sgr_clarify_intro": "Я хочу убедиться, что правильно понял ваш запрос. Вы упомянули {user_intent}, но мне нужно уточнение:"

"sgr_clarify_outro": "Could you please provide more details so I can assist you better?"
"sgr_clarify_outro": "Не могли бы вы предоставить больше деталей, чтобы я мог лучше помочь?"
```

#### 3. Spam Template (High Spam Score)

```markdown
## Request Analysis
- **Assessment**: Off-topic or spam request
- **Validity**: Request unrelated to Comindware Platform [spam_score: {spam_score}]
- **Reason**: {spam_reason}
- **Action**: {action}

## Response

{i18n_sgr_spam_response}
```

**UI Output:**
- User Intent (already shown)
- Response: i18n text from `sgr_spam_response` key

**i18n Keys:**
```python
"sgr_spam_response": "I notice this request doesn't appear to be related to Comindware Platform support.\n\nI'm designed to help with Comindware Platform configuration, troubleshooting, and features. Please let me know if you'd like assistance with any of these topics."
"sgr_spam_response": "Я заметил, что этот запрос, похоже, не связан с поддержкой Comindware Platform.\n\nЯ предназначен для помощи с настройкой, устранением неполадок и функциями Comindware Platform. Пожалуйста, дайте мне знать, если вам нужна помощь с любой из этих тем."
```

#### 4. Guardian Blocked Template (Danger Score)

```markdown
## Request Analysis
- **Assessment**: Request blocked by safety policy
- **Validity**: Potentially harmful [guard_categories: {guard_categories}]
- **Category**: Unsafe request
- **Action**: {action}

## Response

{i18n_sgr_guardian_response}
```

**UI Output:**
- User Intent (already shown)
- Response: i18n text from `sgr_guardian_response` key

**i18n Keys:**
```python
"sgr_guardian_response": "I can't process this request as it may involve potentially harmful actions or content that could affect system security or stability.\n\nIf you need assistance with this type of request, please contact your system administrator or Comindware support directly."
"sgr_guardian_response": "Я не могу обработать этот запрос, так как он может включать потенциально вредоносные действия или контент, который может повлиять на безопасность или стабильность системы.\n\nЕсли вам нужна помощь с таким типом запроса, пожалуйста, свяжитесь с системным администратором или службой поддержки Comindware напрямую."
```

### Template Variables

```python
TEMPLATE_VARIABLES = {
    "topic": "Inferred topic from request",
    "user_intent": "Parsed user intent from SGR",
    "context": "Business/domain context",
    "spam_score": "0.0-1.0",
    "spam_reason": "Explanation if spam",
    "intent_confidence": "0.0-1.0",
    "category": "Request classification",
    "action_plan": "List of steps",
    "uncertainties": "List if confidence low",
    "clarification_question": "Generated question",
    "action": "Enum: normal, clarify, block, guardian_block (ALSO template name)"
    # Note: guard_categories passed via prompt context from guardian call
    # Note: i18n_* variables resolved via get_text() from rag_engine.api.i18n
}
```

**Key Design:** 
- `action` enum values ARE the template names (normal, clarify, block, guardian_block). No mapping needed.
- UI response texts use i18n keys resolved via existing `get_text()` function.

---

## 5. Architecture: Clean Separation (Handler Rendering)

### Enhanced Pydantic Schema with Reasoning Descriptions

The schema descriptions are **critical** - they enforce structured reasoning by guiding the model on how to think through each field:

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
    
    Note: guard_categories comes from preceding guardian call (in prompt context),
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
            "Write in Russian, avoid duplicates."
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
    
    # STEP 5: Confidence Assessment
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
    
    # STEP 6: Routing Decision (Action = Template Name)
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
    
    # STEP 7: Clarification (if needed)
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

### Why Detailed Descriptions Matter

The descriptions enforce **step-by-step structured reasoning**:

1. **Sequential thinking:** Each field builds on previous ones
2. **Explicit criteria:** Clear rubrics for scoring (spam_score, confidence)
3. **Contextual guidance:** "Think beyond keywords", "as if explaining to a colleague"
4. **Decision logic:** Clear rules for routing decisions
5. **Self-consistency:** Later fields reference earlier reasoning

### Tool Responsibility (Pure Function)

```python
@tool("analyse_user_request", args_schema=SGRPlanResult)
async def analyse_user_request(
    spam_score: float,
    spam_reason: str,
    user_intent: str,
    subqueries: list[str],
    action_plan: list[str],
    intent_confidence: float,
    uncertainties: list[str],
    action: SGRAction,
    clarification_question: str | None,
    runtime: ToolRuntime,
) -> str:
    """
    Pure function - returns analysis results only.
    
    Note: guard_categories comes from preceding guardian call (in prompt context),
    not generated by this tool.
    
    The detailed schema descriptions guide the model through structured reasoning.
    Handler decides how to use results.
    """
    plan = {
        "spam_score": spam_score,
        "spam_reason": spam_reason,
        "user_intent": user_intent,
        "subqueries": subqueries,
        "action_plan": action_plan,
        "intent_confidence": intent_confidence,
        "uncertainties": uncertainties,
        "action": action,
        "clarification_question": clarification_question,
    }
    
    return json.dumps(plan, ensure_ascii=False)
```

### Handler Responsibility (Orchestration with Three Outputs)

```python
from rag_engine.api.i18n import get_text

# In agent_chat_handler:

# 0. Pre-step: Guardian assessment (if enabled)
guardian_result = await guardian.assess(message) if guardian_enabled else None
guard_categories = guardian_result.categories if guardian_result else {}

# Include guardian context in SGR tool prompt
sgr_prompt_context = f"""
Guardian Safety Assessment: {guard_categories}
Use this information for routing decisions.
"""

# 1. Execute SGR tool (returns JSON)
sgr_result_json = await execute_sgr_tool(messages, context=sgr_prompt_context)
sgr_plan = json.loads(sgr_result_json)

# 2. Store structured data for downstream processing
agent_context.sgr_plan = sgr_plan  # OUTPUT 3: Structured Metadata

# 3. Render synthetic analysis template
# action IS the template name (normal, clarify, block, guardian_block)
template_name = sgr_plan["action"]  # No mapping needed!
synthetic_analysis = render_template(template_name, sgr_plan)

# 4. Inject synthetic analysis to agent context
messages.append({
    "role": "assistant",
    "content": synthetic_analysis  # OUTPUT 1: Synthetic Analysis → Context only
})

# 5. Build UI response using i18n (following existing pattern)
user_intent_prefix = get_text("user_intent_prefix")  # "How I understood your request:"

# Get template-specific response text from i18n
if sgr_plan["action"] == "normal":
    response_text = get_text("sgr_normal_response", user_intent=sgr_plan['user_intent'])
elif sgr_plan["action"] == "clarify":
    intro = get_text("sgr_clarify_intro", user_intent=sgr_plan['user_intent'])
    outro = get_text("sgr_clarify_outro")
    response_text = f"{intro}\n\n{sgr_plan['clarification_question']}\n\n{outro}"
elif sgr_plan["action"] == "block":
    response_text = get_text("sgr_spam_response")
elif sgr_plan["action"] == "guardian_block":
    response_text = get_text("sgr_guardian_response")

# 6. Inject response to agent context (model sees this as its own response)
messages.append({
    "role": "assistant",
    "content": response_text  # Part of OUTPUT 2: Synthetic Response → Context
})

# 7. UI emission: User intent + Response
ui_message = f"**{user_intent_prefix}**\n\n{sgr_plan['user_intent']}\n\n{response_text}"
gradio_history.append({
    "role": "assistant",
    "content": ui_message,  # OUTPUT 2: Synthetic Response → UI
    "metadata": {"ui_type": "sgr_response_with_intent"}
})

# 8. Route based on action enum
if sgr_plan["action"] in ["block", "guardian_block"]:
    # Skip further tool calls for blocking cases
    yield gradio_history
    return
elif sgr_plan["action"] == "clarify":
    # Continue but wait for user clarification
    yield gradio_history
    return
else:
    # Proceed with normal agent flow
    pass
```

### SGRAction Enum (Action = Template Name)

```python
from enum import Enum

class SGRAction(str, Enum):
    """Action enum values ARE template names - no mapping dictionary needed."""
    NORMAL = "normal"           # Proceed with normal assistance (template: normal)
    CLARIFY = "clarify"         # Need clarification (template: clarify)
    BLOCK = "block"             # Spam/off-topic (template: block)
    GUARDIAN_BLOCK = "guardian_block"  # Safety concern (template: guardian_block)

# Handler usage:
template_name = sgr_plan["action"]  # "normal", "clarify", "block", or "guardian_block"
template = SGR_TEMPLATES[template_name]
```

**Key Design Decision:** No mapping dictionary needed - action values are semantic AND match template names directly.

---

## 6. Handling Sensitive Data

### Data Visibility Matrix

| Field | Synthetic Analysis (Model) | Synthetic Response (Context + UI) | Structured Metadata |
|-------|---------------------------|-----------------------------------|---------------------|
| user_intent | ✅ Yes | ✅ Yes (in UI) | ✅ Yes |
| action_plan | ✅ Yes | ❌ No | ✅ Yes |
| spam_score | ✅ Yes (bracketed) | ❌ No | ✅ Yes |
| spam_reason | ✅ Yes | ❌ No | ✅ Yes |
| intent_confidence | ✅ Yes | ❌ No | ✅ Yes |
| uncertainties | ✅ Yes | ❌ No | ✅ Yes |
| subqueries | ✅ Yes | ❌ No | ✅ Yes |
| action | ✅ Yes | ❌ No | ✅ Yes |

**Note:** `guard_categories` comes from preceding guardian call (prompt context), not from SGR tool.

### Spam Score Constraint

**Requirement:** Model must see spam score from the templated assistant message, UI must NOT display it.

**Solution:** 
- Spam score appears in `## Internal Assessment` section (synthetic analysis only)
- UI only sees `## Response` section which excludes internal scores
- Structured metadata captures all fields for downstream processing

---

## 7. Implementation Phases

### Phase 1: Core Template System
- [ ] **Update SGRPlanResult schema** with enhanced field descriptions (REASONING STEP N pattern)
- [ ] Add new fields: `action` enum (values: normal, clarify, block, guardian_block), `intent_confidence`, `uncertainties`
- [ ] **Remove from schema**: `template_hint` (derived from action), `guard_categories` (from prompt context)
- [ ] Define template catalog (normal, clarify, block, guardian_block - all with ## Response)
- [ ] **Add i18n keys** to `rag_engine/api/i18n.py`:
  - `sgr_normal_response`
  - `sgr_clarify_intro`, `sgr_clarify_outro`
  - `sgr_spam_response`
  - `sgr_guardian_response`
- [ ] Implement template rendering functions in handler
- [ ] Update `analyse_user_request` to return new schema fields
- [ ] Modify `agent_chat_handler` to implement three-output architecture with i18n
- [ ] Add guardian context passing via prompt (separate from SGR tool schema)
- [ ] Keep existing structured output flow for external use

### Phase 2: Guardian Integration
- [ ] Integrate guardian model call in handler (before SGR tool)
- [ ] Pass guardian assessment via prompt context to SGR tool
- [ ] Update SGR tool system prompt to reference guardian context
- [ ] Add fallback logic when guardian is unavailable
- [ ] Test guardian routing in staging environment

### Phase 3: Testing & Refinement
- [ ] Unit tests for template rendering
- [ ] Integration tests for full three-output flow
- [ ] A/B test: synthetic message vs standard tool result
- [ ] Measure: plan-to-answer consistency, user satisfaction

### Phase 4: Documentation & IP Protection
- [ ] Document architecture for team
- [ ] Consider provisional patent application
- [ ] Prepare conference talk/paper outline

---

## 8. Open Questions

1. **Guardian Integration:** Should guard_categories come from:
   - [ ] Separate guardian tool call (current branch)
   - [ ] Integrated into analyse_user_request
   - [ ] Both (guardian enriches the plan)

2. **Template Language:**
   - [x] Python `.format()` (simple, chosen)
   - [ ] Jinja2 (more powerful, conditionals)
   - [ ] Custom templating (overkill)

3. **Template Storage:**
   - [x] Hardcoded in Python (current draft)
   - [ ] External YAML/JSON config
   - [ ] Database (admin-editable)

4. **Schema Field Descriptions:**
   - [x] Enhanced with REASONING STEP N pattern (chosen)
   - [ ] Standard descriptions
   - [ ] Minimal descriptions (rely on examples)

5. **A/B Testing:**
   - [ ] Feature flag to toggle synthetic message
   - [ ] Compare metrics between approaches
   - [ ] Gradual rollout

---

## 9. Success Metrics

### Quantitative
- **Plan-to-Action Consistency:** BLEU/ROUGE between plan and final answer
- **User Satisfaction:** Blind test comparing responses
- **Error Recovery:** Fewer "apology loops" when plans change
- **Token Efficiency:** Context usage comparison

### Qualitative
- **Response Quality:** Does model follow plan better?
- **User Perception:** More coherent, less disjointed?
- **Debugging:** Easier to trace model reasoning?

---

## 10. References

### From Web Research
1. Chen, Z., & Yao, B. (2024). Pseudo-Conversation Injection for LLM Goal Hijacking. arXiv:2410.23678.
2. Meng, W., et al. (2025). Dialogue Injection Attack. arXiv:2503.08195.
3. Instructor Library - https://python.useinstructor.com/
4. sgr-agent-core - https://github.com/vamplabAI/sgr-agent-core
5. Anthropic. (2024). Building Effective Agents. https://www.anthropic.com/research/building-effective-agents
6. 12-Factor Agents. https://github.com/humanlayer/12-factor-agents

### From Prior Analysis
7. Kimi Report 1-5 - Novelty analysis and deep verification
8. Current codebase analysis - rag_engine/tools/analyse_user_request.py

---

## 11. Decision Log

| Decision | Date | Rationale |
|----------|------|-----------|
| Use `role: "assistant"` not "reasoning" | 2026-02-13 | No API support for reasoning role, assistant is standard and natural |
| Hybrid markdown structure | 2026-02-13 | Best of both: structure for system, natural flow for model |
| Template-based generation | 2026-02-13 | Deterministic, testable, maintainable vs creative generation |
| Tool remains pure function | 2026-02-13 | Clean architecture, handler controls orchestration |
| 4-template system | 2026-02-13 | Covers main scenarios: normal, uncertain, spam, guardian-blocked |
| Spam score in synthetic, not UI | 2026-02-13 | Model needs it for routing, users shouldn't see internal scores |
| **Handler renders templates** | 2026-02-13 | Industry best practice (LangChain, Anthropic, 12-Factor Agents) - separation of concerns |
| **Three-output architecture** | 2026-02-13 | Clean separation: analysis (context), response (context+UI), metadata (downstream) |
| **All templates have ## Response** | 2026-02-13 | UI always gets meaningful response preceded by user intent |
| **Include action enum** | 2026-02-13 | Explicit routing decision separate from presentation hint |
| **Enhanced schema descriptions** | 2026-02-13 | Field descriptions enforce step-by-step structured reasoning (REASONING STEP N) |
| **Action IS template name** | 2026-02-13 | Simplify: action enum values match template names directly, no mapping needed |
| **Guardian categories in prompt** | 2026-02-13 | Pass guardian assessment via prompt context, not as LLM-generated field |
| **Remove template_hint** | 2026-02-13 | Handler derives template from action directly, no separate field needed |
| **UI responses use i18n** | 2026-02-13 | Follow existing pattern: all user-facing texts go through i18n system |

---

**Next Steps:**
1. Review and approve plan
2. Implement Phase 1: Template system with three-output architecture
3. Draft PR with handler-based rendering
4. Set up A/B testing framework

**Contact:** [Your name/team]
