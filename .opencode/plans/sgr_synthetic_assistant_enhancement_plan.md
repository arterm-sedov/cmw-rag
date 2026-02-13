# SGR Synthetic Assistant Message Enhancement Plan

**Status:** DRAFT  
**Created:** 2026-02-13  
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

## 3. Recommended Approach: Hybrid Structured Assistant Message

### Rationale
User preference confirmed: **Option 3 (Hybrid)** - markdown with headers, not pure JSON, not pure natural language.

**Why This Wins:**
1. **Not overengineered** - Just markdown with headers
2. **Sophisticated** - Model gets structured guidance, thinks naturally
3. **Spam score handled** - Internal sections never shown to user
4. **Best of both worlds** - Structure for system, flow for model
5. **Template-friendly** - Deterministic, testable, maintainable

### Message Structure

```markdown
## Request Analysis
- **Topic**: Comindware Platform configuration
- **Intent**: Set up Single Sign-On (SSO) integration
- **Context**: Infrastructure/authentication setup

## Internal Assessment
- Validity: Legitimate support request [spam_score: 0.1, proceed: true]
- Confidence: High (0.92)
- Category: Technical configuration
- Action: Provide detailed guidance

## Approach
1. Retrieve SSO configuration documentation
2. Search for SAML and OAuth setup guides
3. Identify prerequisites and requirements
4. Provide step-by-step configuration steps
5. Include common troubleshooting scenarios

Proceeding with knowledge base search...
```

**Key Sections:**
- `Request Analysis` - User-facing, professional tone
- `Internal Assessment` - System metadata (spam, confidence, danger scores)
- `Approach` - Actionable plan for model to follow

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

### Template Catalog

#### 1. Normal Template (High Confidence, Safe)
```markdown
## Request Analysis
- **Topic**: {topic}
- **Intent**: {user_intent}
- **Category**: {category}
- **Context**: {context}
- **Validity**: Legitimate support request [spam_score: {spam_score}]
- **Confidence**: High ({intent_confidence})
- **Action**: Proceed with the next steps

## Next Steps

Proceed with this plan to answer the user request:

{action_plan_numbered}
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
- **Action**: Clarify the request, decline to proceed

## Response

Please clarify the following, so that I can proceed further:

{clarification_question}
```

#### 3. Spam Template (High Spam Score)
```markdown
## Request Analysis
- **Assessment**: Off-topic or spam request
- **Validity**: Request unrelated to Comindware Platform [spam_score: {spam_score}]
- **Reason**: {spam_reason}
- **Action**: Request clarification, decline to proceed

## Response
Can't proceed, because the request does not seem to be related to Comindware Platform support.
Please let me know if you'd like help with Comindware Platform configuration, troubleshooting, or features.
```

#### 4. Guardian Blocked Template (Danger Score)
```markdown
## Request Analysis
- **Assessment**: Request blocked by safety policy
- **Validity**: Potentially harmful [guard_categories: {guard_categories}]
- **Category**: Unsafe request
- **Action**: Decline to proceed

## Response
Can't process this request as it does not seem to be safe.
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
    "guard_categories": {"object from guardian"},
    "category": "Request classification",
    "action_plan": "List of steps",
    "uncertainties": "List if confidence low",
    "clarification_question": "Generated question",
    "guardian_reason": "Safety category if blocked"
}
```

---

## 5. Architecture: Clean Separation

### Tool Responsibility (Pure Function)

```python
@tool("analyse_user_request", args_schema=SGRPlanResult)
async def analyse_user_request(
    spam_score: float,
    spam_reason: str,
    user_intent: str,
    subqueries: list[str],
    action_plan: list[str],
    intent_confidence: float,  # NEW
    guard_categories: {"object from guardian"},  # NEW (from guardian)
    ask_for_clarification: bool,
    clarification_question: str | None,
    runtime: ToolRuntime,
) -> str:
    """
    Pure function - returns analysis results only.
    No side effects, no context mutation.
    Handler decides how to use results.
    """
    plan = {
        "spam_score": spam_score,
        "spam_reason": spam_reason,
        "user_intent": user_intent,
        "subqueries": subqueries,
        "action_plan": action_plan,
        "intent_confidence": intent_confidence,
        "guard_categories": guard_categories,
        "ask_for_clarification": ask_for_clarification,
        "clarification_question": clarification_question,
        "template_hint": select_template_hint(
            spam_score, intent_confidence, guard_categories
        ),
    }
    
    return json.dumps(plan, ensure_ascii=False)
```

### Handler Responsibility (Orchestration)

```python
# In agent_chat_handler:

# 1. Execute SGR tool (returns JSON)
sgr_result_json = await execute_sgr_tool(messages)
sgr_plan = json.loads(sgr_result_json)

# 2. Store structured data for external use (UI, metrics)
agent_context.sgr_plan = sgr_plan

# 3. Select and render template
template_name = sgr_plan["template_hint"]
synthetic_message = render_template(template_name, sgr_plan)

# 4. Inject synthetic assistant message for MODEL
messages.append({
    "role": "assistant",
    "content": synthetic_message
})

# 5. UI message (excludes sensitive fields like spam_score)
ui_message = render_ui_template(template_name, sgr_plan)
gradio_history.append({
    "role": "assistant",
    "content": ui_message,
    "metadata": {"ui_type": "sgr_analysis"}
})
```

### Template Selection Logic

```python
def select_template_hint(spam_score: float, 
                         confidence: float, 
                         guard_categories: {"object from guardian"}) -> str:
    """Deterministic template selection."""
    
    if guard_categories > unsafe or harmful:
        return "guardian_blocked"
    
    if spam_score > 0.7:
        return "spam"
    
    if confidence < 0.6:
        return "uncertain"
    
    return "normal"
```

---

## 6. Handling Sensitive Data

### Spam Score Constraint

**Requirement:** Model must see spam score from the templated assistant message, UI must NOT display it.

See the templates above 

# UI message for USER (excludes internal data):
ui_msg = """**User Intent**

The user wants help with SSO configuration...
"""
```

### Data Visibility Matrix

| Field | Synthetic Message (Model) | UI Message (User) | Context (External) |
|-------|---------------------------|-------------------|-------------------|
| user_intent | ✅ Yes | ✅ Yes | ✅ Yes |
| action_plan | ✅ Yes | ❌ No | ✅ Yes |
| spam_score | ✅ Yes (bracketed) | ❌ No | ✅ Yes |
| spam_reason | ✅ Yes | ❌ No | ✅ Yes |
| intent_confidence | ✅ Yes | ❌ No | ✅ Yes |
| guard_categories | ✅ Yes | ❌ No | ✅ Yes |
| subqueries | ✅ Yes | ❌ No | ✅ Yes |

---

## 7. Implementation Phases

### Phase 1: Core Template System
- [ ] Define template catalog (normal, uncertain, spam)
- [ ] Implement template rendering functions
- [ ] Add template selection logic
- [ ] Update `analyse_user_request` to return template_hint
- [ ] Modify `agent_chat_handler` to inject synthetic message
- [ ] Keep existing structured output flow for external use

### Phase 2: Enhanced Schema
- [ ] Add `intent_confidence` field to SGRPlanResult
- [ ] Integrate guardian model `guard_categories`
- [ ] Update prompt to populate new fields
- [ ] Add guardian template

### Phase 3: Testing & Refinement
- [ ] Unit tests for template rendering
- [ ] Integration tests for full flow
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
   - [ ] Python `.format()` (simple)
   - [ ] Jinja2 (more powerful, conditionals)
   - [ ] Custom templating (overkill?)

3. **Template Storage:**
   - [ ] Hardcoded in Python (current draft)
   - [ ] External YAML/JSON config
   - [ ] Database (admin-editable)

4. **A/B Testing:**
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

### From Prior Analysis
5. Kimi Report 1-4 - Novelty analysis and sgr-agent-core comparison
6. Current codebase analysis - rag_engine/tools/analyse_user_request.py

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

---

**Next Steps:**
1. Review and approve plan
2. Decide on Phase 1 implementation details
3. Draft PR with template system
4. Set up A/B testing framework

**Contact:** [Your name/team]
