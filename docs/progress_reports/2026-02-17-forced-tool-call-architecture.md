# Forced Tool Call Architecture for SGR, SRP, and retrieve_context Tools

**Date:** 2026-02-17  
**Branch:** `20260217_sgr_tool_choice`  
**Commits:** `d80469b`, `e1908e5`, `520ea26`, `1afd185`, `7793fa1`

---

## Summary

Unified architecture for LangChain tools using `bind_tools + tool_choice` for forced tool execution with single-source-of-truth docstrings and Pydantic validators for robustness. This pattern ensures consistent LLM behavior, graceful error handling, and maintainable code across all RAG tools.

---

## Problem Statement

Before this refactor, tools had inconsistent architecture:

| Issue | SGR (Before) | SRP (Before) | retrieve_context (Before) |
|-------|--------------|--------------|---------------------------|
| Method | `with_structured_output` | `with_structured_output` | N/A |
| Docstring | Split (schema + function) | Split (schema + function) | Split |
| Required fields | 1 (`user_intent`) | 1 (`engineer_intervention_needed`) | 1 (`query`) |
| Validators | Partial | None | Partial |
| Error handling | Manual JSON construction | Manual JSON construction | None for empty |

This led to:
- **Docstring duplication** - LLM could see conflicting guidance
- **Validation gaps** - None values causing runtime errors
- **Inconsistent patterns** - Each tool implemented differently
- **"Failed to parse tool result"** errors when LLM returned empty results

---

## Architecture Principles

### 1. Single Source of Truth
Schema docstring is the ONLY documentation. Function has no docstring.

```python
class SGRPlanResult(BaseModel):
    """Full docstring here - LLM sees this via description="""
    
@tool("analyse_user_request", args_schema=SGRPlanResult, description=SGRPlanResult.__doc__)
async def analyse_user_request(...) -> dict:
    # No docstring - schema is the source
```

### 2. All Fields Optional (Except query)
Every field has a default value. LLM is guided by descriptions, not forced by schema.

```python
user_intent: str = Field(default="", description="...")
steps_completed: list[str] = Field(default_factory=list, description="...")
```

**Exception:** `query` in `retrieve_context` remains required (search without query is absurd).

### 3. Pydantic Validators for Edge Cases
Handle LLM inconsistencies (None → [], str → list):

```python
@field_validator("knowledge_base_search_queries", mode="before")
@classmethod
def _convert_queries(cls, v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v else []
    if isinstance(v, list):
        return [str(item) for item in v]
    return []
```

### 4. Defensive Handling in Function Body
Validators run before function, but LangChain may pass original values:

```python
# In function body
plan = {
    "knowledge_base_search_queries": knowledge_base_search_queries or [],
}
```

### 5. Structured Return Type
SGR and SRP return both JSON and markdown:

```python
return {
    "json": plan,           # For downstream processing
    "markdown": rendered,   # For LLM context visibility
}
```

`retrieve_context` returns JSON string only (no separate markdown format needed).

---

## Implementation Pattern

### Schema Class

```python
class ToolSchema(BaseModel):
    """Complete docstring with examples, guidance, edge cases.
    
    Reason step by step and fill the arguments:
    1. Step one
    2. Step two
    ...
    """
    
    field1: str = Field(default="", description="...")
    field2: list[str] = Field(default_factory=list, description="...")
    
    @field_validator("field2", mode="before")
    @classmethod
    def _convert_field2(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        if isinstance(v, list):
            return [str(item) for item in v]
        return []
```

### Tool Function

```python
@tool("tool_name", args_schema=ToolSchema, description=ToolSchema.__doc__)
async def tool_function(
    field1: str = "",
    field2: list[str] = None,
    runtime: ToolRuntime | None = None,
) -> dict | str:
    # Build result with defensive handling
    result = {
        "field1": field1,
        "field2": field2 or [],
    }
    
    # Store in context if needed
    if runtime and hasattr(runtime, "context"):
        runtime.context.result = result
    
    # Return structured data
    return {"json": result, "markdown": render(result)}
```

---

## Files Changed

| File | Changes |
|------|---------|
| `rag_engine/llm/schemas.py` | Merged docstrings into `SGRPlanResult` and `ResolutionPlanResult`; added validators; made fields optional |
| `rag_engine/tools/analyse_user_request.py` | Added `description=`; removed function docstring; added defaults; return `{json, markdown}` |
| `rag_engine/tools/generate_resolution_plan.py` | Same pattern as SGR |
| `rag_engine/tools/retrieve_context.py` | Moved 85-line docstring to schema; added `description=`; added validator for `exclude_kb_ids` |
| `rag_engine/tools/utils.py` | Added defensive empty check in `parse_tool_result_to_articles()` with logging |
| `rag_engine/api/app.py` | Replaced `with_structured_output` with `bind_tools + tool_choice`; execute tools; extract json/markdown |
| `rag_engine/tests/test_sgr_tool.py` | 17 tests for SGR schema validators, tool execution, template rendering |
| `rag_engine/tests/test_srp_tool.py` | 15 tests for SRP schema validators, tool execution, markdown rendering |

---

## API Changes

### Before (with_structured_output)

```python
structured_llm = llm.with_structured_output(SGRPlanResult, method="function_calling")
plan = await structured_llm.ainvoke(messages)
sgr_plan_dict = plan.model_dump()
# Manual markdown rendering elsewhere
```

### After (bind_tools + tool_choice)

```python
llm_with_tool = llm.bind_tools([analyse_user_request])
llm_forced = llm_with_tool.bind(tool_choice="analyse_user_request")
response = await llm_forced.ainvoke(messages)

if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = await analyse_user_request.ainvoke(tool_call["args"])
    sgr_plan_dict = result["json"]
    sgr_markdown = result["markdown"]
```

---

## Test Coverage

| Tool | Tests | Coverage |
|------|-------|----------|
| SGR | 17 | Schema validators, tool execution, all 3 templates |
| SRP | 15 | Schema validators, tool execution, markdown rendering |
| **Total** | **32** | All passing |

Test patterns:
- Empty args → defaults applied
- None values → converted to []
- String values → converted to single-item list
- Partial args → works correctly
- Tool returns both json and markdown
- Tool description comes from schema docstring

---

## Known Limitations

### 1. Model Reliability
Some models (e.g., `minimax-m2.1`) may skip tool calls with complex schemas on longer contexts:
```
WARNING: SGR LLM did not make tool call
```

**Mitigation:** System gracefully continues without plan. Not all models handle `tool_choice` reliably with 10+ field schemas.

### 2. Both Defaults Required
Schema defaults AND function defaults are both necessary:
- Schema defaults → Pydantic validation
- Function defaults → LangChain direct invocation

Tested and confirmed: omitting either causes failures.

---

## Defensive Error Handling

### Empty Tool Results
```python
def parse_tool_result_to_articles(tool_result: str) -> list[Article]:
    if not tool_result or not tool_result.strip():
        logger.debug("Empty tool result received, returning empty articles list")
        return []
```

Fixes: `Failed to parse tool result: Expecting value: line 1 column 1 (char 0)`

### None to Empty List
```python
# Validator
@field_validator("field", mode="before")
def _convert(cls, v):
    if v is None:
        return []
    ...

# Function body
field = field or []
```

---

## Key Discoveries

1. **`with_structured_output` doesn't execute tool** - creates "dummy tool" for schema only
2. **LangChain passes original args to function** - validators run but function gets raw values
3. **`description=` is required** - ensures schema docstring reaches LLM
4. **Timeouts unnecessary** - LLM has own timeout; user has stop button
5. **Docstring in schema is higher priority** - function docstring would override

---

## References

- LangChain `@tool` decorator: `args_schema`, `description`, `infer_schema`
- Pydantic `@field_validator(mode="before")` for pre-validation
- LangChain `bind_tools()` + `tool_choice` for forced execution
- Related: `docs/progress_reports/2026-01-21-sgr-planning-structured-agent.md`
