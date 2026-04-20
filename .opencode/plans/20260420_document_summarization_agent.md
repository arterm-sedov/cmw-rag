# Plan: Agentic Web Search for Document Summarization

## Problem

Current implementation uses **hardcoded keyword matching** to trigger web search:
- `summary_connector._fetch_search_context()` checks for hardcoded keywords (`"конкурент"`, `"сравни"`, `"цена"`, etc.)
- LLM doesn't decide when to search — it's decided **before** LLM call
- No ReAct loop, no tool exposed to LLM

This is **not agentic** — LLM just receives search results as context text.

## Goal

Convert document summarization endpoint to be a **proper LangChain agent** that:
1. Reuses `create_rag_agent` pattern from `rag_engine/llm/agent_factory.py`
2. Includes `web_search` tool alongside other tools
3. LLM **decides** when to call web search (based on reasoning, not keywords)
4. Proper ReAct loop: think → call tool → get results → continue → answer

## Current Agent Pattern

From `agent_factory.py`, the existing pattern is:
```python
agent = create_agent(
    model=model_with_tools,  # LLM with bind_tools
    tools=all_tools,         # List of LangChain tools
    system_prompt=get_system_prompt(...),
    context_schema=AgentContext,
    middleware=middleware_list,
)
```

Tools available: `retrieve_context`, `get_current_datetime`, math tools

## Implementation Plan

### Step 1: Convert `web_search.py` to LangChain `@tool`

Add `@tool` decorator to expose `web_search` function to LLM:

```python
from langchain.tools import ToolRuntime, tool

@tool("web_search", description="Search the web for current information, prices, weather, competitor data, statistics, etc.")
def web_search(query: str, runtime: ToolRuntime | None = None) -> str:
    """Execute web search using Tavily API.
    
    Use this when user asks for information not present in the document
    (prices, weather, competitor data, etc.).
    """
    tool_instance = get_web_search_tool()
    results = tool_instance.search(query)
    if not results:
        return "No results found or search unavailable"
    return tool_instance.to_json(results)
```

**File:** `rag_engine/tools/web_search.py`

### Step 2: Update `tools/__init__.py`

Export the tool (already done partially, verify):
```python
from rag_engine.tools.web_search import web_search  # @tool decorated
```

### Step 3: Create Summary Agent Factory

Add new function to `rag_engine/llm/agent_factory.py`:

```python
def create_summary_agent(
    system_prompt: str,
    tools: list | None = None,
    force_tool_choice: bool = False,
) -> any:
    """Create a document summarization agent with web search capability.
    
    Reuses create_rag_agent pattern but with custom prompt and tools.
    """
    from rag_engine.tools.web_search import web_search
    
    # Default tools for summary agent
    default_tools = [
        web_search,           # NEW: web search for external data
        get_current_datetime, # For date/time queries
        add, subtract, multiply, divide, power, square_root, modulus,  # Math
    ]
    
    all_tools = tools or default_tools
    
    # Use same LLM setup as create_rag_agent
    temp_llm_manager = LLMManager(
        provider=settings.default_llm_provider,
        model=settings.default_model,
        temperature=settings.llm_temperature,
    )
    base_model = temp_llm_manager._chat_model()
    
    # bind_tools with optional force
    tool_choice = "auto" if force_tool_choice else None
    model_with_tools = base_model.bind_tools(all_tools, tool_choice=tool_choice)
    
    # Create agent (no memory compression needed for single doc summary)
    agent = create_agent(
        model=model_with_tools,
        tools=all_tools,
        system_prompt=system_prompt,
    )
    
    return agent
```

**File:** `rag_engine/llm/agent_factory.py`

### Step 4: Refactor `summary_connector._summarize()`

Replace hardcoded logic with agent:

```python
def _summarize(self, text: str, user_prompt: str) -> str:
    from rag_engine.llm.agent_factory import create_summary_agent
    
    system_prompt = self._get_system_prompt()
    agent = create_summary_agent(system_prompt=system_prompt)
    
    # Build input with document
    document_context = f"""Document to summarize:
{text[:50000]}

User request: {user_prompt}

Provide a concise summary following the user's instructions.
If user asks for information not in the document (prices, weather, competitor data),
use the web_search tool to find current information."""
    
    # Run agent (handles tool calls automatically)
    result = agent.invoke({"input": document_context})
    return result["output"]
```

**Alternative:** If simpler, just use `bind_tools` without full agent:
```python
def _summarize(self, text: str, user_prompt: str) -> str:
    from rag_engine.tools.web_search import web_search
    
    llm = self._get_llm()
    llm_with_tools = llm.bind_tools([web_search])
    
    document_context = f"""Document:\n{text[:50000]}\n\nRequest: {user_prompt}"""
    
    response = llm_with_tools.invoke(document_context)
    
    # Handle tool calls in loop (if any)
    while hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "web_search":
                # Execute tool
                tool_result = web_search.invoke(tool_call["args"])
                # Append result and continue
                response = llm_with_tools.invoke([
                    ("user", document_context),
                    response,
                    ("tool", tool_result),
                ])
    
    return response.content
```

### Step 5: Update System Prompt

Add to `rag_engine/config/cmw_platform_secondary.yaml`:
```
## Web Search

When user asks for information not present in the document (prices, weather, 
competitor data, statistics, current events, etc.) — use the web_search tool 
to find current information. Always cite sources from search results.
```

### Step 6: Remove Hardcoded Logic

Remove from `summary_connector.py`:
- `_fetch_search_context()` method
- `_build_search_queries()` method

### Step 7: Tests

Add tests for:
- Tool decorator works correctly
- Agent can call web_search when needed
- Fallback when no API key

---

## Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `rag_engine/tools/web_search.py` | UPDATE | Add `@tool` decorator |
| `rag_engine/tools/__init__.py` | VERIFY | Export `web_search` |
| `rag_engine/llm/agent_factory.py` | ADD | `create_summary_agent()` function |
| `rag_engine/cmw_platform/summary_connector.py` | REFACTOR | Use agent instead of hardcoded logic |
| `rag_engine/config/cmw_platform_secondary.yaml` | UPDATE | Add web search instructions |
| `rag_engine/tests/test_tools_web_search.py` | UPDATE | Add tool decorator tests |

---

## Benefits

1. **Agentic** — LLM decides when to search (not hardcoded keywords)
2. **Reuses existing patterns** — Same `create_agent` pattern as main RAG agent
3. **Flexible** — Can handle any query type, not just predefined keywords
4. **DRY** — Tool is standalone, reusable across agents
5. **Extensible** — Easy to add more tools to summary agent

---

## Verification

```bash
# Run tests
pytest rag_engine/tests/test_tools_web_search.py rag_engine/tests/test_cmw_platform_summary_connector.py -v

# Manual: Process document with prompt "Сравни цены конкурентов"
# Should see web_search tool called in logs
```
