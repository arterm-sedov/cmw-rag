# Plan: Robust Web Search Tool

## Goal

Implement a reusable web search tool that can be used by any endpoint or agent, replacing the current inline Tavily search in `summary_connector._fetch_search_context()`.

## References

### Current Implementation

**File:** `rag_engine/cmw_platform/summary_connector.py` (lines 61-104)

```python
def _fetch_search_context(self, user_prompt: str, document_text: str) -> str:
    """Fetch web search results if prompt asks for external data."""
    import os

    if not any(kw in user_prompt.lower() for kw in ["конкурент", "сравни", "цена", "weather", "погода", "москва"]):
        return ""

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return ""

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return ""

    try:
        search = TavilySearch(max_results=3)
        # ... search logic
    except Exception:
        return ""
```

**Issues:**
- Hardcoded keywords for trigger detection (Russian-specific)
- Inline implementation — not reusable
- No streaming, no tool-like interface
- No error handling for API failures
- Keywords hardcoded per language

### cmw-platform-agent Implementation

**File:** `~/cmw-platform-agent/tools/tools.py` (lines 689-760)

Features:
- `@tool` decorator for LangChain compatibility
- Returns JSON with structured response
- Handles Tavily errors gracefully
- Configurable max results
- Tool interface pattern

```python
@tool
def web_search(input: str) -> str:
    """
    Search the web using Tavily for a query and return up to 3 results.
    ...
    """
    if not TAVILY_AVAILABLE:
        return json.dumps({"error": "Tavily not available..."})
    if not os.environ.get("TAVILY_API_KEY"):
        return json.dumps({"error": "TAVILY_API_KEY not found..."})

    search_result = TavilySearch(max_results=SEARCH_LIMIT).invoke(input)
    # Handle different response types
```

---

## Architecture

```
rag_engine/tools/
├── __init__.py
├── web_search.py          # NEW: Reusable web search tool
├── tavily_client.py        # NEW: Tavily client wrapper (optional abstraction)
└── ...

summary_connector.py         # UPDATED: Use tools.web_search instead of inline
```

---

## Implementation Steps

### Step 1: Create `rag_engine/tools/web_search.py`

```python
"""Web search tool using Tavily."""

import json
import os
from typing import Any

try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


class WebSearchResult:
    """Structured web search result."""

    def __init__(self, title: str, url: str, content: str):
        self.title = title
        self.url = url
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "content": self.content}


class WebSearchTool:
    """Reusable web search tool with configurable providers."""

    def __init__(self, provider: str = "tavily", max_results: int = 3):
        self.provider = provider
        self.max_results = max_results

    def search(self, query: str) -> list[WebSearchResult]:
        """Execute web search and return results."""
        if self.provider == "tavily":
            return self._search_tavily(query)
        return []

    def _search_tavily(self, query: str) -> list[WebSearchResult]:
        """Search using Tavily API."""
        if not TAVILY_AVAILABLE:
            return []
        if not os.getenv("TAVILY_API_KEY"):
            return []

        try:
            search = TavilySearch(max_results=self.max_results)
            result = search.invoke(query)

            if isinstance(result, dict):
                results = result.get("results", [])
            elif isinstance(result, list):
                results = result
            else:
                return []

            return [
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", "")[:500]
                )
                for item in results[:self.max_results]
                if isinstance(item, dict)
            ]
        except Exception:
            return []

    def search_multiple(self, queries: list[str]) -> list[WebSearchResult]:
        """Execute multiple searches and merge results."""
        all_results = []
        for query in queries:
            all_results.extend(self.search(query))
        return all_results

    def to_json(self, results: list[WebSearchResult]) -> str:
        """Serialize results to JSON."""
        return json.dumps({
            "type": "tool_response",
            "tool_name": "web_search",
            "results": [r.to_dict() for r in results]
        })


# Singleton instance for reuse
_web_search_tool: WebSearchTool | None = None


def get_web_search_tool() -> WebSearchTool:
    """Get or create singleton web search tool."""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool(max_results=3)
    return _web_search_tool


def web_search(query: str) -> str:
    """LangChain-compatible web search tool function.

    Args:
        query: Search query string

    Returns:
        JSON string with search results or error
    """
    tool = get_web_search_tool()
    results = tool.search(query)

    if not results:
        return json.dumps({
            "type": "tool_response",
            "tool_name": "web_search",
            "error": "No results found or search unavailable"
        })

    return tool.to_json(results)
```

### Step 2: Update `rag_engine/tools/__init__.py`

Add exports:

```python
from rag_engine.tools.web_search import web_search, WebSearchTool, get_web_search_tool
```

### Step 3: Refactor `summary_connector._fetch_search_context()`

Replace inline Tavily logic with tool call:

```python
def _fetch_search_context(self, user_prompt: str, document_text: str) -> str:
    """Fetch web search results if prompt asks for external data."""
    from rag_engine.tools.web_search import get_web_search_tool

    trigger_keywords = {
        "конкурент", "сравни", "цена", "weather",
        "погода", "москва", "competitor", "compare", "price"
    }

    if not any(kw in user_prompt.lower() for kw in trigger_keywords):
        return ""

    # Determine search queries from prompt
    queries = self._build_search_queries(user_prompt, document_text)
    if not queries:
        return ""

    tool = get_web_search_tool()
    results = tool.search_multiple(queries)

    if not results:
        return ""

    return "\n".join([f"[{r.title}]: {r.content}" for r in results]) + "\n\n"

def _build_search_queries(self, prompt: str, text: str) -> list[str]:
    """Build search queries from prompt and document."""
    queries = []

    competitor_keywords = {"конкурент", "сравни", "цена", "competitor", "compare", "price"}
    weather_keywords = {"погода", "weather", "москва", "moscow"}

    if any(kw in prompt.lower() for kw in competitor_keywords):
        # Extract product hint from document
        product = " ".join(text.split("\n")[:3])[:100]
        queries.append(f"competitor price {product}")

    if any(kw in prompt.lower() for kw in weather_keywords):
        queries.append("Moscow Russia current temperature weather")

    return queries
```

### Step 4: Add Configuration Options

Support providers via config:

```yaml
# rag_engine/config/cmw_platform_secondary.yaml
pipeline:
  search:
    enabled: true
    provider: tavily  # or future: exa, SerpAPI, etc.
    max_results: 3
    trigger_keywords:
      - конкурент
      - сравни
      - цена
      - погода
      - москва
```

---

## Verification

```bash
# Test web search tool directly
.venv/bin/python -c "
from rag_engine.tools.web_search import web_search
result = web_search('Moscow weather today')
print(result)
"

# Test with summary connector
.venv/bin/python -c "
from rag_engine.cmw_platform.summary_connector import DocumentSummaryConnector
c = DocumentSummaryConnector()
ctx = c._fetch_search_context('Сравни цены конкурентов', 'Test product')
print(f'Search context: {len(ctx)} chars')
"

# Run tests
.venv/bin/python -m pytest rag_engine/tests/test_tools_web_search.py -v
```

---

## Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `rag_engine/tools/web_search.py` | CREATE | New web search tool |
| `rag_engine/tools/__init__.py` | UPDATE | Export new tool |
| `rag_engine/cmw_platform/summary_connector.py` | UPDATE | Use tool instead of inline Tavily |
| `rag_engine/tests/test_tools_web_search.py` | CREATE | Tests for web search tool |

---

## Benefits

1. **Reusable** — any agent/endpoint can import and use `web_search`
2. **Testable** — unit testable independently
3. **Configurable** — provider, max_results via config or constructor
4. **LangChain compatible** — works with `@tool` decorator pattern
5. **DRY** — removes duplicate Tavily code from summary_connector
6. **Extensible** — easy to add new providers (Exa, SerpAPI, etc.)
