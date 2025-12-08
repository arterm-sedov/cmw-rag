# MCP Server Configuration

## Overview

The Comindware RAG Engine exposes an MCP (Model Context Protocol) server via Gradio, providing access to documentation search and Q&A capabilities.

## Server URL

**Base URL:** `http://skepseis1.slickjump.org:7860/gradio_api/mcp/`

## Recommended Configuration

**⚠️ IMPORTANT:** Always use the **filtered endpoint** to expose only working tools:

```json
{
  "mcpServers": {
    "gradio": {
      "url": "http://skepseis1.slickjump.org:7860/gradio_api/mcp/?tools=get_knowledge_base_articles,ask_comindware"
    }
  }
}
```

### Why Use the Filtered Endpoint?

**The filtered endpoint (`?tools=...`) is REQUIRED** because:

1. **Gradio's ChatInterface auto-exposes its function** - The `agent_chat_handler` generator function is automatically exposed by Gradio's ChatInterface, and it has a "pop index out of range" error in Gradio's MCP wrapper
2. **Filtered endpoint excludes broken function** - Using `?tools=...` allows you to specify exactly which tools to expose, excluding the broken generator function
3. **Both functions are exposed without filter** - The unfiltered endpoint exposes 3 tools:
   - ✅ `get_knowledge_base_articles` - Working
   - ✅ `ask_comindware` - Working Q&A function (business-oriented name)
   - ❌ `agent_chat_handler` - Broken generator function (auto-exposed by ChatInterface)

**Note:** Attempts to prevent ChatInterface from auto-exposing its function (via `api_name=False` or similar) are not effective. The filtered endpoint is the reliable solution.

**Unfiltered endpoint exposes:**
- ✅ `get_knowledge_base_articles` - Working
- ✅ `ask_comindware` - Working Q&A function (business-oriented name)
- ❌ `agent_chat_handler` - Broken (pop index error)

**Filtered endpoint exposes:**
- ✅ `get_knowledge_base_articles` - Working
- ✅ `ask_comindware` - Working

## Available Tools

### 1. `get_knowledge_base_articles`

**Purpose:** Search the Comindware Platform documentation knowledge base and retrieve relevant articles with full content.

**When to use:** Use this for programmatic access to documentation content. For conversational answers, use `ask_comindware` instead.

**Parameters:**
- `query` (string, required): Search query or question to find relevant documentation articles
  - Examples: "authentication", "API integration", "user management"
- `top_k` (integer, optional): Limit on number of articles to return. If not specified, returns default number (typically 10-20)

**Returns:**
JSON string containing an array of articles, each with:
- `kb_id`: Article identifier
- `title`: Article title
- `url`: Link to the article
- `content`: Full article content (markdown format)
- `metadata`: Additional metadata including rerank scores and source information

**Example Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_knowledge_base_articles",
    "arguments": {
      "query": "authentication methods",
      "top_k": 3
    }
  }
}
```

**Example Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"articles\":[{\"kb_id\":\"4656\",\"title\":\"Аутентификация, авторизация и сеансы пользователей\",\"url\":\"https://kb.comindware.ru/article.php?id=4656\",\"content\":\"...\",\"metadata\":{...}}]}"
    }]
  }
}
```

### 2. `ask_comindware`

**Purpose:** Ask questions about Comindware Platform documentation and get intelligent answers with citations.

**When to use:** Use this for conversational Q&A. The assistant automatically searches the knowledge base to find relevant articles and provides comprehensive answers based on official documentation. Use for technical questions, configuration help, API usage, troubleshooting, and general platform guidance.

**Note:** This tool has a business-oriented name (`ask_comindware`) to make it intuitive for MCP consumers. It's the recommended tool for getting answers about Comindware Platform.

**Parameters:**
- `message` (string, required): User's current message or question

**Returns:**
Complete response text with citations formatted in markdown, including:
- AI-generated answer based on documentation
- Citations to source articles
- Disclaimer about AI-generated content

**Example Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "ask_comindware",
    "arguments": {
      "message": "What is Comindware Platform?"
    }
  }
}
```

**Example Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [{
      "type": "text",
      "text": "## Сгенерированный ИИ контент\n\n**Comindware Platform** — это российская low-code-платформа...\n\n[Источники: article.php?id=1234, article.php?id=5678]"
    }]
  }
}
```

## Testing Results

### Filtered Endpoint (`?tools=get_knowledge_base_articles,ask_comindware`)

✅ **Tool Discovery:** Successfully lists only 2 tools
- `get_knowledge_base_articles`
- `ask_comindware`

✅ **get_knowledge_base_articles:** Working perfectly
- Successfully retrieves articles
- Returns structured JSON with all expected fields
- Properly filters by `top_k` parameter

✅ **ask_comindware:** Working perfectly
- Successfully generates intelligent answers
- Includes citations to source articles
- Returns formatted markdown response

✅ **Security:** Broken `agent_chat_handler` is not exposed

✅ **Business-oriented naming:** Tool is named `ask_comindware` for clarity
- Attempting to call it returns: "Tool 'agent_chat_handler' is not in the selected tools list"

### Unfiltered Endpoint (no query parameters)

⚠️ **Exposes 3 tools:**
- ✅ `get_knowledge_base_articles` - Working
- ✅ `ask_comindware` - Working Q&A function (business-oriented name)
- ❌ `agent_chat_handler` - Broken (returns "pop index out of range" error)

## Protocol Details

- **Transport:** Streamable HTTP (Server-Sent Events)
- **Content-Type:** `application/json`
- **Accept:** `application/json, text/event-stream`
- **Response Format:** SSE (Server-Sent Events) with JSON-RPC 2.0 messages

## Error Handling

If a tool is not in the filtered list, the server returns:
```json
{
  "jsonrpc": "2.0",
  "id": <id>,
  "result": {
    "content": [{
      "type": "text",
      "text": "Tool '<tool_name>' is not in the selected tools list"
    }],
    "isError": true
  }
}
```

## Notes

- The filtered endpoint is recommended for production use
- Both tools use the OpenRouter Qwen 30B model (via `qwen/qwen3-30b-a3b-instruct-2507`)
- Context window: 262,144 tokens
- Max output tokens: 32,768 tokens
- The server automatically searches the knowledge base before answering questions
