"""Gradio UI with ChatInterface and REST API endpoint."""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Generator
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging

import gradio as gr

from rag_engine.config.settings import get_allowed_fallback_models, settings
from rag_engine.llm.llm_manager import LLMManager
from rag_engine.retrieval.embedder import FRIDAEmbedder
from rag_engine.retrieval.retriever import RAGRetriever
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.utils.formatters import format_with_citations
from rag_engine.utils.logging_manager import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


# Initialize singletons (order matters: llm_manager before retriever)
embedder = FRIDAEmbedder(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
)
vector_store = ChromaStore(
    persist_dir=settings.chromadb_persist_dir,
    collection_name=settings.chromadb_collection,
)
llm_manager = LLMManager(
    provider=settings.default_llm_provider,
    model=settings.default_model,
    temperature=settings.llm_temperature,
)
retriever = RAGRetriever(
    embedder=embedder,
    vector_store=vector_store,
    llm_manager=llm_manager,  # NEW: Pass for dynamic context budgeting
    top_k_retrieve=settings.top_k_retrieve,
    top_k_rerank=settings.top_k_rerank,
    rerank_enabled=settings.rerank_enabled,
)


def _create_rag_agent():
    """Create LangChain agent with forced retrieval tool execution.
    
    Uses tool_choice parameter to enforce tool calling and the standard
    Comindware Platform system prompt for consistent behavior.
    
    Returns:
        Configured LangChain agent with retrieve_context tool
    """
    from langchain.agents import create_agent

    from rag_engine.llm.prompts import SYSTEM_PROMPT
    from rag_engine.tools import retrieve_context

    # Select model based on provider
    if settings.default_llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        base_model = ChatGoogleGenerativeAI(
            model=settings.default_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )
    else:  # openrouter or other
        from langchain_openai import ChatOpenAI
        base_model = ChatOpenAI(
            model=settings.default_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
        )

    # CRITICAL: Use tool_choice to force retrieval tool execution
    # This ensures the agent always searches the knowledge base
    model_with_tools = base_model.bind_tools(
        [retrieve_context],
        tool_choice="retrieve_context"
    )

    agent = create_agent(
        model=model_with_tools,
        tools=[retrieve_context],
        system_prompt=SYSTEM_PROMPT,
    )

    logger.info("RAG agent created with forced tool execution via tool_choice parameter")
    return agent


def agent_chat_handler(
    message: str,
    history: list[dict],
    request: gr.Request | None = None,
) -> Generator[str, None, None]:
    """Agent-based chat handler using LangChain agent with tool calling.
    
    This handler uses a LangChain agent that decides when to call the
    retrieve_context tool. The agent is prompted to always search the
    knowledge base before answering.
    
    Args:
        message: User's current message
        history: Chat history from Gradio
        request: Gradio request object for session management
        
    Yields:
        Streaming response with citations
    """
    if not message or not message.strip():
        yield "Пожалуйста, введите вопрос / Please enter a question."
        return

    # Session management (reuse existing pattern)
    base_session_id = getattr(request, "session_hash", None) if request is not None else None
    session_id = _salt_session_id(base_session_id, history, message)

    # Build messages from history for agent
    messages = []
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    # Create agent and stream execution
    agent = _create_rag_agent()
    tool_results = []
    answer = ""

    try:
        for chunk in agent.stream(
            {"messages": messages},
            stream_mode="values"
        ):
            latest_msg = chunk["messages"][-1]

            # Track tool calls - emit metadata message
            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                tool_name = latest_msg.tool_calls[0].get("name", "retrieve_context")
                logger.debug("Agent calling tool: %s", tool_name)

                # Emit status message to Gradio with metadata
                status_msg = {
                    "role": "assistant",
                    "content": "",
                    "metadata": {"title": "🔍 Searching information in the knowledge base"}
                }
                # In Gradio, this will appear as a collapsible message
                messages.append(status_msg)
                continue

            # Track tool results - emit completion metadata
            if hasattr(latest_msg, "type") and latest_msg.type == "tool":
                tool_results.append(latest_msg.content)
                logger.debug("Tool result received, %d total results", len(tool_results))

                # Parse result to get article count
                try:
                    import json
                    result = json.loads(latest_msg.content)
                    articles_count = result.get("metadata", {}).get("articles_count", 0)

                    # Emit completion message with metadata
                    completion_msg = {
                        "role": "assistant",
                        "content": "",
                        "metadata": {"title": f"✅ Found {articles_count} article{'s' if articles_count != 1 else ''}"}
                    }
                    messages.append(completion_msg)
                except (json.JSONDecodeError, KeyError):
                    # If parsing fails, emit generic completion message
                    completion_msg = {
                        "role": "assistant",
                        "content": "",
                        "metadata": {"title": "✅ Search completed"}
                    }
                    messages.append(completion_msg)

                continue

            # Stream AI response
            if hasattr(latest_msg, "type") and latest_msg.type == "ai" and latest_msg.content:
                answer = latest_msg.content
                yield answer

        # Accumulate articles from tool results and add citations
        from rag_engine.tools import accumulate_articles_from_tool_results
        articles = accumulate_articles_from_tool_results(tool_results)

        # Handle no results case
        if not articles:
            final_text = answer
            logger.info("Agent completed with no retrieved articles")
        else:
            final_text = format_with_citations(answer, articles)
            logger.info("Agent completed with %d articles", len(articles))

        # Save conversation turn (reuse existing pattern)
        if session_id:
            llm_manager.save_assistant_turn(session_id, final_text)

        yield final_text

    except Exception as e:
        logger.error("Error in agent_chat_handler: %s", e, exc_info=True)
        error_msg = f"Извините, произошла ошибка / Sorry, an error occurred: {str(e)}"
        yield error_msg


def _salt_session_id(base_session_id: str | None, history: list[dict], current_message: str = "") -> str | None:
    """Salt session_id with chat history to isolate memory per chat.

    When Gradio's save_history=True is enabled, different chats share the same
    session_hash. Salting ensures each chat (new or loaded from history) gets
    its own isolated session memory.

    Strategy: Generate deterministically from the first user message.
    - For new chats: uses current_message (first message being sent)
    - For loaded chats: uses first user message from history
    - For continuing same chat: same first message → same salt → same session_id → memory preserved

    This approach is robust because:
    1. The first message is stable across Gradio's save/load cycle
    2. Each distinct chat has a different first message → different session_id
    3. Same chat always generates the same session_id → memory continuity

    Args:
        base_session_id: Base session hash from Gradio request
        history: Current chat history from Gradio (includes loaded chats)
        current_message: Current message being sent (used for new chats)

    Returns:
        Salted session_id or None if base_session_id is None
    """
    if not base_session_id:
        return None

    # Extract first user message as salt
    # For loaded chats: from history; for new chats: from current_message
    salt = ""
    if history:
        # Loaded or continuing chat: use first user message from history
        for msg in history:
            role = msg.get("role", "")
            if role != "user":
                continue
            content = msg.get("content", "")
            # Handle both string and dict content (multimodal)
            if isinstance(content, dict):
                # Extract text from dict if available, otherwise use path
                text = content.get("text", "") or str(content.get("path", ""))
            else:
                text = str(content)
            if text:
                salt = text[:100]  # First 100 chars as salt
                break
    elif current_message:
        # New chat: use current message as salt
        salt = str(current_message)[:100]

    # Create salted session_id deterministically
    salted = f"{base_session_id}:{salt}"
    return hashlib.sha256(salted.encode()).hexdigest()[:32]


def chat_handler(message: str, history: list[dict], request: gr.Request | None = None) -> Generator[str, None, None]:
    if not message or not message.strip():
        yield "Пожалуйста, введите вопрос / Please enter a question."
        return

    docs = retriever.retrieve(message)

    # If no documents found, inject a message into the context so LLM knows
    # explicitly that no relevant materials were found
    has_no_results_doc = False
    if not docs:
        # Create a fake document with the "no results" message to inject into context
        from rag_engine.retrieval.retriever import Article
        no_results_msg = "К сожалению, не найдено релевантных материалов / No relevant results found."
        no_results_doc = Article(
            kb_id="",
            content=no_results_msg,
            metadata={"title": "No Results", "kbId": "", "_is_no_results": True}
        )
        docs = [no_results_doc]
        has_no_results_doc = True

    base_session_id = getattr(request, "session_hash", None) if request is not None else None
    session_id = _salt_session_id(base_session_id, history, message)

    answer = ""
    for token in llm_manager.stream_response(
        message,
        docs,
        enable_fallback=settings.llm_fallback_enabled,
        allowed_fallback_models=get_allowed_fallback_models(),
        session_id=session_id,
    ):
        answer += token
        yield answer

    # Save assistant turn with footer appended, once at the end
    # format_with_citations handles empty docs gracefully (no citations)
    # If we injected the "no results" message, don't add citations
    if has_no_results_doc:
        final_text = answer  # Don't add citations for "no results" message
    else:
        final_text = format_with_citations(answer, docs)
    llm_manager.save_assistant_turn(session_id, final_text)
    yield final_text


def query_rag(question: str, provider: str = "gemini", top_k: int = 5) -> str:
    if not question or not question.strip():
        return "Error: Empty question"
    docs = retriever.retrieve(question, top_k=top_k)
    # If no documents found, inject a message into the context
    has_no_results_doc = False
    if not docs:
        from rag_engine.retrieval.retriever import Article
        no_results_msg = "К сожалению, не найдено релевантных материалов / No relevant results found."
        no_results_doc = Article(
            kb_id="",
            content=no_results_msg,
            metadata={"title": "No Results", "kbId": "", "_is_no_results": True}
        )
        docs = [no_results_doc]
        has_no_results_doc = True

    answer = llm_manager.generate(question, docs, provider=provider)
    # If we injected the "no results" message, don't add citations
    if has_no_results_doc:
        return answer  # Don't add citations for "no results" message
    return format_with_citations(answer, docs)


# Configure chatbot height and UI elements based on embedded widget setting
if settings.gradio_embedded_widget:
    # For embedded widget
    chatbot_height = "400px"
    chatbot_max_height = "65vh"
    chat_title = None
    chat_description = None
else:
    # For standalone app
    chatbot_height = "85vh"
    chatbot_max_height = "80vh"
    chat_title = "Ассистент базы знаний Comindware Platform"
    chat_description = None  # "RAG-агент базы знаний Comindware Platform"

chatbot_config = gr.Chatbot(
    type="messages",
    show_copy_button=True,
    min_height="30vh",
    height=chatbot_height,
    max_height=chatbot_max_height,
    resizable=True,
    elem_classes=["gradio-chatbot"],
)

# Select handler based on agent mode setting
handler_fn = agent_chat_handler if settings.use_agent_mode else chat_handler
handler_mode = "agent-based (LangChain)" if settings.use_agent_mode else "direct retrieval"
logger.info("Using %s handler for chat interface", handler_mode)

demo = gr.ChatInterface(
    fn=handler_fn,
    title=chat_title,
    description=chat_description,
    type="messages",
    save_history=True,
    #fill_width=True,
    chatbot=chatbot_config,
)
# Explicitly set a plain attribute for tests and downstream code to read
demo.title = "Comindware Platform Documentation Assistant"

try:
    gr.api(fn=query_rag, api_name="query_rag")
except Exception:  # noqa: BLE001
    # Older/newer Gradio builds without gr.api support
    pass


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting Gradio server at %s:%s (share=%s)",
        settings.gradio_server_name,
        settings.gradio_server_port,
        settings.gradio_share,
    )

    if settings.gradio_share:
        logger.info(
            "Share link enabled. If share link creation fails, the app will still run locally."
        )

    demo.queue().launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        share=settings.gradio_share,
    )


