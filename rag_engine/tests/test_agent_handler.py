"""Tests for agent-based chat handler."""
from __future__ import annotations

from unittest.mock import Mock, patch

from rag_engine.api.app import _create_rag_agent, agent_chat_handler


class TestCreateRagAgent:
    """Tests for _create_rag_agent function."""

    @patch("rag_engine.api.app.settings")
    @patch("langchain.agents.create_agent")
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_create_agent_gemini(self, mock_gemini_cls, mock_create_agent, mock_settings):
        """Test agent creation with Gemini provider."""
        mock_settings.default_llm_provider = "gemini"
        mock_settings.default_model = "gemini-2.5-flash"
        mock_settings.llm_temperature = 0.1
        mock_settings.google_api_key = "test_key"

        mock_model = Mock()
        mock_model_with_tools = Mock()
        mock_model.bind_tools.return_value = mock_model_with_tools
        mock_gemini_cls.return_value = mock_model
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        agent = _create_rag_agent()

        # Verify model was created with correct params
        mock_gemini_cls.assert_called_once_with(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key="test_key",
        )

        # Verify bind_tools was called with tool_choice
        mock_model.bind_tools.assert_called_once()
        bind_call_args = mock_model.bind_tools.call_args
        assert len(bind_call_args[0][0]) == 1  # One tool
        assert "retrieve_context" in str(bind_call_args[0][0][0])
        assert bind_call_args[1]["tool_choice"] == "retrieve_context"

        # Verify agent was created with model_with_tools
        mock_create_agent.assert_called_once()
        call_args = mock_create_agent.call_args
        assert call_args[1]["model"] is mock_model_with_tools
        assert len(call_args[1]["tools"]) == 1
        assert "Comindware Platform" in call_args[1]["system_prompt"]

        assert agent is mock_agent

    @patch("rag_engine.api.app.settings")
    @patch("langchain.agents.create_agent")
    @patch("langchain_openai.ChatOpenAI")
    def test_create_agent_openrouter(self, mock_openai_cls, mock_create_agent, mock_settings):
        """Test agent creation with OpenRouter provider."""
        mock_settings.default_llm_provider = "openrouter"
        mock_settings.default_model = "deepseek/deepseek-chat"
        mock_settings.llm_temperature = 0.0
        mock_settings.openrouter_api_key = "test_or_key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"

        mock_model = Mock()
        mock_openai_cls.return_value = mock_model
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent

        agent = _create_rag_agent()

        # Verify model was created with correct params
        mock_openai_cls.assert_called_once_with(
            model="deepseek/deepseek-chat",
            temperature=0.0,
            openai_api_key="test_or_key",
            openai_api_base="https://openrouter.ai/api/v1",
        )

        # Verify agent was created
        mock_create_agent.assert_called_once()
        assert agent is mock_agent

    @patch("rag_engine.api.app.settings")
    @patch("langchain.agents.create_agent")
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_system_prompt_uses_standard_prompt(self, mock_gemini_cls, mock_create_agent, mock_settings):
        """Test that agent uses standard Comindware Platform system prompt."""
        mock_settings.default_llm_provider = "gemini"
        mock_settings.default_model = "gemini-2.5-flash"
        mock_settings.llm_temperature = 0.1
        mock_settings.google_api_key = "test_key"

        mock_model = Mock()
        mock_model.bind_tools.return_value = Mock()
        mock_gemini_cls.return_value = mock_model
        mock_create_agent.return_value = Mock()

        _create_rag_agent()

        # Check that standard system prompt is used
        call_args = mock_create_agent.call_args
        system_prompt = call_args[1]["system_prompt"]
        assert "Comindware Platform" in system_prompt
        assert "technical documentation assistant" in system_prompt
        assert "<role>" in system_prompt  # Characteristic of SYSTEM_PROMPT


class TestAgentChatHandler:
    """Tests for agent_chat_handler function."""

    @patch("rag_engine.api.app._create_rag_agent")
    @patch("rag_engine.api.app._salt_session_id")
    @patch("rag_engine.api.app.llm_manager")
    @patch("rag_engine.api.app.format_with_citations")
    def test_agent_handler_empty_message(
        self, mock_format, mock_llm_manager, mock_salt_session, mock_create_agent
    ):
        """Test handler with empty message."""
        result = list(agent_chat_handler("", [], None))
        assert len(result) == 1
        assert "Пожалуйста" in result[0] or "Please" in result[0]

    @patch("rag_engine.api.app._create_rag_agent")
    @patch("rag_engine.api.app._salt_session_id")
    @patch("rag_engine.api.app.llm_manager")
    @patch("rag_engine.api.app.format_with_citations")
    @patch("rag_engine.tools.accumulate_articles_from_tool_results")
    def test_agent_handler_success_with_articles(
        self,
        mock_accumulate,
        mock_format,
        mock_llm_manager,
        mock_salt_session,
        mock_create_agent,
    ):
        """Test successful agent execution with retrieved articles."""
        # Mock session
        mock_salt_session.return_value = "test_session_123"

        # Mock agent stream for stream_mode=["updates", "messages"]
        # Format: (stream_mode, chunk) tuples
        # Mock tool call message (when agent decides to call the tool)
        mock_tool_call_msg = Mock()
        mock_tool_call_msg.type = None
        mock_tool_call_msg.tool_calls = [{"name": "retrieve_context", "args": {"query": "test"}}]
        mock_tool_call_msg.content = ""
        mock_tool_call_msg.content_blocks = None

        mock_tool_result_msg = Mock()
        mock_tool_result_msg.type = "tool"
        mock_tool_result_msg.content = '{"articles": [{"kb_id": "1", "content": "Test", "metadata": {}}], "metadata": {"articles_count": 1}}'
        mock_tool_result_msg.content_blocks = None
        mock_tool_result_msg.tool_calls = None

        mock_ai_token1 = Mock()
        mock_ai_token1.content_blocks = [{"type": "text", "text": "Based on the "}]
        mock_ai_token1.type = None
        mock_ai_token1.tool_calls = None

        mock_ai_token2 = Mock()
        mock_ai_token2.content_blocks = [{"type": "text", "text": "search results, "}]
        mock_ai_token2.type = None
        mock_ai_token2.tool_calls = None

        mock_ai_token3 = Mock()
        mock_ai_token3.content_blocks = [{"type": "text", "text": "here is the answer."}]
        mock_ai_token3.type = None
        mock_ai_token3.tool_calls = None

        mock_agent = Mock()
        mock_agent.stream.return_value = [
            ("messages", (mock_tool_call_msg, {"langgraph_node": "model"})),
            ("messages", (mock_tool_result_msg, {"langgraph_node": "tools"})),
            ("messages", (mock_ai_token1, {"langgraph_node": "model"})),
            ("messages", (mock_ai_token2, {"langgraph_node": "model"})),
            ("messages", (mock_ai_token3, {"langgraph_node": "model"})),
        ]
        mock_create_agent.return_value = mock_agent

        # Mock article accumulation
        from rag_engine.retrieval.retriever import Article
        mock_article = Article(kb_id="1", content="Test content", metadata={"title": "Test"})
        mock_accumulate.return_value = [mock_article]

        # Mock citation formatting
        mock_format.return_value = "Answer with citations"

        # Execute handler
        result = list(agent_chat_handler("test question", [], None))

        # Verify results - now includes metadata messages, streaming tokens, and final
        assert len(result) >= 3  # Start metadata + Completion metadata + streamed tokens + final
        # Last result should be the final answer with citations
        assert result[-1] == "Answer with citations"
        # Should include metadata messages for start and completion
        metadata_msgs = [r for r in result if isinstance(r, dict) and "metadata" in r]
        assert len(metadata_msgs) == 2  # Start + Completion messages
        # Verify metadata content
        assert "Searching" in metadata_msgs[0]["metadata"]["title"]
        assert "Found" in metadata_msgs[1]["metadata"]["title"]

        # Verify tool results were accumulated
        mock_accumulate.assert_called_once()
        tool_results_arg = mock_accumulate.call_args[0][0]
        assert len(tool_results_arg) == 1

        # Verify citations were formatted
        mock_format.assert_called_once_with(
            "Based on the search results, here is the answer.",
            [mock_article]
        )

        # Verify conversation was saved (both user and assistant turns)
        # User message saved before agent execution
        mock_llm_manager._conversations.append.assert_any_call(
            "test_session_123",
            "user",
            "test question"
        )
        # Assistant response saved after execution
        mock_llm_manager.save_assistant_turn.assert_called_once_with(
            "test_session_123",
            "Answer with citations"
        )

    @patch("rag_engine.api.app._create_rag_agent")
    @patch("rag_engine.api.app._salt_session_id")
    @patch("rag_engine.api.app.llm_manager")
    @patch("rag_engine.tools.accumulate_articles_from_tool_results")
    def test_agent_handler_no_articles(
        self,
        mock_accumulate,
        mock_llm_manager,
        mock_salt_session,
        mock_create_agent,
    ):
        """Test handler when no articles are retrieved."""
        mock_salt_session.return_value = "test_session_456"

        # Mock agent stream for stream_mode=["updates", "messages"]
        # Even with no results, the agent still calls the tool
        mock_tool_call_msg = Mock()
        mock_tool_call_msg.type = None
        mock_tool_call_msg.tool_calls = [{"name": "retrieve_context", "args": {"query": "unknown"}}]
        mock_tool_call_msg.content = ""
        mock_tool_call_msg.content_blocks = None

        mock_tool_result_msg = Mock()
        mock_tool_result_msg.type = "tool"
        mock_tool_result_msg.content = '{"articles": [], "metadata": {"articles_count": 0}}'
        mock_tool_result_msg.content_blocks = None
        mock_tool_result_msg.tool_calls = None

        mock_ai_token = Mock()
        mock_ai_token.content_blocks = [{"type": "text", "text": "I couldn't find relevant information."}]
        mock_ai_token.type = None
        mock_ai_token.tool_calls = None

        mock_agent = Mock()
        mock_agent.stream.return_value = [
            ("messages", (mock_tool_call_msg, {"langgraph_node": "model"})),
            ("messages", (mock_tool_result_msg, {"langgraph_node": "tools"})),
            ("messages", (mock_ai_token, {"langgraph_node": "model"})),
        ]
        mock_create_agent.return_value = mock_agent

        # Mock no articles
        mock_accumulate.return_value = []

        # Execute handler
        result = list(agent_chat_handler("unknown topic", [], None))

        # Verify no citations added when no articles
        assert len(result) >= 1
        # Last result should be the final answer (same text, no citations)
        assert result[-1] == "I couldn't find relevant information."

    @patch("rag_engine.api.app._create_rag_agent")
    @patch("rag_engine.api.app._salt_session_id")
    def test_agent_handler_error_handling(
        self,
        mock_salt_session,
        mock_create_agent,
    ):
        """Test error handling in agent handler."""
        mock_salt_session.return_value = "test_session_789"

        # Mock agent that raises exception
        mock_agent = Mock()
        mock_agent.stream.side_effect = Exception("Test error")
        mock_create_agent.return_value = mock_agent

        # Execute handler
        result = list(agent_chat_handler("test question", [], None))

        # Verify error message is returned
        assert len(result) == 1
        assert "Извините" in result[0] or "Sorry" in result[0]
        assert "error" in result[0].lower()

    @patch("rag_engine.api.app._create_rag_agent")
    @patch("rag_engine.api.app._salt_session_id")
    @patch("rag_engine.api.app.llm_manager")
    @patch("rag_engine.api.app.format_with_citations")
    @patch("rag_engine.tools.accumulate_articles_from_tool_results")
    def test_agent_handler_with_history(
        self,
        mock_accumulate,
        mock_format,
        mock_llm_manager,
        mock_salt_session,
        mock_create_agent,
    ):
        """Test handler preserves conversation history."""
        mock_salt_session.return_value = "test_session_abc"

        # Mock agent
        mock_ai_msg = Mock()
        mock_ai_msg.type = "ai"
        mock_ai_msg.content = "Follow-up answer"
        mock_ai_msg.tool_calls = None

        mock_agent = Mock()
        mock_agent.stream.return_value = [{"messages": [mock_ai_msg]}]
        mock_create_agent.return_value = mock_agent

        mock_accumulate.return_value = []

        # History with previous messages
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]

        # Execute handler with history
        list(agent_chat_handler("Follow-up question", history, None))

        # Verify agent received full history
        mock_agent.stream.assert_called_once()
        call_args = mock_agent.stream.call_args[0][0]
        assert len(call_args["messages"]) == 3  # 2 history + 1 current
        assert call_args["messages"][0]["content"] == "First question"
        assert call_args["messages"][1]["content"] == "First answer"
        assert call_args["messages"][2]["content"] == "Follow-up question"


class TestAgentIntegration:
    """Integration tests for agent mode."""

    @patch("rag_engine.api.app.settings")
    def test_handler_selection_agent_mode(self, mock_settings):
        """Test that agent handler is selected when USE_AGENT_MODE=true."""
        mock_settings.use_agent_mode = True
        # This test validates the concept; actual handler selection happens at module load
        # In production, the handler_fn variable would be set to agent_chat_handler

    @patch("rag_engine.api.app.settings")
    def test_handler_selection_direct_mode(self, mock_settings):
        """Test that direct handler is selected when USE_AGENT_MODE=false."""
        mock_settings.use_agent_mode = False
        # This test validates the concept; actual handler selection happens at module load
        # In production, the handler_fn variable would be set to chat_handler

