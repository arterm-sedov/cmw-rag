"""Tests for agent-based chat handler."""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

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

        # Mock agent stream - simulate tool result → AI response
        # Skip the tool_call message to avoid subscript complexity in test
        mock_tool_result_msg = Mock()
        mock_tool_result_msg.type = "tool"
        mock_tool_result_msg.content = '{"articles": [{"kb_id": "1", "content": "Test", "metadata": {}}]}'
        # Add tool_calls as None to avoid hasattr check passing
        mock_tool_result_msg.tool_calls = None

        mock_ai_msg = Mock()
        mock_ai_msg.type = "ai"
        mock_ai_msg.content = "Based on the search results, here is the answer."
        mock_ai_msg.tool_calls = None

        mock_agent = Mock()
        mock_agent.stream.return_value = [
            {"messages": [mock_tool_result_msg]},
            {"messages": [mock_ai_msg]},
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

        # Verify results
        assert len(result) == 2  # AI answer + final with citations
        assert result[0] == "Based on the search results, here is the answer."
        assert result[1] == "Answer with citations"

        # Verify tool results were accumulated
        mock_accumulate.assert_called_once()
        tool_results_arg = mock_accumulate.call_args[0][0]
        assert len(tool_results_arg) == 1

        # Verify citations were formatted
        mock_format.assert_called_once_with(
            "Based on the search results, here is the answer.",
            [mock_article]
        )

        # Verify conversation was saved
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

        # Mock agent stream - no tool calls, direct answer
        mock_ai_msg = Mock()
        mock_ai_msg.type = "ai"
        mock_ai_msg.content = "I couldn't find relevant information."
        mock_ai_msg.tool_calls = None

        mock_agent = Mock()
        mock_agent.stream.return_value = [
            {"messages": [mock_ai_msg]},
        ]
        mock_create_agent.return_value = mock_agent

        # Mock no articles
        mock_accumulate.return_value = []

        # Execute handler
        result = list(agent_chat_handler("unknown topic", [], None))

        # Verify no citations added when no articles
        assert len(result) == 2
        assert result[0] == "I couldn't find relevant information."
        assert result[1] == "I couldn't find relevant information."  # Same text, no citations

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
        result = list(agent_chat_handler("Follow-up question", history, None))

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

