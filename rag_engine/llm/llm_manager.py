"""LLM manager with dynamic token limits (reuses cmw-platform-agent mechanics)."""
from __future__ import annotations

import logging
from typing import Dict, Generator, Iterable, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from rag_engine.llm.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# Model configurations with dynamic token limits (from cmw-platform-agent)
MODEL_CONFIGS: Dict[str, Dict] = {
    # Gemini models (matching cmw-platform-agent)
    "gemini-2.5-flash": {
        "token_limit": 1048576,  # 1M context
        "max_tokens": 65536,
        "temperature": 0,
    },
    "gemini-2.5-pro": {
        "token_limit": 1048576,  # 1M context
        "max_tokens": 65536,
        "temperature": 0,
    },
    # OpenRouter models (matching cmw-platform-agent)
    # DeepSeek Models
    "deepseek/deepseek-v3.1-terminus": {
        "token_limit": 163840,
        "max_tokens": 65536,
        "temperature": 0,
    },
    "deepseek/deepseek-chat-v3.1:free": {
        "token_limit": 163840,
        "max_tokens": 4096,
        "temperature": 0,
    },
    "deepseek/deepseek-r1-0528": {
        "token_limit": 163840,
        "max_tokens": 4096,
        "temperature": 0,
    },
    # Grok (xAI) Models
    "x-ai/grok-4-fast:free": {
        "token_limit": 2000000,
        "max_tokens": 8192,
        "temperature": 0,
    },
    "x-ai/grok-code-fast-1": {
        "token_limit": 256000,
        "max_tokens": 10000,
        "temperature": 0,
    },
    "x-ai/grok-4-fast": {
        "token_limit": 2000000,
        "max_tokens": 30000,
        "temperature": 0,
    },
    # Qwen Models
    "qwen/qwen3-coder:free": {
        "token_limit": 262144,
        "max_tokens": 4096,
        "temperature": 0,
    },
    "qwen/qwen3-coder-flash": {
        "token_limit": 128000,
        "max_tokens": 4096,
        "temperature": 0,
    },
    "qwen/qwen3-max": {
        "token_limit": 256000,
        "max_tokens": 32768,
        "temperature": 0,
    },
    "qwen/qwen3-coder-plus": {
        "token_limit": 128000,
        "max_tokens": 65536,
        "temperature": 0,
    },
    # Other Models
    "anthropic/claude-sonnet-4.5": {
        "token_limit": 1000000,
        "max_tokens": 64000,
        "temperature": 0,
    },
    "openai/gpt-oss-120b": {
        "token_limit": 131072,
        "max_tokens": 32768,
        "temperature": 0,
    },
    "openai/gpt-5-mini": {
        "token_limit": 400000,
        "max_tokens": 32768,
        "temperature": 0,
    },
    "nvidia/nemotron-nano-9b-v2:free": {
        "token_limit": 128000,
        "max_tokens": 4096,
        "temperature": 0,
    },
    "mistralai/codestral-2508": {
        "token_limit": 256000,
        "max_tokens": 4096,
        "temperature": 0,
    },
    # Fallback default
    "default": {
        "token_limit": 8192,
        "max_tokens": 2048,
        "temperature": 0.1,
    },
}


class LLMManager:
    """LLM manager with dynamic token limits and multi-provider support."""

    def __init__(self, provider: str, model: str, temperature: float = 0.1):
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        self._model_config = self._get_model_config(model)
        logger.info(
            f"LLMManager initialized: {provider}/{model} "
            f"(context: {self._model_config['token_limit']} tokens)"
        )

    def _get_model_config(self, model: str) -> Dict:
        """Get model configuration with token limits."""
        # Try exact match first
        if model in MODEL_CONFIGS:
            return MODEL_CONFIGS[model]

        # Try partial match (e.g., "gemini-2.5-flash-latest" → "gemini-2.5-flash")
        for key in MODEL_CONFIGS:
            if key != "default" and key in model:
                logger.info(f"Using config for {key} (matched from {model})")
                return MODEL_CONFIGS[key]

        # Fallback to default
        logger.warning(f"No config for {model}, using default")
        return MODEL_CONFIGS["default"]

    def get_current_llm_context_window(self) -> int:
        """Get the context window size for the current LLM model.

        Returns:
            int: Maximum context tokens for the current model
        """
        return self._model_config["token_limit"]

    def get_max_output_tokens(self) -> int:
        """Get the maximum output tokens for the current model.

        Returns:
            int: Maximum output tokens
        """
        return self._model_config["max_tokens"]

    def _chat_model(self, provider: str | None = None):
        """Create chat model instance."""
        p = (provider or self.provider).lower()
        if p == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self._model_config["max_tokens"],
            )
        if p == "openrouter":
            # OpenRouter via OpenAI-compatible client
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self._model_config["max_tokens"],
                base_url="https://openrouter.ai/api/v1",
            )
        # default fallback to Gemini
        logger.warning(f"Unknown provider {p}, falling back to Gemini")
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self._model_config["max_tokens"],
        )

    def stream_response(
        self, question: str, context_docs: Iterable
    ) -> Generator[str, None, None]:
        """Stream LLM response with context from complete articles."""
        content_blocks: List[str] = []
        for d in context_docs:
            text = getattr(d, "page_content", None) or getattr(d, "content", "")
            content_blocks.append(text)
        context = "\n\n".join(content_blocks)

        # Log context size
        context_tokens = len(context) // 4  # Rough estimate
        logger.info(
            f"Streaming response with ~{context_tokens} context tokens "
            f"({(context_tokens/self._model_config['token_limit']*100):.1f}% of window)"
        )

        model = self._chat_model()
        messages = [
            ("system", SYSTEM_PROMPT + "\n\nContext:\n" + context),
            ("user", question),
        ]
        for chunk in model.stream(messages):
            token = getattr(chunk, "content", None)
            if token:
                yield token

    def generate(
        self, question: str, context_docs: Iterable, provider: str | None = None
    ) -> str:
        """Generate LLM response (non-streaming) with context from complete articles."""
        content_blocks: List[str] = []
        for d in context_docs:
            text = getattr(d, "page_content", None) or getattr(d, "content", "")
            content_blocks.append(text)
        context = "\n\n".join(content_blocks)

        model = self._chat_model(provider)
        messages = [
            ("system", SYSTEM_PROMPT + "\n\nContext:\n" + context),
            ("user", question),
        ]
        resp = model.invoke(messages)
        return getattr(resp, "content", "")

