"""Centralized model configuration registry.

Defines context windows, max output tokens, and default temperatures
for supported models. Kept separate for clarity and easier maintenance.
"""
from __future__ import annotations

from typing import Dict

MODEL_CONFIGS: Dict[str, Dict] = {
    # Gemini models (matching cmw-platform-agent)
    "gemini-2.5-flash": {
        "token_limit": 1_048_576,  # 1M context
        "max_tokens": 65_536,
        "temperature": 0,
    },
    "gemini-2.5-pro": {
        "token_limit": 1_048_576,  # 1M context
        "max_tokens": 65_536,
        "temperature": 0,
    },
    # OpenRouter models (matching cmw-platform-agent)
    # DeepSeek Models
    "deepseek/deepseek-v3.1-terminus": {
        "token_limit": 163_840,
        "max_tokens": 65_536,
        "temperature": 0,
    },
    "deepseek/deepseek-chat-v3.1:free": {
        "token_limit": 163_840,
        "max_tokens": 4_096,
        "temperature": 0,
    },
    "deepseek/deepseek-r1-0528": {
        "token_limit": 163_840,
        "max_tokens": 4_096,
        "temperature": 0,
    },
    # Grok (xAI) Models
    "x-ai/grok-4-fast:free": {
        "token_limit": 2_000_000,
        "max_tokens": 8_192,
        "temperature": 0,
    },
    "x-ai/grok-code-fast-1": {
        "token_limit": 256_000,
        "max_tokens": 10_000,
        "temperature": 0,
    },
    "x-ai/grok-4-fast": {
        "token_limit": 2_000_000,
        "max_tokens": 30_000,
        "temperature": 0,
    },
    # Qwen Models
    "qwen/qwen3-coder:free": {
        "token_limit": 262_144,
        "max_tokens": 4_096,
        "temperature": 0,
    },
    "qwen/qwen3-coder-flash": {
        "token_limit": 128_000,
        "max_tokens": 4_096,
        "temperature": 0,
    },
    "qwen/qwen3-max": {
        "token_limit": 256_000,
        "max_tokens": 32_768,
        "temperature": 0,
    },
    # Additional Qwen Models
    "qwen/qwen3-235b-a22b": {
        # Native window ~40,960; some routes may extend via scaling
        "token_limit": 262_144,
        "max_tokens": 32_768,
        "temperature": 0,
    },
    "qwen/qwen3-30b-a3b-instruct-2507": {
        "token_limit": 262_144,
        "max_tokens": 32_768,
        "temperature": 0,
    },
    # vLLM Qwen model (with capital letters, matches vLLM max_model_len=40000)
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "token_limit": 40_000,  # Matches vLLM max_model_len configuration
        "max_tokens": 40_000,  # Matches LLM_MAX_TOKENS in .env
        "temperature": 0,
    },
    "qwen/qwen3-coder-plus": {
        "token_limit": 128_000,
        "max_tokens": 65_536,
        "temperature": 0,
    },
    "qwen/qwen3-235b-a22b-2507": {
        "token_limit": 262_144,
        "max_tokens": 32_768,
        "temperature": 0,
    },
    "qwen/qwen3-coder": {
        "token_limit": 262_144,
        "max_tokens": 262_144,
        "temperature": 0,
    },
    # Other Models
    "anthropic/claude-sonnet-4.5": {
        "token_limit": 1_000_000,
        "max_tokens": 64_000,
        "temperature": 0,
    },
    "openai/gpt-oss-120b": {
        "token_limit": 131_072,
        "max_tokens": 32_768,
        "temperature": 0,
    },
    "openai/gpt-5-mini": {
        "token_limit": 400_000,
        "max_tokens": 32_768,
        "temperature": 0,
    },
    "nvidia/nemotron-nano-9b-v2:free": {
        "token_limit": 128_000,
        "max_tokens": 4_096,
        "temperature": 0,
    },
    "mistralai/codestral-2508": {
        "token_limit": 256_000,
        "max_tokens": 4_096,
        "temperature": 0,
    },
    # OpenAI specialized models
    "openai/gpt-5-codex": {
        "token_limit": 400_000,
        "max_tokens": 32_768,
        "temperature": 0,
    },
    # Fallback default
    "default": {
        "token_limit": 8_192,
        "max_tokens": 2_048,
        "temperature": 0.1,
    },
}


