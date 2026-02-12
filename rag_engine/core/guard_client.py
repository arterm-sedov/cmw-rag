"""Guardian client for content moderation.

Provides unified interface for content moderation across different providers:
- direct: Local model inference using sentence-transformers
- mosec: Remote HTTP server (Mosec)
- openrouter: Cloud API (OpenRouter)
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from rag_engine.config.settings import settings

logger = logging.getLogger(__name__)


class GuardClient:
    """Unified content moderation client supporting multiple providers.

    Attributes:
        provider_type: Type of provider (direct, mosec, openrouter)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts on failure
    """

    def __init__(
        self,
        provider_type: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the guard client.

        Args:
            provider_type: Override for provider type (defaults to settings)
            timeout: Override for timeout (defaults to settings)
            max_retries: Override for max retries (defaults to settings)
        """
        self._provider_type = provider_type or settings.guard_provider_type
        self._timeout = timeout or settings.guard_timeout
        self._max_retries = max_retries or settings.guard_max_retries
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Direct provider setup (lazy loaded)
        self._direct_model = None

    def _init_direct_model(self) -> Any:
        """Initialize direct provider model (sentence-transformers)."""
        if self._direct_model is None:
            from sentence_transformers import SentenceTransformer

            model_name = settings.guard_model or "ai-forever/FRIDA"
            device = settings.guard_device
            self._direct_model = SentenceTransformer(model_name, device=device)
            logger.info("Guard direct model initialized: %s on %s", model_name, device)
        return self._direct_model

    def _classify_direct(self, content: str) -> dict[str, Any]:
        """Classify content using local model.

        Args:
            content: User message to classify

        Returns:
            Dict with safety_level, categories, is_safe, etc.
        """
        model = self._init_direct_model()
        # Placeholder - actual implementation depends on the model used
        # This is a simplified version for demonstration
        raise NotImplementedError("Direct provider requires specific model implementation")

    def _classify_mosec(self, content: str) -> dict[str, Any]:
        """Classify content via Mosec HTTP server.

        Args:
            content: User message to classify

        Returns:
            Dict with safety_level, categories, is_safe, etc.
        """
        url = f"{settings.guard_mosec_url}:{settings.guard_mosec_port}{settings.guard_mosec_path}"
        payload = {"content": content, "moderation_type": "prompt"}

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self._timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Mosec classification attempt %d/%d failed: %s",
                              attempt + 1, self._max_retries, exc)

        raise last_error

    def _classify_openrouter(self, content: str) -> dict[str, Any]:
        """Classify content via OpenRouter API.

        Args:
            content: User message to classify

        Returns:
            Dict with safety_level, categories, is_safe, etc.
        """
        import openai

        client = openai.OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

        model = settings.guard_openrouter_model or "google/gemini-2.5-flash"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Classify the following text for content safety. "
                    "Return JSON with: safety_level (Safe/Controversial/Unsafe), "
                    "categories (list of categories), is_safe (boolean)."
                },
                {"role": "user", "content": content},
            ],
            temperature=0,
        )

        import json
        result_text = response.choices[0].message.content
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {
                "safety_level": "Safe",
                "categories": [],
                "is_safe": True,
                "raw_output": result_text,
            }

    async def classify(self, content: str) -> dict[str, Any]:
        """Classify content for safety using configured provider.

        Args:
            content: User message to classify

        Returns:
            Dict with:
            - safety_level: "Safe" | "Controversial" | "Unsafe"
            - categories: list[str]
            - is_safe: bool
            - refusal: "Yes" | "No"
            - raw_output: str (original response)
        """
        if self._provider_type == "direct":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._classify_direct(content),
            )
        elif self._provider_type == "mosec":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._classify_mosec(content),
            )
        elif self._provider_type == "openrouter":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._classify_openrouter(content),
            )
        else:
            raise ValueError(f"Unknown guard provider type: {self._provider_type}")

    def is_safe(self, result: dict[str, Any]) -> bool:
        """Check if content is safe based on moderation result.

        Args:
            result: Response from classify() method

        Returns:
            True if content is safe, False otherwise
        """
        if not result:
            return True
        safety_level = result.get("safety_level", "")
        is_safe_flag = result.get("is_safe", True)
        return safety_level == "Safe" and is_safe_flag

    def get_safety_level(self, result: dict[str, Any]) -> str:
        """Extract safety level from moderation result.

        Args:
            result: Response from classify() method

        Returns:
            Safety level string: "Safe" | "Controversial" | "Unsafe"
        """
        return result.get("safety_level", "Safe") if result else "Safe"

    def get_categories(self, result: dict[str, Any]) -> list[str]:
        """Extract categories from moderation result.

        Args:
            result: Response from classify() method

        Returns:
            List of category strings
        """
        return result.get("categories", []) if result else []

    def should_block(self, result: dict[str, Any]) -> bool:
        """Check if content should be blocked (enforce mode).

        Args:
            result: Response from classify() method

        Returns:
            True if content should be blocked
        """
        if not result:
            return False
        return result.get("safety_level") == "Unsafe"


guard_client = GuardClient()
