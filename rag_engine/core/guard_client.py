"""Guardian client for content moderation.

Provides HTTP-based content moderation via Mosec server.
This is the production-ready implementation using async HTTP calls
to a dedicated safety model server.
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
    """Content moderation client using Mosec HTTP server.

    Attributes:
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts on failure
    """

    def __init__(
        self,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the guard client.

        Args:
            timeout: Override for timeout (defaults to settings)
            max_retries: Override for retries (defaults to settings)
        """
        self._timeout = timeout or settings.guard_timeout
        self._max_retries = max_retries or settings.guard_max_retries
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _classify(self, content: str) -> dict[str, Any]:
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
                logger.warning(
                    "Guardian classification attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    import time

                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        logger.error("Guardian classification failed after %d attempts", self._max_retries)
        raise last_error

    async def classify(self, content: str) -> dict[str, Any]:
        """Classify content for safety using Mosec HTTP server.

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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: self._classify(content))

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
