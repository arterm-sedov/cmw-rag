"""Content moderation client for IP model integration.

Provides async wrapper around synchronous requests library using ThreadPoolExecutor.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from rag_engine.config.settings import settings

logger = logging.getLogger(__name__)


class ContentModerationClient:
    """HTTP client for external IP/content moderation model.

    Wraps synchronous requests with async interface using ThreadPoolExecutor
    to avoid blocking the event loop.
    """

    def __init__(
        self,
        url: str | None = None,
        port: int | None = None,
        path: str | None = None,
    ) -> None:
        """Initialize the content moderation client.

        Args:
            url: Base URL for the moderation API (defaults to settings)
            port: Port number (defaults to settings)
            path: API path (defaults to settings)
        """
        self._url = url or settings.content_moderation_url
        self._port = port or settings.content_moderation_port
        self._path = path or settings.content_moderation_path
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _sync_classify(self, content: str) -> dict[str, Any]:
        """Synchronous HTTP call to moderation API.

        Args:
            content: User message to classify

        Returns:
            Parsed JSON response from the moderation API

        Raises:
            requests.RequestException: If the HTTP request fails
        """
        full_url = f"{self._url}:{self._port}{self._path}"
        payload = {"content": content, "moderation_type": "prompt"}

        logger.debug("Calling content moderation API: %s", full_url)
        response = requests.post(full_url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    async def classify(self, content: str) -> dict[str, Any]:
        """Async wrapper for content classification.

        Args:
            content: User message to classify

        Returns:
            Parsed JSON response with:
            - safety_level: "Safe" | "Controversial" | "Unsafe"
            - categories: list[str]
            - is_safe: bool
            - refusal: "Yes" | "No"
            - raw_output: str
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._sync_classify(content),
        )

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


content_moderation_client = ContentModerationClient()
