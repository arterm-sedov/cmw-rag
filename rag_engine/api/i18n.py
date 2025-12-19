"""Internationalization resources for Gradio UI using gr.I18n.

We define both Russian and English strings for tool-progress bubbles.
Locale is determined from GRADIO_LOCALE environment variable (defaults to "ru").
For details, see the Gradio i18n guide:
https://www.gradio.app/guides/internationalization
"""
from __future__ import annotations

import os

import gradio as gr

i18n = gr.I18n(
    en={
        # Tool progress / thinking bubbles
        "language": "en",
        "search_started_title": "🧠 Searching the knowledge base",
        "search_started_content": "Searching for: {query}",
        "search_completed_title_with_count": "✅ Search completed",
        "search_completed_content_with_count": "Found articles: {count}.",
        "sources_header": "**Sources:**",
        "thinking_title": "🧠 Thinking",
        "thinking_content": "Using tool: {tool_name}",
        "model_switch_title": "⚡ Switched to {model} (requires more context)",
        "cancelled_title": "⏹️ Cancelled",
        "cancelled_message": "⚠️ Response cancelled by user.",
    },
    ru={
        "language": "ru",
        "search_started_title": "🧠 Поиск информации в базе знаний",
        "search_started_content": "Ищу: {query}",
        "search_completed_title_with_count": "✅ Поиск завершен",
        "search_completed_content_with_count": "Найдено статей: {count}.",
        "sources_header": "**Источники:**",
        "thinking_title": "🧠 Размышление",
        "thinking_content": "Использую инструмент: {tool_name}",
        "model_switch_title": "⚡ Переключение на {model} (требуется больше контекста)",
        "cancelled_title": "⏹️ Отменено",
        "cancelled_message": "⚠️ Ответ отменён пользователем.",
    },
)

def _get_current_locale() -> str:
    """Get current locale from environment variable.

    Returns:
        Locale code (e.g., "en", "ru"). Defaults to "ru" if not set or invalid.
    """
    locale = os.getenv("GRADIO_LOCALE", "ru").lower()
    # Validate locale is in available translations
    if locale in i18n.translations:
        return locale
    return "ru"


def i18n_resolve(key: str, locale: str | None = None) -> str:
    """Workaround helper to manually resolve i18n translations.

    This function extracts translations directly from i18n.translations
    dictionary, bypassing the frontend resolution that may not work.
    Use this instead of i18n() for Gradio component properties.

    Args:
        key: Translation key to resolve
        locale: Optional locale override. If None, reads from GRADIO_LOCALE env variable

    Returns:
        Resolved translation string, or the key itself if not found

    Example:
        >>> gr.Textbox(label=i18n_resolve("input_label"))
        >>> gr.Button(i18n_resolve("button_label"))
    """
    target_locale = locale or _get_current_locale()
    translations = i18n.translations.get(target_locale, {})
    return translations.get(key, i18n.translations.get("en", {}).get(key, key))


def get_text(key: str, **kwargs: str | int) -> str:
    """Get i18n translated text as a plain string with format arguments.

    Args:
        key: i18n translation key (e.g., "search_started_title")
        **kwargs: Format arguments for the translation string (e.g., query="test", count=5)

    Returns:
        Resolved translation string (never returns i18n metadata objects).

    Notes:
        Locale is read from GRADIO_LOCALE environment variable (defaults to "ru").
        Accesses translations directly from i18n.translations dictionary to avoid i18n metadata objects.
        For simple translations without format arguments, use i18n_resolve() instead.

    Example:
        >>> get_text("search_started_content", query="test")
        "Searching for: test"
    """
    locale = _get_current_locale()
    text = i18n.translations.get(locale, {}).get(key, key)
    return text.format(**kwargs)


