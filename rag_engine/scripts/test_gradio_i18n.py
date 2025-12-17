"""Test script to verify Gradio i18n functionality.

Tests i18n support across various component properties as documented:
https://www.gradio.app/guides/internationalization

Supported Component Properties (varies by component):
- label (most components)
- placeholder (Textbox, Dropdown, etc.)
- info (many components)
- description (some components, NOT Textbox)
- title (Blocks, Accordion)
- value (Button, etc.)

Note: Check component typehints for I18nData to confirm which properties support i18n.

Workaround Helper:
Since Gradio's frontend i18n resolution may not work in all versions,
we provide a helper function that manually resolves translations from
the i18n.translations dictionary as a fallback.

Locale Configuration:
Locale is read from GRADIO_LOCALE environment variable (defaults to "ru").
Set GRADIO_LOCALE=en in .env file to test English translations.
"""
from __future__ import annotations

import os

import gradio as gr


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

    Args:
        key: Translation key to resolve
        locale: Optional locale override. If None, reads from GRADIO_LOCALE env variable

    Returns:
        Resolved translation string, or the key itself if not found
    """
    target_locale = locale or _get_current_locale()
    translations = i18n.translations.get(target_locale, {})
    return translations.get(key, i18n.translations.get("en", {}).get(key, key))


def test_function(text: str) -> str:
    """Simple test function."""
    return f"Output: {text}"


# Create i18n instance with translations for multiple languages
i18n = gr.I18n(
    en={
        "lang_code": "en",
        "greeting": "Hello, welcome to my app!",
        "submit": "Submit",
        "clear": "Clear",
        "input_label": "Input Text",
        "input_placeholder": "Enter your text here...",
        "input_description": "This is an input field",
        "output_label": "Output Text",
        "output_description": "This is an output field",
        "button_label": "Process",
        "info_text": "This is some helpful information",
        "title_text": "Test Application",
        "section_title": "Test Section",
    },
    ru={
        "lang_code": "ru",
        "greeting": "Привет, добро пожаловать в мое приложение!",
        "submit": "Отправить",
        "clear": "Очистить",
        "input_label": "Входной текст",
        "input_placeholder": "Введите ваш текст здесь...",
        "input_description": "Это поле ввода",
        "output_label": "Выходной текст",
        "output_description": "Это поле вывода",
        "button_label": "Обработать",
        "info_text": "Это полезная информация",
        "title_text": "Тестовое приложение",
        "section_title": "Тестовая секция",
    },
)

with gr.Blocks(title=i18n("title_text")) as demo:
    # Display current locale from environment variable
    current_locale = _get_current_locale()
    gr.Markdown(f"**Current locale (from GRADIO_LOCALE env var):** `{current_locale}`")
    gr.Markdown(f"**Locale code translation:** {i18n_resolve('lang_code')}")
    gr.Markdown(f"**Using i18n() directly (may show key):** `{i18n('lang_code')}`")

    # Test Markdown with i18n (using workaround helper)
    gr.Markdown(f"# {i18n_resolve('greeting')}")

    # Test Textbox with label, placeholder, info (description not supported for Textbox)
    # Using workaround helper for manual resolution
    with gr.Row():
        input_text = gr.Textbox(
            label=i18n_resolve("input_label"),
            placeholder=i18n_resolve("input_placeholder"),
            info=i18n_resolve("info_text"),
        )
        output_text = gr.Textbox(
            label=i18n_resolve("output_label"),
            info=i18n_resolve("info_text"),
            interactive=False,
        )

    # Test Button with value (label) - using workaround
    submit_btn = gr.Button(i18n_resolve("button_label"))

    # Test Accordion with title - using workaround
    with gr.Accordion(label=i18n_resolve("section_title"), open=False):
        gr.Markdown("This is inside an accordion")

    # Test Radio with label and info - using workaround
    radio = gr.Radio(
        choices=["Option 1", "Option 2", "Option 3"],
        label=i18n_resolve("input_label"),
        info=i18n_resolve("info_text"),
    )

    # Test Checkbox with label and info - using workaround
    checkbox = gr.Checkbox(
        label=i18n_resolve("input_label"),
        info=i18n_resolve("info_text"),
    )

    # Test Slider with label and info - using workaround
    slider = gr.Slider(
        minimum=0,
        maximum=100,
        value=50,
        label=i18n_resolve("input_label"),
        info=i18n_resolve("info_text"),
    )

    # Test Dropdown with label and info (placeholder not supported for Dropdown) - using workaround
    dropdown = gr.Dropdown(
        choices=["Choice 1", "Choice 2", "Choice 3"],
        label=i18n_resolve("input_label"),
        info=i18n_resolve("info_text"),
    )

    # Test Number with label, description, and info (Number supports description) - using workaround
    number_input = gr.Number(
        label=i18n_resolve("input_label"),
        value=0,
        info=i18n_resolve("info_text"),
    )

    # Test Markdown to display description text (since Textbox doesn't support description)
    gr.Markdown(f"**Description test:** {i18n_resolve('input_description')}")

    # Comparison section: i18n() vs i18n_resolve()
    gr.Markdown("---")
    gr.Markdown("### Comparison: `i18n()` vs `i18n_resolve()` workaround")
    gr.Markdown(f"**i18n('greeting'):** `{i18n('greeting')}`")
    gr.Markdown(f"**i18n_resolve('greeting'):** `{i18n_resolve('greeting')}`")

    # Connect the function
    submit_btn.click(
        fn=test_function,
        inputs=input_text,
        outputs=output_text,
    )


if __name__ == "__main__":
    # Launch with i18n instance
    demo.launch(i18n=i18n, server_name="127.0.0.1", server_port=7860)

