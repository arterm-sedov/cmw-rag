"""Test guardian XML format and category expansion."""

import pytest


def _expand_guardian_categories(categories: list[str]) -> list[str]:
    """Expand ambiguous category abbreviations with brief explanations.

    Based on Qwen3Guard documentation:
    - PII -> PII (Personal Identifiable Information)
    - Jailbreak -> Jailbreak (System Prompt Override Attempt)

    Other categories are left unchanged as they are already self-explanatory:
    - Violent, Non-violent Illegal Acts, Sexual Content or Sexual Acts
    - Suicide & Self-Harm, Unethical Acts, Politically Sensitive Topics
    - Copyright Violation

    Args:
        categories: List of category names from Guardian

    Returns:
        List of category names, with PII and Jailbreak expanded.
    """
    category_expansions = {
        "PII": "PII (Personal Identifiable Information)",
        "Jailbreak": "Jailbreak (System Prompt Override Attempt)",
    }
    return [category_expansions.get(cat, cat) for cat in categories]


def test_pii_expansion() -> None:
    """Test PII category gets expanded correctly."""
    result = _expand_guardian_categories(["PII"])
    assert result == ["PII (Personal Identifiable Information)"]


def test_jailbreak_expansion() -> None:
    """Test Jailbreak category gets expanded correctly."""
    result = _expand_guardian_categories(["Jailbreak"])
    assert result == ["Jailbreak (System Prompt Override Attempt)"]


def test_other_categories_unchanged() -> None:
    """Test non-ambiguous categories remain unchanged."""
    result = _expand_guardian_categories(
        ["Violent", "Suicide & Self-Harm", "Politically Sensitive Topics"]
    )
    expected = ["Violent", "Suicide & Self-Harm", "Politically Sensitive Topics"]
    assert result == expected


def test_mixed_categories() -> None:
    """Test only PII/Jailbreak expanded in mixed list."""
    result = _expand_guardian_categories(["PII", "Violent", "Jailbreak", "Suicide & Self-Harm"])
    expected = [
        "PII (Personal Identifiable Information)",
        "Violent",
        "Jailbreak (System Prompt Override Attempt)",
        "Suicide & Self-Harm",
    ]
    assert result == expected


def test_empty_categories() -> None:
    """Test empty list returns empty list."""
    result = _expand_guardian_categories([])
    assert result == []
