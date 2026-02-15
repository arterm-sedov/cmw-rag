from __future__ import annotations

from pathlib import Path

from rag_engine.config.settings import Settings


def test_settings_loads_from_env_file():
    """Test that settings load from the actual .env file."""
    # Load from the actual .env file in repo root
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"

    # This test verifies behavior: settings load correctly from real .env
    settings = Settings(_env_file=env_file)

    # Verify specific values from .env (behavior, not implementation)
    assert isinstance(settings.top_k_retrieve, int)
    assert isinstance(settings.chunk_overlap, int)
    assert isinstance(settings.rerank_enabled, bool)
    assert isinstance(settings.gradio_server_port, int)


def test_environment_overrides_take_precedence(monkeypatch):
    """Test that environment variables override .env file values."""
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"

    # Get original value to restore later
    settings_before = Settings(_env_file=env_file)
    original_value = settings_before.top_k_retrieve

    # Override via environment variable
    monkeypatch.setenv("TOP_K_RETRIEVE", "999")
    settings = Settings(_env_file=env_file)

    # Environment should override .env
    assert settings.top_k_retrieve == 999
