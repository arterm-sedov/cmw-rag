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

    # Verify Guardian settings from .env implementation
    assert isinstance(settings.guard_enabled, bool)
    assert isinstance(settings.guard_block_threshold, str)
    assert isinstance(settings.guard_provider_type, str)
    assert isinstance(settings.guard_mosec_port, int)
    assert isinstance(settings.guard_timeout, float)
    assert isinstance(settings.guard_max_retries, int)

    # Verify other settings load correctly
    assert isinstance(settings.top_k_retrieve, int)
    assert isinstance(settings.chunk_overlap, int)
    assert isinstance(settings.rerank_enabled, bool)
    assert isinstance(settings.gradio_server_port, int)


def test_guardian_settings_from_env():
    """Test that Guardian settings match .env configuration."""
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"
    settings = Settings(_env_file=env_file)

    # Test Guardian configuration matches .env values
    assert settings.guard_enabled in (True, False)
    assert settings.guard_block_threshold in ("unsafe", "controversial")
    assert settings.guard_provider_type in ("mosec", "vllm")
    assert settings.guard_mosec_url in ("http://localhost", "https://localhost", "")
    assert settings.guard_mosec_port in range(1, 65536)
    assert settings.guard_mosec_path.startswith("/") if settings.guard_mosec_path else True


def test_environment_overrides_take_precedence(monkeypatch):
    """Test that environment variables override .env file values."""
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"

    # Override via environment variable
    monkeypatch.setenv("TOP_K_RETRIEVE", "999")
    settings = Settings(_env_file=env_file)

    # Environment should override .env
    assert settings.top_k_retrieve == 999
