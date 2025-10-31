from __future__ import annotations

from textwrap import dedent

from rag_engine.config.settings import Settings


def _write_env(tmp_path, content: str):
    env_file = tmp_path / "test.env"
    env_file.write_text(dedent(content), encoding="utf-8")
    return env_file


def _base_env() -> str:
    return """
    GOOGLE_API_KEY=test-google
    OPENROUTER_API_KEY=test-openrouter
    EMBEDDING_MODEL=ai-forever/FRIDA
    EMBEDDING_DEVICE=cpu
    CHROMADB_PERSIST_DIR=./data/chroma
    CHROMADB_COLLECTION=test_collection
    TOP_K_RETRIEVE=20
    TOP_K_RERANK=10
    CHUNK_SIZE=500
    CHUNK_OVERLAP=150
    RERANK_ENABLED=true
    RERANKER_MODEL=DiTy/cross-encoder-russian-msmarco
    DEFAULT_LLM_PROVIDER=gemini
    DEFAULT_MODEL=gemini-2.5-flash
    LLM_TEMPERATURE=0.1
    LLM_MAX_TOKENS=4096
    GRADIO_SERVER_NAME=0.0.0.0
    GRADIO_SERVER_PORT=7860
    """


def test_settings_loads_from_env_file(tmp_path):
    env_file = _write_env(tmp_path, _base_env())
    settings = Settings(_env_file=env_file)

    assert settings.embedding_model == "ai-forever/FRIDA"
    assert settings.top_k_retrieve == 20
    assert settings.chunk_overlap == 150
    assert settings.rerank_enabled is True
    assert settings.gradio_server_port == 7860


def test_environment_overrides_take_precedence(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, _base_env())
    monkeypatch.setenv("TOP_K_RETRIEVE", "42")
    settings = Settings(_env_file=env_file)

    assert settings.top_k_retrieve == 42


def test_retrieval_fast_token_char_threshold_default(tmp_path):
    """Test that retrieval_fast_token_char_threshold has correct default."""
    env_file = _write_env(tmp_path, _base_env())
    settings = Settings(_env_file=env_file)

    assert settings.retrieval_fast_token_char_threshold == 200_000


def test_retrieval_fast_token_char_threshold_can_be_overridden(tmp_path):
    """Test that retrieval_fast_token_char_threshold can be set via env."""
    env_content = _base_env() + "\n    RETRIEVAL_FAST_TOKEN_CHAR_THRESHOLD=150000"
    env_file = _write_env(tmp_path, env_content)
    settings = Settings(_env_file=env_file)

    assert settings.retrieval_fast_token_char_threshold == 150_000
