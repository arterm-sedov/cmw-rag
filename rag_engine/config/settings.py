
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    Most configuration is env-driven. Some fields include safe, opinionated
    defaults to ensure resilience when env vars are not set. Secrets and
    provider credentials must be provided via environment variables.
    """

    # LLM Providers
    google_api_key: str
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Embedding
    embedding_model: str
    embedding_device: str

    # ChromaDB
    chromadb_persist_dir: str
    chromadb_collection: str

    # Retrieval
    top_k_retrieve: int
    top_k_rerank: int
    chunk_size: int
    chunk_overlap: int

    # Retrieval – multi-vector query and query decomposition
    # Kept configurable via .env; safe defaults provided
    retrieval_multiquery_enabled: bool = True
    retrieval_multiquery_max_segments: int = 4
    retrieval_multiquery_segment_tokens: int = 448
    retrieval_multiquery_segment_overlap: int = 64
    retrieval_multiquery_pre_rerank_limit: int = 60

    retrieval_query_decomp_enabled: bool = False
    retrieval_query_decomp_max_subqueries: int = 4

    # Reranker
    rerank_enabled: bool
    reranker_model: str

    # LLM
    default_llm_provider: str
    default_model: str
    llm_temperature: float
    llm_max_tokens: int

    # Fallback and summarization
    llm_fallback_enabled: bool = False
    llm_fallback_provider: str | None = None
    llm_allowed_fallback_models: str = ""
    llm_summarization_enabled: bool = False
    # Optional legacy/override; dynamic targets are used by default
    llm_summarization_target_tokens_per_article: int | None = None

    # Gradio
    gradio_server_name: str
    gradio_server_port: int
    # Share link: if True, attempts to create a public shareable link.
    # If share link creation fails (network/service issues), app still runs locally.
    gradio_share: bool = False
    # Embedded widget mode: if True, uses smaller heights suitable for embedded widget.
    # If False, uses larger heights suitable for standalone app.
    gradio_embedded_widget: bool = False

    # Agent Mode
    # If True, uses LangChain agent-based handler with tool calling
    # If False, uses direct retrieval handler (legacy behavior)
    use_agent_mode: bool = False

    # Memory compression (conversation history)
    # Percentage of context window at which we trigger compression
    # For agent mode with tool calls, lower values (70-75) prevent overflow
    memory_compression_threshold_pct: int = 80  # Changed from 85 to 70 for agent compatibility
    # Target tokens for the compressed history turn
    memory_compression_target_tokens: int = 1000
    # Number of recent messages to keep uncompressed (for agent mode)
    memory_compression_messages_to_keep: int = 2  # Match old handler

    # Performance guardrails
    # When counting tokens, approximate for very large strings beyond this char length
    retrieval_fast_token_char_threshold: int = 200000

    # Pydantic v2 configuration: accept extra env vars and set env file
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


# Helpers derived from settings (avoid polluting pydantic model with properties)
def get_allowed_fallback_models() -> list[str]:
    raw = settings.llm_allowed_fallback_models or ""
    return [m.strip() for m in raw.split(",") if m and m.strip()]



