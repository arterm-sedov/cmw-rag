from pydantic_settings import BaseSettings, SettingsConfigDict

# Global configuration constants
# Normalize Unicode escapes (e.g., \u0432\u043e -> воз) in search queries
# Set to False if search engine needs raw Unicode escapes (edge case)
NORMALIZE_SEARCH_QUERIES: bool = False


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    All configuration is env-driven with no hardcoded defaults to ensure
    single source of truth per 12-Factor principles. Secrets and provider
    credentials must be provided via environment variables.

    All variables listed below MUST be configured in environment variables.
    """

    # LLM Providers
    google_api_key: str
    openrouter_api_key: str
    openrouter_base_url: str
    # vLLM configuration (OpenAI-compatible API)
    vllm_base_url: str
    vllm_api_key: str

    # Embedding Configuration (Model-Slug Based)
    # Provider type: direct | infinity | openrouter
    embedding_provider_type: str
    # Model slug (e.g., "ai-forever/FRIDA", "Qwen/Qwen3-Embedding-8B")
    # Case insensitive - "qwen/qwen3-embedding-8b" works too
    embedding_model: str
    # Device for direct embedding/reranker when not using factory (e.g. build_index, legacy retriever).
    # Factory path uses device from models.yaml.
    embedding_device: str

    # ChromaDB (HTTP-only: separate server via chroma run or Docker; no embedded PersistentClient)
    chromadb_persist_dir: str
    chromadb_collection: str
    # ChromaDB HTTP client configuration
    # Must be set in .env - no defaults to ensure single source of truth
    chromadb_host: str
    chromadb_port: int
    chromadb_ssl: bool
    # How long to keep HTTP connections alive (prevents premature disconnects, e.g., 60.0 seconds)
    # Maps to ChromaDB's CHROMA_HTTP_KEEPALIVE_SECS environment variable
    chroma_http_keepalive_secs: float
    # Maximum number of connections in the pool
    # Maps to ChromaDB's CHROMA_HTTP_MAX_CONNECTIONS environment variable
    chroma_http_max_connections: int

    # Retrieval
    top_k_retrieve: int
    top_k_rerank: int
    rerank_score_threshold: float | None = None
    rerank_enabled: bool
    chunk_size: int
    chunk_overlap: int

    # Retrieval – multi-vector query and query decomposition
    retrieval_multiquery_enabled: bool
    retrieval_multiquery_max_segments: int
    retrieval_multiquery_segment_tokens: int
    retrieval_multiquery_segment_overlap: int
    retrieval_multiquery_pre_rerank_limit: int

    retrieval_query_decomp_enabled: bool
    retrieval_query_decomp_max_subqueries: int

    # Reranker Provider Selection
    # Options: direct_crossencoder | infinity_dity | infinity_bge_reranker | infinity_qwen3_reranker_8b | infinity_qwen3_reranker_4b | infinity_qwen3_reranker_0_6b
    # Reranker Configuration (Model-Slug Based)
    # Provider type: direct | infinity | openrouter
    reranker_provider_type: str
    # Model slug (e.g., "DiTy/cross-encoder-russian-msmarco", "Qwen/Qwen3-Reranker-8B")
    # Case insensitive
    reranker_model: str

    # Provider Endpoints (optional, have defaults)
    infinity_embedding_endpoint: str
    infinity_reranker_endpoint: str
    openrouter_endpoint: str

    # Request Configuration
    embedding_timeout: float
    embedding_max_retries: int
    reranker_timeout: float
    reranker_max_retries: int

    # LLM
    default_llm_provider: str
    default_model: str
    llm_temperature: float
    llm_max_tokens: int | None = (
        None  # Optional override for max_tokens from model config (hard cutoff)
    )
    llm_mild_limit: int | None = (
        None  # Optional soft guidance limit for response length (injected into system prompt)
    )
    # Optional overrides for model config (if set, overrides model_configs.py values)
    llm_token_limit: int | None = None  # Optional override for token_limit from model config

    # Fallback and summarization
    llm_fallback_enabled: bool
    llm_fallback_provider: str | None = None
    llm_allowed_fallback_models: str
    llm_summarization_enabled: bool
    # vLLM streaming fallback: if True, falls back to invoke() if tool calls aren't detected in stream
    # Set to False to disable fallback and test pure streaming behavior
    vllm_streaming_fallback_enabled: bool
    # Optional legacy/override; dynamic targets are used by default
    llm_summarization_target_tokens_per_article: int | None = None

    # HuggingFace Configuration
    # Token for authenticated downloads (prevents rate limiting)
    # Get token: https://huggingface.co/settings/tokens
    hf_token: str | None = None
    # Trust locally cached models, skip remote validation (faster, offline-friendly)
    # Set to true to skip HEAD requests checking for model updates
    hf_hub_disable_remote_validation: bool = False

    # Gradio
    gradio_server_name: str
    gradio_server_port: int
    # Share link: if True, attempts to create a public shareable link.
    # If share link creation fails (network/service issues), app still runs locally.
    gradio_share: bool
    # Embedded widget mode: if True, uses smaller heights suitable for embedded widget.
    # If False, uses larger heights suitable for standalone app.
    gradio_embedded_widget: bool
    # Queue configuration: concurrency limit for all event listeners
    # Per Gradio docs: https://www.gradio.app/guides/queuing
    gradio_default_concurrency_limit: int

    # Memory compression (conversation history)
    # Percentage of context window at which we trigger compression
    memory_compression_threshold_pct: int
    # Target tokens for the compressed history turn
    memory_compression_target_tokens: int
    # Number of recent messages to keep uncompressed (for agent mode)
    memory_compression_messages_to_keep: int

    # Context thresholds and compression (env-driven)
    # Pre-agent safety threshold as a fraction of the model context window
    llm_pre_context_threshold_pct: float

    # Context overhead safety margin for formatting and message structure
    # Additional tokens reserved beyond actual system prompt and tool schema counts
    # Accounts for: message formatting, JSON structure overhead, output buffer
    # Note: System prompt and tool schemas are counted directly, this is just a safety buffer
    llm_context_overhead_safety_margin: int

    # JSON overhead percentage for tool results (JSON format adds overhead vs raw content)
    # Applied to accumulated tool result tokens to account for JSON serialization overhead
    llm_tool_results_json_overhead_pct: float

    # Tool-results compression controls
    # When total tokens exceed this fraction, trigger compression
    llm_compression_threshold_pct: float
    # After compression, target total tokens to be at/below this fraction
    llm_compression_target_pct: float
    # Minimum tokens to preserve per article during compression
    llm_compression_min_tokens: int

    # Timezone configuration
    # Default timezone for datetime operations (IANA timezone name, e.g., 'Europe/Moscow', 'UTC')
    default_timezone: str

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
