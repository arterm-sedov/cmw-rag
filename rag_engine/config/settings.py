from pydantic_settings import BaseSettings, SettingsConfigDict

# Global configuration constants
# Normalize Unicode escapes (e.g., \u0432\u043e -> воз) in search queries
# Set to False if search engine needs raw Unicode escapes (edge case)
NORMALIZE_SEARCH_QUERIES: bool = False


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
    # vLLM configuration (OpenAI-compatible API)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"

    # Embedding Configuration (Model-Slug Based)
    # Provider type: direct | infinity | openrouter
    embedding_provider_type: str = "direct"
    # Model slug (e.g., "ai-forever/FRIDA", "Qwen/Qwen3-Embedding-8B")
    # Case insensitive - "qwen/qwen3-embedding-8b" works too
    embedding_model: str = "ai-forever/FRIDA"
    # Device for direct embedding/reranker when not using factory (e.g. build_index, legacy retriever).
    # Factory path uses device from models.yaml. Default auto covers most cases.
    embedding_device: str = "auto"

    # ChromaDB
    chromadb_persist_dir: str
    chromadb_collection: str
    # ChromaDB HTTP client configuration (replaces embedded PersistentClient)
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    chromadb_ssl: bool = False
    chromadb_connection_timeout: float = 30.0
    chromadb_max_connections: int = 100

    # Retrieval
    top_k_retrieve: int
    top_k_rerank: int
    rerank_enabled: bool = True
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

    # Reranker Provider Selection
    # Options: direct_crossencoder | infinity_dity | infinity_bge_reranker | infinity_qwen3_reranker_8b | infinity_qwen3_reranker_4b | infinity_qwen3_reranker_0_6b
    # Reranker Configuration (Model-Slug Based)
    # Provider type: direct | infinity | openrouter
    reranker_provider_type: str = "direct"
    # Model slug (e.g., "DiTy/cross-encoder-russian-msmarco", "Qwen/Qwen3-Reranker-8B")
    # Case insensitive
    reranker_model: str = "DiTy/cross-encoder-russian-msmarco"

    # Provider Endpoints (optional, have defaults)
    infinity_embedding_endpoint: str = "http://localhost:7997"
    infinity_reranker_endpoint: str = "http://localhost:7998"
    openrouter_endpoint: str = "https://openrouter.ai/api/v1"

    # Request Configuration
    embedding_timeout: float = 60.0
    embedding_max_retries: int = 3
    reranker_timeout: float = 60.0
    reranker_max_retries: int = 3

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
    llm_fallback_enabled: bool = False
    llm_fallback_provider: str | None = None
    llm_allowed_fallback_models: str = ""
    llm_summarization_enabled: bool = False
    # vLLM streaming fallback: if True, falls back to invoke() if tool calls aren't detected in stream
    # Set to False to disable fallback and test pure streaming behavior
    vllm_streaming_fallback_enabled: bool = False
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
    # Queue configuration: concurrency limit for all event listeners
    # Per Gradio docs: https://www.gradio.app/guides/queuing
    gradio_default_concurrency_limit: int = 3

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

    # Context thresholds and compression (env-driven)
    # Pre-agent safety threshold as a fraction of the model context window
    llm_pre_context_threshold_pct: float = 0.90

    # Context overhead safety margin for formatting and message structure
    # Additional tokens reserved beyond actual system prompt and tool schema counts
    # Accounts for: message formatting, JSON structure overhead, output buffer
    # Default 2000 tokens (configurable via LLM_CONTEXT_OVERHEAD_SAFETY_MARGIN)
    # Note: System prompt and tool schemas are counted directly, this is just a safety buffer
    llm_context_overhead_safety_margin: int = 2000

    # JSON overhead percentage for tool results (JSON format adds overhead vs raw content)
    # Applied to accumulated tool result tokens to account for JSON serialization overhead
    # Default 0.30 = 30% overhead (configurable via LLM_TOOL_RESULTS_JSON_OVERHEAD_PCT)
    llm_tool_results_json_overhead_pct: float = 0.30

    # Tool-results compression controls
    # When total tokens exceed this fraction, trigger compression
    llm_compression_threshold_pct: float = 0.85
    # After compression, target total tokens to be at/below this fraction
    llm_compression_target_pct: float = 0.80
    # Minimum tokens to preserve per article during compression
    llm_compression_min_tokens: int = 300

    # Timezone configuration
    # Default timezone for datetime operations (IANA timezone name, e.g., 'Europe/Moscow', 'UTC')
    default_timezone: str = "Europe/Moscow"

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
