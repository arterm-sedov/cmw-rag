from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file in project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)


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



