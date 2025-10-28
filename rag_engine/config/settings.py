from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    All configuration values come from environment variables.
    No hardcoded defaults - everything is configurable via .env.
    """

    # LLM Providers
    google_api_key: str
    openrouter_api_key: str

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

    # Reranker
    rerank_enabled: bool
    reranker_model: str

    # LLM
    default_llm_provider: str
    default_model: str
    llm_temperature: float
    llm_max_tokens: int

    # Gradio
    gradio_server_name: str
    gradio_server_port: int

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


