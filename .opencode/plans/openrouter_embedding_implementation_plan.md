# Unified Embedding & Reranker Implementation Plan

## Overview

Add unified embedding and reranker provider support with three categories:
1. **Direct**: Current sentence-transformers implementations (FRIDA, CrossEncoder/DiTy)
2. **Server**: Infinity HTTP servers (FRIDA, Qwen3, BGE, DiTy via OpenAI-compatible API)
3. **API**: OpenRouter cloud API (Qwen3-Embedding)

All providers use **unified Pydantic config schemas** and **interoperable interfaces**.

**Key Decision**: Include DiTy reranker in Infinity configs (it's our battle-tested default).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  cmw-rag (Client Application)                               │
│  ├─ .env: Provider selection (which to use)                 │
│  ├─ config/models.yaml: Client-side Pydantic configs        │
│  │   (endpoints, prefixes, instructions, timeouts)          │
│  └─ rag_engine/                                             │
│      ├─ retrieval/embedder.py: Factory + implementations    │
│      └─ retrieval/reranker.py: Factory + implementations    │
└─────────────────────────────────────────────────────────────┘
                              │
                              HTTP/REST
                              │
┌─────────────────────────────────────────────────────────────┐
│  cmw-infinity (Server Management - NEW REPO)                │
│  ├─ config/models.yaml: Server-side configs                 │
│  │   (ports, dtypes, batch_sizes, device)                   │
│  ├─ cmw_infinity/                                           │
│  │   ├─ cli.py: start/stop/status commands                  │
│  │   ├─ server_manager.py: Process management               │
│  │   └─ server_config.py: Pydantic schemas                  │
│  └─ Thin wrapper around `infinity_emb` CLI                  │
└─────────────────────────────────────────────────────────────┘
```

### Port Allocation Strategy

**Each model has a UNIQUE port** to allow simultaneous running:

```
Embedding Models:
  7997: FRIDA (4GB)
  7998: Qwen3-0.6B (2GB)
  7999: Qwen3-4B (12GB)
  8000: Qwen3-8B (22GB)

Reranker Models:
  8001: BGE-Reranker (2GB)
  8002: DiTy (2GB)
  8003: Qwen3-0.6B (2GB)
  8004: Qwen3-4B (12GB)
  8005: Qwen3-8B (22GB)
```

**Benefits:**
- A/B test different models without stopping/starting
- Run small models alongside large models (if VRAM permits)
- Easy model comparison and benchmarking
- No port conflicts when switching providers

---

## Phase 1: cmw-infinity Package

**Location**: `/c/Repos/cmw-infinity` (new repo, separate from cmw-rag)

**Purpose**: Manage Infinity servers (embedding + reranker) via CLI

### 1.1 Repository Structure
```
cmw-infinity/
├── cmw_infinity/
│   ├── __init__.py
│   ├── cli.py                    # Click CLI: start/stop/status
│   ├── server_config.py          # Pydantic schemas
│   ├── server_manager.py         # Process start/stop
│   └── model_registry.py         # Available models
├── config/
│   └── models.yaml              # Server-side configs
├── tests/
├── pyproject.toml
├── README.md
└── AGENTS.md                    # Adapted from cmw-vllm
```

### 1.2 Server Config Schema (Pydantic)

```python
# server_config.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class InfinityModelConfig(BaseModel):
    """Server-side configuration for Infinity models."""
    model_id: str = Field(description="HuggingFace model ID")
    model_type: Literal["embedding", "reranker"] = Field(description="Model type")
    port: int = Field(description="Server port (must be unique per model)")
    device: str = Field(default="auto", description="Device (auto/cpu/cuda)")
    dtype: Literal["float16", "float32", "int8"] = Field(default="float16")
    batch_size: int = Field(default=32, description="Dynamic batching size")
    memory_gb: float = Field(description="Estimated VRAM usage in GB")
    
    @field_validator('port')
    def validate_port_range(cls, v):
        if not 7000 <= v <= 65535:
            raise ValueError("Port must be between 7000-65535")
        return v
    
    def to_infinity_args(self) -> list[str]:
        """Convert to infinity_emb CLI arguments."""
        args = [
            "v2",
            "--model-name-or-path", self.model_id,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--batch-size", str(self.batch_size),
        ]
        if self.device != "auto":
            args.extend(["--device", self.device])
        return args
```

### 1.3 Server-Side YAML Config

```yaml
# config/models.yaml
# These are SERVER startup configs (ports, dtype, batch_size)
# NOT client connection configs (those are in cmw-rag)
# NOTE: Each model has UNIQUE port to allow simultaneous running

embedding_models:
  # Small models (use when vLLM is running, ~30GB used)
  frida:
    model_id: ai-forever/FRIDA
    model_type: embedding
    port: 7997
    dtype: float16
    batch_size: 32
    memory_gb: 4
    
  qwen3-embedding-0.6b:
    model_id: Qwen/Qwen3-Embedding-0.6B
    model_type: embedding
    port: 7998
    dtype: float16
    batch_size: 32
    memory_gb: 2
    
  # Large models (use when vLLM is NOT running, 48GB available)
  qwen3-embedding-4b:
    model_id: Qwen/Qwen3-Embedding-4B
    model_type: embedding
    port: 7999
    dtype: float16
    batch_size: 16
    memory_gb: 12  # ~10GB weights + overhead
    
  qwen3-embedding-8b:
    model_id: Qwen/Qwen3-Embedding-8B
    model_type: embedding
    port: 8000
    dtype: float16
    batch_size: 8
    memory_gb: 22  # ~16GB weights + 6GB overhead

reranker_models:
  # Small models
  bge-reranker:
    model_id: BAAI/bge-reranker-v2-m3
    model_type: reranker
    port: 8001
    dtype: float16
    batch_size: 32
    memory_gb: 2
    
  dity-reranker:
    model_id: DiTy/cross-encoder-russian-msmarco
    model_type: reranker
    port: 8002
    dtype: float16
    batch_size: 32
    memory_gb: 2
    
  qwen3-reranker-0.6b:
    model_id: Qwen/Qwen3-Reranker-0.6B
    model_type: reranker
    port: 8003
    dtype: float16
    batch_size: 32
    memory_gb: 2
    
  # Large models
  qwen3-reranker-4b:
    model_id: Qwen/Qwen3-Reranker-4B
    model_type: reranker
    port: 8004
    dtype: float16
    batch_size: 16
    memory_gb: 12
    
  qwen3-reranker-8b:
    model_id: Qwen/Qwen3-Reranker-8B
    model_type: reranker
    port: 8005
    dtype: float16
    batch_size: 8
    memory_gb: 22
```

### 1.4 CLI Commands

```bash
# Start servers
cmw-infinity start frida              # Embedding on :7997
cmw-infinity start dity-reranker      # Reranker on :7998
cmw-infinity start qwen3-embedding-8b # Large model on :7997

# Check status
cmw-infinity status
# Output:
# frida                embedding  port:7997   ✓ running  pid:12345
# dity-reranker        reranker   port:7998   ✓ running  pid:12346

# Stop servers
cmw-infinity stop frida
cmw-infinity stop --all
```

### 1.5 Test Plan

**Self-contained tests:**
```bash
# Test 1: Start/Stop FRIDA
cmw-infinity start frida
cmw-infinity status  # Verify running
cmw-infinity stop frida
cmw-infinity status  # Verify stopped

# Test 2: HTTP API working
curl http://localhost:7997/v1/embeddings \
  -X POST \
  -d '{"input": ["test query"], "model": "ai-forever/FRIDA"}' \
  -H "Content-Type: application/json"

# Test 3: Reranker API
curl http://localhost:7998/rerank \
  -X POST \
  -d '{"query": "test", "documents": ["doc1", "doc2"]}' \
  -H "Content-Type: application/json"
```

**Phase 1 Complete When:**
- [ ] cmw-infinity package created
- [ ] `cmw-infinity start/stop/status` commands work
- [ ] FRIDA embedding server responds to HTTP requests
- [ ] DiTy reranker server responds to HTTP requests
- [ ] Tests pass

---

## Phase 2: cmw-rag Client Updates

**Purpose**: Update cmw-rag to use new providers via unified factory pattern

### 2.1 Client-Side Pydantic Schemas (Discriminated Unions)

```python
# rag_engine/config/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
from typing_extensions import Annotated

# ============ EMBEDDING CONFIGS ============

class DirectEmbeddingConfig(BaseModel):
    """Direct sentence-transformers embedder (current implementation)."""
    type: Literal["direct"]
    model: str = Field(..., description="Model name for sentence-transformers")
    device: str = Field(default="auto")
    max_seq_length: int = Field(default=512)

class ServerEmbeddingConfig(BaseModel):
    """HTTP server embedder (Infinity)."""
    type: Literal["server"]
    endpoint: str = Field(..., description="HTTP endpoint (e.g., http://localhost:7997/v1)")
    
    # Model-specific formatting
    query_prefix: Optional[str] = Field(None)  # FRIDA: "search_query: "
    doc_prefix: Optional[str] = Field(None)    # FRIDA: "search_document: "
    default_instruction: Optional[str] = Field(None)  # Qwen3: instruction template

class ApiEmbeddingConfig(BaseModel):
    """Cloud API embedder (OpenRouter)."""
    type: Literal["api"]
    endpoint: str = Field(..., description="API endpoint URL")
    model: str = Field(..., description="Model identifier (e.g., qwen/qwen3-embedding-8b)")
    default_instruction: str = Field(..., description="Default instruction template")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Max retries on failure")

# Discriminated union for type-safe config loading
EmbeddingProviderConfig = Annotated[
    Union[DirectEmbeddingConfig, ServerEmbeddingConfig, ApiEmbeddingConfig],
    Field(discriminator="type")
]

# ============ RERANKER CONFIGS ============

class DirectRerankerConfig(BaseModel):
    """Direct CrossEncoder reranker (current implementation)."""
    type: Literal["direct"]
    model: str = Field(..., description="Model name for CrossEncoder")
    device: str = Field(default="auto")
    batch_size: int = Field(default=16)

class ServerRerankerConfig(BaseModel):
    """HTTP server reranker (Infinity)."""
    type: Literal["server"]
    endpoint: str = Field(..., description="HTTP endpoint (e.g., http://localhost:7998)")
    default_instruction: Optional[str] = Field(None)  # Qwen3 only

RerankerProviderConfig = Annotated[
    Union[DirectRerankerConfig, ServerRerankerConfig],
    Field(discriminator="type")
]

# ============ DIMENSION VALIDATION ============

class ModelDimensions:
    """Embedding dimensions by model - used for validation."""
    DIMENSIONS = {
        "ai-forever/FRIDA": 1024,
        "Qwen/Qwen3-Embedding-0.6B": 1024,
        "Qwen/Qwen3-Embedding-4B": 2560,
        "Qwen/Qwen3-Embedding-8B": 4096,
        "qwen/qwen3-embedding-0.6b": 1024,
        "qwen/qwen3-embedding-4b": 2560,
        "qwen/qwen3-embedding-8b": 4096,
    }
    
    @classmethod
    def get_dimension(cls, model: str) -> int:
        if model not in cls.DIMENSIONS:
            raise ValueError(f"Unknown model: {model}. Must rebuild index when switching models.")
        return cls.DIMENSIONS[model]
```

### 2.2 Client-Side YAML Config

```yaml
# rag_engine/config/models.yaml
# These are CLIENT connection configs
# NOT server startup configs (those are in cmw-infinity)

embedding_providers:
  # FRIDA - Direct (current, unchanged)
  direct_frida:
    type: direct
    model: ai-forever/FRIDA
    device: auto
    max_seq_length: 512
    
  # FRIDA - Via Infinity (new)
  infinity_frida:
    type: server
    endpoint: http://localhost:7997/v1
    query_prefix: "search_query: "      # Match FRIDA's prompt_name
    doc_prefix: "search_document: "
    
  # Qwen3 - Via OpenRouter (new)
  openrouter_qwen3:
    type: api
    endpoint: https://openrouter.ai/api/v1
    model: qwen/qwen3-embedding-8b
    default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
    timeout: 60.0
    max_retries: 3
    
  # Qwen3 - Via Infinity (new)
  infinity_qwen3_8b:
    type: server
    endpoint: http://localhost:8000/v1  # Unique port (not 7997)
    default_instruction: "Given a web search query, retrieve relevant passages that answer the query"

reranker_providers:
  # CrossEncoder/DiTy - Direct (current, unchanged)
  direct_crossencoder:
    type: direct
    model: DiTy/cross-encoder-russian-msmarco
    batch_size: 16
    
  # DiTy - Via Infinity (new, our default)
  infinity_dity:
    type: server
    endpoint: http://localhost:8002  # Unique port (not 8001)
    # DiTy doesn't use instructions - direct (query, doc) pairs
    
  # BGE - Via Infinity (alternative)
  infinity_bge_reranker:
    type: server
    endpoint: http://localhost:8001  # Unique port
    
  # Qwen3 Reranker - Via Infinity (future)
  infinity_qwen3_reranker_8b:
    type: server
    endpoint: http://localhost:8005  # Unique port (not 8001)
    default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
```

### 2.3 .env Provider Selection

```bash
# ============================================
# Provider Selection (Runtime)
# ============================================

# Embedding Provider
# Options: direct_frida | infinity_frida | openrouter_qwen3 | infinity_qwen3_8b
EMBEDDING_PROVIDER=openrouter_qwen3

# Reranker Provider  
# Options: direct_crossencoder | infinity_dity | infinity_bge_reranker | infinity_qwen3_reranker_8b
RERANKER_PROVIDER=infinity_dity

# Server endpoints (when using server providers)
INFINITY_EMBEDDING_ENDPOINT=http://localhost:7997
INFINITY_RERANKER_ENDPOINT=http://localhost:7998

# API keys (when using API providers)
OPENROUTER_API_KEY=sk-...

# Which Infinity models to use (keys from cmw-infinity config)
INFINITY_EMBEDDING_MODEL=frida
INFINITY_RERANKER_MODEL=dity-reranker
```

### 2.4 Unified Embedder Interface

```python
# rag_engine/retrieval/embedder.py
from typing import Protocol, Optional

class Embedder(Protocol):
    """Unified interface for all embedding providers."""
    
    def embed_query(
        self, 
        query: str, 
        instruction: Optional[str] = None
    ) -> list[float]:
        """
        Embed a single query.
        
        Args:
            query: The search query text
            instruction: Optional custom instruction (Qwen3 only, overrides default)
            
        Returns:
            Embedding vector as list of floats
        """
        ...
    
    def embed_documents(
        self, 
        texts: list[str],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> list[list[float]]:
        """Embed multiple documents."""
        ...
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        ...


# Implementation: FRIDA Direct (preserved exactly)
class FRIDAEmbedder:
    """Current implementation - uses sentence-transformers directly."""
    
    def embed_query(self, query: str, instruction: Optional[str] = None) -> list[float]:
        if instruction:
            logger.warning("FRIDA doesn't support dynamic instructions, ignoring")
        return self.model.encode(query, prompt_name="search_query", ...).tolist()
    
    def embed_documents(self, texts: list[str], ...) -> list[list[float]]:
        return self.model.encode(texts, prompt_name="search_document", ...).tolist()


# Implementation: FRIDA via Infinity
class InfinityFRIDAEmbedder:
    """FRIDA via Infinity HTTP server."""
    
    def __init__(self, config: EmbeddingProviderConfig):
        self.client = OpenAI(base_url=config.endpoint, api_key="EMPTY")
        self.query_prefix = config.query_prefix
        self.doc_prefix = config.doc_prefix
    
    def embed_query(self, query: str, instruction: Optional[str] = None) -> list[float]:
        if instruction:
            logger.warning("FRIDA doesn't support dynamic instructions, ignoring")
        formatted = f"{self.query_prefix}{query}"
        response = self.client.embeddings.create(model="ai-forever/FRIDA", input=formatted)
        return response.data[0].embedding
    
    def embed_documents(self, texts: list[str], ...) -> list[list[float]]:
        formatted = [f"{self.doc_prefix}{t}" for t in texts]
        response = self.client.embeddings.create(model="ai-forever/FRIDA", input=formatted)
        return [d.embedding for d in response.data]


# Implementation: Qwen3 via OpenRouter
class OpenRouterEmbedder:
    """Qwen3 via OpenRouter API."""
    
    def __init__(self, config: EmbeddingProviderConfig):
        self.client = OpenAI(
            base_url=config.endpoint, 
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model = config.model
        self.default_instruction = config.default_instruction
    
    def embed_query(self, query: str, instruction: Optional[str] = None) -> list[float]:
        # Dynamic instruction support!
        task = instruction or self.default_instruction
        formatted = f"Instruct: {task}\nQuery: {query}"
        response = self.client.embeddings.create(model=self.model, input=formatted)
        return response.data[0].embedding
    
    def embed_documents(self, texts: list[str], ...) -> list[list[float]]:
        # Documents don't get instruction
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in response.data]


# Factory Function
from rag_engine.config.schemas import EmbeddingProviderConfig, load_config

def create_embedder(settings) -> Embedder:
    """Factory creates appropriate embedder based on .env selection."""
    provider = settings.embedding_provider
    config = load_config("embedding_providers", provider)
    
    if provider == "direct_frida":
        return FRIDAEmbedder(model_name=config.model, device=config.device)
    
    elif provider == "infinity_frida":
        return InfinityFRIDAEmbedder(config)
    
    elif provider == "openrouter_qwen3":
        return OpenRouterEmbedder(config)
    
    elif provider == "infinity_qwen3_8b":
        return InfinityQwen3Embedder(config)  # Same as OpenRouter format
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
```

### 2.5 Unified Reranker Interface

```python
# rag_engine/retrieval/reranker.py
from typing import Protocol, Optional, Any, Sequence

class Reranker(Protocol):
    """Unified interface for all reranker providers."""
    
    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Any, float]],
        top_k: int,
        metadata_boost_weights: Optional[dict[str, float]] = None,
        instruction: Optional[str] = None,  # Qwen3 only
    ) -> list[tuple[Any, float]]:
        """
        Rerank candidates based on query relevance.
        
        Args:
            query: Search query
            candidates: List of (document, initial_score) tuples
            top_k: Number of top results to return
            metadata_boost_weights: Optional metadata-based score boosts
            instruction: Optional custom instruction (Qwen3 reranker only)
            
        Returns:
            Sorted list of (document, reranker_score) tuples
        """
        ...


# Implementation: CrossEncoder Direct (preserved exactly)
class CrossEncoderReranker:
    """Current implementation - uses sentence-transformers directly."""
    
    def rerank(self, query, candidates, top_k, metadata_boost_weights=None, instruction=None):
        if instruction:
            logger.warning("CrossEncoder doesn't support dynamic instructions, ignoring")
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        # ... rest unchanged


# Implementation: DiTy/BGE via Infinity
class InfinityReranker:
    """Reranker via Infinity HTTP server."""
    
    def __init__(self, config: RerankerProviderConfig):
        self.endpoint = config.endpoint
        self.default_instruction = config.default_instruction
    
    def rerank(self, query, candidates, top_k, metadata_boost_weights=None, instruction=None):
        task = instruction or self.default_instruction
        
        # Format based on model type
        if self.default_instruction:
            # Qwen3 format: "Instruct: {task}\nQuery: {query}"
            formatted_query = f"Instruct: {task}\nQuery: {query}"
        else:
            # DiTy/BGE format: raw query
            formatted_query = query
        
        documents = [doc.page_content for doc, _ in candidates]
        
        # Call Infinity /rerank endpoint
        response = requests.post(
            f"{self.endpoint}/rerank",
            json={
                "query": formatted_query,
                "documents": documents,
                "top_k": top_k
            }
        )
        scores = response.json()["scores"]
        
        # Apply metadata boosts if provided
        scored = []
        for (doc, _), score in zip(candidates, scores):
            final_score = self._apply_metadata_boost(score, doc, metadata_boost_weights)
            scored.append((doc, final_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# Factory Function
def create_reranker(settings) -> Reranker:
    """Factory creates appropriate reranker based on .env selection."""
    provider = settings.reranker_provider
    config = load_config("reranker_providers", provider)
    
    if provider == "direct_crossencoder":
        return CrossEncoderReranker(model_name=config.model)
    
    elif provider == "infinity_dity":
        return InfinityReranker(config)
    
    elif provider == "infinity_bge_reranker":
        return InfinityReranker(config)
    
    elif provider == "infinity_qwen3_reranker_8b":
        return InfinityReranker(config)
    
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
```

### 2.5 Error Handling & Resilience

**HTTP clients must handle failures gracefully:**

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class HTTPClientMixin:
    """Mixin providing resilient HTTP client with retries and timeouts."""
    
    def __init__(self, endpoint: str, timeout: float = 60.0, max_retries: int = 3):
        self.endpoint = endpoint
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _post(self, path: str, json_data: dict) -> dict:
        """Make POST request with error handling."""
        url = f"{self.endpoint}{path}"
        try:
            response = self.session.post(
                url,
                json=json_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out after {self.timeout}s")
            raise EmbeddingTimeoutError(f"Server at {url} not responding")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to {url}")
            raise ServerNotAvailableError(f"Server at {url} is not running")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from {url}: {e.response.status_code} - {e.response.text}")
            raise EmbeddingAPIError(f"Server returned error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error calling {url}: {e}")
            raise


# Updated InfinityEmbedder with error handling
class InfinityEmbedder(HTTPClientMixin):
    """FRIDA/Qwen3 via Infinity HTTP server with resilience."""
    
    def __init__(self, config: ServerEmbeddingConfig):
        super().__init__(
            endpoint=config.endpoint,
            timeout=getattr(config, 'timeout', 60.0),
            max_retries=getattr(config, 'max_retries', 3)
        )
        self.query_prefix = config.query_prefix
        self.doc_prefix = config.doc_prefix
        self.default_instruction = config.default_instruction
    
    def embed_query(self, query: str, instruction: Optional[str] = None) -> list[float]:
        if self.default_instruction:
            # Qwen3 format
            task = instruction or self.default_instruction
            formatted = f"Instruct: {task}\nQuery: {query}"
        else:
            # FRIDA format
            if instruction:
                logger.warning("FRIDA doesn't support dynamic instructions, ignoring")
            formatted = f"{self.query_prefix}{query}"
        
        response = self._post("/v1/embeddings", {
            "input": [formatted],
            "model": "auto"  # Infinity ignores this, uses loaded model
        })
        return response["data"][0]["embedding"]


# Updated InfinityReranker with error handling
class InfinityReranker(HTTPClientMixin):
    """DiTy/BGE/Qwen3 via Infinity HTTP server with resilience."""
    
    def __init__(self, config: ServerRerankerConfig):
        super().__init__(
            endpoint=config.endpoint,
            timeout=getattr(config, 'timeout', 60.0),
            max_retries=getattr(config, 'max_retries', 3)
        )
        self.default_instruction = config.default_instruction
    
    def rerank(self, query, candidates, top_k, metadata_boost_weights=None, instruction=None):
        if self.default_instruction:
            # Qwen3 format
            task = instruction or self.default_instruction
            formatted_query = f"Instruct: {task}\nQuery: {query}"
        else:
            # DiTy/BGE format
            if instruction:
                logger.warning("This reranker doesn't support dynamic instructions, ignoring")
            formatted_query = query
        
        documents = [doc.page_content for doc, _ in candidates]
        
        response = self._post("/rerank", {
            "query": formatted_query,
            "documents": documents,
            "top_k": top_k
        })
        
        scores = response["scores"]
        # ... rest of processing
```
```

### 2.6 Usage Examples

```python
# Example 1: Current setup (no changes needed)
# .env:
# EMBEDDING_PROVIDER=direct_frida
# RERANKER_PROVIDER=direct_crossencoder

from rag_engine.retrieval.embedder import create_embedder
from rag_engine.retrieval.reranker import create_reranker

embedder = create_embedder(settings)
reranker = create_reranker(settings)

# Example 2: New setup with Infinity
# .env:
# EMBEDDING_PROVIDER=infinity_frida
# RERANKER_PROVIDER=infinity_dity

embedder = create_embedder(settings)  # HTTP to :7997
reranker = create_reranker(settings)  # HTTP to :7998

# Example 3: OpenRouter + Infinity hybrid
# .env:
# EMBEDDING_PROVIDER=openrouter_qwen3
# RERANKER_PROVIDER=infinity_dity

embedder = create_embedder(settings)  # API call to OpenRouter
reranker = create_reranker(settings)  # HTTP to local :7998

# Example 4: Dynamic instruction (Qwen3 only)
embedding = embedder.embed_query(
    "Python tutorial",
    instruction="Given a code search query, retrieve relevant tutorials"
)
```

### 2.7 Test Plan

**Self-contained tests:**

```python
# tests/test_embedder_factory.py
def test_direct_frida_embedder():
    """Test direct FRIDA embedder (current implementation)."""
    settings = Settings(embedding_provider="direct_frida")
    embedder = create_embedder(settings)
    assert isinstance(embedder, FRIDAEmbedder)
    
    # Test actual embedding
    embedding = embedder.embed_query("test query")
    assert len(embedding) == 1024  # FRIDA dimension

def test_infinity_frida_embedder():
    """Test FRIDA via Infinity (requires server running)."""
    settings = Settings(embedding_provider="infinity_frida")
    embedder = create_embedder(settings)
    assert isinstance(embedder, InfinityFRIDAEmbedder)
    
    # Requires: cmw-infinity start frida
    embedding = embedder.embed_query("test query")
    assert len(embedding) == 1024

# tests/test_reranker_factory.py
def test_direct_crossencoder_reranker():
    """Test direct CrossEncoder reranker (current implementation)."""
    settings = Settings(reranker_provider="direct_crossencoder")
    reranker = create_reranker(settings)
    assert isinstance(reranker, CrossEncoderReranker)

def test_infinity_dity_reranker():
    """Test DiTy via Infinity (requires server running)."""
    settings = Settings(reranker_provider="infinity_dity")
    reranker = create_reranker(settings)
    assert isinstance(reranker, InfinityReranker)
```

**Phase 2 Complete When:**
- [ ] Pydantic schemas created
- [ ] YAML configs with all providers
- [ ] Factory functions work
- [ ] Direct implementations preserved (backward compatibility)
- [ ] Infinity clients implemented (HTTP)
- [ ] OpenRouter client implemented (API)
- [ ] Tests pass for all provider combinations

---

## Summary

### Two Self-Contained Phases:

| Phase | Component | Testable Independently | Duration |
|-------|-----------|----------------------|----------|
| **1** | cmw-infinity package | ✅ Yes (start/stop servers, HTTP API) | 1-2 days |
| **2** | cmw-rag client updates | ✅ Yes (with mock servers) | 2-3 days |

### Key Decisions:

1. **DiTy included**: Yes, in Infinity configs (our default reranker)
2. **Pydantic schemas**: Yes, for type safety and validation
3. **Two packages**: cmw-infinity (server) + cmw-rag (client)
4. **Dynamic instructions**: Optional parameter for Qwen3 (default from config)
5. **Backward compatibility**: Direct implementations preserved unchanged

### Memory Allocation (48GB RTX 4090):

**NOTE: Memory estimates include model weights + activation overhead + batch buffer**

**With vLLM running:**
- vLLM (LLM): ~30GB
- Infinity (FRIDA 4GB + DiTy 2GB): ~6GB
- System overhead: ~2GB
- **Total: ~38GB / 48GB** ✅ (10GB safety margin)

**Without vLLM (remote LLM):**
- Infinity (Qwen3-8B embed 22GB): ~22GB
- Infinity (Qwen3-8B rerank 22GB): ~22GB
- System overhead: ~2GB
- **Total: ~46GB / 48GB** ⚠️ (2GB tight margin)

**Recommendation for 8B models:** Run only one at a time, or use 4B models (~12GB each)

### Next Steps:

1. Review this plan
2. Approve and I'll start Phase 1 (cmw-infinity)
3. Phase 1 testing
4. Phase 2 implementation
5. Integration testing

**Ready to proceed?**
