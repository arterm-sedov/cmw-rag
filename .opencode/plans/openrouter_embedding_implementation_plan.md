# OpenRouter Qwen3-Embedding API Support Implementation Plan

## Overview

Add configurable embedding provider support with provider-based naming convention, preserving existing FRIDA functionality while enabling API-based embeddings through OpenRouter and future local/embedding providers.

## Key Requirements

- **Backward Compatibility**: FRIDA functionality remains unchanged and default
- **Provider-Based Naming**: Consistent `local_*` and API provider naming
- **Configurable Provider**: Switch between providers via `.env` flag
- **Dimension Flexibility**: Configurable embedding dimensions (MRL support)
- **Chunk Size Awareness**: Automatic capping based on model capacity
- **Complete Testing**: Unit tests + integration tests
- **Documentation**: README updates with migration guidance

---

## Implementation Phases

### Phase 1: Configuration Layer

#### 1.1 Update `rag_engine/config/settings.py`

**Add new fields:**
```python
# Provider selection using provider-based naming convention (default: local_frida for backward compatibility)
embedding_provider: str = "local_frida"  # "local_frida", "local_e5", "openrouter", etc.

# OpenRouter embedding configuration
openrouter_embedding_model: str = "qwen/qwen3-embedding-8b"
openrouter_embedding_instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
openrouter_embedding_dim: int | None = None  # Configurable dimension (MRL support)
```

#### 1.2 Add validation helper with backward compatibility:
```python
def validate_embedding_config() -> None:
    """Validate embedding configuration matches provider."""
    provider = settings.embedding_provider.lower()
    
    # Handle backward compatibility: map old "frida" to "local_frida"
    if provider == "frida":
        provider = "local_frida"
        logger.info("Migrating EMBEDDING_PROVIDER from 'frida' to 'local_frida' for consistency")
    
    # Current supported providers (extensible for future)
    local_providers = ("local_frida", "local_e5", "local_bge", "tei_server")
    api_providers = ("openrouter", "openai", "voyage", "together")
    
    if provider not in local_providers + api_providers:
        raise ValueError(f"Invalid embedding_provider: {provider}")
    
    # Provider-specific validation
    if provider == "openrouter" and not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY required when embedding_provider=openrouter")
```

### Phase 2: Embedding Abstraction Layer

#### 2.1 Update `rag_engine/retrieval/embedder.py`

**A. Add imports:**
```python
from openai import OpenAI
from typing import Protocol
```

**B. Define `Embedder` Protocol:**
```python
class Embedder(Protocol):
    """Protocol for embedding backends."""

    def embed_query(self, query: str) -> list[float]: ...

    def embed_documents(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = True
    ) -> list[list[float]]: ...

    def get_embedding_dim(self) -> int: ...

    def get_max_chunk_size(self) -> int: ...
```

**C. Keep `FRIDAEmbedder` unchanged** (implements Protocol implicitly)
**D. Add `OpenRouterEmbedder` class:** (same as original plan)
**E. Add factory function with backward compatibility:**
```python
def create_embedder(settings: Settings) -> Embedder:
    """Factory function to create embedder based on provider."""
    provider = settings.embedding_provider.lower()
    
    # Handle backward compatibility: map old "frida" to "local_frida"
    if provider == "frida":
        provider = "local_frida"
    
    # API Providers
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY required when embedding_provider=openrouter"
            )
        return OpenRouterEmbedder(
            model=settings.openrouter_embedding_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            instruction=settings.openrouter_embedding_instruction,
            dimensions=settings.openrouter_embedding_dim,
        )
    elif provider == "openai":
        # Future: OpenAI embeddings API
        raise NotImplementedError("OpenAI embeddings provider not yet implemented")
    elif provider == "voyage":
        # Future: Voyage AI embeddings
        raise NotImplementedError("Voyage embeddings provider not yet implemented")
    
    # Local Providers
    elif provider == "local_frida":
        # Current FRIDA implementation (uses existing settings)
        return FRIDAEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
    elif provider == "local_e5":
        # Future: Local E5 embeddings
        raise NotImplementedError("Local E5 embeddings provider not yet implemented")
    elif provider == "local_bge":
        # Future: Local BGE embeddings
        raise NotImplementedError("Local BGE embeddings provider not yet implemented")
    elif provider == "tei_server":
        # Future: TEI server client
        raise NotImplementedError("TEI server provider not yet implemented")

    # Default to FRIDA for backward compatibility (should not reach here due to mapping above)
    return FRIDAEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
```

**F. Add `get_max_chunk_size()` to `FRIDAEmbedder`:**
```python
def get_max_chunk_size(self) -> int:
    """Maximum chunk size this embedder supports."""
    return 500  # Current working value
```

#### 2.2 Update `rag_engine/core/indexer.py`

**Add chunk size capping logic:** (same as original plan)

### Phase 3: Update Instantiation Points

All instantiation points updated to use `create_embedder(settings)` (same as original plan).

### Phase 4: Documentation

#### 4.1 Update `rag_engine/.env-example`

**Add new configuration section with consistent provider naming:**
```bash
# ============================================
# Embedding Provider Configuration
# ============================================

# Embedding provider using provider-based naming convention
# Local providers: "local_frida", "local_e5", "local_bge", "tei_server"
# API providers: "openrouter", "openai", "voyage", "together"
# Default: local_frida (backward compatible with old "frida")
EMBEDDING_PROVIDER=local_frida

# FRIDA settings (used when EMBEDDING_PROVIDER=local_frida)
EMBEDDING_MODEL=ai-forever/FRIDA
EMBEDDING_DEVICE=auto

# OpenRouter embedding settings (used when EMBEDDING_PROVIDER=openrouter)
# Available models:
# - qwen/qwen3-embedding-8b (4096 dim, $0.01/M tokens, 32K context)
# - qwen/qwen3-embedding-4b (2560 dim, $0.02/M tokens, 32K context)
# - qwen/qwen3-embedding-0.6b (1024 dim, ~$0.01/M tokens, 32K context)
OPENROUTER_EMBEDDING_MODEL=qwen/qwen3-embedding-8b

# Custom instruction for queries (Qwen3 requires instruction format)
OPENROUTER_EMBEDDING_INSTRUCT="Given a web search query, retrieve relevant passages that answer the query"

# Optional: Reduce embedding dimension (MRL support)
# Range depends on model:
# - 8B: 32-4096
# - 4B: 32-2560
# - 0.6B: 32-1024
# Leave empty for full dimension
OPENROUTER_EMBEDDING_DIM=

# Future local provider settings (examples, NOT implementing now)
# LOCAL_E5_MODEL=intfloat/multilingual-e5-large
# LOCAL_BGE_MODEL=BAAI/bge-large-en-v1.5
# TEI_SERVER_URL=http://localhost:8080

# Future API provider settings (examples, NOT implementing now)
# OPENAI_EMBEDDING_MODEL=text-embedding-3-large
# VOYAGE_API_KEY=your-voyage-key
# TOGETHER_API_KEY=your-together-key

# Chunk size for document splitting (tokens)
# Automatically capped per embedder model capacity:
# - local_frida (FRIDA): max 500 tokens (current working value)
# - openrouter: max 12000 tokens (soft cap for 32K context)
# - Future local models: model-dependent (typically 512-8192)
# - Future API providers: provider-dependent (typically 8192-32768)
CHUNK_SIZE=500
CHUNK_OVERLAP=150
```

#### 4.2 Update `README.md`

**Add section: "Switching Embedding Providers"**

```markdown
## Switching Embedding Providers

The system supports multiple embedding backends with a consistent provider-based naming convention.

### Available Providers

#### Local Providers (Offline)
**local_frida** (Default)
- **Pros**: Free, offline, fast inference, proven in production
- **Cons**: Requires 4GB disk space, 512 token context limit
- **Best for**: Development, local deployments, privacy-sensitive data
- **Settings**: `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`

**Future Local Providers** (Not yet implemented):
- `local_e5`: E5 multilingual embeddings
- `local_bge`: BGE large embeddings  
- `tei_server`: Text Embeddings Inference server client

#### API Providers (Online)

**openrouter** (Current API)
- **Pros**: High quality (32K context), no local resources needed, scalable
- **Cons**: API costs, requires internet connection
- **Models**:
  - `qwen/qwen3-embedding-8b`: 4096 dim, $0.01/M tokens (best quality)
  - `qwen/qwen3-embedding-4b`: 2560 dim, $0.02/M tokens (balanced)
  - `qwen/qwen3-embedding-0.6b`: 1024 dim, ~$0.01/M tokens (fastest)

**Future API Providers** (Not yet implemented):
- `openai`: OpenAI embeddings API
- `voyage`: Voyage AI embeddings
- `together`: Together AI embeddings

### Configuration

```bash
# Current FRIDA users can keep using "frida" (automatically mapped to "local_frida")
EMBEDDING_PROVIDER=frida  # Backward compatible

# Or use new consistent naming
EMBEDDING_PROVIDER=local_frida

# Switch to OpenRouter
EMBEDDING_PROVIDER=openrouter
OPENROUTER_EMBEDDING_MODEL=qwen/qwen3-embedding-8b
OPENROUTER_API_KEY=your-key

# Optional: Reduce dimension for faster processing
OPENROUTER_EMBEDDING_DIM=512
```

### Migration

**Important**: When switching providers or changing dimensions, you **must rebuild the index**:

```bash
# Rebuild index with new embedding provider
python rag_engine/scripts/build_index.py --source "path/to/docs" --mode folder
```

**Backward Compatibility**: Existing `EMBEDDING_PROVIDER=frida` configurations continue to work and are automatically mapped to `local_frida`.

Different models produce incompatible embeddings - vector database must be fully recreated.
```

### Phase 5: Testing

All test files updated to use consistent provider naming with backward compatibility tests.

#### 5.2 Updated `rag_engine/tests/test_embedder_factory.py`

```python
@pytest.fixture
def frida_settings():
    """Settings with FRIDA provider (backward compatible)."""
    return Settings(
        embedding_provider="frida",  # Old name for backward compatibility test
        embedding_model="ai-forever/FRIDA",
        embedding_device="auto",
        openrouter_api_key="test-key",
    )

@pytest.fixture
def local_frida_settings():
    """Settings with local_frida provider."""
    return Settings(
        embedding_provider="local_frida",  # New consistent naming
        embedding_model="ai-forever/FRIDA",
        embedding_device="auto",
        openrouter_api_key="test-key",
    )

def test_factory_backward_compatibility(frida_settings):
    """Test factory handles old 'frida' name correctly."""
    embedder = create_embedder(frida_settings)
    assert isinstance(embedder, FRIDAEmbedder)
    
def test_factory_local_frida_provider(local_frida_settings):
    """Test factory returns FRIDA embedder when provider=local_frida."""
    embedder = create_embedder(local_frida_settings)
    assert isinstance(embedder, FRIDAEmbedder)
```

### Phase 6: Verification & Validation

Same validation steps as original plan, plus:

#### 6.5 Backward Compatibility Test
```bash
# Test that old "frida" name still works
export EMBEDDING_PROVIDER=frida
export OPENROUTER_API_KEY=your-key
python -c "
from rag_engine.retrieval.embedder import create_embedder
from rag_engine.config.settings import settings
embedder = create_embedder(settings)
print(f'Provider: {settings.embedding_provider} -> {type(embedder).__name__}')
print(f'Dimension: {embedder.get_embedding_dim()}')
"
```

---

## Configuration Matrix

| Setting | local_frida | OpenRouter Qwen3-Embedding |
|----------|--------------|----------------------------|
| **Default Model** | `ai-forever/FRIDA` | `qwen/qwen3-embedding-8b` |
| **Context Window** | 512 tokens | 32K tokens |
| **Dimensions** | 1024 | 1024-4096 (configurable) |
| **Max Chunk Size** | 500 tokens | 12000 tokens (soft cap) |
| **Cost** | Free (hardware) | $0.01-$0.02/M tokens |
| **Latency** | Low (local) | Network delay |
| **Setup** | 4GB disk space | API key only |
| **Offline** | Yes | No |

## Provider Naming Convention

| Category | Current | Future Examples | Pattern |
|----------|----------|------------------|----------|
| **Local** | `local_frida` | `local_e5`, `local_bge`, `tei_server` | `local_<model_or_service>` |
| **API** | `openrouter` | `openai`, `voyage`, `together` | `<provider_name>` |

## Chunk Size Behavior

| Configured CHUNK_SIZE | local_frida (effective) | OpenRouter (effective) |
|----------------------|-------------------------|---------------------------|
| 300 | 300 | 300 |
| 500 | 500 | 500 |
| 600 | **500** (capped) | 600 |
| 2000 | **500** (capped) | 2000 |
| 15000 | **500** (capped) | **12000** (capped) |

## Migration Notes

### Backward Compatibility

**Existing `.env` files with `EMBEDDING_PROVIDER=frida` continue to work:**
- Automatically mapped to `local_frida` in factory
- No immediate changes required
- Migration log indicates mapping occurred

### When to Re-index

**Required** when switching:
- Between providers (local ↔ API)
- Between models (local_frida ↔ local_e5 ↔ openrouter 8B ↔ 4B)
- Changing provider-specific dimensions

**Not Required** when:
- Staying with same provider/model/dimension
- Changing other settings (temperature, reranker, etc.)

### Migration Path for Existing Users

```bash
# Step 1: Existing configuration (works unchanged)
EMBEDDING_PROVIDER=frida

# Step 2: Optional - migrate to consistent naming
EMBEDDING_PROVIDER=local_frida

# Step 3: When ready for API
EMBEDDING_PROVIDER=openrouter
```

---

## Summary of Changes

| File | Change Type | Lines Added |
|------|-------------|-------------|
| `rag_engine/config/settings.py` | Add fields + validation + backward compatibility | ~20 |
| `rag_engine/retrieval/embedder.py` | Add Protocol, OpenRouterEmbedder, factory | ~120 |
| `rag_engine/core/indexer.py` | Add chunk size capping | ~15 |
| `rag_engine/api/app.py` | Use factory | ~3 |
| `rag_engine/tools/retrieve_context.py` | Use factory | ~3 |
| `rag_engine/scripts/build_index.py` | Use factory | ~3 |
| `rag_engine/.env-example` | Add config section with consistent naming | ~35 |
| `rag_engine/tests/test_openrouter_embedder.py` | New test file | ~80 |
| `rag_engine/tests/test_embedder_factory.py` | Updated with backward compatibility | ~60 |
| `rag_engine/tests/test_integration_openrouter.py` | New test file | ~60 |
| `README.md` | Update with migration guide + provider naming | ~120 |

**Total:** ~519 lines added, ~0 lines modified (FRIDA untouched)

---

## Implementation Checklist

- [ ] Add configuration fields to `settings.py`
- [ ] Create `Embedder` protocol
- [ ] Implement `OpenRouterEmbedder` class
- [ ] Add `get_max_chunk_size()` to both embedders
- [ ] Implement factory function with backward compatibility
- [ ] Update all instantiation points
- [ ] Add chunk size capping to indexer
- [ ] Update `.env-example` with consistent provider naming
- [ ] Create comprehensive tests with backward compatibility
- [ ] Update README with migration guide and provider naming
- [ ] Verify all existing tests pass
- [ ] Run linting and validation
- [ ] Test with real API key
- [ ] Test backward compatibility with old "frida" name

---

## Tradeoff Guidance

| Use Case | Recommended Provider | Reason |
|----------|-------------------|---------|
| **Development/Testing** | local_frida | Free, fast, no API calls |
| **Production (best quality)** | openrouter 8B | Highest dimensions, best retrieval |
| **Production (cost-sensitive)** | openrouter 0.6B | Fast, cheapest API calls |
| **Privacy-sensitive data** | local_frida | Local processing only |
| **Long documents** | openrouter any | 32K context vs FRIDA's 512 |
| **Offline deployment** | local_frida | No network dependency |
| **Multilingual needs** | local_e5 (future) | Better multilingual support |

---

## Security & Best Practices

- **API Key Security**: Never commit `OPENROUTER_API_KEY` to repository
- **Rate Limits**: OpenRouter has rate limits - implement retry logic if needed
- **Error Handling**: Graceful fallback when API unavailable
- **Cost Monitoring**: Track API usage, especially during bulk indexing
- **Model Validation**: Verify model availability before switching providers
- **Backward Compatibility**: Support old "frida" naming during transition period

---

## Future Extensibility

This implementation enables future providers without code changes using consistent naming:

```python
# Provider examples following naming convention:
local_frida     # Current: FRIDA via sentence-transformers
local_e5          # Future: E5 multilingual embeddings
local_bge          # Future: BGE large embeddings
tei_server         # Future: Text Embeddings Inference server
openrouter         # Current: OpenRouter API
openai            # Future: OpenAI embeddings API
voyage            # Future: Voyage AI embeddings
together           # Future: Together AI embeddings
```

The `Embedder` Protocol and factory pattern provide clean abstraction for any future embedding backend while maintaining consistent naming across all providers.

---

## Migration Timeline

### Phase 1: Implementation (Current Sprint)
- Add new provider naming
- Implement OpenRouter support
- Add backward compatibility mapping

### Phase 2: Transition Period (Optional)
- Existing `frida` users get migration log
- Documentation updated to recommend `local_frida`
- No breaking changes during transition

### Phase 3: Future Deprecation (Optional)
- Remove old "frida" mapping after transition period
- All documentation uses consistent naming