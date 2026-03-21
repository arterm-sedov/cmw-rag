# CMW-RAG Reranker Refactoring Plan - March 21, 2026

## Status: ✅ COMPLETE

**Implemented:** March 21, 2026

**Commits:**
- `d63312c` - docs: add reranker refactoring plan
- `82d8ecc` - feat: add RerankerAdapter with vLLM/Cohere endpoint contracts
- `c23aa15` - fix: update deprecated loader for new schema changes
- `983ab01` - docs: update plan status, add progress report, update README
- `89ae231` - docs: clarify reranker endpoint configuration
- `ad08762` - fix: use configured endpoint directly for reranker

---

## Verification Results

| Test Suite | Result |
|------------|--------|
| `test_reranker_factory.py` | 17 passed, 4 skipped |
| `test_retrieval_reranker.py` | 5 passed |
| `test_reranker_contracts.py` | 13 passed |
| `test_config_loader.py` | 18 passed |
| **Total** | **53 passed, 4 skipped** |

### Live Server Verification (cmw-mosec port 7998)

```
Endpoint: /v1/score (vLLM format)
Request:  {query, documents}
Response: {data: [{index, score}, ...]}
```

### Architecture Decision

The RAG engine uses `/v1/score` endpoint (NOT `/v1/rerank`) because:
1. Documents are already local (ChromaDB)
2. Metadata boosts require client-side processing
3. Sorting is needed anyway after boost application
4. `/v1/score` is lighter weight (no document text in response)

### Cross-Encoder Works
- DiTy/BGE-m3: Raw query/documents (no transformation) ✅
- Client-side sorting with metadata boosts ✅

### LLM Reranker Works
- Qwen3: Prefix + instruction + suffix formatting ✅
- BGE-Gemma: A/B format with prompt ✅
- Client-side formatting before sending to server ✅

---

## Final Implementation

### RerankerAdapter

```python
class RerankerAdapter(HTTPClientMixin):
    """Client-side formatting adapter for server rerankers.
    
    Uses endpoint from config directly (e.g., http://localhost:7998/v1/score).
    Expects vLLM format: {data: [{index, score}, ...]}
    Client-side sorting with metadata boosts.
    """
    
    def rerank(self, query, candidates, top_k, metadata_boost_weights, instruction):
        # 1. Format query/documents
        # 2. Call endpoint, get scores
        # 3. Apply metadata boosts
        # 4. Sort and return top_k
```

## Executive Summary

Refactor cmw-rag reranker to use industry-standard vLLM/Cohere API contracts. Move ALL formatting (prefix, suffix, instruction) to client-side. Server is agnostic.

### Principle: Smart Client, Agnostic Server

- **Client owns:** Model-specific knowledge (prefix, suffix, instruction templates)
- **Server does:** Receive pre-formatted strings, return scores
- **No per-document HTTP loop:** One request, N documents, N scores

---

## Contract Verification (Tested 2026-03-21)

**Server:** cmw-mosec on port 7998 (DiTy cross-encoder)

| Endpoint | Request | Response | Order |
|----------|---------|----------|-------|
| `/v1/score` | `{query, documents}` | `{data: [{index, object: "score", score}, ...]}` | Original |
| `/v1/rerank` | `{query, documents, top_n?}` | `{results: [{index, document: {text}, relevance_score}, ...]}` | Sorted desc |

**Verified:**
- Scores are **identical** in both endpoints (same `_compute_scores()` on server)
- `/v1/score` returns in original document order
- `/v1/rerank` returns sorted by relevance (highest first)
- `top_n` parameter limits `/v1/rerank` results
- Empty documents → `{results: []}`

---

## Design Principles

### TDD (Test-Driven Development)
1. Write test for expected behavior BEFORE implementation
2. Test passes only when implementation is correct
3. Refactor after tests pass

### SDD (Spec-Driven Development)
1. API contract is specification (vLLM/Cohere format)
2. Implementation matches spec exactly
3. No deviation from industry standard

### DDD (Domain-Driven Design)
1. Domain concepts: `cross_encoder`, `llm_reranker`
2. Language aligns with model card terminology
3. Formatting belongs to client domain, not server

### Non-Breaking
1. `CrossEncoderReranker` unchanged (direct model)
2. `IdentityReranker` unchanged (pass-through)
3. `InfinityReranker` deprecated but functional
4. Only `RerankerAdapter` (new) uses new contracts

### Lean / DRY / Minimal
1. Single source of truth: `models.yaml` for formatting
2. No duplicate logic between endpoints
3. Adapter pattern: one class handles both model types
4. Templates in config, not code

---

## Test YAML Fixture

Create `rag_engine/tests/fixtures/test_rerankers.yaml` (aligned with cmw-mosec):

```yaml
# Test harness config for reranker models
# Single source of truth for test cases, formatting templates, and contracts
#
# Endpoint Contracts (industry-standard, vLLM/Cohere/Jina compatible):
#   /v1/score - Raw scores in original order
#     Request: {query, documents}
#     Response: {data: [{index, object: "score", score}, ...]}
#
#   /v1/rerank - Sorted results with documents
#     Request: {query, documents, top_n?}
#     Response: {results: [{index, document: {text}, relevance_score}, ...]}

################################################################################
# CROSS-ENCODER MODELS (no formatting needed)
################################################################################

cross_encoder:
  test_cases:
    - name: "English - Machine Learning"
      query: "What is machine learning?"
      documents:
        - "Machine learning is a method of data analysis that automates analytical model building."
        - "The weather is sunny today in San Francisco."
        - "Deep learning is a subset of machine learning using neural networks."
      expected_ranking: [0, 2]  # Most relevant documents (order matters)

    - name: "Russian - Car/Auto"
      query: "машина"
      documents:
        - "Автомобиль для перевозки грузов и пассажиров."
        - "Куриное блюдо из тушёной курицы с овощами."
        - "Новый автомобиль Tesla model выпуск 2024 года."
      expected_ranking: [0, 2]

    - name: "English - Capital Cities"
      query: "What is the capital of France?"
      documents:
        - "The capital of Brazil is Brasilia."
        - "The capital of France is Paris, known for the Eiffel Tower."
        - "Horses and cows are both animals found on farms."
      expected_ranking: [1]

################################################################################
# LLM RERANKER MODELS (require client-side formatting)
################################################################################

llm_reranker:
  # Qwen3-Reranker formatting (ChatML)
  qwen3:
    query_template: "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    doc_template: "<Document>: {doc}{suffix}"
    prefix: "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix: "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
    instructions:
      - "Given a web search query, retrieve relevant passages that answer the query"
      - "Given a question, retrieve documents that answer the question"
    test_cases:
      - name: "English - Capital Cities"
        query: "What is the capital of France?"
        documents:
          - "The capital of Brazil is Brasilia."
          - "The capital of France is Paris, known for the Eiffel Tower."
          - "Horses and cows are both animals found on farms."
        expected_ranking: [1]

      - name: "English - Machine Learning"
        query: "What is machine learning?"
        documents:
          - "Machine learning is a method of data analysis that automates analytical model building."
          - "The weather is sunny today."
          - "Deep learning is a subset of machine learning using neural networks."
        expected_ranking: [0, 2]

  # BGE-Gemma formatting
  bge_gemma:
    query_template: "A: {query}"
    doc_template: "B: {doc}\n{prompt}"
    prompt: "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    test_cases:
      - name: "English - Capital Cities"
        query: "What is the capital of France?"
        documents:
          - "The capital of Brazil is Brasilia."
          - "The capital of France is Paris, known for the Eiffel Tower."
          - "Horses and cows are both animals found on farms."
        expected_ranking: [1]

################################################################################
# MODEL REGISTRY MAPPINGS
################################################################################

model_mappings:
  DiTy/cross-encoder-russian-msmarco:
    type: cross_encoder

  BAAI/bge-reranker-v2-m3:
    type: cross_encoder

  Qwen/Qwen3-Reranker-0.6B:
    type: llm_reranker
    subtype: qwen3

  Qwen/Qwen3-Reranker-4B:
    type: llm_reranker
    subtype: qwen3

  Qwen/Qwen3-Reranker-8B:
    type: llm_reranker
    subtype: qwen3

  BAAI/bge-reranker-v2-gemma:
    type: llm_reranker
    subtype: bge_gemma
```

---

## Behavior-Based Tests (TDD)

### Test Principles

**DO test:**
- Behavior: inputs → expected outputs
- Contract compliance: response format matches spec
- Formatting correctness: query/document templates applied correctly
- Ranking accuracy: `expected_ranking` matches actual

**DON'T test:**
- Implementation details: which endpoint is called
- Internal state: private attributes
- Mock internals: mock only external HTTP calls

### Test Files

```
rag_engine/tests/
├── fixtures/
│   └── test_rerankers.yaml       # Test cases and formatting
├── test_reranker_contracts.py    # Endpoint contract tests
├── test_reranker_formatting.py   # Client-side formatting tests
├── test_reranker_factory.py      # Factory with new model types
└── test_retrieval_reranker.py    # Integration tests
```

### Contract Tests (test_reranker_contracts.py)

```python
"""Test vLLM/Cohere contract compliance."""
import pytest
from rag_engine.retrieval.reranker import RerankerAdapter
from rag_engine.config.schemas import ServerRerankerConfig

class TestScoreEndpoint:
    """Test /v1/score endpoint contract (vLLM format)."""
    
    def test_returns_data_array(self, mock_server):
        """Response has 'data' key with array of scores."""
        mock_server.set_response({
            "data": [
                {"index": 0, "object": "score", "score": 0.95},
                {"index": 1, "object": "score", "score": 0.05}
            ]
        })
        adapter = RerankerAdapter(config=dity_config())
        scores = adapter.score("query", ["doc1", "doc2"])
        
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert scores[0] == 0.95
    
    def test_preserves_original_order(self, mock_server):
        """Scores returned in original document order."""
        mock_server.set_response({
            "data": [
                {"index": 1, "object": "score", "score": 0.9},
                {"index": 0, "object": "score", "score": 0.1}
            ]
        })
        adapter = RerankerAdapter(config=dity_config())
        scores = adapter.score("query", ["doc1", "doc2"])
        
        # Order follows index, not score value
        assert scores == [0.1, 0.9]  # index 0 first, then index 1


class TestRerankEndpoint:
    """Test /v1/rerank endpoint contract (Cohere format)."""
    
    def test_returns_results_array(self, mock_server):
        """Response has 'results' key with array of ranked items."""
        mock_server.set_response({
            "results": [
                {"index": 1, "document": {"text": "doc2"}, "relevance_score": 0.95},
                {"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.05}
            ]
        })
        adapter = RerankerAdapter(config=dity_config())
        results = adapter.rerank("query", ["doc1", "doc2"])
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["relevance_score"] == 0.95
    
    def test_sorted_by_relevance(self, mock_server):
        """Results sorted by relevance_score descending."""
        mock_server.set_response({
            "results": [
                {"index": 1, "document": {"text": "doc2"}, "relevance_score": 0.95},
                {"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.05}
            ]
        })
        adapter = RerankerAdapter(config=dity_config())
        results = adapter.rerank("query", ["doc1", "doc2"])
        
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_top_n_limits_results(self, mock_server):
        """top_n parameter limits number of results."""
        mock_server.set_response({
            "results": [
                {"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.95}
            ]
        })
        adapter = RerankerAdapter(config=dity_config())
        results = adapter.rerank("query", ["doc1", "doc2", "doc3"], top_n=1)
        
        assert len(results) == 1


class TestIdenticalScores:
    """Both endpoints return identical score values."""
    
    def test_score_values_match(self, mock_server):
        """Same query/documents produces identical scores in both endpoints."""
        # Mock returns same underlying scores
        score_response = {
            "data": [
                {"index": 0, "object": "score", "score": 0.88},
                {"index": 1, "object": "score", "score": 0.12}
            ]
        }
        rerank_response = {
            "results": [
                {"index": 0, "document": {"text": "doc1"}, "relevance_score": 0.88},
                {"index": 1, "document": {"text": "doc2"}, "relevance_score": 0.12}
            ]
        }
        
        adapter = RerankerAdapter(config=dity_config())
        
        # Both should return same score values (just different structure)
        scores = adapter.score("query", ["doc1", "doc2"])
        results = adapter.rerank("query", ["doc1", "doc2"])
        
        score_values = [s for s in scores]
        relevance_values = [r["relevance_score"] for r in results]
        
        assert score_values == relevance_values
```

### Formatting Tests (test_reranker_formatting.py)

```python
"""Test client-side formatting for different reranker types."""
import pytest
from rag_engine.retrieval.reranker import RerankerAdapter
from rag_engine.config.schemas import ServerRerankerConfig, RerankerFormatting

class TestCrossEncoderFormatting:
    """Cross-encoder models require no formatting."""
    
    def test_query_unchanged(self):
        """DiTy/BGE-m3: raw query, no transformation."""
        config = ServerRerankerConfig(
            type="server",
            provider="mosec",
            endpoint="http://localhost:7998",
            reranker_type="cross_encoder",
        )
        adapter = RerankerAdapter(config)
        
        formatted = adapter.format_query("What is Python?", instruction="search")
        
        assert formatted == "What is Python?"
    
    def test_documents_unchanged(self):
        """Cross-encoder: documents pass through unchanged."""
        config = ServerRerankerConfig(
            type="server",
            provider="mosec",
            endpoint="http://localhost:7998",
            reranker_type="cross_encoder",
        )
        adapter = RerankerAdapter(config)
        
        formatted = adapter.format_document("Python is a programming language")
        
        assert formatted == "Python is a programming language"
    
    def test_instruction_ignored_with_warning(self, caplog):
        """Cross-encoder logs warning if instruction provided."""
        config = ServerRerankerConfig(
            type="server",
            provider="mosec",
            endpoint="http://localhost:7998",
            reranker_type="cross_encoder",
        )
        adapter = RerankerAdapter(config)
        
        with caplog.at_level("WARNING"):
            adapter.format_query("query", instruction="custom")
        
        assert "doesn't support dynamic instructions" in caplog.text


class TestLLMRerankerFormatting:
    """LLM reranker models require client-side formatting."""
    
    @pytest.fixture
    def qwen3_config(self):
        """Qwen3 configuration with formatting."""
        return ServerRerankerConfig(
            type="server",
            provider="mosec",
            endpoint="http://localhost:7998",
            reranker_type="llm_reranker",
            default_instruction="Given a web search query, retrieve relevant passages",
            formatting=RerankerFormatting(
                query_template="{prefix}<Instruct>: {instruction}\n<Query>: {query}\n",
                doc_template="<Document>: {doc}{suffix}",
                prefix="<|im_start|>system\nJudge whether...<|im_end|>\n<|im_start|>user\n",
                suffix="<|im_end|>\n<|im_start|>assistant\n\n\n\n\n",
            ),
        )
    
    def test_query_has_prefix_and_instruction(self, qwen3_config):
        """Qwen3 query formatted with prefix + instruction."""
        adapter = RerankerAdapter(qwen3_config)
        
        formatted = adapter.format_query("What is AI?", instruction="search for AI info")
        
        assert "<|im_start|>system" in formatted
        assert "<Instruct>:" in formatted
        assert "search for AI info" in formatted
        assert "<Query>: What is AI?" in formatted
    
    def test_query_uses_default_instruction(self, qwen3_config):
        """Qwen3 uses default instruction if none provided."""
        adapter = RerankerAdapter(qwen3_config)
        
        formatted = adapter.format_query("What is AI?")
        
        assert "Given a web search query" in formatted
    
    def test_document_has_suffix(self, qwen3_config):
        """Qwen3 document formatted with suffix."""
        adapter = RerankerAdapter(qwen3_config)
        
        formatted = adapter.format_document("Python is a language")
        
        assert "<Document>: Python is a language" in formatted
        assert "<|im_end|>" in formatted
        assert "<|im_start|>assistant" in formatted


class TestBgeGemmaFormatting:
    """BGE-Gemma formatting (simpler than Qwen3)."""
    
    @pytest.fixture
    def bge_gemma_config(self):
        """BGE-Gemma configuration."""
        return ServerRerankerConfig(
            type="server",
            provider="mosec",
            endpoint="http://localhost:7998",
            reranker_type="llm_reranker",
            formatting=RerankerFormatting(
                query_template="A: {query}",
                doc_template="B: {doc}\n{prompt}",
                prefix="",
                suffix="",
            ),
        )
    
    def test_query_format(self, bge_gemma_config):
        """BGE-Gemma query: 'A: {query}'"""
        adapter = RerankerAdapter(bge_gemma_config)
        
        formatted = adapter.format_query("What is Python?")
        
        assert formatted == "A: What is Python?"
```

### Expected Ranking Tests (behavior-based)

```python
"""Test ranking accuracy using test_rerankers.yaml."""
import pytest
import yaml
from pathlib import Path

def load_test_cases():
    """Load test cases from YAML fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_rerankers.yaml"
    with open(fixture_path, "r") as f:
        return yaml.safe_load(f)

class TestCrossEncoderRanking:
    """Test cross-encoder ranking matches expected_ranking."""
    
    @pytest.fixture
    def test_cases(self):
        return load_test_cases()["cross_encoder"]["test_cases"]
    
    @pytest.mark.integration
    def test_expected_ranking(self, test_cases, live_mosec_server):
        """Verify top results match expected_ranking."""
        adapter = create_reranker(
            provider="mosec",
            model="DiTy/cross-encoder-russian-msmarco",
            endpoint="http://localhost:7998"
        )
        
        for case in test_cases:
            results = adapter.rerank(
                case["query"],
                case["documents"]
            )
            
            # Verify expected ranking matches
            top_indices = [r["index"] for r in results[:len(case["expected_ranking"])]]
            assert top_indices == case["expected_ranking"], f"Failed: {case['name']}"


class TestLLMRerankerRanking:
    """Test LLM reranker ranking with formatting."""
    
    @pytest.fixture
    def test_cases(self):
        return load_test_cases()["llm_reranker"]["qwen3"]["test_cases"]
    
    @pytest.mark.integration
    def test_expected_ranking_with_formatting(self, test_cases, live_mosec_server):
        """Verify formatting produces correct ranking."""
        adapter = create_reranker(
            provider="mosec",
            model="Qwen/Qwen3-Reranker-0.6B",
            endpoint="http://localhost:7998"
        )
        
        for case in test_cases:
            results = adapter.rerank(
                case["query"],
                case["documents"]
            )
            
            top_indices = [r["index"] for r in results[:len(case["expected_ranking"])]]
            assert top_indices == case["expected_ranking"], f"Failed: {case['name']}"
```

---

## Schema Updates

### File: `rag_engine/config/schemas.py`

```python
class RerankerFormatting(BaseModel):
    """Client-side formatting for LLM rerankers.
    
    Templates use Python format strings with placeholders:
    - {query}: The search query
    - {doc}: The document text
    - {instruction}: The task instruction
    - {prefix}: Query prefix (Qwen3: system prompt)
    - {suffix}: Document suffix (Qwen3: end tokens)
    - {prompt}: Task prompt (BGE-Gemma)
    """
    
    query_template: str = Field(
        default="{query}",
        description="Query formatting template"
    )
    doc_template: str = Field(
        default="{doc}",
        description="Document formatting template"
    )
    prefix: str = Field(
        default="",
        description="Query prefix (e.g., ChatML system prompt)"
    )
    suffix: str = Field(
        default="",
        description="Document suffix (e.g., end tokens)"
    )


class ServerRerankerConfig(BaseModel):
    """HTTP server reranker (Mosec/Infinity)."""
    
    type: Literal["server"]
    provider: str = Field(..., description="Provider: mosec or infinity")
    endpoint: str = Field(..., description="API endpoint URL")
    reranker_type: Literal["cross_encoder", "llm_reranker"] = Field(
        default="cross_encoder",
        description="Model architecture type"
    )
    formatting: RerankerFormatting | None = Field(
        default=None,
        description="Client-side formatting (LLM rerankers only)"
    )
    default_instruction: str | None = Field(
        default=None,
        description="Default instruction for LLM rerankers"
    )
    timeout: float = Field(default=60.0)
    max_retries: int = Field(default=3)
```

### File: `rag_engine/config/models.yaml`

```yaml
models:
  DiTy/cross-encoder-russian-msmarco:
    type: reranker
    reranker_type: cross_encoder
    description: "Russian-optimized cross-encoder reranker"
    provider_formats:
      direct:
        batch_size: 16
        device: auto
      mosec: {}

  BAAI/bge-reranker-v2-m3:
    type: reranker
    reranker_type: cross_encoder
    description: "Multilingual BGE reranker"
    provider_formats:
      direct:
        batch_size: 16
        device: auto
      mosec: {}

  Qwen/Qwen3-Reranker-0.6B:
    type: reranker
    reranker_type: llm_reranker
    description: "Lightweight Qwen3 reranker"
    default_instruction: "Given a web search query, retrieve relevant passages that answer the query"
    formatting:
      query_template: "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
      doc_template: "<Document>: {doc}{suffix}"
      prefix: "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
      suffix: "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
    provider_formats:
      mosec: {}
```

---

## RerankerAdapter Implementation

```python
class RerankerAdapter(HTTPClientMixin):
    """Client-side formatting adapter for server rerankers.
    
    Handles both cross-encoder (no formatting) and LLM reranker (formatting required).
    Uses industry-standard vLLM/Cohere contracts.
    """
    
    def __init__(self, config: ServerRerankerConfig):
        super().__init__(
            endpoint=config.endpoint,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
        self.config = config
    
    def format_query(self, query: str, instruction: str | None = None) -> str:
        """Format query based on reranker type.
        
        Cross-encoder: raw query (no transformation)
        LLM reranker: apply template with prefix/instruction
        """
        if self.config.reranker_type == "cross_encoder":
            if instruction:
                logger.warning("Cross-encoder doesn't support instructions, ignoring")
            return query
        
        # LLM reranker formatting
        if not self.config.formatting:
            return query
        
        fmt = self.config.formatting
        task = instruction or self.config.default_instruction or ""
        
        return fmt.query_template.format(
            prefix=fmt.prefix,
            instruction=task,
            query=query,
        )
    
    def format_document(self, doc: str) -> str:
        """Format document based on reranker type.
        
        Cross-encoder: raw document
        LLM reranker: apply template with suffix
        """
        if self.config.reranker_type == "cross_encoder":
            return doc
        
        if not self.config.formatting:
            return doc
        
        return self.config.formatting.doc_template.format(
            doc=doc,
            suffix=self.config.formatting.suffix,
        )
    
    def score(self, query: str, documents: list[str], 
              instruction: str | None = None) -> list[float]:
        """Get raw scores via /v1/score endpoint.
        
        Returns scores in original document order (vLLM format).
        """
        formatted_query = self.format_query(query, instruction)
        formatted_docs = [self.format_document(d) for d in documents]
        
        response = self._post({
            "query": formatted_query,
            "documents": formatted_docs,
        })
        
        # Parse vLLM format: {data: [{index, object, score}, ...]}
        data = response["data"]
        sorted_data = sorted(data, key=lambda x: x["index"])
        return [item["score"] for item in sorted_data]
    
    def rerank(self, query: str, candidates: Sequence[tuple],
               top_k: int, instruction: str | None = None,
               metadata_boost_weights: dict[str, float] | None = None) -> list[tuple]:
        """Rerank candidates using /v1/rerank endpoint.
        
        Returns (document, score) tuples sorted by relevance.
        """
        documents = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc, _ in candidates
        ]
        
        formatted_query = self.format_query(query, instruction)
        formatted_docs = [self.format_document(d) for d in documents]
        
        payload = {"query": formatted_query, "documents": formatted_docs}
        if top_k:
            payload["top_n"] = top_k
        
        response = self._post(payload)
        
        # Parse Cohere format: {results: [{index, document, relevance_score}, ...]}
        results = response["results"]
        
        # Apply metadata boosts if provided
        scored: list[tuple[Any, float]] = []
        for result in results:
            idx = result["index"]
            doc, _ = candidates[idx]
            score = result["relevance_score"]
            
            boost = 0.0
            if metadata_boost_weights and hasattr(doc, "metadata"):
                meta = getattr(doc, "metadata", {})
                if meta.get("tags") and metadata_boost_weights.get("tag_match"):
                    boost += metadata_boost_weights["tag_match"]
                if meta.get("has_code") and metadata_boost_weights.get("code_presence"):
                    boost += metadata_boost_weights["code_presence"]
            
            final_score = float(score) * (1.0 + boost)
            scored.append((doc, final_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

---

## Non-Breaking Guarantees

| Component | Status | Notes |
|-----------|--------|-------|
| `CrossEncoderReranker` | ✅ Unchanged | Direct model, no HTTP |
| `IdentityReranker` | ✅ Unchanged | Pass-through |
| `Reranker` Protocol | ✅ Unchanged | Same interface |
| `create_reranker()` | ✅ Extended | New `RerankerAdapter` returned |
| `InfinityReranker` | ⚠️ Deprecated | Still works, use `RerankerAdapter` |

---

## Verification Checklist

### Before Implementation
- [ ] Test YAML fixture created (`test_rerankers.yaml`)
- [ ] Contract tests written (pass with mocks)
- [ ] Formatting tests written
- [ ] Schema changes documented

### After Implementation
- [ ] All unit tests pass
- [ ] Cross-encoder works (no formatting)
- [ ] LLM reranker works (with formatting)
- [ ] `/v1/score` returns original order
- [ ] `/v1/rerank` returns sorted by relevance
- [ ] Scores identical across endpoints
- [ ] `expected_ranking` matches for all test cases

### Integration Tests
- [ ] DiTy cross-encoder on mosec server
- [ ] BGE-m3 cross-encoder on mosec server
- [ ] Qwen3 llm_reranker on mosec server (when available)

---

## Implementation Order

1. **Create test fixture** (`test_rerankers.yaml`)
2. **Write contract tests** (TDD: tests fail until implementation)
3. **Write formatting tests** (TDD: tests fail until implementation)
4. **Add schemas** (`RerankerFormatting`, update `models.yaml`)
5. **Implement `RerankerAdapter`**
6. **Update `create_reranker` factory**
7. **Run all tests**
8. **Integration tests with live servers**
9. **Deprecate `InfinityReranker`**

---

## Success Criteria

1. **TDD**: All tests written before implementation
2. **SDD**: Contracts match vLLM/Cohere spec exactly
3. **DDD**: Domain concepts align with model cards
4. **Non-breaking**: CrossEncoderReranker unchanged
5. **Lean/DRY**: Single source of truth in `models.yaml`
6. **Identical scores**: Both endpoints return same values
7. **All test cases pass**: `expected_ranking` matches for all models

---

## Instruction Optimization (2026-03-21)

### Benchmark Results

Tested 11 instructions on Russian/English technical documentation corpus (10,685 docs).

**Winner:** `platform_ru` (0.3459 avg score)
```
Найдите документацию по платформе Comindware: руководства, примеры кода, конфигурации и API на русском или английском
```

**Key Findings:**
1. Russian instructions outperform English for Russian corpus
2. Platform-specific mentions improve relevance
3. Code awareness ("примеры кода") helps
4. Default generic instruction is 2.6x worse

**Recommended Change:**
Update `models.yaml` default_instruction from generic web search to corpus-specific instruction.

**Full analysis:** `rag_engine/docs/analysis/reranker_instruction_analysis_20260321.md`