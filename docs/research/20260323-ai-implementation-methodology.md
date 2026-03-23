# Executive Summary: AI Implementation & Retirement Methodology

**Repositories:** cmw-rag, cmw-mosec, cmw-vllm
**Date:** 2026-03-23
**Prepared for:** Technical Management
**Status:** Final

---

## 1. Executive Overview

This document outlines the methodology for implementing and managing AI infrastructure across the CMW ecosystem. The architecture leverages a **modular, containerized approach** combining a RAG engine (cmw-rag) with specialized inference servers (cmw-mosec, cmw-vllm).

**Key Methodological Principles:**
- **Separation of Concerns:** Distinct layers for data processing, retrieval, inference, and API delivery.
- **Hybrid Retrieval:** Combining vector search (dense) with keyword search (sparse) for optimal accuracy.
- **Agentic Architecture:** Utilizing LangChain agents for dynamic tool calling and reasoning.
- **Infrastructure Flexibility:** Support for both MOSEC (unified server) and vLLM (distributed instances) inference backends.

---

## 2. Implementation Architecture

### 2.1 Core Components

| Component | Repository | Role | Technology |
|-----------|------------|------|------------|
| **RAG Engine** | `cmw-rag` | Orchestrates retrieval, generation, and agent logic | Python, LangChain, Gradio |
| **Inference Server (Unified)** | `cmw-mosec` | Serves embedding, reranker, and guard models on one port | MOSEC, PyTorch |
| **Inference Server (Distributed)** | `cmw-vllm` | Serves LLMs and pooling models via vLLM | vLLM, CUDA |
| **Vector Store** | `cmw-rag` | Persistent storage for document embeddings | ChromaDB (HTTP) |

### 2.2 Data Flow & Pipeline

1.  **Ingestion:**
    *   Documents (Markdown, MkDocs) are processed by `rag_engine/core/document_processor.py`.
    *   Chunked via `rag_engine/core/chunker.py` (token-aware).
    *   Embedded via `rag_engine/retrieval/embedder.py` (FRIDA/Qwen3).
    *   Stored in ChromaDB via `rag_engine/storage/vector_store.py`.

2.  **Retrieval (RAG):**
    *   User query enters `rag_engine/retrieval/retriever.py`.
    *   **Vector Search:** ChromaDB retrieves top-k chunks.
    *   **Reranking:** Cross-encoder or LLM-reranker (`rag_engine/retrieval/reranker.py`) refines results.
    *   **Context Assembly:** Articles reconstructed, summarized if needed (`rag_engine/llm/summarization.py`).

3.  **Generation:**
    *   **Agent Mode (Recommended):** LangChain agent analyzes query, calls `retrieve_context` tool (forced via `tool_choice`), and generates answer with citations.
    *   **Direct Mode:** LLM manager (`rag_engine/llm/llm_manager.py`) generates response directly from retrieved context.

4.  **Delivery:**
    *   **Web UI:** Gradio ChatInterface (`rag_engine/api/app.py`).
    *   **API:** REST endpoint `/api/query_rag`.
    *   **Widget:** Embeddable HTML/JS widget (`ui/gradio-embedded.html`).

### 2.3 Inference Server Configuration

**Option A: MOSEC (Unified Server)**
*   **Command:** `cmw-mosec serve`
*   **Port:** 8001 (default)
*   **Models:** Embedding, Reranker, Guard loaded dynamically.
*   **Pros:** Single process, single port, efficient resource sharing.
*   **Cons:** Limited to models supported by MOSEC runner.

**Option B: vLLM (Distributed Instances)**
*   **Command:** `cmw-vllm start --model <model_id> --port <port>`
*   **Ports:** Separate ports per model (e.g., 8100, 8101, 8105).
*   **Models:** LLMs, Embedding (Qwen3), Reranker, Guard.
*   **Pros:** Supports vLLM optimizations (KV cache, continuous batching), flexible model selection.
*   **Cons:** Higher VRAM overhead, multiple processes.

---

## 3. AI Retirement & Decommissioning Methodology

### 3.1 Data Retirement
*   **ChromaDB Maintenance:** Scripts `maintain_chroma.py` and `inspect_db_schema.py` allow diagnostics, cleanup, and migration.
*   **Vector Store Deletion:** Collections can be deleted via ChromaDB HTTP API or Python client.
*   **Document Archiving:** Source documents (Markdown) remain in filesystem; no vector data is lost if source is preserved.

### 3.2 Model Retirement
*   **Configuration Update:** Switch model IDs in `.env` or `config/models.yaml`.
*   **Hot Reloading:** MOSEC supports dynamic model loading/unloading (restart required for vLLM).
*   **Versioning:** Models tracked via HuggingFace Hub; rollback by changing model ID.

### 3.3 Infrastructure Retirement
*   **Process Termination:** `cmw-mosec stop` or `cmw-vllm stop`.
*   **Container Shutdown:** If deployed via Docker, standard container stop/remove commands.
*   **Resource Cleanup:** GPU memory freed upon process termination; ChromaDB data persists on disk until manually deleted.

---

## 4. Production Best Practices (2026)

Based on the "Advanced RAG Approaches" research:

1.  **Hybrid Retrieval:** Implement BM25 + Dense retrieval for enterprise-grade accuracy (4-7.5% gain).
2.  **Adaptive Routing:** Use query complexity analysis to route simple queries directly to LLM, avoiding unnecessary retrieval.
3.  **Self-Correction:** Implement critique mechanisms for complex queries to reduce hallucinations.
4.  **Monitoring:** Track retrieval precision, context relevance, and hallucination rates.

---

## 5. Recommendations

1.  **For New Implementations:**
    *   Start with `cmw-mosec` for simplicity (single server).
    *   Use Agent Mode in `cmw-rag` for dynamic tool calling.
    *   Implement Hybrid Retrieval (BM25 + Vector) for optimal results.

2.  **For Scaling:**
    *   Move to `cmw-vllm` for LLM inference (better performance).
    *   Use separate vLLM instances for embedding/reranker/guard to distribute load.
    *   Consider Kubernetes for orchestration if multi-node scaling is required.

3.  **For Retirement:**
    *   Archive source documents before deleting vector data.
    *   Use `maintain_chroma.py` to diagnose database health before shutdown.

---

**Conclusion:** The CMW AI ecosystem provides a robust, modular methodology for implementing and managing production RAG systems. The architecture supports flexible deployment (unified vs. distributed) and includes tools for maintenance and retirement.
