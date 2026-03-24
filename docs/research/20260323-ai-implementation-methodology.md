# Executive Summary: AI Implementation Methodology for CMW Ecosystem

**Repositories:** cmw-rag, cmw-mosec, cmw-vllm
**Date:** 2026-03-23
**Prepared for:** Technical Management
**Status:** Final

---

## 1. Executive Overview

This document describes the implementation methodology and AI infrastructure management in the CMW ecosystem, focusing on Russian cloud providers and local inference. The architecture follows a **modular, containerized approach**, combining the RAG engine (cmw-rag) with specialized inference servers (cmw-mosec, cmw-vllm) and integration with Russian cloud platforms.

**Key Methodological Principles:**
- **Separation of Concerns:** Separate layers for data processing, retrieval, inference, and API delivery.
- **Hybrid Search:** Combining vector (dense) with keyword (sparse) search for optimal accuracy.
- **Agent Architecture:** Using LangChain agents for dynamic tool invocation and reasoning.
- **Infrastructure Flexibility:** Support for both MOSEC (unified server) and vLLM (distributed instances) inference backends.
- **Russian Sovereignty:** Prioritizing Russian cloud providers (Cloud.ru, Yandex Cloud, SberCloud) for data and infrastructure compliance.

---

## 2. Implementation Architecture

### 2.1 Core Components

| Component | Repository | Role | Technology |
|-----------|------------|------|------------|
| **RAG Engine** | `cmw-rag` | Retrieval, generation, and agent logic orchestration | Python, LangChain, Gradio |
| **Inference Server (Unified)** | `cmw-mosec` | Serving embedding, reranker, and guard models on one port | MOSEC, PyTorch |
| **Inference Server (Distributed)** | `cmw-vllm` | Serving LLM and pooling models via vLLM | vLLM, CUDA |
| **Vector Storage** | `cmw-rag` | Persistent document embedding storage | ChromaDB (HTTP) |

### 2.2 Data Flow and Pipeline

1.  **Ingestion:**
    *   Documents (Markdown, MkDocs) processed via `rag_engine/core/document_processor.py`.
    *   Chunked via `rag_engine/core/chunker.py` (token-dependent).
    *   Embedded via `rag_engine/retrieval/embedder.py` (FRIDA/Qwen3).
    *   Stored in ChromaDB via `rag_engine/storage/vector_store.py`.

2.  **Retrieval (RAG):**
    *   User query enters `rag_engine/retrieval/retriever.py`.
    *   **Vector Search:** ChromaDB retrieves top-k chunks.
    *   **Reranking:** Cross-encoder or LLM reranker (`rag_engine/retrieval/reranker.py`) refines results.
    *   **Context Assembly:** Articles restored, summarized if needed (`rag_engine/llm/summarization.py`).

3.  **Generation:**
    *   **Agent Mode (Recommended):** LangChain agent analyzes query, calls `retrieve_context` tool (forced via `tool_choice`), generates answer with citations.
    *   **Direct Mode:** LLM manager (`rag_engine/llm/llm_manager.py`) generates answer directly from retrieved context.

4.  **Delivery:**
    *   **Web Interface:** Gradio ChatInterface (`rag_engine/api/app.py`).
    *   **API:** REST endpoint `/api/query_rag`.
    *   **Widget:** Embeddable HTML/JS widget (`ui/gradio-embedded.html`).

### 2.3 Inference Server Configuration

**Option A: MOSEC (Unified Server)**
*   **Command:** `cmw-mosec serve`
*   **Port:** 8001 (default)
*   **Models:** Embedding, Reranker, Guard loaded dynamically.
*   **Pros:** Single process, one port, efficient resource sharing.
*   **Cons:** Limited to models supported by MOSEC runner.

**Option B: vLLM (Distributed Instances)**
*   **Command:** `cmw-vllm start --model <model_id> --port <port>`
*   **Ports:** Separate ports for each model (e.g., 8100, 8101, 8105).
*   **Models:** LLM, Embedding (Qwen3), Reranker, Guard.
*   **Pros:** vLLM optimizations support (KV-cache, continuous batching), flexible model selection.
*   **Cons:** Higher VRAM overhead, multiple processes.

### 2.4 Russian AI Cloud Providers

For data and infrastructure compliance in Russia, use local cloud platforms:

**Cloud.ru (Evolution Foundation Models)** [[source]](https://cloud.ru/products/evolution-foundation-models)
*   **Available Models:** GigaChat-2-Max, GigaChat3-10B-A1.8B, Qwen3-235B-A22B-Instruct-2507, GLM-4.6, MiniMax-M2 [[source]](https://cloud.ru/documents/tariffs/evolution/foundation-models)
*   **Pricing:** Pay-per-token (input/output), from ₽10/million tokens (GigaChat3-10B-A1.8B) [[source]](https://cloud.ru/documents/tariffs/evolution/foundation-models)
*   **Infrastructure:** Russian infrastructure, data compliance

**Yandex Cloud (YandexGPT)** [[source]](https://aistudio.yandex.ru/docs/ru/ai-studio/pricing.html)
*   **Available Models:** YandexGPT Pro 5.1, Alice AI LLM
*   **Pricing:** ~50 kopecks per 1,000 tokens [[source]](https://www.akm.ru/eng/press/yandex-b2b-tech-has-opened-access-to-the-largest-language-model-on-the-russian-market/)
*   **Features:** Yandex ecosystem integration, Russian language support

**SberCloud (GigaChat API)** [[source]](https://developers.sber.ru/portal/products/gigachat-api)
*   **Available Models:** GigaChat-2 Lite, GigaChat-2 Pro, GigaChat-2 Max
*   **Pricing:** Token packages from ₽19,500 for 300M tokens (GigaChat 2 Lite) [[source]](https://developers.sber.ru/docs/ru/gigachat/tariffs/legal-tariffs)
*   **Features:** Developed by Sber, optimized for Russian language

---

## 3. AI Disposal Methodology

### 3.1 Data Disposal
*   **ChromaDB Support:** `maintain_chroma.py` and `inspect_db_schema.py` scripts allow diagnosis, cleanup, and migration.
*   **Vector Store Deletion:** Collections can be deleted via ChromaDB HTTP API or Python client.
*   **Document Archival:** Source documents (Markdown) remain in filesystem; vector data not lost if source preserved.

### 3.2 Model Disposal
*   **Configuration Update:** Change model IDs in `.env` or `config/models.yaml`.
*   **Hot Reload:** MOSEC supports dynamic model loading/unloading (vLLM requires restart).
*   **Versioning:** Models tracked via HuggingFace Hub; rollback via model ID change.

### 3.3 Infrastructure Disposal
*   **Process Termination:** `cmw-mosec stop` or `cmw-vllm stop`.
*   **Container Termination:** For Docker deployment, standard container stop/remove commands.
*   **Resource Cleanup:** GPU memory freed on process termination; ChromaDB data persists on disk until manual deletion.

---

## 4. Production Recommendations (2026)

Based on "Advanced RAG Approaches" research:

1.  **Hybrid Search:** Implement BM25 + Dense search for enterprise-level accuracy (4-7.5% improvement).
2.  **Adaptive Routing:** Use query complexity analysis to directly route simple queries to LLM, avoiding unnecessary retrieval.
3.  **Self-Correction:** Implement critique mechanisms for complex queries to reduce hallucinations.
4.  **Monitoring:** Track retrieval accuracy, context relevance, and hallucination rates.

---

## 5. Recommendations

1.  **For new deployments:**
    *   Start with `cmw-mosec` for simplicity (unified server).
    *   Use agent mode in `cmw-rag` for dynamic tool invocation.
    *   Implement hybrid search (BM25 + Vector) for optimal results.

2.  **For scaling:**
    *   Migrate to `cmw-vllm` for LLM inference (better performance).
    *   Use separate vLLM instances for embedding/reranker/guard to distribute load.
    *   Consider Kubernetes for orchestration when scaling to multiple nodes.

3.  **For disposal:**
    *   Archive source documents before deleting vector data.
    *   Use `maintain_chroma.py` to diagnose database state before shutdown.

---

## 6. RAG Best Practices from NeuralDeep Community

Based on **NeuralDeep** channel (t.me/neuraldeep) and expert community:

### 6.1 ETL and Data Preparation [[source]](https://t.me/neuraldeep/1758)

*   **markitdown** — convert documents to Markdown [[GitHub]](https://github.com/microsoft/markitdown)
*   **marker** — fast PDF text extraction [[GitHub]](https://github.com/datalab-to/marker)
*   **docling** — advanced document data extraction [[GitHub]](https://github.com/docling-project/docling)

### 6.2 Chunking [[source]](https://habr.com/ru/companies/raft/articles/954158/)

*   **Chonkie** — fast and lightweight chunking library [[GitHub]](https://github.com/chonkie-inc/chonkie)
*   LangChain text splitters [[GitHub]](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters)

### 6.3 Vector Models for Russian Language [[source]](https://t.me/neuraldeep/1758)

*   **ai-forever/FRIDA** — Russian-optimized model
*   **BAAI/bge-m3** — multilingual model
*   **intfloat/multilingual-e5-large** — multilingual embeddings
*   **Qwen3-Embedding-8B** — large multilingual model

### 6.4 LLM and vLLM Models for Russian Segment [[source]](https://t.me/neuraldeep/1758)

**Community price/quality recommendations:**

*   **t-tech/T-lite-it-1.0** — lightweight Russian language model
*   **t-tech/T-pro-it-2.0** — advanced Russian language model
*   **Qwen3-30B-A3B-Instruct-2507** — recommended for Agentic RAG [[GitHub]](https://github.com/vamplabAI/sgr-agent-core/tree/tool-confluence)
*   **RefalMachine/RuadaptQwen2.5-14B-Instruct** — Russian-adapted

### 6.5 Rerankers [[source]](https://t.me/neuraldeep/1758)

*   **BAAI/bge-reranker-v2-m3** — multilingual cross-encoder
*   **Qwen3-Reranker-8B** — large reranking model

### 6.6 RAG Frameworks [[source]](https://t.me/neuraldeep/1758)

Community-approved:
*   **Dify** — Low-code AI app platform [[GitHub]](https://github.com/langgenius/dify/)
*   **AutoRAG** — automatic RAG optimizer [[GitHub]](https://github.com/Marker-Inc-Korea/AutoRAG)
*   **LlamaIndex** — structured data work [[GitHub]](https://github.com/run-llama/llama_index)
*   **Mastra** — production AI framework [[GitHub]](https://github.com/mastra-ai/mastra)

### 6.7 Agentic RAG Architecture [[source]](https://t.me/neuraldeep/1605)

**SGR (Schema-Guided Reasoning)** — agent framework from neuraldeep:
*   SGR Agent Core [[GitHub]](https://github.com/vamplabAI/sgr-agent-core) — 1k+ stars
*   Launch and philosophy | SGR vs Tools | Benchmarks
*   Agentic RAG on local models (Qwen3-30B-A3B)

### 6.8 Evaluation (Eval) [[source]](https://t.me/neuraldeep/1758)

*   **RAGAS** — RAG metrics [[Docs]](https://docs.ragas.io/en/stable/)
*   **ARES** — automatic RAG evaluation [[GitHub]](https://github.com/stanford-futuredata/ARES)

### 6.9 Security [[source]](https://t.me/neuraldeep/1758)

*   **NVIDIA NeMo Guardrails** — keeping bot on topic [[GitHub]](https://github.com/NVIDIA-NeMo/Guardrails)
*   **Lakera / Rebuff** — injection detectors [[Platform]](https://platform.lakera.ai/) [[GitHub]](https://github.com/protectai/rebuff)
*   **Garak** — LLM vulnerability scanner [[GitHub]](https://github.com/NVIDIA/garak)

### 6.10 Case Study: RAG for FSC (Construction Company) [[source]](https://habr.com/ru/companies/redmadrobot/articles/892882/)

*   **Task:** RAG chatbot for FSC (5M+ tokens) — B2B
*   **Result:** Support team workload reduced by **30-40%**
*   **Architecture:** Router component + two AI agent workflows
*   **Focus:** Hallucination prevention for reputational risk minimization

---

## 7. Next-Generation RAG Architectures from @ai_archnadzor Channel

Based on **@ai_archnadzor** channel materials (practicing AI architect):

### 7.1 Semantic Gravity Framework: Physics Against LLM Hallucinations [[source]](https://t.me/ai_archnadzor/155)

**Problem:** LLM "Yes Man" syndrome — model tries to help even with harmful requests, or hallucinates to "please" user queries.

**Solution:** Use physics and geometry instead of prompt persuasion:

**Geometry as Truth Map:**
1.  **Matryoshka Slicing:** Trim vectors to 256 dimensions (from 1536) — 83% faster calculations without quality loss.
2.  **SGI (Semantic Grounding Index):** Measures where model "leans":
    *   If response (R) is too close to query (Q) but far from context (C) → Sycophancy
    *   SGI = Distance(R, Q) / Distance(R, C)
    *   If SGI < 1.0 → Agent "hallucinates"

**Physics as Engine:**
*   Chain-of-Thought viewed as particle motion in energy field
*   High energy = Conflict between query and context
*   Dynamic Beta: "Temperature" of system (lower for slang, higher for legal questions)

**Feedback Loop:**
1.  LLM generates thought
2.  Physics Engine calculates SGI and Energy
3.  If energy too high → thought rejected
4.  System message injected: "Previous thought rejected. Stick strictly to facts."
5.  LLM retries

**Results:** 100% Safety Compliance in tests across 10 industries.

### 7.2 GraphOS for RAG: 16-Layer Architecture [[source]](https://t.me/ai_archnadzor/151)

**Architecture:**
*   Request Routers — 47% cost savings
*   Three-tier Memory (Redis + Neo4j)
*   Full Observability stack

### 7.3 Nested Learning: Transformer 2.0 [[source]](https://t.me/ai_archnadzor/157)

Google DeepMind (NeurIPS 2025) concept:
*   Models with "fast weights" for current task and "slow" for fundamental knowledge
*   HOPE module (Higher-Order Processing Engine)
*   Continual Learning capability

### 7.4 LEANN: World's Smallest Vector Index [[source]](https://t.me/ai_archnadzor/161)

**Innovation:** On-demand embedding computation instead of stored vectors

**Results:**
*   60M text chunks indexed
*   Classic VectorDB: ~201 GB
*   LEANN: ~6 GB (97% reduction)

**Use cases:**
*   True Offline-First
*   Privacy: 100% local without cloud data transmission
*   Integrates with Claude Code and Ollama

### 7.5 Hybrid Search: 7 Patterns for High Relevance [[source]](https://t.me/ai_archnadzor/162)

1.  Native RRF in Elasticsearch/OpenSearch
2.  Postgres: "one table — two indexes" (tsvector + pgvector)
3.  Parallel queries at App-layer (asyncio.gather)
4.  Filters-first: metadata filters before search
5.  Lightweight lexical reranking (BM25F)
6.  Dynamic k budget (80% BM25 for exact, 80% vector for fuzzy)
7.  Streaming results for perceived performance

### 7.6 30B Model on Raspberry Pi [[source]](https://t.me/ai_archnadzor/163)

**Performance:** Qwen3-30B-A3B-Instruct on Raspberry Pi 5: 8-8.5 tokens/sec (human reading speed)

**Key insight:** 5-bit or mixed-bit formats often faster than 4-bit due to better llama.cpp kernels

### 7.7 OpenClaw (ex-Moltbot): Self-Hosted AI Agent [[source]](https://t.me/ai_archnadzor/165)

**GitHub:** 170,000 stars in one week

**Architecture (4 layers):**
1.  Gateway: WebSocket routing
2.  Runtime: State management, tool execution
3.  Providers: Unified API for Cloud/Local models
4.  Integration: Channel adapters (Telegram, Slack, Discord, WhatsApp)

**Security:**
*   Docker isolation required
*   Enable auth.mode: "token" and sandbox: "docker"
*   Security audit: `openclaw security audit --deep`

### 7.8 Perplexica: Open Source Perplexity Clone [[source]](https://t.me/ai_archnadzor/166)

**Architecture:**
1.  Intent Classification
2.  Search Query Generation
3.  Aggregated Search (SearXNG)
4.  Reranking (Embeddings)
5.  Answer Generation with Citations

**Features:** Ollama support (full local), API-first (/api/search, /api/chat)

### 7.9 Local LLMs for Coding: Replacing Claude Code [[source]](https://t.me/ai_archnadzor/167)

**Top models for local coding:**
*   **Qwen3-Coder-Next** (MoE, ~80B → 3B active) — VRAM: ~24 GB (Q4_K_M)
*   **GLM-4.7-Flash** (Code MoE, ~30B → 3B active) — Context up to 200K tokens
*   **CodeGemma v1.1** (9B/27B) — Clean syntax, VRAM: 8 GB
*   **Phi-4-mini-instruct** (3.8B) — Autocomplete on phones
*   **IBM Granite-20B-Code** — Enterprise compliance

**Deployment strategy:** Brain + Edge (heavy model server + micro local autocomplete)

### 7.10 Guardrails: #1 Architectural Pattern for Agent Era [[source]](https://t.me/ai_archnadzor/168)

**2026 Stack:**
*   **Guardrails AI (v2.x):** Structural validation, cross-model hallucination audit
*   **NeMo Guardrails:** Dialog flow control, topic restriction
*   **Llama Guard 4 / ShieldGemma:** Real-time safety classifiers

**Monitoring:** HiveTrace for deep agentic tracing, Chain-of-Thought visualization, safety observability

**Layer Architecture:**
1.  Input Rails: Prompt injection protection
2.  Logic/Tool Rails: Parameter validation before function calls
3.  Output Rails: Brand policy compliance, hallucination check
4.  HiveTrace: Full visibility and debugging

### 7.11 Top-10 Graph DBs for GraphRAG [[source]](https://t.me/ai_archnadzor/169)

**Open-Source (Self-managed):**
1.  **FalkorDB** — 496x faster, 6x less RAM than Neo4j
2.  **Neo4j** — Index-free adjacency, APOC library
3.  **Memgraph** — In-memory C++, MAGE library, native vector search
4.  **NebulaGraph** — Distributed, trillions of edges
5.  **ArangoDB** — Multi-model (Graph + Document + KV)

**Closed-Source/Enterprise:**
6.  **Amazon Neptune** — Serverless, multi-stack (Gremlin/openCypher/SPARQL)
7.  **Neo4j Enterprise** — Sharding, RBAC, online backups
8.  **TigerGraph** — MPP, 10x compression, GSQL
9.  **AllegroGraph** — RDF, Neuro-symbolic AI, per-triple security
10. **Azure Cosmos DB** — Global distribution, guaranteed SLA

### 7.12 EffGen: Native Agentic Framework for SLM [[source]](https://t.me/ai_archnadzor/171)

**Problem:** LangChain/AutoGen "waste" SLM context with framework overhead

**Results on 13 benchmarks:**
*   1.5B model + EffGen beats LangChain/AutoGen
*   +11.2% efficiency for 1.5B models
*   +2.4% efficiency for 32B models

**Architecture (4 pillars):**
1.  **Prompt Compression:** 70-80% reduction
2.  **Complexity-Based Routing:** 5-factor triage
3.  **Intelligent Task Decomposition:** Parallel subtasks
4.  **Unified Memory System:** Short/long-term + vector store

**Protocols:** MCP + A2A + ACP support

### 7.13 5 AI Agent Architectures [[source]](https://t.me/ai_archnadzor/173)

1.  **Reactive:** Stimulus → Reaction (thermostat metaphor)
2.  **Deliberative:** Think first, then act (chess player)
3.  **Hybrid:** Fast reflexes + deep planning (car driver)
4.  **BDI (Cognitive):** Beliefs, Desires, Intentions
5.  **Multi-Agent Systems (MAS):** Team of specialists

### 7.14 GenAI in Production: Technology Manifesto [[source]](https://t.me/ai_archnadzor/175)

**5-Layer Architecture:**

1.  **Retrieval Layer (RAG 2.0):**
    *   Ingestion: Unstructured.io / LlamaParse
    *   Hybrid Search: BM25 + Vector
    *   Reranking: BGE-Reranker / Cohere
    *   Query Expansion: Multi-query generation

2.  **Orchestration Layer:**
    *   State Management: LangGraph (not linear chains)
    *   Tool Calling: Native JSON-output (instructor/outlines)
    *   Router: Fast model for query classification

3.  **Prompt Engineering as Code:**
    *   Templating: Jinja2
    *   Optimization: DSPy
    *   Structured Output: JSON Schema

4.  **Evaluation & Guardrails:**
    *   Metrics: RAGAS / G-Eval
    *   Safety: NeMo Guardrails / Pydantic Guard
    *   HITL: Human confirmation for critical actions

5.  **Infrastructure:**
    *   Inference: vLLM / TGI
    *   Caching: GPTCache (Redis)
    *   Observability: Arize Phoenix / LangSmith

**Gold Standard Stack:**
*   LLM: Claude 3.5 Sonnet / Llama 3.1 70B (local)
*   Orchestrator: LangGraph
*   Vector DB: Qdrant / pgvector
*   Eval: DeepEval
*   Monitoring: Arize Phoenix

### 7.15 OpenClaw + Ollama: Desktop Autonomous Agent [[source]](https://t.me/ai_archnadzor/176)

**Setup:**
```bash
# Install
curl -fsSL https://molt.bot/install.sh | bash -s -- --install-method git

# Model
ollama pull gpt-oss:20b
ollama launch openclaw

# Onboarding
npm install -g openclaw@latest
openclaw onboard --install-daemon
```

**Security:**
*   Regular: `openclaw security audit --deep`
*   Isolation: Run in VM/container
*   Skill Check: 26% of third-party skills have vulnerabilities

### 7.16 Full Local Observability Stack [[source]](https://t.me/ai_archnadzor/177)

**Stack:** vLLM + LangGraph + Arize Phoenix

**Setup:**
```bash
pip install arize-phoenix-otel openinference-instrumentation-langchain langchain-openai

from phoenix.otel import register
tracer_provider = register(
    project_name="local-agent-research",
    endpoint="http://localhost:6006/v1/traces",
    auto_instrument=True
)
```

**What you get:** Span & Trace visualization, tool input/output, token and latency metrics

### 7.17 REFRAG: 30x RAG Speedup [[source]](https://t.me/ai_archnadzor/178)

**Problem:** Large context kills TTFT and "eats" KV-Cache

**Solution:** Compress raw chunks into compact embeddings via RoBERTa + selective expansion via RL policy

**Results:**
*   TTFT speedup: 30.85x vs native Llama-3-7B
*   KV-Cache savings: 16x context extension on same hardware
*   Quality: Maintained or improved

**Verdict:** For Tier-1 systems with millions of queries

### 7.18 Cog-RAG: Hypergraphs and "Thematic" Thinking [[source]](https://t.me/ai_archnadzor/179)

**Concept:** Dual hypergraphs (themes + entities) mimicking human "general to specific" approach

**Results:**
*   Win Rate: 84.5% vs Naive RAG
*   Best improvement: Medicine (neurology +21%, pathology +26.4%)
*   LLM-agnostic: Works consistently from GPT-4o-mini to LLaMA-3.3-70B

**Verdict:** Powerful but expensive for indexing. Ideal for medicine and science.

### 7.19 HippoRAG 2: 12x Cheaper Graph Indexing [[source]](https://t.me/ai_archnadzor/180)

**Innovation:** Dual-Node architecture (entity nodes + passage nodes)

**Economics:**
*   Indexing tokens: 9M vs 115M (12x reduction)
*   Quality: +7.1 points (MuSiQue F1), +13.9% (2Wiki R@5)

**Stack:** `pip install hipporag`

### 7.20 Topo-RAG: Conquering "Table Blindness" [[source]](https://t.me/ai_archnadzor/182)

**Problem:** Linearizing tables (Markdown/JSON) into one vector creates "semantic noise"

**Solution:** Multi-vector index (each cell gets own vector) + smart router

**Results:**
*   +18.4% nDCG@10 on hybrid queries
*   Hallucinations on numbers: 45% → 8%
*   Index size: 12.4 GB → 4.1 GB

**Verdict:** Must-have for fintech and logistics.

### 7.21 Disco-RAG: Logic Instead of "Flat Fact Soup" [[source]](https://t.me/ai_archnadzor/183)

**Concept:** Rhetorical Structure Theory (RST) for understanding arguments vs contradictions vs conditions.

**Architecture:**
1.  **Intra-chunk RST Trees:** Nucleus/Satellite relationships
2.  **Inter-chunk Rhetorical Graph:** Chunks that complement vs contradict
3.  **Discourse-Aware Planning:** Response plan based on relationship graph

**Implementation (DIY since no ready library):**
1.  Indexing: LLM parser extracts RST structure → store in vector metadata
2.  Query processing: LLM builds "rhetorical graph" after Top-K retrieval
3.  Planning: Model creates response plan before generation

**Verdict:** For tasks with high cost of logical error (jurisprudence, medicine). Transforms RAG from "reader" to "analyst".

### 7.22 DSPy 3 and GEPA: Industrial Prompt Engineering [[source]](https://t.me/ai_archnadzor/189)

**DSPy 3:** Treat LLM as computational device. Architect describes Signatures, system generates and optimizes prompt code.

**GEPA (Genetic-Pareto Prompt Optimizer):**
*   Genetic algorithms for "cross-breeding" best prompts
*   Language reflection — model analyzes its own errors in text
*   **Results:** 35x faster than MIPROv2, 9x shorter prompts, 10% more accurate

### 7.23 New "Old" OCR: NEMOTRON-PARSE, Chandra, DOTS.OCR [[source]](https://t.me/ai_archnadzor/188)

| Model | Focus | Output | For Whom |
|-------|-------|--------|----------|
| **NVIDIA Nemotron (885M)** | Speed and Enterprise RAG | Markdown / LaTeX | High-load RAG systems |
| **Chandra (~1B)** | Handwriting and accuracy | MD / JSON / HTML | Archives, digitization |
| **dots.ocr (1.7B, MIT)** | Agents and license | MD / HTML (tables) | Commercial SaaS |

### 7.24 CLI Replacing MCP: Secret AI Agent Weapon [[source]](https://t.me/ai_archnadzor/190)

**Problem with MCP:**
*   Context Bloat: 30-40% overhead from schema dumping
*   Fragile dependencies: Server, handshake, WebSocket
*   Hard testing: Can't just run one command

**Why CLI is ideal for agents:**
1.  Zero overhead: Just `--help` to understand
2.  Composition: Native pipes and jq
3.  Structured output: `--json` flag
4.  Exit Codes: 0 = success, 1 = error

**Pattern:** Binary + SKILL.md file

### 7.25 AI Agent Memory: Why Vector Search Isn't Always Enough [[source]](https://t.me/ai_archnadzor/191)

**Thesis:** Future of autonomous agents is not "flat soup" of vectors, but combination of Ontologies + Graphs.

**Why ontologies are must-have:**
*   Without: Fact "User switched to Pro plan last Tuesday" = text fragment
*   With: Structured PlanChange event linking Customer + Subscription + timestamp

**4 Types of Graph Memory:**
1.  Knowledge Graphs: Factual skeleton
2.  Hierarchical Graphs: Zoom in/out capability
3.  Temporal Graphs: Change history
4.  Hypergraphs: Complex N-ary relationships

### 7.26 Multimodal LLM: How Vision and Hearing Work [[source]](https://t.me/ai_archnadzor/192)

**Architecture:**
1.  Modal Encoders: ViT (images), Whisper (audio)
2.  Alignment Mechanism: Projector (MLP/Q-Former)
3.  LLM Backbone: Unified reasoning

**Two approaches:**
*   **Modular:** LLaVA, Qwen-VL (frozen LLM + encoder + projector)
*   **Monolithic:** Fuyu-8B (native multi-modal training)

**Use cases:**
*   Document AI 2.0: No separate OCR needed
*   VQA: Charts, medical images, defect detection
*   GUI Agents: See screen like humans, no DOM access needed

### 7.27 BitNet: 1-Bit LLM for CPU Inference [[source]](https://t.me/ai_archnadzor/189)

**Concept:** 1-bit weights for Attention/MLP layers + 8/16 bit for activations

**Why it matters:**
*   **Edge AI:** Massive models can now run locally
*   **TCO Reduction:** CPU instances much cheaper than GPU
*   **Hybrid Clusters:** Train on GPU, deploy on CPU

**Verdict:** Not "GPU killer" for training, but undermining GPU monopoly on inference.

### 7.28 Doc-to-LoRA: End of "Context Tax" [[source]](https://t.me/ai_archnadzor/191)

**Problem:** KV-Cache consumes gigabytes of VRAM for long contexts

**Solution:** Hypernetwork generates LoRA adapter from document in one pass

**Results:**
*   VRAM consumption: **12 GB → 50 MB** (99% savings)
*   Assimilation speed: **<1 second** (vs 100+ seconds for fine-tuning)
*   Requirements: **<2 GB VRAM** (vs 40+ GB for gradient methods)

---

**Conclusion:** The CMW AI ecosystem provides a robust, modular methodology for deploying and managing production RAG systems. The architecture supports flexible deployment (unified vs distributed) and includes tools for maintenance and disposal. The NeuralDeep community, @ai_archnadzor channel, and Habr best practices confirm the effectiveness of chosen approaches and point to the trend of "end of vanilla RAG era" toward composite architectures.
