# CMW-RAG Project Index

**Generated:** 2026-03-22

All plans, reports, analysis, code, tests, scripts, and configuration files in cmw-rag.

---

## Plans

| File | Date | Description |
|------|------|-------------|
| <plans/20260321-reranker-refactor.md> | Mar 21 | Reranker refactoring with vLLM/Cohere endpoint contracts |
| <plans/20260321-cmw-rag-analysis.md> | Mar 21 | Model sizing and API parameters analysis |
| <plans/20260321_reranker_instruction_optimize.md> | Mar 21 | Reranker instruction optimization session log |
| <plans/20260306-slm-regex-pipeline-report.md> | Mar 7 | SLM + Regex pipeline evaluation (20 samples, 5 models) |
| <plans/20260306-slm-evaluation-report.md> | Mar 6 | Small Language Model evaluation for Russian/English NER |
| <plans/20260302-anonymization-research-geminy.md> | Mar 2 | Gemini research on English/USA NER models for anonymization |
| <plans/20260227-anonymization-implementation-plan.md> | Mar 3 | Anonymization pipeline: 3-stage Regex → dslim → Gherman |
| <plans/20260227_cmw_api_endpoint_plan.md> | Feb 27 | CMW Platform API endpoint implementation (COMPLETED) |
| <plans/20260227_cmw_agent_pipeline_plan.md> | Feb 27 | CMW Platform agent pipeline script plan |
| <plans/20260227_cmw_enum_attribute_handling_analysis.md> | Feb 27 | CMW Platform enum attribute handling analysis |
| <plans/comindware_integration_plan.md> | Mar 3 | Comindware Platform API integration (read/create records) |
| <plans/Anonymization.ipynb> | Mar 3 | Comparative analysis of Russian PII anonymization models |
| <plans/openrouter_embedding_implementation_plan.md> | Feb 21 | Unified embedding & reranker provider support (Direct/Server/API) |
| <plans/chromadb-http-migration.md> | Feb 21 | ChromaDB HTTP migration (eliminate 21s init delay) |
| <plans/chroma_host_rename_plan.md> | Feb 21 | ChromaDB host variable rename (CHROMADB_HOST → client/server) |
| <plans/scoring_system_enhancement_plan.md> | Feb 21 | RAG scoring enhancement (rerank_score_raw preservation) |
| <plans/cacheable-system-prompt.md> | Feb 17 | Cacheable system prompt: 100% static, dynamic → user messages |
| <plans/llm-architecture-skills.md> | Feb 17 | LLM architecture patterns: 3 focused skills |
| <plans/support_resolution_plan_tool_plan.md> | Feb 14 | Support Resolution Plan (SRP) tool implementation |
| <plans/sgr_synthetic_assistant_enhancement_plan.md> | Feb 13 | SGR formatted tool result enhancement |
| <plans/reverse_normalized_rank.md> | Feb 18 | Reverse normalized rank (CANCELLED — semantic confusion) |

## Session Logs

| File | Date | Description |
|------|------|-------------|
| <plans_reranker_benchmark_20260321.md> | Mar 21 | Reranker benchmark session (ses_2efe) |
| <../session-ses_2efe.md> | Mar 21 | Reranker benchmark session (ses_2efe, duplicate) |

## Cursor Plans (`.cursor/plans/`)

| File | Description |
|------|-------------|
| <../.cursor/plans/async_chroma_retriever_f90a0838.plan.md> | Async Chroma Retriever |
| <../.cursor/plans/chat-7ca608fd.plan.md> | Conversational Memory, Citations, and Copy (Gradio-only) |
| <../.cursor/plans/chunk_size_model_cap_rewire_1e8b497d.plan.md> | Chunk size model cap rewire |
| <../.cursor/plans/context-budgeting-with-fallback-ea534041.plan.md> | Summarization-First Budgeting with Immediate Fallback |
| <../.cursor/plans/direct-ml-cc8b035d.plan.md> | Add Optional DirectML Backend With CUDA-First Fallback |
| <../.cursor/plans/floating-embedded-gradio-widget-966b8bcb.plan.md> | Floating Embedded Gradio Widget |
| <../.cursor/plans/git-timestamps-with-file-fallback-03d4dbdb.plan.md> | Three-Tier Timestamp Fallback: Frontmatter, Git, File Modification |
| <../.cursor/plans/incremental-indexing-optimization-014a8ce2.plan.md> | Incremental Indexing Optimization |
| <../.cursor/plans/mk-c94e6ce4.plan.md> | Phase 1 — MkDocs RAG MVP (Chroma + FRIDA) with Lean Reranking |
| <../.cursor/plans/mkdocs-rag-engine.plan.md> | MkDocs RAG Engine Implementation Plan |
| <../.cursor/plans/multi-a9035e3d.plan.md> | Multi-vector + LLM Query Decomposition Retrieval |
| <../.cursor/plans/phase-2-interfaces-evaluation-observability-87cc6fb0.plan.md> | Phase 2: Abstract Interfaces, Evaluation, and Observability |
| <../.cursor/plans/rag-158a02a8.plan.md> | Add and run unit tests for rag_engine (WSL, real providers) |
| <../.cursor/plans/reasoning-safety-net-alignment_5e883665.plan.md> | Align Harmony reasoning interception with think-tag safety net |
| <../.cursor/plans/refactor-app.plan.md> | Refactor rag_engine/api/app.py (eliminate ~200-250 lines of duplication) |
| <../.cursor/plans/remote-chromadb-support-and-vectorstore-interface-ed550de1.plan.md> | Remote ChromaDB Support and VectorStore Interface |
| <../.cursor/plans/remove_fast_path_token_counting_ca73f0e3.plan.md> | Remove Fast Path Token Counting |
| <../.cursor/plans/retriever-refactor.plan.md> | Retriever Refactor: Clean Separation of Concerns |
| <../.cursor/plans/sgr_planning_extended_agent_6b047977.plan.md> | SGR Planning Extended Agent |
| <../.cursor/plans/test-remote-chromadb-connection-83c91b11.plan.md> | Simple ChromaDB Remote Connection Test |
| <../.cursor/plans/wrap-rag-context-retrieval-into-langchain-tool-8f6bd171.plan.md> | Wrap RAG Context Retrieval into LangChain Tool |

## Progress Reports

| File | Date | Description |
|------|------|-------------|
| <../docs/progress_reports/20260322-reranking-boost-config.md> | Mar 22 | Reranking boost configuration (configurable via .env) |
| <../docs/progress_reports/20260321-reranker-refactor.md> | Mar 21 | Reranker refactoring (vLLM/Cohere contracts) |
| <../docs/progress_reports/opencode-troubleshooting.md> | Mar 2026 | OpenCode TUI troubleshooting guide |
| <../docs/progress_reports/2026-02-28-cmw-api-endpoint-implementation.md> | Feb 28 | CMW Platform API endpoint implementation |
| <../docs/progress_reports/2026-02-27-cmw-dynamic-category-integration.md> | Feb 27 | CMW dynamic category integration |
| <../docs/progress_reports/2026-02-25-cmw-platform-integration.md> | Feb 25 | CMW Platform integration |
| <../docs/progress_reports/2026-02-18-endpoint-refactor.md> | Feb 18 | Endpoint refactor: single source of truth |
| <../docs/progress_reports/2026-02-17-forced-tool-call-architecture.md> | Feb 17 | Forced tool call architecture for SGR/SRP tools |
| <../docs/progress_reports/2026-01-21-sgr-planning-structured-agent.md> | Jan 21 | SGR planning + structured agent callable |
| <../docs/progress_reports/2025-12-19-native-gradio-spinners.md> | Dec 19 | Native Gradio spinners implementation |
| <../docs/progress_reports/token_estimation_russian_issue_20251209.md> | Dec 9 | Token estimation issue for Russian text |
| <../docs/progress_reports/clean_architecture_refactor.md> | Nov 3 | Clean architecture refactor: agent-driven context tracking |
| <../docs/progress_reports/streaming_implementation.md> | Nov 3 | Token-level streaming implementation |
| <../docs/progress_reports/memory_and_context_management_implementation.md> | Nov 3 | Memory and context management |
| <../docs/progress_reports/accumulated_context_tracking_fix.md> | Nov 3 | Accumulated context tracking fix |
| <../docs/progress_reports/centralized_token_counting.md> | Nov 3 | Centralized token counting utility |
| <../docs/progress_reports/aggressive_memory_compression_fix.md> | Nov 3 | Aggressive memory compression fix |
| <../docs/progress_reports/tool_architecture_fix.md> | Nov 3 | Tool architecture fix: self-contained retrieve_context |
| <../docs/progress_reports/tool_result_leak_fix.md> | Nov 3 | Tool result leak fix |
| <../docs/progress_reports/tool_call_leak_fix.md> | Nov 3 | Tool call leak fix |
| <../docs/progress_reports/generic_tool_filtering_fix.md> | Nov 3 | Generic tool filtering fix |
| <../docs/progress_reports/context_fallback_implementation.md> | Nov 3 | Context window fallback implementation |
| <../docs/progress_reports/progressive_budgeting_fix.md> | Nov 3 | Progressive context budgeting fix |
| <../docs/progress_reports/article_deduplication_fix.md> | Nov 3 | Article deduplication fix |
| <../docs/progress_reports/agent_mode_implementation.md> | Nov 2 | Agent mode implementation |
| <../docs/progress_reports/agent_tool_choice_update.md> | Nov 2 | Agent mode: tool_choice parameter |
| <../docs/progress_reports/agent_final_implementation_summary.md> | Nov 2 | Agent mode: final implementation summary |
| <../docs/progress_reports/multi_tool_call_implementation.md> | Nov 2 | Multi-tool-call citation accumulation |
| <../docs/progress_reports/multi_tool_call_final_summary.md> | Nov 2 | Multi-tool-call: final summary |
| <../docs/progress_reports/tool_refactoring_completion.md> | Nov 2 | RAG context retrieval tool refactoring |
| <../docs/progress_reports/tool_implementation_validation.md> | Nov 2 | Tool implementation validation |
| <../docs/progress_reports/gradio_metadata_implementation.md> | Nov 2 | Gradio metadata messages for tool execution |
| <../docs/progress_reports/2025-10-28-dynamic-token-limits-and-llm-mechanics-reuse.md> | Oct 28 | Dynamic token limits & LLM mechanics reuse |
| <../docs/progress_reports/2025-10-28-testing-and-verification.md> | Oct 28 | Testing & verification report |
| <../docs/progress_reports/2025-10-28-implementation-audit-and-fixes.md> | Oct 28 | Implementation audit & fixes |
| <../docs/progress_reports/2025-01-28-indexer-separation-validation.md> | Jan 28 | Indexer separation validation |
| <../docs/progress_reports/VALIDATION_REPORT.md> | — | Async ChromaDB retriever validation (PASSED) |
| <../docs/progress_reports/dynamic_tool_result_compression.md> | — | Dynamic tool result compression |
| <../docs/progress_reports/json_overhead_safety_margin.md> | — | JSON overhead safety margin |
| <../docs/progress_reports/typed_context_refactor.md> | — | Typed context refactor |
| <../docs/progress_reports/summary_context_overflow_fix.md> | — | Context overflow fix summary |
| <../docs/progress_reports/summary_typed_context_and_json_overhead.md> | — | Typed context + JSON overhead summary |
| <rag_engine/docs/progress_reports/20260321-embedding-instruction-fix-applied.md> | Mar 21 | Embedding instruction fix applied to models.yaml |

## Analysis

| File | Date | Description |
|------|------|-------------|
| <../docs/analysis/context_compression_analysis_20251209.md> | Dec 9 | Context compression analysis for small context windows |
| <../docs/analysis/memory_management_limits_explanation.md> | Jan 28 | Memory management limits: timing, precedence, defaults |
| <../docs/analysis/retriever-refactor-validation.md> | Jan 28 | Retriever refactor plan validation |
| <../docs/analysis/system_prompt_redundancy_analysis.md> | Jan 28 | System prompt redundancy analysis |
| <rag_engine/docs/analysis/20260321-qwen3-reranker-instruction-optimization-report.md> | Mar 21 | Reranker instruction research (30+ variants tested) |
| <rag_engine/docs/analysis/20260321-reranker-benchmark-final.md> | Mar 21 | Reranker benchmark (52 questions, 38 Russian) |
| <rag_engine/docs/analysis/20260321-reranker-instruction-analysis.md> | Mar 21 | Qwen3 reranker instruction optimization (10,685 docs in ChromaDB) |

### Experiments

| Directory | Description |
|-----------|-------------|
| <../docs/analysis/experiments/2026-02-19-backend-inference-tests/> | Backend inference tests (mosec vs vLLM comparisons) |
| <../docs/analysis/experiments/2026-02-20-qwen3-embedding-validation/> | Qwen3 embedding validation test scripts |

### rag_engine/docs/plans

| File | Date | Description |
|------|------|-------------|
| <rag_engine/docs/plans/20260321-reranker-instruction-optimization-final-report.md> | Mar 21 | Qwen3 reranker instruction optimization final report (58-65% improvement) |
| <rag_engine/docs/plans/20260321-file-inventory-reranker-embedding-experiments.md> | Mar 21 | Inventory of reranker and embedding experiment files |

---

## Core Code (`rag_engine/`)

### core/

| File | Description |
|------|-------------|
| <../rag_engine/core/chunker.py> | Token-aware, code-safe chunker |
| <../rag_engine/core/document_processor.py> | Unified document processor (MkDocs/folders/single files) |
| <../rag_engine/core/indexer.py> | RAG indexer: chunk, embed, write to vector store (async) |
| <../rag_engine/core/metadata_enricher.py> | Minimal metadata enrichment utilities |
| <../rag_engine/core/guard_client.py> | Guardian client for content moderation |
| <../rag_engine/core/vllm_guard_adapter.py> | vLLM guard adapter |

### retrieval/

| File | Description |
|------|-------------|
| <../rag_engine/retrieval/embedder.py> | Unified embedding provider (Direct/OpenAI-compatible) |
| <../rag_engine/retrieval/reranker.py> | Unified reranker provider (Direct/Server) |
| <../rag_engine/retrieval/retriever.py> | RAG retriever: search, rerank, load articles (async) |
| <../rag_engine/retrieval/confidence.py> | Retrieval confidence metrics from reranker scores |
| <../rag_engine/retrieval/vector_search.py> | Vector search wrapper (async only) |

### llm/

| File | Description |
|------|-------------|
| <../rag_engine/llm/llm_manager.py> | LLM manager with dynamic token limits |
| <../rag_engine/llm/agent_factory.py> | Agent factory for RAG agents (LangChain) |
| <../rag_engine/llm/prompts.py> | System/user prompts |
| <../rag_engine/llm/schemas.py> | Pydantic schemas for structured agent outputs |
| <../rag_engine/llm/compression.py> | Context compression utilities |
| <../rag_engine/llm/fallback.py> | Model fallback management |
| <../rag_engine/llm/summarization.py> | Summarization logic |
| <../rag_engine/llm/token_utils.py> | Token counting utilities |
| <../rag_engine/llm/usage_accounting.py> | Usage/cost accounting |
| <../rag_engine/llm/openrouter_native.py> | OpenRouter chat model via native OpenAI SDK |
| <../rag_engine/llm/model_configs.py> | Centralized model configuration registry |

### tools/

| File | Description |
|------|-------------|
| <../rag_engine/tools/retrieve_context.py> | RAG context retrieval tool (self-sufficient) |
| <../rag_engine/tools/analyse_user_request.py> | User request analysis tool (SGR) |
| <../rag_engine/tools/generate_resolution_plan.py> | Support Resolution Plan tool (SRP) |
| <../rag_engine/tools/get_datetime.py> | Current date/time tool |
| <../rag_engine/tools/math_tools.py> | Math tools for arithmetic operations |
| <../rag_engine/tools/utils.py> | Tool result parsing utilities |

### storage/

| File | Description |
|------|-------------|
| <../rag_engine/storage/vector_store.py> | Chroma vector store wrapper (async HTTP client) |

### api/

| File | Description |
|------|-------------|
| <../rag_engine/api/app.py> | Main Gradio UI + REST API |
| <../rag_engine/api/harmony_parser.py> | GPT-OSS Harmony format streaming parser |
| <../rag_engine/api/i18n.py> | Internationalization (ru/en) |
| <../rag_engine/api/stream_helpers.py> | Streaming and UI metadata helpers |

### config/

| File | Description |
|------|-------------|
| <../rag_engine/config/schemas.py> | Pydantic schemas for embedding/reranker configs |
| <../rag_engine/config/settings.py> | Application settings from .env |
| <../rag_engine/config/loader.py> | Config loader (deprecated, use schemas.ModelRegistry) |

### cmw_platform/

| File | Description |
|------|-------------|
| <../rag_engine/cmw_platform/api.py> | CMW API client |
| <../rag_engine/cmw_platform/connector.py> | Platform connector |
| <../rag_engine/cmw_platform/records.py> | Record operations |
| <../rag_engine/cmw_platform/request_builder.py> | Request builder |
| <../rag_engine/cmw_platform/mapping.py> | Field mapping |
| <../rag_engine/cmw_platform/models.py> | CMW data models |
| <../rag_engine/cmw_platform/config.py> | CMW config |
| <../rag_engine/cmw_platform/attribute_types.py> | Attribute type definitions |
| <../rag_engine/cmw_platform/category_enum.py> | Category enums |

### utils/

| File | Description |
|------|-------------|
| <../rag_engine/utils/context_tracker.py> | Context tracking |
| <../rag_engine/utils/conversation_store.py> | Conversation storage |
| <../rag_engine/utils/logging_manager.py> | Logging manager |
| <../rag_engine/utils/formatters.py> | Output formatters |
| <../rag_engine/utils/message_utils.py> | Message utilities |
| <../rag_engine/utils/metadata_utils.py> | Metadata utilities |
| <../rag_engine/utils/trace_formatters.py> | Trace formatters |
| <../rag_engine/utils/device_utils.py> | Device detection |
| <../rag_engine/utils/disk_space.py> | Disk space utilities |
| <../rag_engine/utils/git_utils.py> | Git utilities |
| <../rag_engine/utils/huggingface_utils.py> | HuggingFace utilities |
| <../rag_engine/utils/path_utils.py> | Path utilities |
| <../rag_engine/utils/thread_pool.py> | Thread pool |
| <../rag_engine/utils/vllm_fallback.py> | vLLM fallback |

## Tests (`rag_engine/tests/`)

56 test files:

<../rag_engine/tests/conftest.py> — Fixtures and configuration
<../rag_engine/tests/test_smoke.py> — Smoke tests
<../rag_engine/tests/test_agent_factory.py> — Agent factory
<../rag_engine/tests/test_agent_handler.py> — Agent handler
<../rag_engine/tests/test_api_app.py> — Gradio API app
<../rag_engine/tests/test_async_retrieval_integration.py> — Async retrieval
<../rag_engine/tests/test_chat_with_metadata_analysis.py> — Chat metadata
<../rag_engine/tests/test_cmw_platform.py> — CMW Platform
<../rag_engine/tests/test_config_loader.py> — Config loader
<../rag_engine/tests/test_config_settings.py> — Config settings
<../rag_engine/tests/test_core_chunker.py> — Chunker
<../rag_engine/tests/test_core_document_processor.py> — Document processor
<../rag_engine/tests/test_core_indexer.py> — Indexer
<../rag_engine/tests/test_core_metadata_enricher.py> — Metadata enricher
<../rag_engine/tests/test_embedder_factory.py> — Embedder factory
<../rag_engine/tests/test_guard_client.py> — Guard client
<../rag_engine/tests/test_guardian_xml_format.py> — Guardian XML format
<../rag_engine/tests/test_llm_manager.py> — LLM manager
<../rag_engine/tests/test_llm_prompts.py> — LLM prompts
<../rag_engine/tests/test_llm_summarization.py> — LLM summarization
<../rag_engine/tests/test_llm_token_utils.py> — Token utils
<../rag_engine/tests/test_llm_integration_openrouter_vllm.py> — OpenRouter vLLM
<../rag_engine/tests/test_reranker_contracts.py> — Reranker contracts
<../rag_engine/tests/test_reranker_factory.py> — Reranker factory
<../rag_engine/tests/test_retrieval_confidence.py> — Retrieval confidence
<../rag_engine/tests/test_retrieval_embedder.py> — Retrieval embedder
<../rag_engine/tests/test_retrieval_reranker.py> — Retrieval reranker
<../rag_engine/tests/test_retrieval_vector_search.py> — Vector search
<../rag_engine/tests/test_retriever.py> — Retriever
<../rag_engine/tests/test_sgr_tool.py> — SGR tool
<../rag_engine/tests/test_srp_tool.py> — SRP tool
<../rag_engine/tests/test_structured_agent.py> — Structured agent
<../rag_engine/tests/test_storage_vector_store.py> — Vector store
<../rag_engine/tests/test_tools_retrieve_context.py> — Retrieve context tool
<../rag_engine/tests/test_tools_utils.py> — Tool utils
<../rag_engine/tests/test_usage_accounting.py> — Usage accounting
<../rag_engine/tests/test_thread_safety.py> — Thread safety
<../rag_engine/tests/test_utils_conversation_store.py> — Conversation store
<../rag_engine/tests/test_utils_formatters.py> — Formatters
<../rag_engine/tests/test_utils_git_utils.py> — Git utils
<../rag_engine/tests/test_utils_logging_manager.py> — Logging manager
<../rag_engine/tests/test_utils_message_utils.py> — Message utilities
<../rag_engine/tests/test_ui_metadata.py> — UI metadata
<../rag_engine/tests/test_unified_bubble.py> — Unified bubble
<../rag_engine/tests/test_device_utils.py> — Device utils
<../rag_engine/tests/test_gradio_spinners.py> — Gradio spinners
<../rag_engine/tests/test_generating_answer_spinner.py> — Answer spinner
<../rag_engine/tests/test_spinner_lifecycle.py> — Spinner lifecycle
<../rag_engine/tests/test_path_utils.py> — Path utils
<../rag_engine/tests/test_process_requests_xlsx.py> — XLSX processing
<../rag_engine/tests/test_scripts_build_index.py> — Build index script
<../rag_engine/tests/test_scripts_run_mkdocs_export.py> — MkDocs export script
<../rag_engine/tests/test_mkdocs_hook_import.py> — MkDocs hook import
<../rag_engine/tests/test_mcp_get_knowledge_base_articles.py> — MCP KB articles

## Scripts (`rag_engine/scripts/`)

82 scripts in categories:

### Data & Indexing
<../rag_engine/scripts/build_index.py> — Build RAG index from MkDocs
<../rag_engine/scripts/process_requests_xlsx.py> — Process XLSX requests
<../rag_engine/scripts/generate_synthetic_dataset.py> — Generate synthetic dataset
<../rag_engine/scripts/enrich_synthetic_dataset.py> — Enrich synthetic dataset
<../rag_engine/scripts/run_mkdocs_export.py> — Run MkDocs export

### Reranker Benchmarking
<../rag_engine/scripts/reranker_benchmark_from_dataset.py> — Benchmark from dataset
<../rag_engine/scripts/reranker_comprehensive_benchmark.py> — Comprehensive benchmark
<../rag_engine/scripts/reranker_instruction_benchmark.py> — Instruction benchmark
<../rag_engine/scripts/reranker_batched_benchmark.py> — Batched benchmark
<../rag_engine/scripts/reranker_extended_benchmark.py> — Extended benchmark
<../rag_engine/scripts/reranker_bilingual_benchmark.py> — Bilingual benchmark
<../rag_engine/scripts/reranker_fast_benchmark.py> — Fast benchmark
<../rag_engine/scripts/reranker_quick_benchmark.py> — Quick benchmark
<../rag_engine/scripts/reranker_realistic_benchmark.py> — Realistic benchmark
<../rag_engine/scripts/reranker_semantic_benchmark.py> — Semantic benchmark
<../rag_engine/scripts/reranker_hybrid_benchmark.py> — Hybrid benchmark
<../rag_engine/scripts/reranker_continue_benchmark.py> — Continue benchmark
<../rag_engine/scripts/reranker_prepare_datasets.py> — Prepare benchmark datasets
<../rag_engine/scripts/reranker_research.py> — Reranker research
<../rag_engine/scripts/generate_benchmark_report.py> — Generate benchmark report

### Embedding
<../rag_engine/scripts/test_embedding_format.py> — Test embedding format
<../rag_engine/scripts/test_embedding_instruction.py> — Test embedding instruction
<../rag_engine/scripts/investigate_embed_rerank_instructions.py> — Investigate instructions

### Anonymization
<../rag_engine/scripts/test_anonymization_stages.py> — Test anonymization stages
<../rag_engine/scripts/test_4stage_cascade.py> — Test 4-stage cascade
<../rag_engine/scripts/evaluate_full_cascade.py> — Evaluate full cascade
<../rag_engine/scripts/evaluate_regex_baseline.py> — Evaluate regex baseline

### ChromaDB
<../rag_engine/scripts/check_chroma.py> — Check ChromaDB
<../rag_engine/scripts/maintain_chroma.py> — Maintain ChromaDB
<../rag_engine/scripts/start_chroma_server.py> — Start ChromaDB server
<../rag_engine/scripts/test_chroma_connection.py> — Test ChromaDB connection

### CMW Platform
<../rag_engine/scripts/process_cmw_record.py> — Process CMW record
<../rag_engine/scripts/process_cmw_range.py> — Process CMW range
<../rag_engine/scripts/test_cmw_platform_integration.py> — CMW integration test

### OpenRouter Probes
<../rag_engine/scripts/openrouter_langchain_reasoning_probe.py> — LangChain reasoning
<../rag_engine/scripts/openrouter_langchain_stream_usage_probe.py> — Stream usage
<../rag_engine/scripts/openrouter_langchain_usage_probe.py> — Usage probe
<../rag_engine/scripts/openrouter_multi_model_compat_test.py> — Multi-model compat
<../rag_engine/scripts/openrouter_native_full_agent_test.py> — Native agent test
<../rag_engine/scripts/openrouter_reasoning_agent_invoke_test.py> — Reasoning invoke
<../rag_engine/scripts/openrouter_reasoning_agent_stream_test.py> — Reasoning stream
<../rag_engine/scripts/openrouter_reasoning_gpt_oss_120b_test.py> — GPT-OSS 120B
<../rag_engine/scripts/openrouter_toolchoice_test.py> — Tool choice test
<../rag_engine/scripts/openrouter_usage_accounting_adhoc.py> — Usage accounting
<../rag_engine/scripts/openrouter_usage_conversation_probe.py> — Usage conversation

### SLM/Regex
<../rag_engine/scripts/test_slm_regex_pipeline.py> — SLM + Regex pipeline
<../rag_engine/scripts/test_robust_slm_regex_pipeline.py> — Robust pipeline
<../rag_engine/scripts/find_slm_models.py> — Find SLM models

### UI & Testing
<../rag_engine/scripts/demo_spinners.py> — Spinner demo
<../rag_engine/scripts/test_gradio_chatinterface_history.py> — Gradio history
<../rag_engine/scripts/test_gradio_i18n.py> — Gradio i18n
<../rag_engine/scripts/test_thinking_streaming.py> — Thinking streaming
<../rag_engine/scripts/test_vllm_tool_calling.py> — vLLM tool calling
<../rag_engine/scripts/test_conversation_stopping.py> — Conversation stopping

## Configuration

| File | Description |
|------|-------------|
| <../rag_engine/config/models.yaml> | Model registry (embedding/reranker specs) |
| <../rag_engine/config/cmw_platform.yaml> | CMW Platform pipeline config |
| <../rag_engine/config/anonymization.yaml> | Anonymization pipeline config |
| <../mkdocs_for_rag_indexing.yml> | MkDocs config for RAG indexing |
| <../pyproject.toml> | Build config (setuptools, ruff, pytest) |
| <../.env-example> | Environment variable template |

## UI (`ui/`)

| File | Description |
|------|-------------|
| <../ui/run_widget.py> | Widget runner (HTTP server for embedded Gradio) |
| <../ui/kb_proxy.py> | KB proxy for Comindware KB |
| <../ui/index.html> | Main index HTML |
| <../ui/gradio-embedded.html> | Embedded Gradio widget |
| <../ui/cmw-kb-copilot.html> | CMW KB copilot |
| <../ui/embed-snippet.html> | Embed snippet |

## Root Documentation

| File | Description |
|------|-------------|
| <../README.md> | Project documentation: RAG engine features, API, architecture |
| <../AGENTS.md> | Agent guide: conventions, testing, TDD/SDD practices |
| <../LICENSE> | License |
| <../session-ses_2efe.md> | Session log: reranker benchmark research (2026-03-21) |

## docs/ Documentation

| File | Description |
|------|-------------|
| <../docs/TESTING.md> | Testing guide |
| <../docs/MCP_CONFIGURATION.md> | MCP server configuration via Gradio |
| <../docs/ASYNC_IMPLEMENTATION_SUMMARY.md> | Async ChromaDB retriever implementation |
| <../docs/CHANGES_SUMMARY_20251208.md> | MCP tool improvements summary |
| <../docs/GRADIO_SPINNERS_IMPLEMENTATION.md> | Gradio native spinners (19/19 tests) |
| <../docs/RAG_architecture_improvements_test_task.md> | RAG architecture test task (Russian) |

### Troubleshooting

| File | Description |
|------|-------------|
| <../docs/troubleshooting/vllm-streaming-tool-calls-issue.md> | vLLM streaming tool calls (resolved) |
| <../docs/troubleshooting/wsl-disk-space.md> | WSL disk space solutions |

### Presentations

| Directory | Description |
|-----------|-------------|
| <../docs/presentations/comindware-demo-2026-03-05/> | Comindware demo slides (en/ru) |
