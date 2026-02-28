# Backend Inference Testing - 2026-02-19/20

Comprehensive testing of vLLM vs Mosec inference backends for embedding, reranking, and guard models.

**Update (2026-02-20):** vLLM now working with 8GB VRAM cap, Mosec fixed with last-token pooling support

## Test Objective

Compare three inference backends for serving embedding/reranker/guard models:
- **Direct Transformers** (ground truth)
- **vLLM** (high-performance inference engine) - ✅ Now working!
- **Mosec** (custom multi-model server) - ✅ Fixed with pooling config!

## Models Tested

1. **Qwen/Qwen3-Embedding-0.6B** - Multilingual embedding model (1024 dim, last-token pooling)
2. **Qwen/Qwen3-Reranker-0.6B** - Generative reranker (yes/no classification)
3. **Qwen/Qwen3Guard-Gen-0.6B** - Content safety guard (119 languages)
4. **DiTy/cross-encoder-russian-msmarco** - Russian cross-encoder (CLS pooling)

## Main Report

**📊 Full Analysis:** [`2026-02-19-inference-backend-comprehensive-test-report.md`](./2026-02-19-inference-backend-comprehensive-test-report.md)

The comprehensive report includes:
- Executive summary with winner recommendations
- Detailed test results for all 4 models across 3 backends
- VRAM usage and performance metrics
- **Mosec fix documentation** (last-token pooling support added)
- vLLM working configuration (8GB VRAM cap)
- Technical notes on why each backend works/fails

## Directory Structure

```
2026-02-19-backend-inference-tests/
├── 2026-02-19-inference-backend-comprehensive-test-report.md  # Main report
├── README.md                                                   # This file
├── test-scripts/                                              # Python test files
│   ├── test_comprehensive_all_models.py
│   ├── test_direct_embeddings.py
│   ├── test_final_comparison_2026-02-20.py
│   ├── test_mosec_*.py (5 files)
│   ├── test_vllm_*.py (3 files)
│   └── test_comparison_summary.py
├── data/                                                      # Test data & results
│   ├── comprehensive_test_results.json
│   ├── qwen3_direct.pkl
│   └── qwen3_emb_direct.pkl
└── logs/                                                      # Execution logs
    ├── comprehensive_test_output.log
    ├── mosec_proper_output.log
    └── vllm_proper_output.log
```

## Test Scripts

### Core Test Files
- **`test-scripts/test_comprehensive_all_models.py`** - Main comprehensive test suite (all backends)
- **`test-scripts/test_direct_embeddings.py`** - Direct Transformers ground truth
- **`test-scripts/test_final_comparison_2026-02-20.py`** - Final vLLM vs Mosec comparison (2026-02-20)

### vLLM Tests
- `test-scripts/test_vllm_thorough.py` - vLLM comprehensive tests
- `test-scripts/test_vllm_proper_instructions.py` - vLLM with proper Qwen instructions
- `test-scripts/test_vllm_pooling.py` - vLLM pooling mechanism tests

### Mosec Tests
- `test-scripts/test_mosec_server.py` - Mosec server tests
- `test-scripts/test_mosec_thorough.py` - Mosec comprehensive tests
- `test-scripts/test_mosec_proper_instructions.py` - Mosec with proper instructions

### Utilities
- `test-scripts/test_comparison_summary.py` - Comparison summary generator

## Test Data

- `data/comprehensive_test_results.json` - Machine-readable test results
- `data/qwen3_direct.pkl` - Direct Transformers embeddings (reference)
- `data/qwen3_emb_direct.pkl` - Qwen3 embedding test data

## Logs

- `logs/comprehensive_test_output.log` - Full test execution log (83KB)
- `logs/vllm_proper_output.log` - vLLM test output with proper instructions
- `logs/mosec_proper_output.log` - Mosec test output with proper instructions

## Quick Start

To reproduce tests:

```bash
# Navigate to test scripts
cd test-scripts

# Activate environment
source ../../../.venv/bin/activate

# Run comprehensive test (all backends)
python test_comprehensive_all_models.py

# Run final comparison test (2026-02-20)
python test_final_comparison_2026-02-20.py

# Run specific backend tests
python test_vllm_proper_instructions.py
python test_mosec_proper_instructions.py
python test_direct_embeddings.py
```

## Key Findings (TL;DR)

### Updated Results (2026-02-20)

#### vLLM - ✅ NOW WORKING

**Configuration:**
```python
LLM(
    model="Qwen/Qwen3-Embedding-0.6B",
    runner="pooling",
    dtype="float16",
    gpu_memory_utilization=0.17,  # 8GB cap
    enforce_eager=True,           # Disable CUDA graphs
    max_model_len=8192,
)
```

**Performance:**
- ✅ Latency: 9.0ms (4 texts)
- ✅ VRAM: ~1.1GB
- ✅ Accuracy: 99.9998% match with Direct
- ✅ Load time: 10.7s

#### Mosec - ✅ FIXED

**Changes Made:**
1. Added `pooling` field to `models.yaml`
2. Added `last_token_pool()` method
3. Worker now uses configured pooling method

**Supported Pooling:**
- `mean` - Default (BERT-style)
- `cls` - For T5/FRIDA
- `last_token` - For Qwen3 (NEW!)

#### Direct Transformers

✅ **Status:**
- Gold standard for accuracy (100%)
- Latency: 9.8ms
- VRAM: ~2.4GB
- Used for validation

## Recommendation (Updated)

**Hybrid Setup:**
- **vLLM** (8GB VRAM): Qwen3-Embedding-0.6B (best performance)
- **Mosec**: FRIDA, DiTy, Qwen3Guard (native support)
- **Direct**: Ground truth testing

## Date
2026-02-19 (Updated: 2026-02-20)

## Test Duration
~2 hours initial + 1 hour for vLLM/Mosec fixes

## Models × Backends Tested
12 combinations (4 models × 3 backends)

## Files Summary

**Total: 15 files organized in 3 subdirectories**

### Root (2 files)
- `2026-02-19-inference-backend-comprehensive-test-report.md` - Main report
- `README.md` - This documentation

### test-scripts/ (10 files)
- Python test scripts for all backends

### data/ (3 files)
- Test results and embedding data

### logs/ (3 files)
- Execution logs from test runs
