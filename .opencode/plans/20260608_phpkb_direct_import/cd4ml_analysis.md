# CD4ML Analysis: CMW RAG Content Pipeline

**Date:** 2026-06-08  
**Status:** analysis (not implemented)  
**References:**
- [Continuous Delivery for Machine Learning (CD4ML)](https://martinfowler.com/articles/cd4ml.html)
- [Ragas — Context Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)
- [Write the Docs — Docs as Code](https://www.writethedocs.org/guide/docs-as-code/)

## The 3 Axes of Change

CD4ML identifies three axes that drive change in ML systems. Applied to CMW RAG:

| Axis | CMW RAG State | Churn | Risk |
|---|---|---|---|
| **Data** — corpus articles | PHPKB articles, ~605 total, ~1-5 changes per sync | **High** | **Primary risk** — bad import silently pollutes retrieval |
| **Model** — embedding/reranker | `Qwen3-Embedding-0.6B`, fixed at deploy time | Low | Embedding dim mismatch if model changed without collection migration |
| **Code** — `cmw-rag` app | Git versioned, manually deployed | Low | Standard software risk (regressions, bugs) |

**Data is where the risk lives.** The embedding model and application code change infrequently. Every 6 hours the corpus can change. A single corrupted article, a bad HTML→Markdown conversion, or a PHPKB article with broken encoding lands directly in the production ChromaDB collection. There is no pre-promotion quality gate.

## Current Pipeline vs CD4ML Ideal

```
Current:  import → index → prod (fire-and-forget, no gate)
Ideal:    import → validate → index to staging → eval → promote | reject
```

## Gap Analysis

| CD4ML Practice | Current CMW RAG | Gap |
|---|---|---|
| **Discoverable Data** | PHPKB MySQL via SSH tunnel (planned) | Direct import closes this |
| **Reproducible Pipeline** | Content-hash-based incremental indexing, `--ff-only` git sync | ✅ Solid |
| **Data Versioning** | Git for corpus files, ChromaDB has no history | Missing: point-in-time manifest per index run |
| **Content Validation** | kbId required, empty docs skipped | Missing: frontmatter integrity gate, count drift detection |
| **Testing & Quality** | `evaluate_full_cascade.py` exists | Missing: pre-promotion eval gate in pipeline |
| **Deployment Strategy** | Direct index into production collection | Missing: staging → promotion pattern |
| **Monitoring** | Index summary logs, systemd journal | Missing: freshness metrics, drift alerts |
| **Rollback** | None automated | Missing: collection versioning for fast revert |

## Proposed Improvement Tiers

### Tier 1 — Content Integrity Gate
**Effort:** ~30 lines in `indexer.py`  
**Impact:** Catches malformed articles before they reach ChromaDB

- Validate frontmatter completeness per article (numeric kbId, non-empty title, parseable `updated` timestamp)
- Detect article count drop >20% from previous run — halt pipeline, don't promote
- Already partially in place (kbId check exists in `document_processor.py:103`)

### Tier 2 — Staging Collection + Eval Gate
**Effort:** ~80 lines in new `promote_collection` step  
**Impact:** Prevents bad content from reaching users — the single highest-value addition

Architecture using ChromaDB collection aliases (already supported by HTTP API):

```
collection_v6                    ← alias → points to active prod
collection_v6_20260608T180000    ← timestamped staging, freshly indexed
collection_v6_20260608T120000    ← previous prod (kept for rollback)
collection_v6_20260608T060000    ← older (eventually pruned)
```

Flow:
1. Index into `collection_v6_<timestamp>` (staging)
2. Run `evaluate_full_cascade.py` against golden query set
3. If recall ≥ threshold → update `collection_v6` alias to point to staging → delete old
4. If recall < threshold → keep previous prod, log alert, keep staging for inspection
5. Prune old timestamped collections (keep last 2)

Leverages existing:
- `evaluate_full_cascade.py` for eval
- `maintain_chroma.py` collection management primitives
- `Chromadb.AsyncHttpClient` for `create_collection`, `update_collection` (alias)

### Tier 3 — Manifest & Audit Trail
**Effort:** ~20 lines at end of indexer run  
**Impact:** Instant answer to "what changed in the last index run?"

Per-run `manifest_YYYYMMDDHHMMSS.json`:
```json
{
  "timestamp": "2026-06-08T18:00:00Z",
  "collection": "collection_v6_20260608T180000",
  "source": "phpkb_direct",
  "summary": {
    "total_articles": 605,
    "new": 0, "updated": 4, "skipped": 601, "deleted": 0
  },
  "articles": [
    {"kbId": 5283, "action": "updated", "title": "Calculate Role Accounts", "chunks": 2}
  ]
}
```
Store alongside the collection. Commit to git optionally. Enables diffing between runs.

### Tier 4 — Embedding Drift Detection
**Effort:** ~40 lines, post-index step  
**Impact:** Catch semantic corruption (article changed meaning unintentionally)

For each updated article:
- Retrieve old embedding from current prod collection
- Compute cosine distance to new embedding from staging
- Flag articles with distance >0.3 for human review
- High-distance articles indicate: major rewrite, broken encoding, content injection

## What NOT to Build (overengineering at 605-article scale)

| Pattern | Why skip |
|---|---|
| **Canary routing** | Needs traffic splitting infra. Zero benefit for a dev support chatbot without production traffic. |
| **DVC for data versioning** | Git is sufficient for 605 markdown files. ChromaDB already holds vector state. |
| **Multi-model shadow evaluation** | Embedding/reranker models are fixed, not retrained. No model selection needed. |
| **Push-triggered CI reindex** | The systemd timer is the update frequency. A push hook would just duplicate the timer. |
| **Online learning / multi-arm bandits** | Not applicable — this is a KB retrieval system, not a recommendation model. |
| **Full ML model monitoring (Kibana/EFK)** | Embedding model is fixed. Monitoring the corpus freshness is sufficient. |

## Summary

Two gaps separate this pipeline from a mature CD4ML content pipeline:

1. **Pre-promotion quality gate** (Tiers 1+2) — index to staging, validate integrity, run eval queries, promote only on pass. This is the RAG equivalent of a CI pipeline's test-then-deploy gate.

2. **Auditability** (Tier 3) — point-in-time manifest of what changed in each run. Enables post-hoc investigation and diffing between versions.

Everything else — content-hash incremental indexing, systemd orchestration, keyring-backed credentials, git-based corpus tracking — already follows CD4ML best practices. Tiers 1-2 are the missing layer between "a good script" and "a mature CD pipeline for content."
