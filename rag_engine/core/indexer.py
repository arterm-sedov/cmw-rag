"""RAG indexer: chunk, embed, and write documents to vector store."""
from __future__ import annotations

import json
import logging
from hashlib import sha1
from typing import Any

from rag_engine.core.chunker import split_text
from rag_engine.core.metadata_enricher import enrich_metadata
from rag_engine.utils.git_utils import get_file_timestamp
from rag_engine.utils.metadata_utils import extract_numeric_kbid

logger = logging.getLogger(__name__)


def _stable_id(kb_id: str, chunk_index: int, content: str, source_file: str | None = None) -> str:
    """Generate a globally unique but stable ID for a chunk.

    Uses a document seed derived from source_file (preferred) or kb_id to
    disambiguate chunks from different documents that might otherwise share
    kbId and chunk content at the same index.
    """
    doc_seed = (source_file or kb_id).encode("utf-8")
    doc_hash = sha1(doc_seed).hexdigest()[:8]
    text_hash = sha1(content.encode("utf-8")).hexdigest()[:10]
    return f"{doc_hash}:{chunk_index}:{text_hash}"


class RAGIndexer:
    """Indexes documents into vector store with incremental processing and timestamp-based deduplication."""

    def __init__(self, embedder, vector_store):
        """Initialize RAG indexer.

        Args:
            embedder: Embedder instance for generating embeddings
            vector_store: Vector store instance (e.g., ChromaStore) for storing chunks
        """
        self.embedder = embedder
        self.store = vector_store
        logger.info("RAGIndexer initialized")

    def index_documents(
        self,
        documents: list[Any],
        chunk_size: int,
        chunk_overlap: int,
        max_files: int | None = None,
    ) -> None:
        """Index documents incrementally with immediate database writes.

        Processes documents one at a time: chunk → embed → write to vector store.
        This enables partial progress recovery and reduces memory usage.
        Uses timestamp-based incremental reindexing to skip unchanged documents.

        Features:
        - Three-tier timestamp detection: frontmatter → Git → file modification date
        - Incremental reindexing: skips unchanged docs, replaces outdated ones
        - Numeric kbId normalization for consistent document identification
        - Automatic deduplication via stable IDs

        Args:
            documents: List of document objects with content and metadata
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            max_files: Maximum number of documents to index (None = no limit)
        """
        total_docs = len(documents)
        processed_docs = 0
        total_chunks_indexed = 0

        for doc_idx, doc in enumerate(documents, start=1):
            base_meta = dict(getattr(doc, "metadata", {}))
            kb_id = base_meta.get("kbId", base_meta.get("source_file", "doc"))
            content = getattr(doc, "content", "")

            # Skip empty documents
            if not content or not content.strip():
                logger.warning("Skipping empty document %d/%d (kbId: %s)", doc_idx, total_docs, kb_id)
                continue

            # Respect max_files limit (counts only documents we actually index)
            if max_files is not None and processed_docs >= max_files:
                logger.info("Max files limit reached: %d", max_files)
                break

            # Generate chunks for this document
            chunks = list(split_text(content, chunk_size, chunk_overlap))
            if not chunks:
                logger.warning("No chunks generated for document %d/%d (kbId: %s)", doc_idx, total_docs, kb_id)
                continue

            # Prepare chunk data
            texts: list[str] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []

            # Compute per-document invariants
            # Normalize kbId to numeric for consistent doc_stable_id and storage
            numeric_kb_id = extract_numeric_kbid(kb_id) or str(kb_id)
            doc_stable_id = sha1(numeric_kb_id.encode("utf-8")).hexdigest()[:12]
            source_file = base_meta.get("source_file")
            # Three-tier fallback: frontmatter → Git → file stat
            file_mtime_epoch, file_modified_at_iso, _timestamp_source = get_file_timestamp(source_file, base_meta)

            # Incremental reindexing: skip unchanged docs, replace outdated
            found_via_fallback = False
            if file_mtime_epoch is not None:
                existing = self.store.get_any_doc_meta({"doc_stable_id": doc_stable_id})
                # Fallback: search by numeric kbId if exact match not found
                if existing is None and extract_numeric_kbid(kb_id):
                    try:
                        all_docs = self.store.collection.get(limit=1000, include=["metadatas"])
                        for meta_item in all_docs.get("metadatas", []):
                            if extract_numeric_kbid(meta_item.get("kbId")) == numeric_kb_id:
                                old_doc_stable_id = meta_item.get("doc_stable_id")
                                if old_doc_stable_id:
                                    existing = self.store.get_any_doc_meta({"doc_stable_id": old_doc_stable_id})
                                    if existing:
                                        # Found old document with suffix - delete it and reindex with normalized kbId
                                        self.store.delete_where({"doc_stable_id": old_doc_stable_id})
                                        logger.info(
                                            "Deleted old document chunks with kbId=%s (doc_stable_id=%s), reindexing with normalized kbId",
                                            meta_item.get("kbId"),
                                            old_doc_stable_id,
                                        )
                                        found_via_fallback = True
                                        existing = None  # Clear existing so we always reindex with new normalized kbId
                                        break
                    except Exception:  # noqa: BLE001
                        pass

                # Only skip if we found an exact match (same doc_stable_id) and it's up to date
                if existing is not None and not found_via_fallback:
                    existing_epoch = existing.get("file_mtime_epoch")
                    if isinstance(existing_epoch, int) and existing_epoch >= file_mtime_epoch:
                        logger.info("Skipping unchanged document %d/%d (kbId: %s)", doc_idx, total_docs, kb_id)
                        continue
                    # Delete outdated chunks for this document before re-adding
                    try:
                        self.store.delete_where({"doc_stable_id": doc_stable_id})
                        logger.info("Deleted outdated chunks for kbId=%s (doc_stable_id=%s)", kb_id, doc_stable_id)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to delete outdated chunks for kbId=%s: %s", kb_id, exc)

            for idx, chunk in enumerate(chunks):
                # Generate stable ID using normalized kbId
                stable_id = _stable_id(numeric_kb_id, idx, chunk, base_meta.get("source_file"))

                meta = enrich_metadata(base_meta, chunk, idx)
                # Store normalized kbId in vector store
                meta["kbId"] = numeric_kb_id
                meta["stable_id"] = stable_id
                meta["doc_stable_id"] = doc_stable_id
                if file_modified_at_iso is not None:
                    meta["file_modified_at"] = file_modified_at_iso
                if file_mtime_epoch is not None:
                    meta["file_mtime_epoch"] = file_mtime_epoch

                texts.append(chunk)
                metadatas.append(meta)
                ids.append(stable_id)

            # Sanitize metadata values for vector store (drop None, stringify lists/dicts)
            def _sanitize_metadata(m: dict[str, Any]) -> dict[str, Any]:
                out: dict[str, Any] = {}
                for k, v in (m or {}).items():
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        out[k] = v
                    elif isinstance(v, list):
                        if not v:
                            continue
                        out[k] = ", ".join(str(x) for x in v)
                    else:
                        try:
                            out[k] = json.dumps(v, ensure_ascii=False)
                        except Exception:  # noqa: BLE001
                            out[k] = str(v)
                return out

            # Deduplicate by ID within this document batch before embedding
            seen_ids: set[str] = set()
            dedup_texts: list[str] = []
            dedup_metadatas: list[dict[str, Any]] = []
            dedup_ids: list[str] = []
            for t, m, i in zip(texts, metadatas, ids):
                if i in seen_ids:
                    continue
                seen_ids.add(i)
                dedup_texts.append(t)
                dedup_metadatas.append(_sanitize_metadata(m))
                dedup_ids.append(i)

            # Embed chunks for this (deduplicated) document
            embeddings = self.embedder.embed_documents(dedup_texts, show_progress=False)

            # Write immediately to vector store
            self.store.add(texts=dedup_texts, metadatas=dedup_metadatas, ids=dedup_ids, embeddings=embeddings)

            total_chunks_indexed += len(chunks)
            processed_docs += 1
            logger.info(
                "Indexed document %d/%d: %d chunks (kbId: %s, total chunks: %d)",
                doc_idx,
                total_docs,
                len(chunks),
                kb_id,
                total_chunks_indexed,
            )

