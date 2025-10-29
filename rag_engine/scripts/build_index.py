from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path if not already installed
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag_engine.config.settings import settings
from rag_engine.core.document_processor import DocumentProcessor
from rag_engine.llm.llm_manager import LLMManager
from rag_engine.retrieval.embedder import FRIDAEmbedder
from rag_engine.retrieval.retriever import RAGRetriever
from rag_engine.storage.vector_store import ChromaStore
from rag_engine.utils.logging_manager import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG index from markdown sources")
    parser.add_argument("--source", required=True, help="Path to folder or file")
    parser.add_argument("--mode", choices=["folder", "file", "mkdocs"], default="folder")
    parser.add_argument("--reindex", action="store_true", help="Force reindex (ignored in MVP)")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of source files to scan/process (for quick runs)",
    )
    args = parser.parse_args()

    setup_logging()

    dp = DocumentProcessor(mode=args.mode)
    # Always process full set; apply limiting in retriever so skipped files don't count
    docs = dp.process(args.source)

    embedder = FRIDAEmbedder(model_name=settings.embedding_model, device=settings.embedding_device)
    store = ChromaStore(persist_dir=settings.chromadb_persist_dir, collection_name=settings.chromadb_collection)
    llm_manager = LLMManager(
        provider=settings.default_llm_provider,
        model=settings.default_model,
        temperature=settings.llm_temperature,
    )
    retriever = RAGRetriever(
        embedder=embedder,
        vector_store=store,
        llm_manager=llm_manager,
        top_k_retrieve=settings.top_k_retrieve,
        top_k_rerank=settings.top_k_rerank,
        rerank_enabled=settings.rerank_enabled,
    )

    # Call retriever with optional max_files, but keep backward compatibility
    try:
        retriever.index_documents(
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_files=args.max_files,
        )
    except TypeError:
        # Older retrievers without max_files support
        retriever.index_documents(
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    print("Index build complete.")


if __name__ == "__main__":
    main()


