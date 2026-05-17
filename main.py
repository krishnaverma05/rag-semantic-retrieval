#!/usr/bin/env python3
"""
RAG Semantic Retrieval System — Main Application

Orchestrates the full pipeline:
1. Load technical documentation corpus
2. Generate dense embeddings with sentence-transformers
3. Build a FAISS cosine-similarity index
4. Benchmark raw vs. semantic-enhanced retrieval strategies
5. Persist structured results and a human-readable report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.benchmark import RetrievalBenchmark
from embeddings.embedding_service import EmbeddingService
from retrieval.query_expander import QueryExpander
from retrieval.retriever import Retriever
from vectorstore.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "technical_docs.json"
RESULTS_PATH = PROJECT_ROOT / "retrieval_results.json"
REPORT_PATH = PROJECT_ROOT / "retrieval_benchmark.md"
TOP_K = 5


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load the technical documentation corpus from disk.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        List of document dicts containing ``id``, ``title``, ``content``,
        and ``tags``.
    """
    with open(path, "r", encoding="utf-8") as f:
        docs: List[Dict[str, Any]] = json.load(f)
    print(f"✅  Loaded {len(docs)} documents from {path.name}")
    return docs


def main() -> None:
    """Run the full retrieval benchmark pipeline."""
    print("=" * 64)
    print("  RAG Semantic Retrieval System — Benchmark Runner")
    print("=" * 64)
    print()

    # 1. Load dataset ---------------------------------------------------
    docs = load_dataset(DATA_PATH)
    texts = [doc["content"] for doc in docs]
    doc_ids = [doc["id"] for doc in docs]
    metadata = [{"title": doc["title"], "tags": doc.get("tags", [])} for doc in docs]

    # 2. Generate embeddings -------------------------------------------
    print("🔄  Initialising embedding service (all-MiniLM-L6-v2)…")
    embedding_service = EmbeddingService()
    print(f"   Model loaded: {embedding_service}")

    print("🔄  Generating document embeddings…")
    doc_embeddings = embedding_service.embed_documents(texts)
    print(f"   Embedding matrix shape: {doc_embeddings.shape}")

    # 3. Build FAISS index ---------------------------------------------
    print("🔄  Building FAISS vector index…")
    vector_store = FAISSVectorStore(dimension=embedding_service.embedding_dim)
    vector_store.add_embeddings(
        embeddings=doc_embeddings,
        documents=texts,
        doc_ids=doc_ids,
        metadata=metadata,
    )
    print(f"   Index built: {vector_store}")

    # 4. Initialise retriever ------------------------------------------
    query_expander = QueryExpander()
    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        query_expander=query_expander,
        default_top_k=TOP_K,
    )
    print(f"   Retriever ready: {retriever}")
    print()

    # 5. Run benchmark -------------------------------------------------
    print("=" * 64)
    print("  Running Benchmark")
    print("=" * 64)
    print()

    benchmark = RetrievalBenchmark(retriever=retriever, top_k=TOP_K)
    benchmark.run()

    # 6. Save results --------------------------------------------------
    benchmark.save_json(RESULTS_PATH)
    print(f"📄  JSON results saved to {RESULTS_PATH.name}")

    benchmark.save_report(REPORT_PATH)
    print(f"📄  Markdown report saved to {REPORT_PATH.name}")
    print()

    # 7. Print readable report -----------------------------------------
    print("=" * 64)
    print("  Benchmark Report")
    print("=" * 64)
    print()
    print(benchmark.generate_report())

    print("=" * 64)
    print("  ✅  Pipeline complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
