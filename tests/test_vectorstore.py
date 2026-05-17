"""
Tests for the FAISS Vector Store.

Validates:
- Embedding ingestion
- Cosine similarity search
- Top-K retrieval
- Score ordering
- Edge cases (empty index, dimension mismatch)
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from vectorstore.faiss_store import FAISSVectorStore, SearchResult
from tests.conftest import SAMPLE_DOCUMENTS, SAMPLE_DOC_IDS


class TestAddEmbeddings:
    """Tests for ``FAISSVectorStore.add_embeddings``."""

    def test_index_size(self, vector_store: FAISSVectorStore) -> None:
        """Index size must equal number of added documents."""
        assert vector_store.size == len(SAMPLE_DOCUMENTS)

    def test_dimension_mismatch_raises(self) -> None:
        """Wrong embedding dimension must raise ValueError."""
        store = FAISSVectorStore(dimension=384)
        wrong_dim = np.random.randn(3, 128).astype(np.float32)
        with pytest.raises(ValueError, match="shape"):
            store.add_embeddings(wrong_dim, ["a", "b", "c"])

    def test_count_mismatch_raises(
        self, sample_embeddings: np.ndarray
    ) -> None:
        """Mismatched embedding/document counts must raise ValueError."""
        store = FAISSVectorStore(dimension=sample_embeddings.shape[1])
        with pytest.raises(ValueError, match="match"):
            store.add_embeddings(sample_embeddings, ["only_one"])

    def test_auto_generated_ids(
        self, sample_embeddings: np.ndarray
    ) -> None:
        """When no doc_ids are provided, IDs are auto-generated."""
        store = FAISSVectorStore(dimension=sample_embeddings.shape[1])
        store.add_embeddings(sample_embeddings.copy(), SAMPLE_DOCUMENTS)
        assert len(store.doc_ids) == len(SAMPLE_DOCUMENTS)
        assert all(did.startswith("doc_") for did in store.doc_ids)


class TestSearch:
    """Tests for ``FAISSVectorStore.search``."""

    def test_returns_results(
        self,
        vector_store: FAISSVectorStore,
        embedding_service,
    ) -> None:
        """Search must return a non-empty list of SearchResult."""
        query_emb = embedding_service.embed_query("autoscaling compute")
        results = vector_store.search(query_emb, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_k_limit(
        self,
        vector_store: FAISSVectorStore,
        embedding_service,
    ) -> None:
        """Number of results must not exceed top_k."""
        query_emb = embedding_service.embed_query("caching data")
        results = vector_store.search(query_emb, top_k=2)
        assert len(results) <= 2

    def test_scores_descending(
        self,
        vector_store: FAISSVectorStore,
        embedding_service,
    ) -> None:
        """Results must be sorted by descending similarity score."""
        query_emb = embedding_service.embed_query("load balancing traffic")
        results = vector_store.search(query_emb, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(
        self,
        vector_store: FAISSVectorStore,
        embedding_service,
    ) -> None:
        """Cosine similarity scores must be in [-1, 1]."""
        query_emb = embedding_service.embed_query("circuit breaker pattern")
        results = vector_store.search(query_emb, top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-5, f"Score out of range: {r.score}"

    def test_empty_index_raises(self) -> None:
        """Searching an empty index must raise ValueError."""
        store = FAISSVectorStore(dimension=384)
        query = np.random.randn(384).astype(np.float32)
        with pytest.raises(ValueError, match="empty"):
            store.search(query)

    def test_result_contains_document(
        self,
        vector_store: FAISSVectorStore,
        embedding_service,
    ) -> None:
        """Each result must contain a document string from the corpus."""
        query_emb = embedding_service.embed_query("message queue processing")
        results = vector_store.search(query_emb, top_k=3)
        for r in results:
            assert r.document in SAMPLE_DOCUMENTS

    def test_result_contains_doc_id(
        self,
        vector_store: FAISSVectorStore,
        embedding_service,
    ) -> None:
        """Each result must have a valid doc_id."""
        query_emb = embedding_service.embed_query("fault tolerance")
        results = vector_store.search(query_emb, top_k=3)
        for r in results:
            assert r.doc_id in SAMPLE_DOC_IDS
