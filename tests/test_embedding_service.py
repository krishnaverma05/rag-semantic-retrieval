"""
Tests for the Embedding Service.

Validates:
- Correct output shapes and dtypes
- Normalisation (L2 unit vectors)
- Semantic similarity ordering
- Error handling for edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from embeddings.embedding_service import EmbeddingService


class TestEmbedDocuments:
    """Tests for ``EmbeddingService.embed_documents``."""

    def test_output_shape(self, embedding_service: EmbeddingService) -> None:
        """Embedding matrix must be (n_docs, embedding_dim)."""
        docs = ["Hello world", "Another document"]
        embeddings = embedding_service.embed_documents(docs)
        assert embeddings.shape == (2, embedding_service.embedding_dim)

    def test_output_dtype(self, embedding_service: EmbeddingService) -> None:
        """Embeddings must be float32."""
        embeddings = embedding_service.embed_documents(["test"])
        assert embeddings.dtype == np.float32

    def test_normalised_vectors(self, embedding_service: EmbeddingService) -> None:
        """Each row must have L2 norm ≈ 1 (unit vector)."""
        embeddings = embedding_service.embed_documents(
            ["first doc", "second doc", "third doc"]
        )
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_list_raises(self, embedding_service: EmbeddingService) -> None:
        """Empty document list must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            embedding_service.embed_documents([])

    def test_single_document(self, embedding_service: EmbeddingService) -> None:
        """Single-document embedding must still be 2-D."""
        embeddings = embedding_service.embed_documents(["single doc"])
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == 1


class TestEmbedQuery:
    """Tests for ``EmbeddingService.embed_query``."""

    def test_output_shape(self, embedding_service: EmbeddingService) -> None:
        """Query embedding must be 1-D with correct dimension."""
        emb = embedding_service.embed_query("test query")
        assert emb.ndim == 1
        assert emb.shape[0] == embedding_service.embedding_dim

    def test_output_dtype(self, embedding_service: EmbeddingService) -> None:
        """Query embedding must be float32."""
        emb = embedding_service.embed_query("test query")
        assert emb.dtype == np.float32

    def test_normalised(self, embedding_service: EmbeddingService) -> None:
        """Query vector must be L2-normalised."""
        emb = embedding_service.embed_query("test query")
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_query_raises(self, embedding_service: EmbeddingService) -> None:
        """Empty / blank query must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            embedding_service.embed_query("")
        with pytest.raises(ValueError, match="non-empty"):
            embedding_service.embed_query("   ")


class TestSemanticSimilarity:
    """Validate that semantically similar texts produce close embeddings."""

    def test_similar_texts_closer(self, embedding_service: EmbeddingService) -> None:
        """Two similar sentences must have higher cosine sim than dissimilar."""
        a = embedding_service.embed_query("autoscaling compute nodes")
        b = embedding_service.embed_query("dynamically provisioning servers")
        c = embedding_service.embed_query("chocolate cake recipe")

        sim_ab = float(np.dot(a, b))
        sim_ac = float(np.dot(a, c))

        assert sim_ab > sim_ac, (
            f"Expected sim(a,b)={sim_ab:.4f} > sim(a,c)={sim_ac:.4f}"
        )
