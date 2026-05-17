"""
Embedding Service Module

Provides semantic embedding generation using sentence-transformers.
Wraps the all-MiniLM-L6-v2 model for producing dense vector representations
of documents and queries for downstream similarity search.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Default model – a lightweight, high-quality embedding model suitable for
# semantic similarity tasks.  384-dimensional output, ~22 M params.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Generates dense vector embeddings for documents and queries.

    This service encapsulates the sentence-transformers library and exposes
    a clean interface for the rest of the retrieval pipeline.

    Attributes:
        model_name: Identifier of the HuggingFace sentence-transformer model.
        model: Loaded ``SentenceTransformer`` instance.
        embedding_dim: Dimensionality of the produced embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialise the embedding service.

        Args:
            model_name: Name or path of a sentence-transformer model to load.
        """
        self.model_name: str = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embedding_dim: int = self.model.get_embedding_dimension()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed a batch of document strings.

        Args:
            documents: List of document texts to embed.

        Returns:
            A 2-D numpy array of shape ``(len(documents), embedding_dim)``
            with dtype ``float32``.

        Raises:
            ValueError: If *documents* is empty.
        """
        if not documents:
            raise ValueError("Cannot embed an empty document list.")

        embeddings: np.ndarray = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalised → cosine ≡ dot-product
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: The search query to embed.

        Returns:
            A 1-D numpy array of shape ``(embedding_dim,)`` with dtype
            ``float32``.

        Raises:
            ValueError: If *query* is empty or blank.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        embedding: np.ndarray = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EmbeddingService(model_name={self.model_name!r}, "
            f"embedding_dim={self.embedding_dim})"
        )
