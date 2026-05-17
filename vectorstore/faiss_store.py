"""
FAISS Vector Store Module

Provides a thin, production-ready wrapper around Facebook AI Similarity Search
(FAISS) for high-performance approximate nearest-neighbour retrieval using
cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import faiss
import numpy as np


@dataclass
class SearchResult:
    """Structured container for a single search hit.

    Attributes:
        document: The raw text of the matched document.
        score: Cosine similarity score (higher → more similar).
        doc_id: Identifier of the matched document.
        metadata: Optional auxiliary metadata (tags, title, etc.).
    """

    document: str
    score: float
    doc_id: str
    metadata: Dict = field(default_factory=dict)


class FAISSVectorStore:
    """FAISS-backed vector store with cosine similarity search.

    Uses ``IndexFlatIP`` (inner-product) on **L2-normalised** embeddings so
    that the inner product is numerically equivalent to cosine similarity.

    Attributes:
        dimension: Embedding dimensionality.
        index: Underlying FAISS index.
        documents: Ordered list of stored document texts.
        doc_ids: Ordered list of corresponding document IDs.
        metadata_store: Ordered list of metadata dicts.
    """

    def __init__(self, dimension: int) -> None:
        """Initialise the vector store.

        Args:
            dimension: The dimensionality of the embedding vectors to be
                stored (e.g. 384 for *all-MiniLM-L6-v2*).
        """
        self.dimension: int = dimension
        # Inner-product index – combined with L2-normalised vectors this
        # computes cosine similarity directly.
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadata_store: List[Dict] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Add pre-computed embeddings and their source documents to the index.

        Args:
            embeddings: 2-D ``float32`` array of shape
                ``(n_documents, dimension)``.  **Must** be L2-normalised.
            documents: Corresponding document texts.
            doc_ids: Optional document identifiers.  Auto-generated when
                omitted.
            metadata: Optional list of metadata dicts aligned to *documents*.

        Raises:
            ValueError: On dimension mismatches or length inconsistencies.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dimension}), "
                f"got {embeddings.shape}."
            )
        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings and documents must match."
            )

        # Ensure normalisation (idempotent for already-normalised vectors).
        faiss.normalize_L2(embeddings)

        # Generate sequential IDs when none are supplied.
        if doc_ids is None:
            start = len(self.documents)
            doc_ids = [f"doc_{start + i:04d}" for i in range(len(documents))]

        if metadata is None:
            metadata = [{}] * len(documents)

        self.index.add(embeddings)
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)
        self.metadata_store.extend(metadata)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Retrieve the *top_k* most similar documents to a query embedding.

        Args:
            query_embedding: 1-D or 2-D ``float32`` array.  Will be
                L2-normalised internally.
            top_k: Number of results to return.

        Returns:
            A list of :class:`SearchResult` instances sorted by descending
            similarity score.

        Raises:
            ValueError: If the index is empty.
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty — add documents first.")

        # Reshape to (1, dim) if needed, then normalise.
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32).copy()
        faiss.normalize_L2(query_embedding)

        # Clamp top_k to index size.
        effective_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, effective_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS sentinel for missing results
            results.append(
                SearchResult(
                    document=self.documents[idx],
                    score=float(score),
                    doc_id=self.doc_ids[idx],
                    metadata=self.metadata_store[idx],
                )
            )
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of indexed vectors."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore(dimension={self.dimension}, "
            f"indexed={self.index.ntotal})"
        )
