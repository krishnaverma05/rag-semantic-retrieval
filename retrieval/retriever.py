"""
Retriever Module

Implements the two core retrieval strategies:

- **Strategy A (Raw)**: Query → Embedding → Vector Search → Results
- **Strategy B (Enhanced)**: Query → Expansion → Embedding → Vector Search → Results

Both strategies share the same embedding service and vector store, differing
only in whether the query undergoes expansion before embedding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from embeddings.embedding_service import EmbeddingService
from retrieval.query_expander import QueryExpander
from vectorstore.faiss_store import FAISSVectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Structured output from a retrieval operation.

    Attributes:
        strategy: ``"raw"`` or ``"enhanced"``.
        original_query: The query as submitted by the user.
        processed_query: The query actually used for embedding (may be expanded).
        results: List of matched documents with scores.
    """

    strategy: str
    original_query: str
    processed_query: str
    results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return asdict(self)


class Retriever:
    """Unified retrieval engine supporting both raw and enhanced strategies.

    Attributes:
        embedding_service: Service used to encode queries.
        vector_store: FAISS-backed index of document embeddings.
        query_expander: Module used for Strategy B query rewriting.
        default_top_k: Default number of results per search.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        query_expander: QueryExpander | None = None,
        default_top_k: int = 5,
    ) -> None:
        """Initialise the retriever.

        Args:
            embedding_service: Pre-initialised embedding service.
            vector_store: Pre-populated FAISS vector store.
            query_expander: Optional query expander (created lazily if
                ``None``).
            default_top_k: Default number of results to return.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.query_expander = query_expander or QueryExpander()
        self.default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Strategy A — Raw vector search
    # ------------------------------------------------------------------

    def retrieve_raw(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute Strategy A: direct query → embedding → vector search.

        Args:
            query: The user's natural-language query.
            top_k: Number of results to return.

        Returns:
            A :class:`RetrievalResult` with strategy ``"raw"``.
        """
        k = top_k or self.default_top_k
        query_embedding = self.embedding_service.embed_query(query)
        search_results: List[SearchResult] = self.vector_store.search(
            query_embedding, top_k=k,
        )
        return RetrievalResult(
            strategy="raw",
            original_query=query,
            processed_query=query,
            results=self._format_results(search_results),
        )

    # ------------------------------------------------------------------
    # Strategy B — Semantic-enhanced retrieval
    # ------------------------------------------------------------------

    def retrieve_enhanced(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute Strategy B: query expansion → embedding → vector search.

        Args:
            query: The user's natural-language query.
            top_k: Number of results to return.

        Returns:
            A :class:`RetrievalResult` with strategy ``"enhanced"``.
        """
        k = top_k or self.default_top_k
        expanded_query = self.query_expander.expand(query)
        query_embedding = self.embedding_service.embed_query(expanded_query)
        search_results: List[SearchResult] = self.vector_store.search(
            query_embedding, top_k=k,
        )
        return RetrievalResult(
            strategy="enhanced",
            original_query=query,
            processed_query=expanded_query,
            results=self._format_results(search_results),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(
        search_results: List[SearchResult],
    ) -> List[Dict[str, Any]]:
        """Convert :class:`SearchResult` objects to plain dicts."""
        return [
            {
                "rank": rank,
                "doc_id": sr.doc_id,
                "score": round(sr.score, 6),
                "document": sr.document,
                "metadata": sr.metadata,
            }
            for rank, sr in enumerate(search_results, start=1)
        ]

    def __repr__(self) -> str:
        return (
            f"Retriever(top_k={self.default_top_k}, "
            f"store_size={self.vector_store.size})"
        )
