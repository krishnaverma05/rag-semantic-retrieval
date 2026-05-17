"""
Shared test fixtures for the RAG assessment test suite.

Provides reusable, session-scoped fixtures that avoid redundant model loading
during test runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.embedding_service import EmbeddingService
from retrieval.query_expander import QueryExpander
from vectorstore.faiss_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Sample corpus used across tests
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENTS: List[str] = [
    "The platform provisions additional compute nodes dynamically when incoming request rates exceed predefined throughput thresholds.",
    "Incoming network traffic is distributed across healthy backend instances using a weighted round-robin algorithm.",
    "Frequently accessed data is served from an in-memory caching layer that sits between the application tier and the primary datastore.",
    "The system implements circuit breaker patterns and bulkhead isolation to prevent cascading failures across service boundaries.",
    "Long-running operations are offloaded to durable message queues that decouple producers from consumers.",
]

SAMPLE_DOC_IDS: List[str] = [f"test_{i:03d}" for i in range(len(SAMPLE_DOCUMENTS))]


@pytest.fixture(scope="session")
def embedding_service() -> EmbeddingService:
    """Session-scoped embedding service to avoid repeated model loading."""
    return EmbeddingService()


@pytest.fixture(scope="session")
def sample_embeddings(embedding_service: EmbeddingService) -> np.ndarray:
    """Pre-computed embeddings for the sample corpus."""
    return embedding_service.embed_documents(SAMPLE_DOCUMENTS)


@pytest.fixture
def vector_store(sample_embeddings: np.ndarray) -> FAISSVectorStore:
    """Fresh FAISS index populated with sample documents."""
    store = FAISSVectorStore(dimension=sample_embeddings.shape[1])
    store.add_embeddings(
        embeddings=sample_embeddings.copy(),
        documents=SAMPLE_DOCUMENTS,
        doc_ids=SAMPLE_DOC_IDS,
    )
    return store


@pytest.fixture
def query_expander() -> QueryExpander:
    """Default query expander instance."""
    return QueryExpander()
