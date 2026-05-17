"""
Tests for the Retrieval Engine and Mock Vertex AI Services.

Validates:
- Strategy A (raw) and Strategy B (enhanced) retrieval
- Result structure and typing
- Enhanced retrieval uses expanded queries
- MockTextEmbeddingModel behaviour
- MockGenerativeModel behaviour
- Benchmark generation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.benchmark import RetrievalBenchmark
from embeddings.embedding_service import EmbeddingService
from mocks.vertex_ai_mock import (
    MockGenerativeModel,
    MockTextEmbeddingModel,
)
from retrieval.query_expander import QueryExpander
from retrieval.retriever import Retriever, RetrievalResult
from tests.conftest import SAMPLE_DOCUMENTS, SAMPLE_DOC_IDS
from vectorstore.faiss_store import FAISSVectorStore


# =========================================================================
# Retrieval Strategy Tests
# =========================================================================

class TestRetrieveRaw:
    """Tests for Strategy A — raw vector search."""

    def test_returns_retrieval_result(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Strategy A must return a RetrievalResult."""
        retriever = Retriever(embedding_service, vector_store)
        result = retriever.retrieve_raw("autoscaling compute nodes")
        assert isinstance(result, RetrievalResult)
        assert result.strategy == "raw"

    def test_processed_query_unchanged(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """In raw mode, processed_query must equal original_query."""
        retriever = Retriever(embedding_service, vector_store)
        query = "autoscaling compute nodes"
        result = retriever.retrieve_raw(query)
        assert result.processed_query == result.original_query == query

    def test_results_not_empty(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Raw retrieval must return at least one result."""
        retriever = Retriever(embedding_service, vector_store)
        result = retriever.retrieve_raw("load balancing")
        assert len(result.results) > 0

    def test_top_k_respected(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Number of results must respect top_k."""
        retriever = Retriever(embedding_service, vector_store, default_top_k=2)
        result = retriever.retrieve_raw("caching")
        assert len(result.results) <= 2


class TestRetrieveEnhanced:
    """Tests for Strategy B — Semantic-enhanced retrieval."""

    def test_returns_retrieval_result(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Strategy B must return a RetrievalResult."""
        retriever = Retriever(embedding_service, vector_store)
        result = retriever.retrieve_enhanced("How does the system handle peak load?")
        assert isinstance(result, RetrievalResult)
        assert result.strategy == "enhanced"

    def test_processed_query_expanded(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Enhanced mode must produce a processed_query different from original."""
        retriever = Retriever(embedding_service, vector_store)
        query = "How does the system handle peak load?"
        result = retriever.retrieve_enhanced(query)
        assert result.processed_query != result.original_query
        assert len(result.processed_query) > len(result.original_query)

    def test_results_not_empty(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Enhanced retrieval must return at least one result."""
        retriever = Retriever(embedding_service, vector_store)
        result = retriever.retrieve_enhanced("How is downtime prevented?")
        assert len(result.results) > 0

    def test_result_structure(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Each result dict must have the expected keys."""
        retriever = Retriever(embedding_service, vector_store)
        result = retriever.retrieve_enhanced("How are slow requests reduced?")
        for r in result.results:
            assert "rank" in r
            assert "doc_id" in r
            assert "score" in r
            assert "document" in r
            assert "metadata" in r


class TestRetrievalResultSerialization:
    """Validate RetrievalResult.to_dict()."""

    def test_to_dict(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        retriever = Retriever(embedding_service, vector_store)
        result = retriever.retrieve_raw("test query")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "strategy" in d
        assert "original_query" in d
        assert "results" in d


# =========================================================================
# Mock Vertex AI Tests
# =========================================================================

class TestMockTextEmbeddingModel:
    """Tests for the mocked embedding model."""

    def test_output_length(self) -> None:
        """Must return one embedding per input text."""
        model = MockTextEmbeddingModel()
        results = model.get_embeddings(["hello", "world"])
        assert len(results) == 2

    def test_embedding_dimension(self) -> None:
        """Mock embeddings must match the configured dimension."""
        model = MockTextEmbeddingModel(dimension=768)
        results = model.get_embeddings(["test"])
        assert len(results[0].values) == 768

    def test_deterministic(self) -> None:
        """Same input must produce identical embeddings across calls."""
        model = MockTextEmbeddingModel()
        a = model.get_embeddings(["deterministic"])[0].values
        b = model.get_embeddings(["deterministic"])[0].values
        np.testing.assert_array_equal(a, b)

    def test_normalised(self) -> None:
        """Mock embeddings must be L2-normalised."""
        model = MockTextEmbeddingModel()
        values = model.get_embeddings(["normalised test"])[0].values
        norm = float(np.linalg.norm(values))
        assert abs(norm - 1.0) < 1e-5

    def test_from_pretrained(self) -> None:
        """Factory method must return a valid instance."""
        model = MockTextEmbeddingModel.from_pretrained("custom-model")
        assert model.model_name == "custom-model"


class TestMockGenerativeModel:
    """Tests for the mocked generative model."""

    def test_generate_returns_response(self) -> None:
        """generate_content must return a MockGenerationResponse."""
        model = MockGenerativeModel()
        resp = model.generate_content("How does autoscaling work?")
        assert hasattr(resp, "text")
        assert hasattr(resp, "metadata")
        assert len(resp.text) > 0

    def test_peak_load_expansion(self) -> None:
        """Prompt mentioning 'peak load' must trigger domain expansion."""
        model = MockGenerativeModel()
        resp = model.generate_content("Explain peak load handling")
        assert "autoscale" in resp.text.lower() or "traffic" in resp.text.lower()

    def test_downtime_expansion(self) -> None:
        """Prompt mentioning 'downtime' must trigger availability expansion."""
        model = MockGenerativeModel()
        resp = model.generate_content("Preventing downtime")
        assert "availability" in resp.text.lower() or "failover" in resp.text.lower()

    def test_metadata_present(self) -> None:
        """Response metadata must include model name and finish reason."""
        model = MockGenerativeModel()
        resp = model.generate_content("test prompt")
        assert "model" in resp.metadata
        assert "finish_reason" in resp.metadata


# =========================================================================
# Benchmark Tests
# =========================================================================

class TestBenchmark:
    """Tests for the benchmarking engine."""

    def test_benchmark_runs(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Benchmark must execute without errors and produce results."""
        retriever = Retriever(embedding_service, vector_store)
        benchmark = RetrievalBenchmark(
            retriever=retriever,
            queries=["How does autoscaling work?"],
            top_k=3,
        )
        results = benchmark.run()
        assert len(results) == 1

    def test_benchmark_json_output(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """JSON export must produce a valid list of dicts."""
        retriever = Retriever(embedding_service, vector_store)
        benchmark = RetrievalBenchmark(
            retriever=retriever,
            queries=["test query"],
            top_k=3,
        )
        benchmark.run()
        json_out = benchmark.to_json()
        assert isinstance(json_out, list)
        assert len(json_out) == 1
        assert "query" in json_out[0]
        assert "analysis" in json_out[0]

    def test_benchmark_report(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Markdown report must be a non-empty string with headers."""
        retriever = Retriever(embedding_service, vector_store)
        benchmark = RetrievalBenchmark(
            retriever=retriever,
            queries=["test query"],
            top_k=3,
        )
        benchmark.run()
        report = benchmark.generate_report()
        assert isinstance(report, str)
        assert "# Retrieval Benchmark Report" in report

    def test_benchmark_analysis_keys(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        """Analysis dict must contain expected comparison metrics."""
        retriever = Retriever(embedding_service, vector_store)
        benchmark = RetrievalBenchmark(
            retriever=retriever,
            queries=["autoscaling"],
            top_k=3,
        )
        benchmark.run()
        analysis = benchmark.results[0].analysis
        expected_keys = {
            "overlap_count",
            "overlap_doc_ids",
            "only_in_raw",
            "only_in_enhanced",
            "raw_avg_score",
            "enhanced_avg_score",
            "score_improvement",
            "rank_shifts",
        }
        assert expected_keys.issubset(analysis.keys())
