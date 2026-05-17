"""
Mock Vertex AI SDK Module

Provides drop-in mock implementations of Google Cloud Vertex AI services so
that the retrieval system can be tested and demonstrated without requiring
real cloud credentials or API calls.

Simulated services:
- ``MockTextEmbeddingModel`` — mirrors ``vertexai.language_models.TextEmbeddingModel``
- ``MockGenerativeModel``   — mirrors ``vertexai.generative_models.GenerativeModel``

All outputs are deterministic and reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Mock Embedding Model
# ---------------------------------------------------------------------------

@dataclass
class MockEmbeddingValue:
    """Mimics the embedding response object from Vertex AI.

    Attributes:
        values: The embedding vector (list of floats).
    """
    values: List[float]


class MockTextEmbeddingModel:
    """Simulates ``vertexai.language_models.TextEmbeddingModel``.

    Produces deterministic pseudo-embeddings by hashing the input text.
    The output vectors are L2-normalised so they behave correctly with
    cosine-similarity indices.

    Attributes:
        model_name: Name of the simulated model.
        dimension: Dimensionality of the produced embeddings.
    """

    def __init__(
        self,
        model_name: str = "textembedding-gecko@003",
        dimension: int = 768,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension

    def get_embeddings(
        self,
        texts: List[str],
    ) -> List[MockEmbeddingValue]:
        """Generate mock embeddings for a batch of texts.

        Each embedding is a deterministic function of the input string,
        ensuring reproducible results across runs.

        Args:
            texts: List of strings to embed.

        Returns:
            List of :class:`MockEmbeddingValue` instances.
        """
        results: List[MockEmbeddingValue] = []
        for text in texts:
            vector = self._deterministic_embed(text)
            results.append(MockEmbeddingValue(values=vector.tolist()))
        return results

    def _deterministic_embed(self, text: str) -> np.ndarray:
        """Create a deterministic pseudo-embedding from *text*.

        Uses SHA-256 to seed a numpy RNG, producing a reproducible
        unit-length vector.
        """
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)  # L2 normalise
        return vec

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockTextEmbeddingModel":
        """Factory matching the Vertex AI SDK pattern."""
        return cls(model_name=model_name)

    def __repr__(self) -> str:
        return f"MockTextEmbeddingModel(model={self.model_name!r}, dim={self.dimension})"


# ---------------------------------------------------------------------------
# Mock Generative Model
# ---------------------------------------------------------------------------

@dataclass
class MockGenerationResponse:
    """Mimics a Vertex AI ``GenerationResponse``.

    Attributes:
        text: The generated text content.
        metadata: Simulated response metadata.
    """
    text: str
    metadata: Dict[str, Any]


class MockGenerativeModel:
    """Simulates ``vertexai.generative_models.GenerativeModel``.

    Provides deterministic "generated" responses for query expansion and
    other generative tasks without calling any cloud service.

    Attributes:
        model_name: Name of the simulated generative model.
        temperature: Simulated temperature parameter (unused in mock).
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def generate_content(
        self,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> MockGenerationResponse:
        """Generate a mock text response for the given prompt.

        Applies simple rule-based expansion to simulate an LLM rewrite.

        Args:
            prompt: The input prompt / instruction.
            generation_config: Optional generation parameters (ignored).

        Returns:
            A :class:`MockGenerationResponse` with generated text.
        """
        expanded = self._mock_expand(prompt)
        return MockGenerationResponse(
            text=expanded,
            metadata={
                "model": self.model_name,
                "finish_reason": "STOP",
                "token_count": len(expanded.split()),
            },
        )

    @staticmethod
    def _mock_expand(prompt: str) -> str:
        """Produce a simple simulated expansion of the prompt."""
        expansions = {
            "peak load": (
                "How does the platform autoscale infrastructure during "
                "high traffic spikes and increased system demand?"
            ),
            "downtime": (
                "How does the system ensure high availability and prevent "
                "service interruptions through failover mechanisms and "
                "redundancy strategies?"
            ),
            "slow requests": (
                "How does the system reduce request latency through "
                "caching, connection pooling, query optimisation, and "
                "performance engineering?"
            ),
        }
        prompt_lower = prompt.lower()
        for trigger, expansion in expansions.items():
            if trigger in prompt_lower:
                return expansion
        # Default: return the prompt with a generic expansion suffix.
        return f"{prompt} — expanded for improved semantic retrieval."

    def __repr__(self) -> str:
        return (
            f"MockGenerativeModel(model={self.model_name!r}, "
            f"temperature={self.temperature})"
        )
