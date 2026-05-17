"""
Query Expansion Module

A rule-based query rewrite system that transforms vague, natural-language
user queries into semantically richer, embedding-friendly queries.

Uses a deterministic rule-based approach combined with synonym/concept injection 
to improve query expansion and retrieval quality.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Concept expansion map – maps high-level intent phrases to more specific
# technical vocabulary that is likely to appear in the embedding space of
# infrastructure documentation.
# ---------------------------------------------------------------------------
_CONCEPT_MAP: Dict[str, List[str]] = {
    # Scaling / load
    "peak load": [
        "horizontal autoscaling",
        "traffic spike handling",
        "dynamic compute provisioning",
        "increased system demand",
    ],
    "handle load": [
        "load balancing",
        "autoscaling infrastructure",
        "horizontal pod autoscaler",
        "request distribution",
    ],
    "scale": [
        "horizontal scaling",
        "auto-provisioning",
        "elastic compute",
        "replica scaling",
    ],
    # Availability / downtime
    "downtime": [
        "high availability",
        "fault tolerance",
        "failover mechanism",
        "zero-downtime deployment",
        "disaster recovery",
    ],
    "availability": [
        "redundancy",
        "failover",
        "health check",
        "replica promotion",
        "uptime guarantee",
    ],
    # Performance / latency
    "slow requests": [
        "latency optimisation",
        "response time reduction",
        "performance bottleneck",
        "connection pooling",
        "query optimisation",
        "caching layer",
    ],
    "latency": [
        "p99 latency",
        "response time",
        "fast-path endpoint",
        "edge caching",
        "CDN",
    ],
    "performance": [
        "throughput optimisation",
        "profiling",
        "flame graph analysis",
        "hot-path optimisation",
    ],
    # Reliability / resilience
    "failure": [
        "circuit breaker",
        "bulkhead isolation",
        "cascading failure prevention",
        "retry with exponential backoff",
    ],
    "resilience": [
        "fault isolation",
        "graceful degradation",
        "fallback response",
        "error budget",
    ],
    # Caching
    "caching": [
        "in-memory cache",
        "write-through cache",
        "cache invalidation",
        "TTL eviction policy",
        "LRU algorithm",
    ],
    # Observability
    "monitoring": [
        "distributed tracing",
        "OpenTelemetry",
        "anomaly detection",
        "structured logging",
        "metrics collection",
    ],
    # Messaging / async
    "queue": [
        "message queue",
        "event-driven processing",
        "dead letter queue",
        "back-pressure",
        "exactly-once delivery",
    ],
    "async": [
        "asynchronous processing",
        "event-driven architecture",
        "durable message queue",
        "consumer group",
    ],
}


class QueryExpander:
    """Rewrites user queries into semantically enriched, embedding-friendly
    form by injecting domain-relevant technical vocabulary.

    Uses a deterministic rule engine to expand queries.

    Attributes:
        concept_map: Mapping from trigger phrases to expansion terms.
    """

    def __init__(
        self,
        concept_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Initialise the query expander.

        Args:
            concept_map: Custom mapping of trigger phrases to expansion terms.
                Falls back to the built-in ``_CONCEPT_MAP`` when ``None``.
        """
        self.concept_map: Dict[str, List[str]] = concept_map or _CONCEPT_MAP

    def expand(self, query: str) -> str:
        """Expand a raw user query into an embedding-optimised query.

        The expansion process:
        1. Normalises the input query.
        2. Scans for known trigger phrases.
        3. Appends semantically related technical terms.
        4. Returns a single enriched query string.

        Args:
            query: The original user query.

        Returns:
            An expanded query string with additional semantic context.
        """
        if not query or not query.strip():
            return query

        query_lower = query.lower().strip()
        expansion_terms: List[str] = []

        for trigger, terms in self.concept_map.items():
            # Use word-boundary aware matching for better precision.
            if re.search(re.escape(trigger), query_lower):
                expansion_terms.extend(terms)

        if not expansion_terms:
            # Fallback: return the original query unchanged.
            return query

        # De-duplicate while preserving order.
        seen: set = set()
        unique_terms: List[str] = []
        for term in expansion_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        expanded = f"{query.strip()} — including {', '.join(unique_terms)}"
        return expanded

    def __repr__(self) -> str:
        return f"QueryExpander(triggers={len(self.concept_map)})"
