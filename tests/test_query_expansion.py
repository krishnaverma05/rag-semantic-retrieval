"""
Tests for the Query Expansion Module.

Validates:
- Known trigger expansion
- Passthrough for unrecognised queries
- Empty / blank input handling
- Custom concept maps
"""

from __future__ import annotations

import pytest

from retrieval.query_expander import QueryExpander


class TestQueryExpansion:
    """Core expansion behaviour."""

    def test_peak_load_expansion(self, query_expander: QueryExpander) -> None:
        """Queries containing 'peak load' must be expanded."""
        result = query_expander.expand("How does the system handle peak load?")
        assert len(result) > len("How does the system handle peak load?")
        assert "autoscaling" in result.lower() or "traffic" in result.lower()

    def test_downtime_expansion(self, query_expander: QueryExpander) -> None:
        """Queries containing 'downtime' must be expanded."""
        result = query_expander.expand("How is downtime prevented?")
        assert "availability" in result.lower() or "failover" in result.lower()

    def test_slow_requests_expansion(self, query_expander: QueryExpander) -> None:
        """Queries about slow requests must include performance terms."""
        result = query_expander.expand("How are slow requests reduced?")
        assert "latency" in result.lower() or "caching" in result.lower()

    def test_no_match_passthrough(self, query_expander: QueryExpander) -> None:
        """Queries with no trigger phrase must be returned unchanged."""
        original = "What is the meaning of life?"
        result = query_expander.expand(original)
        assert result == original

    def test_empty_input(self, query_expander: QueryExpander) -> None:
        """Empty / blank strings must be returned as-is."""
        assert query_expander.expand("") == ""
        assert query_expander.expand("   ") == "   "

    def test_expansion_is_longer(self, query_expander: QueryExpander) -> None:
        """Expanded query must always be longer than the original."""
        queries = [
            "How does the system handle peak load?",
            "How is downtime prevented?",
            "How are slow requests reduced?",
        ]
        for q in queries:
            expanded = query_expander.expand(q)
            assert len(expanded) > len(q), f"Expansion failed for: {q}"


class TestCustomConceptMap:
    """Verify that custom concept maps work correctly."""

    def test_custom_trigger(self) -> None:
        """Custom concept map should trigger on custom terms."""
        custom_map = {"banana": ["fruit", "potassium", "tropical"]}
        expander = QueryExpander(concept_map=custom_map)
        result = expander.expand("I like banana smoothies")
        assert "fruit" in result.lower()
        assert "potassium" in result.lower()

    def test_custom_no_match(self) -> None:
        """Custom map with no matching trigger returns original."""
        custom_map = {"banana": ["fruit"]}
        expander = QueryExpander(concept_map=custom_map)
        original = "I like apple juice"
        assert expander.expand(original) == original
