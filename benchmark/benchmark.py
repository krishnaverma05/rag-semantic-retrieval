"""
Benchmarking Module

Compares Strategy A (Raw Vector Search) and Strategy B (Semantic-Enhanced Retrieval)
across a suite of complex semantic queries.  Produces structured JSON output
and a human-readable markdown report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from retrieval.retriever import Retriever, RetrievalResult


# ---------------------------------------------------------------------------
# Default benchmark queries – intentionally vague/semantic so that query
# expansion can demonstrate measurable improvement.
# ---------------------------------------------------------------------------
DEFAULT_BENCHMARK_QUERIES: List[str] = [
    "How does the system handle peak load?",
    "How is downtime prevented?",
    "How are slow requests reduced?",
]


@dataclass
class QueryBenchmark:
    """Benchmark results for a single query across both strategies.

    Attributes:
        query: The original user query.
        raw_result: Strategy A output.
        enhanced_result: Strategy B output.
        analysis: Computed comparison metrics.
    """

    query: str
    raw_result: RetrievalResult
    enhanced_result: RetrievalResult
    analysis: Dict[str, Any] = field(default_factory=dict)


class RetrievalBenchmark:
    """Orchestrates comparative benchmarking of retrieval strategies.

    Attributes:
        retriever: The retrieval engine under test.
        queries: Queries to benchmark.
        top_k: Number of results to retrieve per query.
        results: Collected benchmark records.
    """

    def __init__(
        self,
        retriever: Retriever,
        queries: List[str] | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.queries = queries or DEFAULT_BENCHMARK_QUERIES
        self.top_k = top_k
        self.results: List[QueryBenchmark] = []

    # ------------------------------------------------------------------
    # Benchmark execution
    # ------------------------------------------------------------------

    def run(self) -> List[QueryBenchmark]:
        """Execute the benchmark suite.

        For each query, runs both strategies and computes comparative
        analytics.

        Returns:
            List of :class:`QueryBenchmark` instances.
        """
        self.results = []
        for query in self.queries:
            raw = self.retriever.retrieve_raw(query, top_k=self.top_k)
            enhanced = self.retriever.retrieve_enhanced(query, top_k=self.top_k)
            analysis = self._analyse(raw, enhanced)
            self.results.append(
                QueryBenchmark(
                    query=query,
                    raw_result=raw,
                    enhanced_result=enhanced,
                    analysis=analysis,
                )
            )
        return self.results

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _analyse(
        raw: RetrievalResult,
        enhanced: RetrievalResult,
    ) -> Dict[str, Any]:
        """Compute comparative metrics between two retrieval runs."""
        raw_ids = [r["doc_id"] for r in raw.results]
        enh_ids = [r["doc_id"] for r in enhanced.results]

        raw_scores = [r["score"] for r in raw.results]
        enh_scores = [r["score"] for r in enhanced.results]

        overlap = set(raw_ids) & set(enh_ids)
        only_raw = set(raw_ids) - set(enh_ids)
        only_enh = set(enh_ids) - set(raw_ids)

        # Rank shift: for overlapping docs, compare rank positions.
        rank_shifts: Dict[str, int] = {}
        for doc_id in overlap:
            raw_rank = raw_ids.index(doc_id) + 1
            enh_rank = enh_ids.index(doc_id) + 1
            rank_shifts[doc_id] = raw_rank - enh_rank  # +ve → promoted

        return {
            "overlap_count": len(overlap),
            "overlap_doc_ids": sorted(overlap),
            "only_in_raw": sorted(only_raw),
            "only_in_enhanced": sorted(only_enh),
            "raw_avg_score": round(sum(raw_scores) / max(len(raw_scores), 1), 6),
            "enhanced_avg_score": round(sum(enh_scores) / max(len(enh_scores), 1), 6),
            "score_improvement": round(
                (sum(enh_scores) / max(len(enh_scores), 1))
                - (sum(raw_scores) / max(len(raw_scores), 1)),
                6,
            ),
            "rank_shifts": rank_shifts,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_json(self) -> List[Dict[str, Any]]:
        """Serialise benchmark results to a JSON-compatible list."""
        output: List[Dict[str, Any]] = []
        for qb in self.results:
            output.append(
                {
                    "query": qb.query,
                    "expanded_query": qb.enhanced_result.processed_query,
                    "strategy_a_raw": qb.raw_result.to_dict(),
                    "strategy_b_enhanced": qb.enhanced_result.to_dict(),
                    "analysis": qb.analysis,
                }
            )
        return output

    def save_json(self, path: str | Path) -> None:
        """Persist benchmark results as a JSON file.

        Args:
            path: Destination file path.
        """
        payload = {
            "benchmark_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "num_queries": len(self.results),
                "top_k": self.top_k,
            },
            "results": self.to_json(),
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def generate_report(self) -> str:
        """Generate a human-readable markdown benchmark report.

        Returns:
            A multi-line markdown string.
        """
        lines: List[str] = [
            "# Retrieval Benchmark Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Queries:** {len(self.results)}  ",
            f"**Top-K:** {self.top_k}",
            "",
            "---",
            "",
        ]

        for idx, qb in enumerate(self.results, start=1):
            a = qb.analysis
            lines.append(f"## Query {idx}")
            lines.append("")
            lines.append(f"**Original:** {qb.query}  ")
            lines.append(f"**Expanded:** {qb.enhanced_result.processed_query}")
            lines.append("")

            # Strategy A table
            lines.append("### Strategy A — Raw Vector Search")
            lines.append("")
            lines.append("| Rank | Doc ID | Score | Snippet |")
            lines.append("|------|--------|-------|---------|")
            for r in qb.raw_result.results:
                snippet = r["document"][:80].replace("|", "\\|") + "…"
                lines.append(f"| {r['rank']} | {r['doc_id']} | {r['score']:.4f} | {snippet} |")
            lines.append("")

            # Strategy B table
            lines.append("### Strategy B — Semantic-Enhanced Retrieval")
            lines.append("")
            lines.append("| Rank | Doc ID | Score | Snippet |")
            lines.append("|------|--------|-------|---------|")
            for r in qb.enhanced_result.results:
                snippet = r["document"][:80].replace("|", "\\|") + "…"
                lines.append(f"| {r['rank']} | {r['doc_id']} | {r['score']:.4f} | {snippet} |")
            lines.append("")

            # Comparison summary
            lines.append("### Comparison")
            lines.append("")
            lines.append(f"- **Overlapping documents:** {a['overlap_count']}")
            lines.append(f"- **Only in Raw:** {a['only_in_raw']}")
            lines.append(f"- **Only in Enhanced:** {a['only_in_enhanced']}")
            lines.append(f"- **Raw avg score:** {a['raw_avg_score']:.4f}")
            lines.append(f"- **Enhanced avg score:** {a['enhanced_avg_score']:.4f}")
            lines.append(f"- **Score improvement:** {a['score_improvement']:+.4f}")
            if a["rank_shifts"]:
                lines.append(f"- **Rank shifts:** {a['rank_shifts']}")
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def save_report(self, path: str | Path) -> None:
        """Write the markdown report to disk.

        Args:
            path: Destination file path (typically ``retrieval_benchmark.md``).
        """
        Path(path).write_text(self.generate_report(), encoding="utf-8")
