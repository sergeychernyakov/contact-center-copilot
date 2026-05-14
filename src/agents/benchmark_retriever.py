"""Benchmark Retriever agent — fetches industry benchmark chunks for each metric."""

from __future__ import annotations

from pathlib import Path

from ..graph.state import CopilotState
from ..rag.retriever import BenchmarkRetriever

# Lazy singleton — initialised on first call to avoid slow imports
_retriever: BenchmarkRetriever | None = None


def _get_retriever() -> BenchmarkRetriever:
    global _retriever
    if _retriever is None:
        root = Path(__file__).resolve().parents[2]
        _retriever = BenchmarkRetriever(
            benchmarks_dir=root / "data" / "benchmarks",
            index_path=root / "vector_store" / "benchmarks",
        )
    return _retriever


async def retrieve_benchmarks(state: CopilotState) -> dict:
    """LangGraph node: fetch benchmark chunks for the extracted metrics."""
    metrics = state.get("metrics")
    industry = state.get("industry", "cross_industry")

    if metrics is None or not metrics.metrics:
        return {
            "benchmark_chunks": [],
            "log": ["[Retriever] No metrics to look up benchmarks for"],
        }

    retriever = _get_retriever()

    # Build a single query that covers all metric names — efficient & cheap
    metric_names = ", ".join(m.name for m in metrics.metrics[:10])
    query = f"Industry benchmarks for: {metric_names}"

    chunks = await retriever.retrieve(query, industry=industry, k=8)

    cross_ind_count = sum(1 for c in chunks if c.is_cross_industry)
    msg = f"[Retriever] {len(chunks)} chunks ({cross_ind_count} cross-industry fallback)"

    return {
        "benchmark_chunks": chunks,
        "log": [msg],
    }
