"""FastMCP server exposing Contact Center Copilot tools.

This makes the pipeline's capabilities available to any MCP-compatible client
(Claude Desktop, external orchestrators, other agents). Demonstrates how the
system integrates into a broader AI tooling ecosystem.

Usage: python -m src.mcp_server.server
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

from ..agents.extractor import _extract_from_sheet
from ..rag.retriever import BenchmarkRetriever
from ..tools.roi_calculator import aht_reduction_savings, fcr_improvement_savings

mcp = FastMCP("Contact Center Copilot")

# Lazy-init retriever
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


@mcp.tool()
def read_excel_sheets(path: str) -> dict:
    """List sheet names and column headers from an Excel file."""
    xls = pd.ExcelFile(path)
    out: dict[str, list[str]] = {}
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name, nrows=1)
        out[name] = [str(c) for c in df.columns]
    return out


@mcp.tool()
def extract_metrics(path: str, sheet_name: str) -> list[dict]:
    """Extract KPI metrics from a single sheet (deterministic, no LLM)."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    metrics = _extract_from_sheet(df, sheet_name)
    return [m.model_dump() for m in metrics]


@mcp.tool()
async def search_benchmarks(query: str, industry: str = "cross_industry", k: int = 5) -> list[dict]:
    """Search industry benchmark corpus and return ranked chunks."""
    retriever = _get_retriever()
    chunks = await retriever.retrieve(query, industry=industry, k=k)
    return [c.model_dump() for c in chunks]


@mcp.tool()
def estimate_fcr_roi(
    current_fcr_pct: float,
    target_fcr_pct: float,
    daily_call_volume: int,
) -> dict:
    """Estimate annual savings from improving First Call Resolution."""
    scenario = fcr_improvement_savings(current_fcr_pct, target_fcr_pct, daily_call_volume)
    return scenario.model_dump()


@mcp.tool()
def estimate_aht_roi(
    current_aht_seconds: float,
    target_aht_seconds: float,
    num_agents: int,
) -> dict:
    """Estimate annual savings from reducing Average Handle Time."""
    scenario = aht_reduction_savings(current_aht_seconds, target_aht_seconds, num_agents)
    return scenario.model_dump()


if __name__ == "__main__":
    mcp.run()
