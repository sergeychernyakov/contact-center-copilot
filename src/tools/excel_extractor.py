"""Excel extraction helpers — pure pandas, no LLM.

Kept thin: the heavy lifting lives in `src/agents/extractor.py` as a LangGraph
node. This module just exposes utility functions for the FastMCP server and
standalone use.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..graph.models import MetricsSet


def list_sheets(path: str | Path) -> dict[str, list[str]]:
    """Return {sheet_name: [column_headers]} for an Excel file."""
    xls = pd.ExcelFile(path)
    out: dict[str, list[str]] = {}
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name, nrows=1)
        out[name] = [str(c) for c in df.columns]
    return out


def extract_all(path: str | Path) -> MetricsSet:
    """Extract metrics from all sheets (no schema filtering)."""
    from ..agents.extractor import _extract_from_sheet

    xls = pd.ExcelFile(path)
    all_metrics = []
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name)
        all_metrics.extend(_extract_from_sheet(df, name))
    return MetricsSet(metrics=all_metrics, source_file=str(path))
