"""Pandas Extractor — DETERMINISTIC metric extraction (no LLM).

This is the core anti-hallucination pattern: numbers come from pandas, not LLM.
LLMs in later stages can only describe these numbers, never invent new ones.
See docs/challenge-1-hallucinations.md.
"""

from __future__ import annotations

import re

import pandas as pd

from ..graph.models import Metric, MetricsSet, MetricType
from ..graph.state import CopilotState

# Heuristic mapping of column name patterns to canonical metric types
KEYWORD_MAP: dict[str, tuple[MetricType, str]] = {
    r"\bfcr\b|first[\s_-]?call[\s_-]?resolution": (MetricType.FCR, "%"),
    r"\baht\b|average[\s_-]?handle|handle[\s_-]?time": (MetricType.AHT, "s"),
    r"\bcsat\b|customer[\s_-]?satisfaction": (MetricType.CSAT, "%"),
    r"\bnps\b|net[\s_-]?promoter": (MetricType.NPS, ""),
    r"agent[\s_-]?(util|occ)|utilization": (MetricType.AGENT_UTILIZATION, "%"),
    r"abandon[\s_-]?rate": (MetricType.ABANDON_RATE, "%"),
    r"service[\s_-]?level": (MetricType.SERVICE_LEVEL, "%"),
    r"occupancy": (MetricType.OCCUPANCY, "%"),
}


def _classify(col_name: str) -> tuple[MetricType, str]:
    """Map a column name to a canonical metric type + unit."""
    lower = col_name.lower()
    for pattern, (mtype, unit) in KEYWORD_MAP.items():
        if re.search(pattern, lower):
            return mtype, unit
    return MetricType.OTHER, ""


def _extract_from_sheet(df: pd.DataFrame, sheet_name: str) -> list[Metric]:
    """Pull numeric KPIs from a single sheet using simple heuristics."""
    metrics: list[Metric] = []

    for col in df.columns:
        col_str = str(col)
        if df[col].dtype not in ("float64", "int64", "float32", "int32"):
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        mtype, unit = _classify(col_str)
        # Aggregation: prefer mean unless the metric is a count
        value = float(series.mean())

        # Heuristic unit detection from values
        if not unit:
            if series.between(0, 1).all():
                unit = "%"
                value = value * 100
            elif series.between(0, 100).all():
                unit = "%"

        metrics.append(
            Metric(
                name=col_str,
                metric_type=mtype,
                value=round(value, 2),
                unit=unit,
                source_sheet=sheet_name,
            )
        )

    return metrics


async def extract_metrics(state: CopilotState) -> dict:
    """LangGraph node: deterministic pandas-based metric extraction."""
    inspection = state.get("schema_inspection")
    excel_path = state["excel_path"]

    if inspection is None:
        # Fallback: process all sheets
        xls = pd.ExcelFile(excel_path)
        sheets = xls.sheet_names
    else:
        sheets = inspection.relevant_sheets

    all_metrics: list[Metric] = []
    for sheet_name in sheets:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            all_metrics.extend(_extract_from_sheet(df, sheet_name))
        except Exception as exc:  # noqa: BLE001
            return {"log": [f"[Extractor] Failed to read '{sheet_name}': {exc}"]}

    metrics_set = MetricsSet(metrics=all_metrics, source_file=excel_path)

    return {
        "metrics": metrics_set,
        "log": [f"[Extractor] Extracted {len(all_metrics)} metrics from {len(sheets)} sheet(s)"],
    }
