"""Smoke tests for the deterministic Pandas Extractor (no LLM)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.agents.extractor import _classify, _extract_from_sheet, extract_metrics
from src.graph.models import MetricType


def test_classify_fcr():
    mtype, unit = _classify("FCR")
    assert mtype == MetricType.FCR
    assert unit == "%"


def test_classify_aht_variations():
    for name in ["AHT", "AHT_seconds", "Average Handle Time", "average-handle-time"]:
        mtype, _ = _classify(name)
        assert mtype == MetricType.AHT, f"Failed for {name}"


def test_classify_unknown_column():
    mtype, unit = _classify("Random_Column_X")
    assert mtype == MetricType.OTHER
    assert unit == ""


def test_extract_from_sheet_simple():
    df = pd.DataFrame(
        {
            "FCR": [0.68, 0.70, 0.66, 0.72],
            "AHT_seconds": [420, 410, 430, 415],
            "CallVolume": [3500, 3800, 4100, 3900],
        }
    )
    metrics = _extract_from_sheet(df, "Q3_Performance")
    assert len(metrics) == 3

    fcr = next(m for m in metrics if m.name == "FCR")
    assert fcr.metric_type == MetricType.FCR
    assert fcr.unit == "%"
    # mean of [0.68, 0.70, 0.66, 0.72] = 0.69 → scaled to 69%
    assert 68.0 <= fcr.value <= 70.0


def test_extract_skips_non_numeric():
    df = pd.DataFrame(
        {
            "AgentName": ["Alice", "Bob"],
            "FCR": [0.75, 0.78],
        }
    )
    metrics = _extract_from_sheet(df, "Agents")
    names = {m.name for m in metrics}
    assert "AgentName" not in names
    assert "FCR" in names


@pytest.mark.asyncio
async def test_extract_metrics_node_with_no_inspection(tmp_path: Path):
    """Extractor should fall back to all sheets when inspection is missing."""
    excel_path = tmp_path / "tiny.xlsx"
    pd.DataFrame({"FCR": [0.7, 0.72]}).to_excel(excel_path, sheet_name="Perf", index=False)

    state = {"excel_path": str(excel_path)}
    result = await extract_metrics(state)

    assert "metrics" in result
    assert result["metrics"].metrics
    assert any(m.name == "FCR" for m in result["metrics"].metrics)
