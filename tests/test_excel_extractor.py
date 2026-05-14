"""Tests for the pandas-based Excel extraction helpers (no LLM)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.graph.models import MetricsSet
from src.tools.excel_extractor import extract_all, list_sheets


def _make_workbook(path: Path) -> None:
    """Write a two-sheet workbook used by the extraction tests."""
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"FCR": [0.68, 0.70], "AHT_seconds": [410, 420]}).to_excel(
            writer, sheet_name="Q3", index=False
        )
        pd.DataFrame({"AgentName": ["Alice", "Bob"]}).to_excel(
            writer, sheet_name="Roster", index=False
        )


def test_list_sheets_returns_headers(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    _make_workbook(path)

    sheets = list_sheets(path)

    assert set(sheets) == {"Q3", "Roster"}
    assert sheets["Q3"] == ["FCR", "AHT_seconds"]
    assert sheets["Roster"] == ["AgentName"]


def test_extract_all_collects_metrics_across_sheets(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    _make_workbook(path)

    result = extract_all(path)

    assert isinstance(result, MetricsSet)
    assert result.source_file == str(path)
    names = {m.name for m in result.metrics}
    assert {"FCR", "AHT_seconds"} <= names
    assert "AgentName" not in names  # non-numeric column is skipped
