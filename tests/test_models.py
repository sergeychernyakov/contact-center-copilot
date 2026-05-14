"""Tests for Pydantic models and LangGraph state."""

from __future__ import annotations

from src.graph.models import (
    BenchmarkChunk,
    CriticReport,
    Metric,
    MetricsSet,
    MetricType,
    NarrativeReport,
)


def test_metric_formatted():
    m = Metric(name="FCR", metric_type=MetricType.FCR, value=68.5, unit="%")
    assert "FCR" in m.formatted()
    assert "68.5" in m.formatted()


def test_metricsset_to_dict():
    ms = MetricsSet(
        metrics=[
            Metric(name="FCR", value=68.0, metric_type=MetricType.FCR),
            Metric(name="AHT", value=420.0, metric_type=MetricType.AHT),
        ]
    )
    d = ms.to_dict()
    assert d == {"FCR": 68.0, "AHT": 420.0}


def test_benchmark_chunk_defaults():
    c = BenchmarkChunk(content="x")
    assert c.industry == "cross_industry"
    assert c.is_cross_industry is False


def test_critic_report_bounds():
    cr = CriticReport(faithfulness_score=0.9, numeric_accuracy_score=1.0)
    assert 0.0 <= cr.faithfulness_score <= 1.0
    assert cr.passes is False  # default


def test_narrative_report_default_empty():
    nr = NarrativeReport()
    assert nr.executive_summary == ""
    assert nr.requires_human_review is False
    assert nr.key_findings == []
