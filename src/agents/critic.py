"""Critic agent — validates the narrative for hallucinations and faithfulness."""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from ..config import get_settings
from ..graph.models import CriticReport, NarrativeReport
from ..graph.state import CopilotState
from ._llm import get_heavy_llm, llm_with_structured_output

NUMBER_RE = re.compile(r"\b\d+\.?\d*\b")

PROMPT = ChatPromptTemplate.from_template(
    """You are a strict reviewer checking an AI-generated report for faithfulness.

THE REPORT:
{report_text}

ALLOWED FACTS (client metrics + benchmark sources):
{allowed_facts}

CURRENT ATTEMPT: {attempt} of {max_attempts}

Rate the report on:
- faithfulness_score (0.0-1.0): are all claims supported by allowed facts?
- numeric_accuracy_score (0.0-1.0): are all numbers traceable to allowed facts?

List specific issues and suggestions. Set passes=True only if
faithfulness_score >= {threshold}.

If this is the FINAL attempt ({attempt} == {max_attempts}) and faithfulness
is at least {threshold_relaxed}, set passes=True.
"""
)


def _allowed_numbers(state: CopilotState) -> set[str]:
    allowed: set[str] = set()
    metrics = state.get("metrics")
    if metrics:
        for m in metrics.metrics:
            allowed.add(f"{m.value:g}")
            allowed.add(f"{round(m.value):d}")
            allowed.add(f"{round(m.value, 1):g}")
    for chunk in state.get("benchmark_chunks", []):
        for num in NUMBER_RE.findall(chunk.content):
            allowed.add(num)
    return allowed


def _report_text(report: NarrativeReport) -> str:
    parts = [report.executive_summary]
    parts.extend(report.key_findings)
    parts.extend(report.recommendations)
    return "\n".join(parts)


def _numeric_accuracy(report: NarrativeReport, allowed: set[str]) -> tuple[float, list[str]]:
    text = _report_text(report)
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return 1.0, []
    bad = [n for n in numbers if n not in allowed]
    score = 1.0 - (len(bad) / len(numbers))
    issues = [f"Unsupported number: {n}" for n in set(bad)]
    return score, issues


async def critique_report(state: CopilotState) -> dict:
    """LangGraph node: critique the narrative report."""
    settings = get_settings()
    report = state.get("narrative")
    attempt = state.get("attempts", 0)
    max_att = settings.max_reporter_attempts

    if report is None:
        return {"log": ["[Critic] No report to review"]}

    # 1. Deterministic numeric check
    allowed = _allowed_numbers(state)
    num_score, num_issues = _numeric_accuracy(report, allowed)

    # 2. LLM-based faithfulness check
    metrics = state.get("metrics")
    allowed_facts = "Metrics:\n" + "\n".join(
        m.formatted() for m in (metrics.metrics if metrics else [])
    )
    allowed_facts += "\n\nBenchmarks:\n" + "\n".join(
        f"- {c.source}: {c.content[:200]}..." for c in state.get("benchmark_chunks", [])[:5]
    )

    llm = llm_with_structured_output(get_heavy_llm(temperature=0), CriticReport)
    critic_llm: CriticReport = await (PROMPT | llm).ainvoke(
        {
            "report_text": _report_text(report),
            "allowed_facts": allowed_facts,
            "attempt": attempt,
            "max_attempts": max_att,
            "threshold": settings.faithfulness_threshold,
            "threshold_relaxed": max(0.7, settings.faithfulness_threshold - 0.15),
        }
    )

    final = CriticReport(
        faithfulness_score=critic_llm.faithfulness_score,
        numeric_accuracy_score=num_score,
        issues=num_issues + critic_llm.issues,
        suggestions=critic_llm.suggestions,
        passes=critic_llm.passes and num_score >= 0.99,
    )

    history = state.get("feedback_history", [])
    if not final.passes:
        history = history + final.issues[:3]

    return {
        "critic_report": final,
        "feedback_history": history,
        "log": [
            f"[Critic] faithfulness={final.faithfulness_score:.2f}, "
            f"numeric={final.numeric_accuracy_score:.2f}, "
            f"passes={final.passes}"
        ],
    }
