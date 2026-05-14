"""Reporter agent — generates a NarrativeReport using Sonnet.

The LLM cannot invent numbers: only values from `state["metrics"]` are
allowed. Benchmark chunks are passed as the only external context.
"""

from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from ..config import get_settings
from ..graph.models import NarrativeReport
from ..graph.state import CopilotState

PROMPT = ChatPromptTemplate.from_template(
    """You are an HR-grade analyst writing a contact-center diagnostic report.

CLIENT METRICS (these are the ONLY numbers you may cite for the client):
{metrics_json}

INDUSTRY BENCHMARKS (cite these as 'source':'industry'):
{benchmarks_text}

{feedback_block}

Generate a narrative report with:
- executive_summary (2-3 sentences)
- key_findings (3-5 bullets, each referencing a metric)
- benchmark_analysis (compare each client metric to benchmark)
- roi_scenarios (1-2 scenarios with rough annual savings if FCR or AHT improve)
- recommendations (3 actionable items)
- citations (link claims to benchmark sources)

CRITICAL RULES:
1. Never invent numbers. Only use values from CLIENT METRICS or INDUSTRY BENCHMARKS.
2. If a benchmark is marked 'cross-industry', state so explicitly.
3. Keep tone professional, factual, no marketing fluff.
"""
)


def _feedback_block(history: list[str]) -> str:
    if not history:
        return ""
    items = "\n".join(f"- {item}" for item in history[-3:])
    return f"\nPREVIOUS CRITIC FEEDBACK — ADDRESS EACH:\n{items}\n"


async def generate_report(state: CopilotState) -> dict:
    """LangGraph node: generate (or regenerate) the narrative report."""
    settings = get_settings()
    metrics = state.get("metrics")
    chunks = state.get("benchmark_chunks", [])
    history = state.get("feedback_history", [])
    attempts = state.get("attempts", 0)

    metrics_json = json.dumps(
        [m.model_dump() for m in metrics.metrics] if metrics else [],
        indent=2,
    )
    benchmarks_text = "\n\n".join(
        f"[{c.source} | industry={c.industry}"
        + (" | cross-industry fallback" if c.is_cross_industry else "")
        + f"]\n{c.content}"
        for c in chunks
    )

    llm = ChatAnthropic(
        model=settings.model_heavy,
        temperature=0.2,
        api_key=settings.anthropic_api_key,
    ).with_structured_output(NarrativeReport)

    report: NarrativeReport = await (PROMPT | llm).ainvoke(
        {
            "metrics_json": metrics_json,
            "benchmarks_text": benchmarks_text or "(none retrieved)",
            "feedback_block": _feedback_block(history),
        }
    )

    return {
        "narrative": report,
        "attempts": attempts + 1,
        "log": [f"[Reporter] Generated report (attempt {attempts + 1})"],
    }
