"""Reporter agent — generates a NarrativeReport using the heavy LLM.

The LLM cannot invent numbers: only values from `state["metrics"]` are
allowed. Benchmark chunks are passed as the only external context.

The report is assembled from four small, focused structured-output calls
rather than one monolithic schema: smaller local models (e.g. Llama 3.1 8B
via Ollama) fill a 3-field schema reliably but leave large nested schemas
half-empty.
"""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..graph.models import BenchmarkComparison, Citation, NarrativeReport, ROIScenario
from ..graph.state import CopilotState
from ._llm import get_heavy_llm, llm_with_structured_output

# ─── Focused sub-schemas (reporter-internal report sections) ────────────────


class _ProseBlock(BaseModel):
    """Narrative prose section of the report."""

    executive_summary: str = Field(default="", description="2-3 sentence overview.")
    key_findings: list[str] = Field(
        default_factory=list,
        description="3-5 bullets, each referencing a specific metric.",
    )
    recommendations: list[str] = Field(default_factory=list, description="3 actionable items.")


class _AnalysisBlock(BaseModel):
    """Benchmark-comparison section of the report."""

    benchmark_analysis: list[BenchmarkComparison] = Field(default_factory=list)


class _RoiBlock(BaseModel):
    """ROI-projection section of the report."""

    roi_scenarios: list[ROIScenario] = Field(default_factory=list)


class _CitationsBlock(BaseModel):
    """Source-attribution section of the report."""

    citations: list[Citation] = Field(default_factory=list)


# ─── Prompts ────────────────────────────────────────────────────────────────

_CONTEXT = """You are an analyst writing a contact-center diagnostic report.

CLIENT METRICS (the ONLY numbers you may cite for the client):
{metrics_json}

INDUSTRY BENCHMARKS (cite as 'source':'industry'):
{benchmarks_text}
{feedback_block}
RULES: Never invent numbers — use only the values above. Flag any benchmark
marked 'cross-industry'. Professional, factual tone; no marketing fluff.

"""

_PROSE_PROMPT = ChatPromptTemplate.from_template(
    _CONTEXT
    + """Write the narrative prose of the report:
- executive_summary: 2-3 sentences
- key_findings: 3-5 bullets, each referencing a specific metric by name
- recommendations: 3 concrete, actionable items
"""
)

_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    _CONTEXT
    + """For each client metric that has a comparable industry benchmark, emit one
benchmark_analysis entry with: metric_name, client_value, benchmark_value,
benchmark_source, gap_percentage (= (client - benchmark) / benchmark * 100),
and direction (one of: above, below, on_par).
"""
)

_ROI_PROMPT = ChatPromptTemplate.from_template(
    _CONTEXT
    + """Produce 1-2 roi_scenarios for improving FCR or AHT, each with: name,
description, estimated_annual_savings_usd (a rough number), and confidence
(one of: low, medium, high).
"""
)

# Citations run last and receive the already-generated claims, so the model
# attributes real statements instead of guessing with no claims in context.
_CITATIONS_PROMPT = ChatPromptTemplate.from_template(
    """You are attributing the claims in a contact-center report to their sources.

REPORT CLAIMS:
{report_claims}

AVAILABLE BENCHMARK SOURCES:
{benchmark_sources}

For every claim above that draws on an industry benchmark, emit one citation
with: claim (the statement, quoted or closely paraphrased) and source (the
benchmark document name it came from). Emit at least one citation for each
claim that references a benchmark.
"""
)


def _feedback_block(history: list[str]) -> str:
    if not history:
        return ""
    items = "\n".join(f"- {item}" for item in history[-3:])
    return f"\nPREVIOUS CRITIC FEEDBACK — ADDRESS EACH:\n{items}\n"


async def generate_report(state: CopilotState) -> dict:
    """LangGraph node: generate (or regenerate) the narrative report.

    The report is built from four focused structured-output calls (prose,
    benchmark analysis, ROI, citations) so smaller local models can fill each
    section reliably instead of half-filling one large schema. Citations run
    last with the generated claims in context so they attribute real statements.
    """
    metrics = state.get("metrics")
    chunks = state.get("benchmark_chunks", [])
    history = state.get("feedback_history", [])
    attempts = state.get("attempts", 0)

    metrics_json = json.dumps(
        [m.model_dump() for m in metrics.metrics] if metrics else [],
        indent=2,
    )
    benchmarks_text = (
        "\n\n".join(
            f"[{c.source} | industry={c.industry}"
            + (" | cross-industry fallback" if c.is_cross_industry else "")
            + f"]\n{c.content}"
            for c in chunks
        )
        or "(none retrieved)"
    )
    context = {
        "metrics_json": metrics_json,
        "benchmarks_text": benchmarks_text,
        "feedback_block": _feedback_block(history),
    }

    llm = get_heavy_llm(temperature=0.2)
    prose: _ProseBlock = await (
        _PROSE_PROMPT | llm_with_structured_output(llm, _ProseBlock)
    ).ainvoke(context)
    analysis: _AnalysisBlock = await (
        _ANALYSIS_PROMPT | llm_with_structured_output(llm, _AnalysisBlock)
    ).ainvoke(context)
    roi: _RoiBlock = await (_ROI_PROMPT | llm_with_structured_output(llm, _RoiBlock)).ainvoke(
        context
    )
    # Citations run last: feed them the claims that were actually generated.
    report_claims = "\n".join(
        f"- {claim}" for claim in (prose.executive_summary, *prose.key_findings) if claim
    )
    cites: _CitationsBlock = await (
        _CITATIONS_PROMPT | llm_with_structured_output(llm, _CitationsBlock)
    ).ainvoke(
        {
            "report_claims": report_claims or "(no claims generated)",
            "benchmark_sources": ", ".join(sorted({c.source for c in chunks})) or "(none)",
        }
    )

    report = NarrativeReport(
        executive_summary=prose.executive_summary,
        key_findings=prose.key_findings,
        recommendations=prose.recommendations,
        benchmark_analysis=analysis.benchmark_analysis,
        roi_scenarios=roi.roi_scenarios,
        citations=cites.citations,
    )

    return {
        "narrative": report,
        "attempts": attempts + 1,
        "log": [f"[Reporter] Generated report (attempt {attempts + 1})"],
    }
