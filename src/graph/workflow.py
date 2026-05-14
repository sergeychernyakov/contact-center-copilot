"""LangGraph workflow assembly — wires agents into a self-correcting graph.

Flow:
    schema_inspector → extractor → benchmark_retriever → reporter
                                                          ↓
                                                       critic
                                                       ↙    ↘
                                      (retry if attempts<N)  (pass or maxed)
                                              ↑                    ↓
                                              └─── reporter ←   guardrails → END

The Critic→Reporter loop is the LangGraph killer feature — impossible
in plain LangChain because chains are DAGs.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from ..agents.benchmark_retriever import retrieve_benchmarks
from ..agents.critic import critique_report
from ..agents.extractor import extract_metrics
from ..agents.guardrails import apply_guardrails
from ..agents.reporter import generate_report
from ..agents.schema_inspector import inspect_schema
from ..config import get_settings
from .state import CopilotState


def _route_after_critic(state: CopilotState) -> str:
    """Conditional edge: retry reporter if critic failed and attempts < max."""
    settings = get_settings()
    critic = state.get("critic_report")
    attempts = state.get("attempts", 0)

    if critic is None:
        return "guardrails"
    if critic.passes:
        return "guardrails"
    if attempts >= settings.max_reporter_attempts:
        # Out of retries — pass to guardrails which will flag for human review
        return "guardrails"
    return "reporter"


def build_graph():
    """Compile the LangGraph workflow."""
    graph = StateGraph(CopilotState)

    # Nodes
    graph.add_node("schema_inspector", inspect_schema)
    graph.add_node("extractor", extract_metrics)
    graph.add_node("retriever", retrieve_benchmarks)
    graph.add_node("reporter", generate_report)
    graph.add_node("critic", critique_report)
    graph.add_node("guardrails", apply_guardrails)

    # Edges
    graph.set_entry_point("schema_inspector")
    graph.add_edge("schema_inspector", "extractor")
    graph.add_edge("extractor", "retriever")
    graph.add_edge("retriever", "reporter")
    graph.add_edge("reporter", "critic")

    # Conditional: critic decides retry vs progress
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"reporter": "reporter", "guardrails": "guardrails"},
    )

    graph.add_edge("guardrails", END)

    return graph.compile()


# Compile once at import time
WORKFLOW = build_graph()


async def run_pipeline(excel_path: str, industry: str = "cross_industry") -> CopilotState:
    """Convenience entry point — run the full pipeline end-to-end."""
    initial: CopilotState = {
        "excel_path": excel_path,
        "industry": industry,
        "attempts": 0,
        "feedback_history": [],
        "benchmark_chunks": [],
        "log": [],
    }
    return await WORKFLOW.ainvoke(initial)
