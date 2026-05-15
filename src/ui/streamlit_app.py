# src/ui/streamlit_app.py
"""Streamlit UI for the Contact Center Copilot.

Usage: streamlit run src/ui/streamlit_app.py

The page streams pipeline progress live (per-agent ✅/⏳ rows, Critic iteration
cards), then renders the final NarrativeReport with KPI metrics, a Plotly
benchmark chart, ROI scenarios, recommendations and citations.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from typing import Any, Literal

import plotly.graph_objects as go
import streamlit as st

from src.config import get_settings
from src.graph.models import BenchmarkComparison, CriticReport, NarrativeReport
from src.graph.workflow import run_pipeline_streaming

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Contact Center Copilot",
    page_icon="📞",
    layout="wide",
)

st.title("📞 Contact Center Copilot")
st.caption(
    "Multi-agent diagnostic system — upload an Excel, watch six agents run "
    "live, and get a narrative report with benchmark comparisons, ROI "
    "scenarios and citations."
)


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    industry = st.selectbox(
        "Industry",
        ["banking", "retail", "cross_industry"],
        index=0,
        help="Only industries with a benchmark dataset in data/benchmarks/.",
    )
    settings = get_settings()
    st.caption(f"**Provider:** `{settings.llm_provider}`  ·  model: `{settings.model_heavy}`")
    st.divider()
    st.markdown(
        "**Pipeline:**\n"
        "1. Schema Inspector (fast LLM)\n"
        "2. Pandas Extractor (deterministic)\n"
        "3. Benchmark Retriever (FAISS + BM25)\n"
        "4. Reporter (heavy LLM)\n"
        "5. Critic → loop if needed\n"
        "6. Guardrails (PII / tone)"
    )
    st.divider()
    st.markdown(
        "[GitHub repo](https://github.com/sergeychernyakov/contact-center-copilot) · "
        "Built for PwC AppDev"
    )


# ─── Node metadata ──────────────────────────────────────────────────────────

NODE_ORDER: list[tuple[str, str, str]] = [
    ("schema_inspector", "🔍 Schema Inspector", "scanning sheets for KPI columns"),
    ("extractor", "📊 Pandas Extractor", "reading values deterministically"),
    ("retriever", "📚 Benchmark Retriever", "hybrid search over benchmarks"),
    ("reporter", "✍️ Reporter", "drafting narrative + ROI"),
    ("critic", "🔬 Critic", "checking faithfulness"),
    ("guardrails", "🛡️ Guardrails", "PII redaction + tone"),
]
NODE_LABEL = {n: (label, hint) for n, label, hint in NODE_ORDER}

# Metrics where a lower value is better (so a positive gap is bad).
LOWER_IS_BETTER = ("aht", "abandon", "handle", "occupancy", "wait")


def _next_node(just_completed: str, state: dict, max_attempts: int) -> str | None:
    """Mirror of workflow._route_after_critic — tells the UI which node fires next."""
    if just_completed == "schema_inspector":
        return "extractor"
    if just_completed == "extractor":
        return "retriever"
    if just_completed == "retriever":
        return "reporter"
    if just_completed == "reporter":
        return "critic"
    if just_completed == "critic":
        critic = state.get("critic_report")
        attempts = state.get("attempts", 0)
        if critic is None or critic.passes or attempts >= max_attempts:
            return "guardrails"
        return "reporter"
    return None  # after guardrails: done


# ─── Live progress helpers ──────────────────────────────────────────────────


def _row_pending(label: str, hint: str) -> str:
    """Render a progress row for a node that has not started yet."""
    return f"⚪&nbsp;&nbsp;{label} &nbsp;·&nbsp; <span style='opacity:.55'>{hint}</span>"


def _row_running(label: str, hint: str, attempt: int) -> str:
    """Render a progress row for the currently executing node."""
    suffix = f" (attempt {attempt})" if attempt > 1 else ""
    return (
        f"⏳&nbsp;&nbsp;**{label}**{suffix} &nbsp;·&nbsp; <span style='opacity:.7'>{hint}…</span>"
    )


def _row_done(label: str, elapsed: float, run_count: int) -> str:
    """Render a progress row for a completed node, with timing and run-count suffix."""
    suffix = f" (x{run_count})" if run_count > 1 else ""
    return f"✅&nbsp;&nbsp;{label}{suffix} &nbsp;·&nbsp; `{elapsed:.1f}s`"


def _render_critic_cards(holder: Any, attempts: list[dict]) -> None:
    """Stack one card per Critic attempt, latest at the bottom."""
    if not attempts:
        holder.empty()
        return
    lines = ["**🔬 Critic iterations**"]
    for a in attempts:
        icon = "✅" if a["passes"] else "❌"
        lines.append(
            f"- {icon} **Attempt {a['attempt']}** &nbsp;·&nbsp; "
            f"faithfulness `{a['faithfulness']:.2f}` &nbsp;·&nbsp; "
            f"numeric `{a['numeric']:.2f}` &nbsp;·&nbsp; "
            f"{'PASS' if a['passes'] else 'retry'}"
        )
    holder.markdown("\n".join(lines), unsafe_allow_html=True)


# ─── Final-report rendering ─────────────────────────────────────────────────


def _delta_color(name: str, gap: float) -> Literal["normal", "inverse", "off"]:
    """Pick `st.metric` delta color based on whether the metric is better when lower."""
    lower_is_better = any(token in name.lower() for token in LOWER_IS_BETTER)
    if abs(gap) < 0.5:
        return "off"
    going_up = gap > 0
    good = (going_up and not lower_is_better) or (not going_up and lower_is_better)
    return "normal" if good else "inverse"


def _kpi_cards(comparisons: list[BenchmarkComparison]) -> None:
    """Big KPI tiles for up to 4 comparisons, with delta arrows vs benchmark."""
    if not comparisons:
        return
    top = comparisons[:4]
    cols = st.columns(len(top))
    for col, c in zip(cols, top, strict=False):
        with col:
            st.metric(
                label=c.metric_name,
                value=f"{c.client_value:g}",
                delta=f"{c.gap_percentage:+.1f}% vs benchmark",
                delta_color=_delta_color(c.metric_name, c.gap_percentage),
            )


def _benchmark_chart(comparisons: list[BenchmarkComparison]) -> go.Figure:
    """Horizontal grouped bars — client vs benchmark for every comparison."""
    labels = [c.metric_name for c in comparisons]
    client = [c.client_value for c in comparisons]
    bench = [c.benchmark_value for c in comparisons]
    fig = go.Figure(
        data=[
            go.Bar(
                name="Client",
                y=labels,
                x=client,
                orientation="h",
                marker_color="#3B82F6",
                text=[f"{v:g}" for v in client],
                textposition="outside",
                hovertemplate="%{y}: %{x:g}<extra>Client</extra>",
            ),
            go.Bar(
                name="Benchmark",
                y=labels,
                x=bench,
                orientation="h",
                marker_color="#9CA3AF",
                text=[f"{v:g}" for v in bench],
                textposition="outside",
                hovertemplate="%{y}: %{x:g}<extra>Benchmark</extra>",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        height=80 + 70 * len(labels),
        margin={"t": 20, "l": 10, "r": 30, "b": 20},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        font={"color": "#E5E7EB"},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(autorange="reversed")
    return fig


def _render_report(state: dict, run_seconds: float) -> None:
    """Render the final NarrativeReport with KPI tiles, chart and sections."""
    report: NarrativeReport | None = state.get("narrative")
    critic: CriticReport | None = state.get("critic_report")

    if state.get("requires_human_review") and report is not None:
        st.warning("⚠️ This report is flagged for human review.")
        for reason in report.review_reasons:
            st.caption(f"• {reason}")

    if report is None:
        st.error("Pipeline finished without producing a report.")
        return

    if report.benchmark_analysis:
        st.subheader("📊 Headline KPIs vs benchmarks")
        _kpi_cards(report.benchmark_analysis)
        st.caption(
            f"Generated in **{run_seconds:.1f}s** on `{get_settings().llm_provider}` · "
            f"final Critic faithfulness "
            f"**{critic.faithfulness_score:.2f}**"
            if critic
            else f"Generated in **{run_seconds:.1f}s**"
        )

    st.subheader("📋 Executive Summary")
    st.write(report.executive_summary)

    st.subheader("🔑 Key Findings")
    for finding in report.key_findings:
        st.markdown(f"- {finding}")

    if report.benchmark_analysis:
        st.subheader("📊 Benchmark Analysis")
        st.plotly_chart(_benchmark_chart(report.benchmark_analysis), use_container_width=True)

    if report.roi_scenarios:
        st.subheader("💰 ROI Scenarios")
        for s in report.roi_scenarios:
            st.markdown(f"**{s.name}** — {s.description}")
            st.caption(
                f"Estimated annual savings: ${s.estimated_annual_savings_usd:,.0f} "
                f"(confidence: {s.confidence})"
            )

    if report.recommendations:
        st.subheader("✅ Recommendations")
        for r in report.recommendations:
            st.markdown(f"- {r}")

    if report.citations:
        with st.expander("📚 Citations"):
            for c in report.citations:
                st.markdown(f"- **{c.source}** — {c.claim}")

    with st.expander("🔍 Pipeline diagnostics"):
        if critic:
            st.markdown(
                f"**Critic scores:** faithfulness {critic.faithfulness_score:.2f}, "
                f"numeric accuracy {critic.numeric_accuracy_score:.2f}"
            )
        metrics = state.get("metrics")
        if metrics:
            st.markdown(f"**Extracted metrics:** {len(metrics.metrics)}")
            st.json([m.model_dump() for m in metrics.metrics])
        if state.get("pii_redactions"):
            st.markdown(f"**PII redacted:** {len(state['pii_redactions'])} items")
        st.markdown("**Logs:**")
        for line in state.get("log", []):
            st.code(line, language=None)


# ─── Streaming driver ───────────────────────────────────────────────────────


async def _run_with_live_progress(
    excel_path: str, industry: str, placeholders: dict, critic_holder: Any
) -> tuple[dict, float]:
    """Consume run_pipeline_streaming and update the UI as each node finishes."""
    max_attempts = get_settings().max_reporter_attempts
    run_counts: dict[str, int] = {n: 0 for n, _, _ in NODE_ORDER}
    started_at: dict[str, float] = {}
    elapsed_total: dict[str, float] = {n: 0.0 for n, _, _ in NODE_ORDER}
    critic_attempts: list[dict] = []

    # The pipeline starts at schema_inspector
    first = "schema_inspector"
    label, hint = NODE_LABEL[first]
    started_at[first] = time.time()
    placeholders[first].markdown(_row_running(label, hint, 1), unsafe_allow_html=True)

    pipeline_start = time.time()
    final_state: dict = {}
    async for node, state in run_pipeline_streaming(excel_path, industry):
        now = time.time()
        run_counts[node] += 1
        elapsed_total[node] += now - started_at.get(node, now)
        label, _hint = NODE_LABEL[node]
        placeholders[node].markdown(
            _row_done(label, elapsed_total[node], run_counts[node]),
            unsafe_allow_html=True,
        )

        if node == "critic":
            critic = state.get("critic_report")
            if critic is not None:
                critic_attempts.append(
                    {
                        "attempt": state.get("attempts", len(critic_attempts) + 1),
                        "faithfulness": critic.faithfulness_score,
                        "numeric": critic.numeric_accuracy_score,
                        "passes": critic.passes,
                    }
                )
                _render_critic_cards(critic_holder, critic_attempts)

        nxt = _next_node(node, state, max_attempts)
        if nxt is not None:
            label_nxt, hint_nxt = NODE_LABEL[nxt]
            attempt_nxt = run_counts[nxt] + 1
            started_at[nxt] = now
            placeholders[nxt].markdown(
                _row_running(label_nxt, hint_nxt, attempt_nxt),
                unsafe_allow_html=True,
            )
        final_state = state

    return final_state, time.time() - pipeline_start


# ─── Main flow ──────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload contact-center diagnostic Excel",
    type=["xlsx", "xls"],
    help="Sheets with FCR, AHT, CSAT, agent utilization, etc.",
)
run_button = st.button("▶️ Run Analysis", type="primary", disabled=uploaded is None)

if run_button and uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        excel_path = tmp.name

    progress_box = st.container(border=True)
    with progress_box:
        st.markdown("**🚀 Pipeline progress**")
        node_placeholders = {n: st.empty() for n, _, _ in NODE_ORDER}
        for node_id, lbl, hnt in NODE_ORDER:
            node_placeholders[node_id].markdown(_row_pending(lbl, hnt), unsafe_allow_html=True)
        critic_holder = st.empty()

    final_state: dict = {}
    duration: float = 0.0
    try:
        final_state, duration = asyncio.run(
            _run_with_live_progress(excel_path, industry, node_placeholders, critic_holder)
        )
        st.toast(f"✅ Pipeline complete in {duration:.1f}s", icon="🎉")
    except Exception as exc:
        st.error(f"❌ Pipeline failed: {exc}")
        st.exception(exc)
        final_state = {}

    if final_state:
        st.divider()
        _render_report(final_state, duration)
elif not uploaded:
    st.info("👆 Upload an Excel file to begin. A sample is at `data/sample_diagnostic.xlsx`.")
