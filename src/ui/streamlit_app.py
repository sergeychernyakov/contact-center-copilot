# src/ui/streamlit_app.py
"""Streamlit UI for the Contact Center Copilot.

Usage: streamlit run src/ui/streamlit_app.py

The page does four things visually:
    1. Right after the Excel is dropped, render a sheets-preview card so the
       page is never empty while the user decides to run.
    2. While the pipeline runs, paint a Graphviz workflow diagram that
       highlights the active node + colour-codes completed ones, and a list
       of agent rows that each expand into a detail card on completion
       (sheets, KPI chips, retrieved benchmarks, etc.) so the screen fills
       with real content as agents finish — not just a checklist.
    3. Stack Critic-iteration cards live as the self-correction loop runs.
    4. After the loop ends, render the final NarrativeReport with KPI tiles
       (delta arrows vs benchmark), a Plotly bar chart, sections and
       citations.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
import time
from typing import Any, Literal

import pandas as pd
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


# ─── Architecture diagram (live-updating graphviz) ──────────────────────────

_DOT_PALETTE: dict[str, dict[str, str]] = {
    "pending": {"fillcolor": "#1F2937", "fontcolor": "#9CA3AF", "color": "#374151"},
    "active": {"fillcolor": "#1D4ED8", "fontcolor": "white", "color": "#60A5FA", "penwidth": "2"},
    "done": {"fillcolor": "#065F46", "fontcolor": "#D1FAE5", "color": "#10B981"},
}


def _arch_dot(active: str = "", completed: set[str] | None = None) -> str:
    """Return a graphviz DOT diagram of the workflow with node states baked in."""
    completed = completed or set()

    def status(n: str) -> str:
        if n == active:
            return "active"
        if n in completed:
            return "done"
        return "pending"

    def attrs(node_id: str) -> str:
        return ", ".join(f'{k}="{v}"' for k, v in _DOT_PALETTE[status(node_id)].items())

    nodes_dot = "\n".join(
        f'  {nid} [label="{label}", style="filled,rounded", {attrs(nid)}];'
        for nid, label, _ in NODE_ORDER
    )
    end_color = "#10B981" if "guardrails" in completed else "#374151"
    end_fill = "#065F46" if "guardrails" in completed else "#111827"
    return f"""digraph G {{
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box, fontname="Inter", fontsize=12, margin="0.18,0.1"];
  edge [color="#4B5563", fontcolor="#9CA3AF", fontsize=10, fontname="Inter"];
{nodes_dot}
  END [label="📄 Report", shape=oval, style="filled,rounded",
       fillcolor="{end_fill}", color="{end_color}", fontcolor="#D1FAE5"];

  schema_inspector -> extractor -> retriever -> reporter -> critic;
  critic -> guardrails [label="pass / max"];
  critic -> reporter [label="retry", style=dashed, color="#F59E0B", fontcolor="#F59E0B"];
  guardrails -> END;
}}"""


# ─── File preview (shown right after upload, before Run) ────────────────────


def _render_file_preview(uploaded: Any) -> None:
    """Render a sheets-summary card for the uploaded Excel."""
    try:
        xls = pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
    except Exception as exc:
        st.error(f"Couldn't read the Excel file: {exc}")
        return

    rows = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        rows.append(
            {
                "Sheet": sheet,
                "Rows": len(df),
                "Columns": len(df.columns),
                "First columns": ", ".join(str(c) for c in df.columns[:4])
                + (" …" if len(df.columns) > 4 else ""),
            }
        )
    with st.container(border=True):
        st.markdown(
            f"**📂 `{uploaded.name}`** &nbsp;·&nbsp; "
            f"{len(xls.sheet_names)} sheets &nbsp;·&nbsp; "
            f"{len(uploaded.getvalue()) / 1024:.1f} KB"
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ─── Per-agent live detail rendering ────────────────────────────────────────


def _detail_schema(state: dict) -> str:
    """Render Schema Inspector results as a chip list of relevant sheets."""
    insp = state.get("schema_inspection")
    if not insp:
        return ""
    rel = " ".join(f"`{s}`" for s in insp.relevant_sheets[:6])
    more = f" +{len(insp.relevant_sheets) - 6}" if len(insp.relevant_sheets) > 6 else ""
    skip = f" &nbsp;·&nbsp; skipped {len(insp.skipped_sheets)}" if insp.skipped_sheets else ""
    return f"&nbsp;&nbsp;&nbsp;&nbsp;➜ {len(insp.relevant_sheets)} relevant: {rel}{more}{skip}"


def _detail_metrics(state: dict) -> str:
    """Render extracted KPIs as inline value chips."""
    mset = state.get("metrics")
    if not mset or not mset.metrics:
        return ""
    chips = " ".join(f"`{m.name} {m.value:g}{m.unit}`" for m in mset.metrics[:6])
    more = f" +{len(mset.metrics) - 6}" if len(mset.metrics) > 6 else ""
    return f"&nbsp;&nbsp;&nbsp;&nbsp;➜ {len(mset.metrics)} KPIs: {chips}{more}"


def _detail_retriever(state: dict) -> str:
    """Render retrieved benchmark chunks as a sources list with a cross-industry flag."""
    chunks = state.get("benchmark_chunks", [])
    if not chunks:
        return ""
    sources = sorted({c.source for c in chunks})
    src_chips = " ".join(f"`{s}`" for s in sources[:3])
    flag = (
        " &nbsp;·&nbsp; ⚠️ cross-industry fallback"
        if any(c.is_cross_industry for c in chunks)
        else ""
    )
    return f"&nbsp;&nbsp;&nbsp;&nbsp;➜ {len(chunks)} chunks from {src_chips}{flag}"


def _detail_reporter(state: dict) -> str:
    """Render the Reporter output stats — counts per report section."""
    report = state.get("narrative")
    if not report:
        return ""
    return (
        f"&nbsp;&nbsp;&nbsp;&nbsp;➜ {len(report.key_findings)} findings &nbsp;·&nbsp; "
        f"{len(report.benchmark_analysis)} benchmark comparisons &nbsp;·&nbsp; "
        f"{len(report.roi_scenarios)} ROI scenarios &nbsp;·&nbsp; "
        f"{len(report.citations)} citations"
    )


def _detail_critic(state: dict) -> str:
    """Render the Critic scores + pass/retry decision for the most recent attempt."""
    critic = state.get("critic_report")
    if not critic:
        return ""
    mark = "✓ pass" if critic.passes else "↻ retry"
    return (
        f"&nbsp;&nbsp;&nbsp;&nbsp;➜ faithfulness `{critic.faithfulness_score:.2f}` &nbsp;·&nbsp; "
        f"numeric `{critic.numeric_accuracy_score:.2f}` &nbsp;·&nbsp; {mark}"
    )


def _detail_guardrails(state: dict) -> str:
    """Render Guardrails outcome — PII counts and HITL flag if any."""
    redactions = state.get("pii_redactions") or []
    flagged = state.get("requires_human_review", False)
    bits = [f"{len(redactions)} PII items redacted" if redactions else "clean (no PII)"]
    if flagged:
        bits.append("⚠️ flagged for human review")
    return f"&nbsp;&nbsp;&nbsp;&nbsp;➜ {' &nbsp;·&nbsp; '.join(bits)}"


_DETAIL_FUNCS = {
    "schema_inspector": _detail_schema,
    "extractor": _detail_metrics,
    "retriever": _detail_retriever,
    "reporter": _detail_reporter,
    "critic": _detail_critic,
    "guardrails": _detail_guardrails,
}


def _row_markdown(
    node: str, status: str, elapsed: float, run_count: int, state: dict | None
) -> str:
    """Render a single progress row (status line + detail card when done)."""
    label, hint = NODE_LABEL[node]
    if status == "pending":
        return (
            f"⚪&nbsp;&nbsp;<span style='font-size:1.05em'>{label}</span> &nbsp;·&nbsp; "
            f"<span style='opacity:.55'>{hint}</span>"
        )
    if status == "running":
        att = f" &nbsp;·&nbsp; attempt {run_count + 1}" if run_count else ""
        return (
            f"⏳&nbsp;&nbsp;<span style='font-size:1.05em'>**{label}**</span>{att} "
            f"&nbsp;·&nbsp; <span style='opacity:.75'>{hint}…</span>"
        )
    # done
    suffix = f" &nbsp;·&nbsp; ran {run_count} times" if run_count > 1 else ""
    head = (
        f"✅&nbsp;&nbsp;<span style='font-size:1.05em'>**{label}**</span> "
        f"&nbsp;·&nbsp; `{elapsed:.1f}s`{suffix}"
    )
    if state is not None:
        detail = _DETAIL_FUNCS[node](state)
        if detail:
            return head + "\n\n" + detail
    return head


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


def _typewriter_reveal(text: str, word_delay: float = 0.04) -> None:
    """Reveal ``text`` word-by-word in a single Streamlit placeholder.

    Used to animate the Executive Summary so the final report doesn't appear
    as a wall of text — the viewer sees the narrative landing live. Words are
    chunked rather than streamed token-by-token because the report is already
    generated; this is presentational, not actual LLM streaming.
    """
    placeholder = st.empty()
    words = text.split()
    accumulated = ""
    for word in words:
        accumulated = f"{accumulated} {word}".lstrip()
        placeholder.write(accumulated)
        time.sleep(word_delay)


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
            f"final Critic faithfulness **{critic.faithfulness_score:.2f}**"
            if critic
            else f"Generated in **{run_seconds:.1f}s**"
        )

    st.subheader("📋 Executive Summary")
    _typewriter_reveal(report.executive_summary, word_delay=0.04)

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
    excel_path: str,
    industry: str,
    arch_holder: Any,
    placeholders: dict,
    critic_holder: Any,
) -> tuple[dict, float]:
    """Consume run_pipeline_streaming and update the UI as each node finishes."""
    max_attempts = get_settings().max_reporter_attempts
    run_counts: dict[str, int] = {n: 0 for n, _, _ in NODE_ORDER}
    started_at: dict[str, float] = {}
    elapsed_total: dict[str, float] = {n: 0.0 for n, _, _ in NODE_ORDER}
    critic_attempts: list[dict] = []
    completed: set[str] = set()

    # Light up the first node before the LangGraph stream produces anything.
    first = "schema_inspector"
    started_at[first] = time.time()
    placeholders[first].markdown(
        _row_markdown(first, "running", 0, 0, None), unsafe_allow_html=True
    )
    arch_holder.graphviz_chart(_arch_dot(active=first, completed=completed))

    pipeline_start = time.time()
    final_state: dict = {}
    async for node, state in run_pipeline_streaming(excel_path, industry):
        now = time.time()
        run_counts[node] += 1
        elapsed_total[node] += now - started_at.get(node, now)
        placeholders[node].markdown(
            _row_markdown(node, "done", elapsed_total[node], run_counts[node], state),
            unsafe_allow_html=True,
        )
        completed.add(node)

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
            started_at[nxt] = now
            # When we loop back to reporter, its row goes from ✅ → ⏳ again.
            if nxt in completed:
                completed.discard(nxt)
            placeholders[nxt].markdown(
                _row_markdown(nxt, "running", 0, run_counts[nxt], None),
                unsafe_allow_html=True,
            )
            arch_holder.graphviz_chart(_arch_dot(active=nxt, completed=completed))
        else:
            arch_holder.graphviz_chart(_arch_dot(active="", completed=completed))
        final_state = state

    return final_state, time.time() - pipeline_start


# ─── Main flow ──────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload contact-center diagnostic Excel",
    type=["xlsx", "xls"],
    help="Sheets with FCR, AHT, CSAT, agent utilization, etc.",
)

if uploaded is not None:
    _render_file_preview(uploaded)

run_button = st.button("▶️ Run Analysis", type="primary", disabled=uploaded is None)

if run_button and uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        excel_path = tmp.name

    st.markdown("### 🗺️ Workflow")
    arch_holder = st.empty()
    arch_holder.graphviz_chart(_arch_dot())

    progress_box = st.container(border=True)
    with progress_box:
        st.markdown("**🚀 Pipeline progress**")
        node_placeholders = {n: st.empty() for n, _, _ in NODE_ORDER}
        for node_id, _lbl, _hnt in NODE_ORDER:
            node_placeholders[node_id].markdown(
                _row_markdown(node_id, "pending", 0, 0, None), unsafe_allow_html=True
            )
        critic_holder = st.empty()

    final_state: dict = {}
    duration: float = 0.0
    try:
        final_state, duration = asyncio.run(
            _run_with_live_progress(
                excel_path, industry, arch_holder, node_placeholders, critic_holder
            )
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
