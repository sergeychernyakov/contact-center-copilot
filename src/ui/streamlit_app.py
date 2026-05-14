"""Streamlit UI for the Contact Center Copilot.

Usage: streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import tempfile

import streamlit as st

from src.graph.state import CopilotState
from src.graph.workflow import run_pipeline

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Contact Center Copilot",
    page_icon="📞",
    layout="wide",
)

st.title("📞 Contact Center Copilot")
st.caption(
    "Multi-agent diagnostic system — upload an Excel, get a narrative report "
    "with benchmark comparisons, ROI scenarios, and citations."
)


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    # Only industries with a benchmark dataset in data/benchmarks/ — picking an
    # unbacked industry would silently fall back to cross-industry data.
    industry = st.selectbox(
        "Industry",
        ["banking", "retail", "cross_industry"],
        index=0,
    )
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


# ─── Main area ──────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload contact center diagnostic Excel",
    type=["xlsx", "xls"],
    help="Sheets with FCR, AHT, CSAT, agent utilization, etc.",
)

run = st.button("▶️ Run Analysis", type="primary", disabled=uploaded is None)


def _render_report(state: dict) -> None:
    """Pretty-print the final state in the UI."""
    report = state.get("narrative")
    metrics = state.get("metrics")
    critic = state.get("critic_report")

    if state.get("requires_human_review"):
        st.warning("⚠️ This report is flagged for human review.")
        for reason in report.review_reasons if report else []:
            st.caption(f"• {reason}")

    if report:
        st.subheader("📋 Executive Summary")
        st.write(report.executive_summary)

        st.subheader("🔑 Key Findings")
        for f in report.key_findings:
            st.markdown(f"- {f}")

        if report.benchmark_analysis:
            st.subheader("📊 Benchmark Analysis")
            for b in report.benchmark_analysis:
                cols = st.columns([2, 1, 1, 1])
                cols[0].markdown(f"**{b.metric_name}**")
                cols[1].metric("Client", f"{b.client_value:g}")
                cols[2].metric("Benchmark", f"{b.benchmark_value:g}")
                cols[3].metric("Gap %", f"{b.gap_percentage:+.1f}%")

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
        st.markdown("**Logs:**")
        for line in state.get("log", []):
            st.code(line, language=None)

        if critic:
            st.markdown(
                f"**Critic scores:** faithfulness {critic.faithfulness_score:.2f}, "
                f"numeric accuracy {critic.numeric_accuracy_score:.2f}"
            )

        if metrics:
            st.markdown(f"**Extracted metrics:** {len(metrics.metrics)}")
            st.json([m.model_dump() for m in metrics.metrics])

        if state.get("pii_redactions"):
            st.markdown(f"**PII redacted:** {len(state['pii_redactions'])} items")


if run and uploaded is not None:
    # Save upload to a temp file the pipeline can read
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        excel_path = tmp.name

    with st.status("Running pipeline...", expanded=True) as status:
        final_state: CopilotState | None = None
        try:
            final_state = asyncio.run(run_pipeline(excel_path, industry=industry))
            status.update(label="✅ Pipeline complete", state="complete")
        except Exception as exc:
            status.update(label=f"❌ Pipeline failed: {exc}", state="error")
            st.exception(exc)
            final_state = None

    if final_state is not None:
        _render_report(dict(final_state))

else:
    st.info("👆 Upload an Excel file to begin. A sample is at `data/sample_diagnostic.xlsx`.")
