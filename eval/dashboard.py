# eval/dashboard.py
"""Streamlit dashboard for eval results.

Usage: streamlit run eval/dashboard.py

Reads eval/last_run.json (written by run_eval.py) plus the thresholds in
golden_dataset.json, then renders per-case scores with pass/fail gates — the
same regression gates run_eval.py enforces in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

EVAL_DIR = Path(__file__).resolve().parent
LAST_RUN_PATH = EVAL_DIR / "last_run.json"
DATASET_PATH = EVAL_DIR / "golden_dataset.json"

GATED_METRICS = ("numeric_accuracy", "must_mention_coverage")


# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Eval Dashboard — Contact Center Copilot",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Eval Dashboard")
st.caption(
    "Regression gates for the multi-agent pipeline — per-case scores from the "
    "golden dataset, checked against the same thresholds run_eval.py enforces "
    "in CI. Run `python eval/run_eval.py` to refresh."
)


# ─── Load thresholds ────────────────────────────────────────────────────────

with DATASET_PATH.open() as f:
    dataset = json.load(f)
thresholds = dataset["thresholds"]
cases_by_id = {case["id"]: case for case in dataset["cases"]}

with st.sidebar:
    st.header("⚙️ Thresholds")
    st.caption("From `golden_dataset.json` — CI-gated metrics in **bold**.")
    for name, value in thresholds.items():
        label = f"**{name}**" if name in GATED_METRICS else name
        st.markdown(f"{label}: `{value}`")
    st.divider()
    st.markdown(
        "[GitHub repo](https://github.com/sergeychernyakov/contact-center-copilot) · "
        "Built for PwC AppDev"
    )

if not LAST_RUN_PATH.exists():
    st.warning(
        "No `eval/last_run.json` yet. Run `python eval/run_eval.py` to generate "
        "results, then reload this page."
    )
    st.stop()

with LAST_RUN_PATH.open() as f:
    results = json.load(f)


# ─── Gate logic ─────────────────────────────────────────────────────────────


def _gate_results(scores: dict) -> list[dict]:
    """Evaluate the CI-gated quality metrics for one case's scores."""
    numeric = scores.get("numeric_accuracy", 0.0)
    mention = scores.get("must_mention_coverage", 0.0)
    return [
        {
            "label": "Numeric accuracy",
            "passed": numeric >= thresholds["numeric_accuracy"],
            "detail": f"{numeric:.2f} (need ≥ {thresholds['numeric_accuracy']})",
        },
        {
            "label": "Must-mention coverage",
            "passed": mention >= thresholds["must_mention_coverage"],
            "detail": f"{mention:.2f} (need ≥ {thresholds['must_mention_coverage']})",
        },
    ]


def _case_passed(result: dict) -> bool:
    """True if a case ran without error and cleared every gate."""
    if "error" in result:
        return False
    return all(gate["passed"] for gate in _gate_results(result["scores"]))


# ─── Summary ────────────────────────────────────────────────────────────────

total = len(results)
passed = sum(1 for r in results if _case_passed(r))
failed = total - passed

cols = st.columns(3)
cols[0].metric("Cases", total)
cols[1].metric("Passed", passed)
cols[2].metric("Failed", failed)

if failed:
    st.error(f"❌ {failed} of {total} cases failed a regression gate.")
else:
    st.success(f"✅ All {total} cases passed every gate.")

st.caption(
    "Gates: **numeric accuracy** (no hallucinated numbers) and **must-mention "
    "coverage** (required KPIs named). Latency is shown per case but not gated — "
    "it tracks the provider (seconds on hosted APIs, minutes on local Ollama), "
    "not the code."
)

st.divider()


# ─── Per-case detail ────────────────────────────────────────────────────────

for result in results:
    case_id = result["id"]
    industry = cases_by_id.get(case_id, {}).get("industry", "—")

    if "error" in result:
        st.subheader(f"❌ {case_id} · {industry}")
        st.error(f"Pipeline errored: {result['error']}")
        st.divider()
        continue

    scores = result["scores"]
    gates = _gate_results(scores)
    icon = "✅" if all(gate["passed"] for gate in gates) else "❌"
    st.subheader(f"{icon} {case_id} · {industry}")

    score_cols = st.columns(3)
    score_cols[0].metric("Numeric accuracy", f"{scores.get('numeric_accuracy', 0.0):.2f}")
    score_cols[1].metric("Must-mention coverage", f"{scores.get('must_mention_coverage', 0.0):.2f}")
    score_cols[2].metric("Latency", f"{scores.get('latency_seconds', 0.0):.1f}s")

    for gate in gates:
        mark = "✅" if gate["passed"] else "❌"
        st.markdown(f"{mark} **{gate['label']}** — {gate['detail']}")

    st.divider()
