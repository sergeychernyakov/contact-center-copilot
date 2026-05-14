"""Eval runner — executes the pipeline against the golden dataset and
checks regression gates.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --fail-on-regression  # exit non-zero if thresholds missed

In production this would be triggered by GitHub Actions on every PR.
The POC version is intentionally simple — replace `_score_*` stubs with
real RAGAS metrics when running with the `eval` extras installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from src.graph.workflow import run_pipeline

NUMBER_RE = re.compile(r"\b\d+\.?\d*\b")


def _score_numeric_accuracy(state: dict) -> float:
    """Fraction of numbers in the narrative traceable to source metrics."""
    report = state.get("narrative")
    metrics = state.get("metrics")
    if report is None or metrics is None:
        return 0.0

    allowed: set[str] = set()
    for m in metrics.metrics:
        allowed.add(f"{m.value:g}")
        allowed.add(f"{round(m.value):d}")
        allowed.add(f"{round(m.value, 1):g}")
    for chunk in state.get("benchmark_chunks", []):
        allowed.update(NUMBER_RE.findall(chunk.content))

    text = report.executive_summary + " ".join(report.key_findings)
    found = NUMBER_RE.findall(text)
    if not found:
        return 1.0
    good = sum(1 for n in found if n in allowed)
    return good / len(found)


def _score_must_mention(state: dict, must: list[str]) -> float:
    report = state.get("narrative")
    if report is None:
        return 0.0
    text = (
        report.executive_summary + " ".join(report.key_findings) + " ".join(report.recommendations)
    ).lower()
    if not must:
        return 1.0
    hits = sum(1 for term in must if term.lower() in text)
    return hits / len(must)


async def run_case(case: dict) -> dict:
    start = time.time()
    state = await run_pipeline(case["excel_path"], industry=case["industry"])
    duration = time.time() - start

    expected = case["expected"]
    scores = {
        "numeric_accuracy": _score_numeric_accuracy(dict(state)),
        "must_mention_coverage": _score_must_mention(dict(state), expected.get("must_mention", [])),
        "latency_seconds": round(duration, 2),
    }
    return {"id": case["id"], "scores": scores}


async def main(fail_on_regression: bool) -> int:
    root = Path(__file__).resolve().parent
    dataset_path = root / "golden_dataset.json"
    with dataset_path.open() as f:
        dataset = json.load(f)

    results = []
    for case in dataset["cases"]:
        print(f"▶ Running {case['id']}...")
        try:
            result = await run_case(case)
            results.append(result)
            print(f"  ✓ {result['scores']}")
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")
            results.append({"id": case["id"], "error": str(exc)})

    thresholds = dataset["thresholds"]
    regressions: list[str] = []
    for r in results:
        if "error" in r:
            regressions.append(f"{r['id']}: errored ({r['error']})")
            continue
        if r["scores"]["numeric_accuracy"] < thresholds["numeric_accuracy"]:
            regressions.append(
                f"{r['id']}: numeric_accuracy {r['scores']['numeric_accuracy']:.2f} "
                f"< {thresholds['numeric_accuracy']}"
            )
        if r["scores"]["must_mention_coverage"] < thresholds["must_mention_coverage"]:
            regressions.append(
                f"{r['id']}: must_mention_coverage "
                f"{r['scores']['must_mention_coverage']:.2f} "
                f"< {thresholds['must_mention_coverage']}"
            )
    # latency_seconds is recorded for observability but not gated: it is
    # provider-dependent (seconds on hosted APIs, minutes on local Ollama),
    # so it tracks the deployment, not a code regression.

    # Persist results
    out_path = root / "last_run.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    if regressions:
        print("\n❌ REGRESSIONS:")
        for r in regressions:
            print(f"  • {r}")
        return 1 if fail_on_regression else 0

    print("\n✅ All gates passed")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.fail_on_regression)))
