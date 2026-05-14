# Challenge 5: Regression-proof evaluation pipeline

## The problem

After the system "worked", every prompt tweak felt like progress — but I had
no way to **prove** it. Did changing the Reporter's system prompt improve
narrative quality, or did I just remember the good outputs and forget the bad?

Without evaluation, every change is a coin flip. Over months, a system
slowly degrades through unmeasured "improvements". That's the most insidious
class of bug because it's invisible.

## What didn't work

### Attempt 1: Ad-hoc "looks-good" review
Run a few queries by hand, eyeball outputs. Worked for the first prompt
iteration. By the third, I couldn't remember what the second version
produced for comparison.

### Attempt 2: RAGAS scores without thresholds
Plugged in RAGAS, got faithfulness and answer_relevancy numbers. Cool. Are
they good? Compared to what? No way to tell if a 0.82 was acceptable or a
regression from a previous 0.89.

### Attempt 3: Production logs only
Tail the logs, look for obvious errors. Misses everything that's wrong but
not obviously wrong — exactly the failure mode that matters most.

## The fix: golden dataset + multi-dim metrics + CI gates

### 1. Golden dataset (`eval/golden_dataset.json`)
50 input/expected pairs, hand-curated. Each entry:

```json
{
  "id": "case_001",
  "excel_path": "data/fixtures/banking_q3.xlsx",
  "industry": "banking",
  "expected": {
    "key_metrics": {"FCR": 68.0, "AHT_seconds": 412},
    "must_mention": ["FCR", "below benchmark"],
    "must_not_invent": true,
    "max_cost_usd": 0.10
  }
}
```

### 2. Multi-dimensional metrics (`eval/run_eval.py`)
A single number ("quality") is meaningless. We track five:

| Metric | How measured | Threshold |
|---|---|---|
| **Faithfulness** | RAGAS faithfulness | ≥ 0.85 |
| **Numeric accuracy** | regex extract numbers, compare to source | = 1.00 |
| **Must-mention coverage** | required KPIs named in the narrative | = 1.00 |
| **Citation coverage** | % of claims with source attribution | ≥ 0.80 |
| **Narrative quality** | Opus as LLM judge (1-5 scale) | ≥ 4.0 |
| **Cost** | actual token spend | ≤ $0.10/report |

Different metrics catch different failure modes. A change might improve
faithfulness while hurting narrative readability — without separate metrics
you'd never notice. Latency is recorded too, but not gated: it's
provider-dependent (seconds on hosted APIs, minutes on local Ollama), so it
tracks the deployment, not the code.

### 3. Regression gates in CI (`.github/workflows/ci.yml`)
On every PR, the eval suite runs:

```bash
pytest -q tests/  # unit tests
python eval/run_eval.py --golden eval/golden_dataset.json --fail-on-regression
```

The eval script compares current scores to the baseline stored in the repo.
If **any** metric drops below threshold, the build fails. The PR can't merge.

This is the same shape as performance regression tests in compiled-language
projects, applied to AI systems.

### 4. Versioned baselines
The baseline scores aren't just printed — they're committed to
`eval/baselines.json`. Updating baselines is an explicit PR step, forcing
intentional acknowledgement of any quality shift.

## Impact

Live for two weeks, the eval caught **3 regressions** that would otherwise
have shipped:

1. **LangChain 0.3.5 → 0.3.7 upgrade** silently broke structured output
   adherence for `NarrativeReport` — 15% of reports failed schema validation
   on the new version. Caught by failing unit test in CI.
2. **Reporter prompt tweak** ("make narrative more concise") dropped citation
   coverage from 0.87 to 0.62 because the model elided source attributions
   to save space. Caught by citation coverage metric.
3. **Chunking change** (800 → 1200 chars) improved faithfulness marginally
   but spiked cost +40% because more context made it into every prompt.
   Caught by cost gate.

None of these would have been caught by manual review — they're all
"works fine on the case I happened to try."

## What I'd add for production

The POC's eval is comprehensive but synchronous and gate-only. In production:

- **Continuous eval** on sampled production traffic (10% sample, anonymised)
- **Drift detection** — if production distributions of inputs shift, re-run
  the eval suite even without code changes
- **LLM-judge calibration** — periodically compare LLM-judge scores to human
  expert ratings on a small sample to detect judge drift
- **Per-customer baselines** — different clients tolerate different trade-offs
  (compliance-heavy clients need higher faithfulness; consumer clients want
  faster latency)

The principle stays the same: **if you can't measure regression, you don't
know if you're regressing.**
