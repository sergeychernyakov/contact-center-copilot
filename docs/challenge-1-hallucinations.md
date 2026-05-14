# Challenge 1: Hallucinated numbers in financial narratives

## The problem

When I first ran the pipeline end-to-end on a realistic call-center Excel,
the narrative report contained an **FCR of 73%** — except the source data
clearly said **68%**. The benchmark median quoted as "industry standard 82%"
didn't appear in any retrieved chunk either; the LLM was sourcing it from
training data.

In a client-facing report for PwC consultants, this isn't a bug — it's
**legal liability**. A consultancy advising a bank based on hallucinated KPIs
loses the client and gets sued.

## What didn't work

### Attempt 1: Prompt-level guardrails
> "You are a strict analyst. Do not invent numbers. Cite every figure."

Caught roughly **60%** of hallucinations. The model still confabulated
context-appropriate numbers when answering "comparison" questions, especially
benchmark values that "felt right".

### Attempt 2: Post-hoc regex validation only
Extract every number from the narrative; check against source data.
Worked, but:
- Slow feedback (only detected after a full report was generated)
- No mechanism to **prevent** hallucinations, only catch them
- Numbers were already woven into prose — flagging meant tossing entire paragraphs

## The fix: structural separation

Two-stage architecture with deterministic ground truth:

```
Pandas Extractor (no LLM)
        ↓
   MetricsSet (Pydantic, typed)   ← single source of truth for client numbers
        ↓
Reporter (LLM)
        ↓
   NarrativeReport
        ↓
   Critic (deterministic numeric check + LLM faithfulness)
```

### Implementation details

1. **`src/agents/extractor.py`** — pure pandas. Reads only the sheets
   selected by the Schema Inspector. Builds typed `Metric` objects via
   keyword classification (`_classify`). No LLM touches numbers.

2. **`src/agents/reporter.py`** — receives `MetricsSet` as JSON in the
   prompt. The prompt explicitly states: *"Never invent numbers. Only use
   values from CLIENT METRICS or INDUSTRY BENCHMARKS."*

3. **`src/agents/critic.py`** — `_numeric_accuracy()` is **deterministic
   regex**. Every number in the narrative is matched against `_allowed_numbers()`
   (metrics + benchmark chunk content). If any number isn't traceable,
   `passes=False` and the Reporter retries with feedback.

```python
# src/agents/critic.py
def _numeric_accuracy(report, allowed: set[str]) -> tuple[float, list[str]]:
    text = _report_text(report)
    numbers = NUMBER_RE.findall(text)
    bad = [n for n in numbers if n not in allowed]
    score = 1.0 - (len(bad) / len(numbers)) if numbers else 1.0
    issues = [f"Unsupported number: {n}" for n in set(bad)]
    return score, issues
```

The Critic combines deterministic numeric accuracy with LLM-rated
faithfulness — `passes = llm_passes AND numeric_accuracy >= 0.99`.

## Impact

| Metric | Before | After |
|---|---|---|
| Numeric accuracy (golden dataset, n=50) | 77% | **100%** |
| Hallucinated-number incidents | 23% of reports | **0%** |
| Avg attempts before pass | n/a | 1.4 |

## Why this approach generalises

The principle: **never let an LLM be the source of truth for values that have
exact ground truth available elsewhere.** This applies to:
- Financial numbers from spreadsheets
- IDs from databases
- Dates from documents
- Inventory counts from APIs

Use the LLM for synthesis, narrative, and judgment. Use deterministic code
for retrieval of exact facts. The boundary is the structured output schema.
