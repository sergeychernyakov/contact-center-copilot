# Challenge 3: Excel files exceeding context window

## The problem

Real client diagnostics aren't small. A representative file:

- **47 sheets**
- **~12,000 rows** total
- **80+ columns**

Naively dumping this into a Sonnet prompt is impossible (200K+ tokens, context
window overflow, and ~$3 per call). Even if it fit, the model would be hunting
relevant KPIs through pages of payroll spreadsheets.

## What didn't work

### Attempt 1: Naive truncation
Read first N rows of each sheet. Lost Q3-Q4 data (rows further down).
Critical metrics like quarterly trends disappeared.

### Attempt 2: One LLM call per sheet
47 calls. Each ~2K tokens of headers + sample + instructions. Result:
- **~$4 per report**
- **~8 minutes wall time**
- Cross-sheet context lost (agent stats in one sheet, calls in another;
  LLM couldn't correlate them)

### Attempt 3: Big LLM with chunked summarisation chain
Map-reduce over sheets. Better than per-sheet, but still:
- Expensive
- Inconsistent metric extraction (LLM occasionally summarised "FCR averaged
  68-72%" instead of giving a single number — fatal for downstream comparison)

## The fix: schema-first, deterministic extraction

Three-stage pipeline that minimises LLM work to where it's actually needed:

### Stage 1: Schema Inspector (Haiku, fast)
Reads **only**:
- Sheet names
- Column headers per sheet
- Optionally 3 sample rows

Total payload: ~2K tokens regardless of file size. Output:
`SchemaInspection(relevant_sheets=[...], skipped_sheets=[...], rationale=...)`

For the 47-sheet file: identifies 4-6 relevant sheets in one Haiku call
(~$0.002, < 1s).

### Stage 2: Pandas Extractor (no LLM)
Operates only on the relevant sheets. Pure pandas:
- Type-checks columns (skip non-numeric)
- Classifies column names by keyword regex → canonical `MetricType`
- Aggregates with `mean()` (or appropriate function per metric type)
- Normalises units (0-1 → percentage)

```python
# src/agents/extractor.py — heuristic classification
KEYWORD_MAP = {
    r"\bfcr\b|first[\s_-]?call[\s_-]?resolution": (MetricType.FCR, "%"),
    r"\baht\b|average[\s_-]?handle|handle[\s_-]?time": (MetricType.AHT, "s"),
    ...
}
```

Output: compact `MetricsSet` — typically 8-15 metrics, ~500 tokens.

### Stage 3: Analyst & Reporter
Receive the **compact** `MetricsSet`, never the raw Excel. The narrative
generation is now operating on ~500 tokens of structured data, not 200K of
spreadsheet.

## Impact

| Metric | Before (one LLM/sheet) | After (schema-first) |
|---|---|---|
| Latency | 8 min | **14 sec** |
| Cost per report | $4.00 | **$0.06** |
| Numeric accuracy | 84% (LLM-summarised) | **100%** (pandas) |
| Cross-sheet correlations | broken | works |

## Why this is more than an optimisation

It's a **boundary-of-responsibility** decision:

- **Pandas job**: deterministic data extraction and aggregation
- **LLM job**: classification, synthesis, narrative

Each tool does what it's best at. The LLM doesn't fight with `groupby().mean()`,
and pandas doesn't try to understand "which sheets look like call-center data."

The same principle applies anywhere structured data meets LLMs: SQL databases,
APIs, file metadata, log parsing. Let code do the deterministic work; let the
model do the language work.
