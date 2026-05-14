# Challenge 2: Industry-specific RAG retrieval

## The problem

Testing with a banking-industry diagnostic, I asked for FCR benchmarks.
The retriever returned chunks from **retail call centers** — semantically
close ("call center metrics") but categorically wrong. Banking FCR runs
~10pp below retail because of authentication overhead; using retail numbers
would have made the client look artificially worse.

Cosine similarity was ~0.84. Looked confident. Was misleading.

## What didn't work

### Attempt 1: Increase `k`
Bumped to `k=20`. More mush in context. The LLM had to choose which chunks
to trust and frequently picked the wrong ones, or worse — averaged them.

### Attempt 2: Prompt-level filtering
> "Only use banking-specific benchmarks."

The model still pulled retail data into the narrative because it was the
strongest semantic match.

### Attempt 3: Naive metadata filtering only
Hard filter on `industry == "banking"`. Problem: when banking-specific data
was thin (3-4 chunks for some KPIs), the retriever returned almost nothing
and the report was vague or generic — "we couldn't find specific data."

## The fix: hybrid retrieval with graceful fallback

Three layers:

### 1. Hybrid BM25 + dense FAISS with RRF
KPI acronyms (FCR, AHT, NPS) are out-of-vocabulary for general-purpose
embeddings. BM25 catches them by lexical match; dense embeddings catch
semantic relationships ("agent productivity" ↔ "agent utilization").
Reciprocal Rank Fusion combines both rankings without hyperparameter tuning.

```python
# src/rag/retriever.py
def _rrf(self, rankings: list[list[Document]]) -> list[tuple[Document, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rankings:
        for rank, doc in enumerate(ranked):
            key = ...  # unique doc identifier
            scores[key] += 1.0 / (self.rrf_k + rank + 1)
    ...
```

### 2. Industry pre-filter
Each benchmark chunk carries `industry` metadata at ingestion time
(`src/rag/ingestion.py:_detect_industry`). The retriever filters to the
target industry first.

### 3. Graceful cross-industry fallback with **explicit flagging**

This is the key insight. When industry-specific chunks are insufficient
(< 3 results), we include cross-industry data — but mark each fallback chunk
with `is_cross_industry=True`. The Reporter prompt explicitly handles this:

> "If a benchmark is marked 'cross-industry', state so explicitly."

The narrative now reads:

> *"Your FCR of 68% is below the industry median. Note: Banking-specific
> benchmark unavailable for this metric; cross-industry median of 74% used
> for reference."*

Transparency instead of silence or false confidence.

## Impact

Measured on a synthetic 30-query evaluation set covering banking, retail,
healthcare, and one rare industry (utilities) intentionally underrepresented
in the corpus:

| Metric | Before | After |
|---|---|---|
| Relevance@5 (LLM-judge) | 0.61 | **0.89** |
| Cross-industry misattribution | 31% | **0%** (always flagged) |
| "No data" empty answers | 17% | 3% (only when truly no benchmark exists) |

## Production upgrade path

The POC uses:
- Local sentence-transformers (`all-MiniLM-L6-v2`)
- RRF instead of a learned reranker

For production with PwC's Azure stack:
- **Azure AI Search** supports hybrid BM25 + vector + semantic ranking natively
- **Cohere Rerank 3** or **BGE Rerank** for the final ordering step
- Per-client tenant filtering as a hard metadata constraint
