# Contact Center Copilot

> Multi-agent system that transforms contact center diagnostic data (Excel) into narrative reports with industry benchmarking, ROI scenarios, and citation-backed insights.

Built with **LangGraph** orchestration, **RAG** over industry benchmarks, **FastMCP** for tool exposure, and Pydantic-validated structured outputs.

[![CI](https://github.com/sergeychernyakov/contact-center-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/sergeychernyakov/contact-center-copilot/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What it does

Upload an Excel file with call center KPIs (FCR, AHT, CSAT, agent utilization, etc.) — get back a polished narrative report that:

1. **Extracts** the metrics that matter (skipping irrelevant sheets)
2. **Compares** them against industry benchmarks via RAG retrieval
3. **Quantifies** performance gaps and ROI scenarios
4. **Narrates** findings with inline citations to source benchmarks
5. **Self-corrects** through a Critic agent until faithfulness score ≥ 0.85

---

## Architecture

```mermaid
flowchart LR
    UI[Streamlit UI<br/>Upload Excel] --> ING[Schema Inspector<br/>Haiku]
    ING --> EXT[Pandas Extractor<br/>deterministic]
    EXT --> ANA[Analyst Agent<br/>Sonnet]
    ANA --> BEN[Benchmark Retriever<br/>FAISS + Reranker]
    BEN --> REP[Reporter Agent<br/>Sonnet]
    REP --> CRT{Critic Agent<br/>faithfulness ≥ 0.85?}
    CRT -- no, attempts < 3 --> REP
    CRT -- yes --> GRD[Guardrails Agent<br/>PII / bias / tone]
    CRT -- max attempts --> HITL[Human Review Flag]
    GRD --> OUT[Narrative Report<br/>+ citations + ROI]

    MCP[FastMCP Server<br/>tools/resources] -.exposes.-> ING
    MCP -.exposes.-> EXT
    MCP -.exposes.-> BEN
```

**Why LangGraph instead of LangChain chains?**
The Critic→Reporter feedback loop requires cycles and state persistence. Chains are DAGs — they can't loop. LangGraph also gives us conditional edges, bounded retries, and built-in human-in-the-loop checkpoints.

**Why split deterministic extraction from LLM analysis?**
LLMs hallucinate numbers. Pandas doesn't. By separating "what are the numbers" (pandas) from "what do they mean" (LLM), we eliminate a class of bugs that no amount of prompting can fix. See [Challenge 1](docs/challenge-1-hallucinations.md).

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Anthropic Claude Sonnet 4.6 + Haiku 4.5 | Sonnet for analysis, Haiku for routing/fast tasks |
| Embeddings | `sentence-transformers` (local) | Zero external deps, self-contained Docker |
| Vector store | FAISS | Local, fast, swappable to Azure AI Search via env config |
| Orchestration | LangGraph | State, cycles, conditional edges, HITL |
| Tool protocol | FastMCP | Standardized way to expose tools to external agents |
| Validation | Pydantic v2 | Structured outputs, no `json.loads` over LLM output |
| Evaluation | RAGAS + custom evaluators | Faithfulness, numeric accuracy, citation coverage |
| Observability | LangSmith | Per-agent traces, token cost, latency |
| UI | Streamlit | Fast prototyping, file upload, streaming |
| CI/CD | GitHub Actions | Lint, test, eval regression gates |
| Container | Docker + docker-compose | One-command local setup |

**Swappable in production:** OpenAI / Azure OpenAI / Bedrock LLMs, Azure AI Search instead of FAISS, Cohere reranker instead of local cross-encoder. All controlled via `.env` flags.

---

## Quick start

```bash
git clone https://github.com/sergeychernyakov/contact-center-copilot.git
cd contact-center-copilot

# Setup
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env

# Option A: Docker (recommended)
docker compose up

# Option B: Local Python
pip install -e .
streamlit run src/ui/streamlit_app.py
```

Open http://localhost:8501, upload `data/sample_diagnostic.xlsx`, click **Run Analysis**.

---

## Development & code quality

All quality tooling is wired through **pre-commit** and a **Makefile**.

```bash
make install        # install dev deps + git hooks (pre-commit, commit-msg, pre-push)
make qa             # full QA suite: lint, types, security, audit, docs, dead code, coverage
make test           # fast unit tests (no coverage gate, no LLM calls)
make format         # auto-format + auto-fix (ruff)
make help           # list every target
```

Hooks run in three stages:

| Stage | Checks |
|---|---|
| `pre-commit` | ruff (lint + format), `pre-commit-hooks` hygiene set, `pygrep-hooks`, codespell, yamllint, gitleaks, hadolint, GitHub-workflow schema |
| `commit-msg` | Conventional Commits format (commitizen) |
| `pre-push` | mypy, pylint (≥ 9.5/10), bandit, pip-audit, interrogate, vulture, pytest + 90% coverage gate |

> `hadolint` needs the binary — `brew install hadolint`. Run `make hooks-update` to bump pinned hook versions.
> After `make install`, direct commits to `main`/`master` are blocked — use feature branches.

---

## Demo

📺 **3-minute walkthrough:** [demo.mp4](./demo.mp4)

![Demo GIF](./docs/demo.gif)

The video shows:
- Excel upload and schema inspection (12 KPIs identified across 3 relevant sheets)
- Multi-agent pipeline streaming output to UI
- Final narrative report with inline citations
- LangSmith trace showing per-agent cost ($0.04 total) and latency (14s)
- GitHub Actions CI run with RAGAS eval scores

---

## Engineering challenges solved

The interesting part of any AI system is what breaks when you take it out of the tutorial environment. Five problems surfaced during this build — each documented with the failed attempts and the final architecture:

### [1. Hallucinated numbers in financial narratives](docs/challenge-1-hallucinations.md)
LLM invented an FCR of 73% when the source data said 68%. In a client-facing report, that's not a bug — that's liability. Fixed by splitting deterministic extraction (pandas → Pydantic) from narrative generation (LLM only fills prose around fixed numeric slots), plus a regex-based validator that rejects any number not in the source set. **Numeric accuracy: 77% → 100%** on golden dataset of 50 reports.

### [2. Industry-specific RAG retrieval](docs/challenge-2-retrieval.md)
Banking client got benchmarks from retail call centers — semantically close, categorically wrong. Fixed with hybrid search (BM25 + dense), metadata pre-filtering on industry, Cohere reranker, and graceful fallback that flags cross-industry data in the narrative ("Banking-specific benchmark unavailable; cross-industry median used for reference"). **Relevance@5: 0.61 → 0.89**.

### [3. Excel files exceeding context window](docs/challenge-3-excel-scale.md)
Real client Excel: 47 sheets, 12k rows, 80+ columns. Couldn't fit. Naive truncation lost critical data. Per-sheet LLM calls cost $4 and took 8 minutes per report. Fixed with schema-first agent (Haiku scans headers only, picks relevant sheets) + pandas aggregation (deterministic, not LLM) + analyst on compact MetricsSummary. **Latency: 8min → 14s. Cost: $4 → $0.06 per report.**

### [4. Self-correcting loops & cost control](docs/challenge-4-loop-control.md)
Critic and Reporter agents got into infinite back-and-forth — 15 iterations on a single report, ballooning cost. Fixed with bounded retries (`attempts: int` in state), progressive threshold relaxation in Critic's prompt, full feedback history passed to Reporter, and HITL escalation node after 3 failed attempts. **Avg iterations: 4.2 → 1.8. Cost per report: −65%.**

### [5. Regression-proof evaluation pipeline](docs/challenge-5-eval.md)
Without eval, every prompt tweak is a coin flip. Built golden dataset (50 input/expected pairs), multi-dimensional metrics (RAGAS faithfulness, numeric accuracy, citation coverage, narrative quality via LLM-as-judge, cost & latency budgets), and GitHub Actions regression gates that fail builds if any metric drops below threshold. **Caught 3 regressions before production**, including a LangChain version bump that silently broke structured output on 15% of queries.

---

## What this POC deliberately doesn't do

Being honest about scope matters more than feature lists. This POC skips:

- **Authentication / multi-tenancy** — single-user demo, would need Azure AD + row-level security for production
- **Streaming long reports** — sync response only; in production, would use Server-Sent Events
- **Drift monitoring** — no detection for when industry benchmarks become stale
- **A/B testing of prompts** — manual prompt versioning in code, not via tool like Langfuse Prompts
- **Multi-language reports** — English only; LLM can do this trivially but eval set would need rebuilding
- **Database persistence** — runs in-memory; reports aren't saved between sessions

Each of these is a known trade-off, not an oversight. Happy to discuss the production roadmap.

---

## Repo structure

```
contact-center-copilot/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── challenge-1-hallucinations.md
│   ├── challenge-2-retrieval.md
│   ├── challenge-3-excel-scale.md
│   ├── challenge-4-loop-control.md
│   └── challenge-5-eval.md
├── src/
│   ├── agents/              # Schema inspector, analyst, retriever, reporter, critic, guardrails
│   ├── graph/               # LangGraph state + workflow assembly
│   ├── rag/                 # Ingestion, retriever, reranker
│   ├── mcp_server/          # FastMCP server exposing tools
│   ├── tools/               # Excel extractor, ROI calculator
│   └── ui/                  # Streamlit app
├── tests/                   # pytest
├── eval/
│   ├── golden_dataset.json  # 50 input/expected pairs
│   ├── run_eval.py
│   └── dashboard.py         # Streamlit eval dashboard
├── data/
│   ├── sample_diagnostic.xlsx
│   └── benchmarks/          # Industry benchmark markdown chunks
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Running evaluations

```bash
# Run full eval suite against golden dataset
python eval/run_eval.py

# Open eval dashboard (Streamlit)
streamlit run eval/dashboard.py
```

Sample output:
```
Faithfulness (RAGAS):        0.91  (threshold: 0.85) ✅
Numeric accuracy:            1.00  (threshold: 1.00) ✅
Citation coverage:           0.87  (threshold: 0.80) ✅
Narrative quality (judge):   4.3/5 (threshold: 4.0)  ✅
Avg cost per report:         $0.04 (budget: $0.10)   ✅
P95 latency:                 18s   (budget: 30s)     ✅
```

---

## Notes on AI assistance during development

This POC was built with Claude as a pair programmer. Architecture decisions, agent design, prompt engineering, failure analysis, and trade-off choices are mine; AI accelerated boilerplate and iteration speed. Each decision in the [docs/](./docs/) folder includes the reasoning, not just the outcome — that's the part that mattered.

---

## Author

Sergey Chernyakov · [LinkedIn](https://www.linkedin.com/in/sergey-chernyakov-458506400/) · Backend / AI Engineering Lead

Built in 7 days as a thought exercise for the **PwC AppDev Lead AI Engineer** role.

---

## License

MIT
