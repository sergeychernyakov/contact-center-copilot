# PwC HackerRank Test — HR Policy RAG Assistant

> Source code from the PwC AppDev technical screening (May 2026).
> Used as the architectural foundation and inspiration for the Contact Center Copilot project.

## Original task

Build an HR Policy Assistant using **Retrieval-Augmented Generation (RAG)** that:

1. Indexes both HR policies and employee data using **FAISS** for efficient vector search
2. Retrieves relevant information based on the employee's query
3. Enhances answers using **GPT-4o** by combining policy information with personal HRIS data
4. Automatically detects when queries refer to specific employees
5. Delivers comprehensive responses through a user-friendly **Streamlit** interface

## Use case

Emma, an employee at a multinational company, struggles to find answers about HR policies. She has questions like *"How many sick leaves do I get?"* or *"What's my remaining PTO balance?"*. Searching through multiple policy documents and contacting HR for personal information is time-consuming and frustrating.

## Stack

- **LangChain** for chain composition and prompts
- **FAISS** for local vector storage
- **OpenAI** (`text-embedding-3-small` + `gpt-4o`)
- **Streamlit** for the UI
- **Pydantic** for structured data

## Architecture (as implemented)

```
PolicyDocumentLoader → list[Document]
        ↓
PolicyVectorStore → FAISS index (with dedup)
        ↓
HRPolicyRAG → query + employee_id → narrative answer
        ↓
Streamlit UI
```

## Files

- `src/document_loader.py` — JSON → LangChain Documents with policy filtering
- `src/vector_store.py` — FAISS wrapper with async similarity search + title-based dedup
- `src/rag_chain.py` — RAG pipeline with employee detection and HRIS integration
- `app.py` — Streamlit UI with session state and employee selector

## What I'd improve for production (notes for the interview)

These are gaps that became obvious during the test but were out of scope:

1. **Hallucinated numbers** — `pto_balance: 17` could become `pto_balance: 22` in the LLM output. Needs deterministic extraction + validator (→ Challenge 1 in main repo).
2. **No reranker** — `similarity_search` returns top-K by cosine alone. Cohere or BGE reranker would improve precision (→ Challenge 2).
3. **No evaluation framework** — there's no way to measure if a prompt change made things better or worse (→ Challenge 5).
4. **Single-shot generation** — no self-correction loop if the answer is unfaithful to context (→ Challenge 4).
5. **No observability** — no per-query traces, token costs, or latency tracking.

The main `contact-center-copilot` project addresses all five.
