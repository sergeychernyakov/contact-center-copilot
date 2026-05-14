"""Hybrid retriever — BM25 + dense FAISS with Reciprocal Rank Fusion.

Implements Challenge 2: industry-specific retrieval with graceful fallback
to cross-industry data when domain-specific chunks are unavailable.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..graph.models import BenchmarkChunk
from .ingestion import build_index, load_benchmarks, load_index


class BenchmarkRetriever:
    """Hybrid retriever combining lexical BM25 and dense FAISS."""

    def __init__(
        self,
        benchmarks_dir: str | Path,
        index_path: str | Path,
        *,
        rrf_k: int = 60,
    ) -> None:
        self.benchmarks_dir = Path(benchmarks_dir)
        self.index_path = Path(index_path)
        self.rrf_k = rrf_k

        if (self.index_path / "index.faiss").exists():
            self.dense: FAISS = load_index(self.index_path)
        else:
            self.dense = build_index(self.benchmarks_dir, self.index_path)

        # BM25 over the raw (un-chunked) docs for now — good enough for POC.
        self._raw_docs = load_benchmarks(self.benchmarks_dir)
        self.bm25 = BM25Retriever.from_documents(self._raw_docs)
        self.bm25.k = 10

    def _filter_by_industry(
        self, docs: list[Document], industry: str
    ) -> tuple[list[Document], bool]:
        """Filter docs by industry, falling back to cross-industry if too few."""
        target = [d for d in docs if d.metadata.get("industry") == industry]
        if len(target) >= 3:
            return target, False
        # Fallback: include cross-industry chunks, flagged in metadata
        fallback = [d for d in docs if d.metadata.get("industry") != industry]
        for d in fallback:
            d.metadata["is_cross_industry"] = True
        return target + fallback, True

    def _rrf(
        self, rankings: list[list[Document]]
    ) -> list[tuple[Document, float]]:
        """Reciprocal Rank Fusion of multiple ranking lists."""
        scores: dict[str, float] = defaultdict(float)
        doc_by_key: dict[str, Document] = {}

        for ranked in rankings:
            for rank, doc in enumerate(ranked):
                key = f"{doc.metadata.get('source', '')}::{doc.page_content[:60]}"
                scores[key] += 1.0 / (self.rrf_k + rank + 1)
                doc_by_key[key] = doc

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(doc_by_key[k], s) for k, s in ranked]

    async def retrieve(
        self,
        query: str,
        industry: str = "cross_industry",
        k: int = 5,
    ) -> list[BenchmarkChunk]:
        """Hybrid search, RRF, industry filter; returns BenchmarkChunks."""
        dense_docs = await asyncio.to_thread(
            self.dense.similarity_search, query, k=10
        )
        bm25_docs = await asyncio.to_thread(self.bm25.invoke, query)

        fused = self._rrf([dense_docs, bm25_docs])
        filtered, fell_back = self._filter_by_industry(
            [d for d, _ in fused], industry
        )
        # Re-attach RRF scores
        score_lookup = {
            f"{d.metadata.get('source', '')}::{d.page_content[:60]}": s
            for d, s in fused
        }

        chunks: list[BenchmarkChunk] = []
        for doc in filtered[:k]:
            key = f"{doc.metadata.get('source', '')}::{doc.page_content[:60]}"
            chunks.append(
                BenchmarkChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", ""),
                    industry=doc.metadata.get("industry", "cross_industry"),
                    is_cross_industry=bool(doc.metadata.get("is_cross_industry", False)),
                    relevance_score=float(score_lookup.get(key, 0.0)),
                )
            )
        return chunks
