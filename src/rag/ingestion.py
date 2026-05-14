"""Build FAISS index over industry benchmark markdown files."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings

logger = logging.getLogger(__name__)


def _detect_industry(filename: str) -> str:
    name = filename.lower()
    for industry in ("banking", "retail", "healthcare", "telecom", "insurance"):
        if industry in name:
            return industry
    return "cross_industry"


def load_benchmarks(benchmarks_dir: str | Path) -> list[Document]:
    """Read all .md files from a directory into LangChain Documents."""
    benchmarks_dir = Path(benchmarks_dir)
    docs: list[Document] = []

    if not benchmarks_dir.exists():
        logger.warning("Benchmarks dir does not exist: %s", benchmarks_dir)
        return docs

    for md_file in benchmarks_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        industry = _detect_industry(md_file.name)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": md_file.name,
                    "industry": industry,
                    "title": md_file.stem.replace("_", " ").title(),
                },
            )
        )
    return docs


def build_index(benchmarks_dir: str | Path, index_path: str | Path) -> FAISS:
    """Chunk benchmarks, embed them, and persist a FAISS index to disk."""
    settings = get_settings()

    raw_docs = load_benchmarks(benchmarks_dir)
    if not raw_docs:
        raise ValueError(f"No benchmark documents found in {benchmarks_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    store = FAISS.from_documents(chunks, embeddings)
    Path(index_path).mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))

    logger.info("Built FAISS index with %d chunks at %s", len(chunks), index_path)
    return store


def load_index(index_path: str | Path) -> FAISS:
    """Load an existing FAISS index from disk."""
    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
