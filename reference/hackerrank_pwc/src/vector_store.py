"""FAISS-based vector store for HR policy and HRIS employee documents.

Wraps LangChain's FAISS integration with OpenAI embeddings, async
similarity search, and title-based deduplication.
"""

import asyncio
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class PolicyVectorStore:
    """FAISS vector store wrapper for HR documents."""

    def __init__(self):
        """
        Initialize FAISS vector store with OpenAI embeddings.

        The vector store will hold both policy documents and HRIS employee data.
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.store: FAISS = None

    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create and store embeddings of policy and HRIS documents.

        Args:
            documents: List of Document objects containing policies and HRIS data.

        Raises:
            ValueError: If no documents are provided.
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        self.store = FAISS.from_documents(documents, self.embeddings)

    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top k unique documents related to the query.

        Steps:
        1. Check if vector store is initialized.
        2. Retrieve more documents than needed (to handle duplicates).
        3. Remove duplicate documents (based on document title).
        4. Return exactly k unique documents (or fewer if not enough unique).

        Args:
            query: The search query.
            k: Number of documents to return.

        Returns:
            List[Document]: List of k relevant unique documents.

        Raises:
            ValueError: If vector store is not initialized.
        """
        if self.store is None:
            raise ValueError("Vector store is not initialized")

        raw_results = await asyncio.to_thread(
            self.store.similarity_search, query, k=k * 3
        )

        seen_titles = set()
        unique = []
        for doc in raw_results:
            title = doc.metadata.get("title") or doc.metadata.get("name")
            if title and title in seen_titles:
                continue
            if title:
                seen_titles.add(title)
            unique.append(doc)
            if len(unique) >= k:
                break
        return unique

    def similarity_search_sync(self, query: str, k: int = 5) -> List[Document]:
        """
        Synchronous version of similarity search for testing purposes.

        Same logic as similarity_search but blocking.

        Args:
            query: The search query.
            k: Number of documents to return.

        Returns:
            List[Document]: List of k relevant unique documents.

        Raises:
            ValueError: If vector store is not initialized.
        """
        if self.store is None:
            raise ValueError("Vector store is not initialized")

        raw_results = self.store.similarity_search(query, k=k * 3)

        seen_titles = set()
        unique = []
        for doc in raw_results:
            title = doc.metadata.get("title") or doc.metadata.get("name")
            if title and title in seen_titles:
                continue
            if title:
                seen_titles.add(title)
            unique.append(doc)
            if len(unique) >= k:
                break
        return unique

    def save_local(self, path: str) -> None:
        """
        Save vector store locally.

        Args:
            path: Directory path to save the vector store.

        Raises:
            ValueError: If vector store is not initialized.
        """
        if self.store is None:
            raise ValueError("Vector store is not initialized")
        os.makedirs(path, exist_ok=True)
        self.store.save_local(path)

    @classmethod
    def load_local(cls, path: str) -> "PolicyVectorStore":
        """
        Load vector store from disk.

        Args:
            path: Directory path to load the vector store from.

        Returns:
            PolicyVectorStore: Instance with loaded vector store.

        Raises:
            FileNotFoundError: If vector store directory doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store directory does not exist: {path}")

        instance = cls()
        instance.store = FAISS.load_local(
            path,
            instance.embeddings,
            allow_dangerous_deserialization=True,
        )
        return instance
