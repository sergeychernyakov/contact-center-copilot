"""RAG pipeline for HR policy queries.

Combines vector retrieval over policy documents with HRIS employee
lookups to produce personalized, citation-aware answers via GPT-4o.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .vector_store import PolicyVectorStore

logger = logging.getLogger(__name__)


class HRPolicyRAG:
    """RAG chain that answers HR policy questions with optional employee context."""

    def __init__(self, vector_store: PolicyVectorStore, hris_data_path: str = None):
        """
        Initialize with vector store and optional HRIS data path.

        If hris_data_path is provided, the method:
        1. Loads HRIS data from the JSON file.
        2. Converts it to a dictionary with employee_id as the key.
        3. Stores it in self.hris_data for efficient lookups.

        Args:
            vector_store: PolicyVectorStore instance for retrieving relevant documents.
            hris_data_path: Optional path to HRIS employee data JSON file.
        """
        self.vector_store = vector_store
        self.hris_data: Dict[str, Dict] = {}

        if hris_data_path and os.path.exists(hris_data_path):
            with open(hris_data_path, "r") as f:
                raw = json.load(f)

            if isinstance(raw, list):
                for emp in raw:
                    emp_id = str(emp.get("employee_id", ""))
                    if emp_id:
                        self.hris_data[emp_id] = emp
            elif isinstance(raw, dict):
                self.hris_data = {str(k): v for k, v in raw.items()}

        self.chain = self._create_chain()

    def _create_chain(self):
        """
        Create a retrieval + generation chain using LangChain.

        The chain uses:
        1. A template prompt for HR policy questions.
        2. The OpenAI LLM (ChatOpenAI / gpt-4o).
        3. String output parsing.

        Returns:
            A configured LangChain chain.
        """
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful HR assistant. Answer the question based ONLY on the provided context from HR policies.
If the context doesn't contain enough information, say so honestly.
Cite policy titles when relevant.

Context (HR Policies):
{context}

{employee_info}

Question: {question}

Answer:"""
        )

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        return prompt | llm | StrOutputParser()

    async def query(self, question: str, employee_id: str = None) -> str:
        """
        Retrieve policies and generate a response.

        Steps:
        1. Retrieve relevant documents using the vector store.
        2. Check if this is a personal HRIS-related query.
        3. Add HRIS data to the context if applicable and employee_id is provided.
        4. Generate a response using the LLM chain.

        Args:
            question: The user's question about HR policies.
            employee_id: Optional employee ID for personalized responses.

        Returns:
            str: AI-generated response based on relevant policies and HRIS data.
        """
        relevant_docs = await self.get_relevant_documents(question, k=5)
        context = "\n\n".join(
            [
                f"[Policy: {doc['metadata'].get('title', 'Unknown')}]\n{doc['content']}"
                for doc in relevant_docs
            ]
        )

        employee_info = ""
        if employee_id:
            hris = await self.call_hris_api(employee_id)
            if "error" not in hris:
                employee_info = (
                    f"Employee personal data:\n{json.dumps(hris, indent=2)}"
                )

        return await self.chain.ainvoke(
            {
                "context": context,
                "employee_info": employee_info,
                "question": question,
            }
        )

    async def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant HR policies based on the query.

        The vector store already deduplicates by title, so this method
        only converts Document objects into dict format for the UI.

        Args:
            query: The search query.
            k: Number of documents to retrieve.

        Returns:
            List[Dict]: List of dictionaries with structure:
                {
                    "metadata": <document metadata dict>,
                    "content": <document content as string>
                }
        """
        docs = await self.vector_store.similarity_search(query, k=k)
        return [
            {"metadata": dict(d.metadata), "content": d.page_content}
            for d in docs
        ]

    async def call_hris_api(self, employee_id: str) -> Dict:
        """
        Retrieve employee-specific data from HRIS.

        Steps:
        1. Convert employee_id to string to ensure lookup works.
        2. Look up the employee data in self.hris_data.
        3. Return the employee data if found.
        4. Return {'error': 'Employee ID {id} not found'} if not found.

        Args:
            employee_id: Employee identifier.

        Returns:
            Dict: Employee data if found, or error dictionary.
        """
        emp_id_str = str(employee_id)
        if emp_id_str in self.hris_data:
            return self.hris_data[emp_id_str]
        return {"error": f"Employee ID {employee_id} not found"}
