"""Document loader for HR policies and HRIS employee data.

Loads JSON sources and converts them into LangChain Documents
with structured metadata for downstream RAG retrieval.
"""

import json
import os
from datetime import datetime
from typing import Dict, List

from langchain_core.documents import Document


class PolicyDocumentLoader:
    """Load HR policies and HRIS employee data from JSON files."""

    def __init__(self, policy_path: str, hris_path: str):
        """
        Initialize document loader with both HR policy and HRIS employee data paths.

        Args:
            policy_path: Path to the HR policies JSON file.
            hris_path: Path to the HRIS employee data JSON file.
        """
        self.policy_path = policy_path
        self.hris_path = hris_path

    def load_policies(self) -> List[Dict]:
        """
        Load HR policies from JSON file with error handling.

        Returns:
            List[Dict]: List of HR policy dictionaries.

        Raises:
            FileNotFoundError: If the policies file is not found.
            json.JSONDecodeError: If the JSON is invalid.
            ValueError: If the loaded data is not a list.
        """
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"Policies file not found: {self.policy_path}")

        with open(self.policy_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("The loaded data is not a list")

        return data

    def load_hris_data(self) -> List[Dict]:
        """
        Load employee data from HRIS JSON file with error handling.

        Returns:
            List[Dict]: List of employee data dictionaries.

        Raises:
            FileNotFoundError: If the HRIS file is not found.
            json.JSONDecodeError: If the JSON is invalid.
            ValueError: If the loaded data is not a list.
        """
        if not os.path.exists(self.hris_path):
            raise FileNotFoundError(f"HRIS file not found: {self.hris_path}")

        with open(self.hris_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("The loaded data is not a list")

        return data

    def create_documents(self) -> List[Document]:
        """
        Create LangChain Documents from HR policies and HRIS data.

        Filters out inactive policies older than 2 years. Only policies
        from the last 2 years OR marked as active are included.

        Returns:
            List[Document]: List of preprocessed documents for RAG system.
        """
        documents: List[Document] = []
        current_year = datetime.now().year
        cutoff_year = current_year - 2

        # === POLICIES ===
        for policy in self.load_policies():
            is_active = policy.get("active", False)
            is_recent = policy.get("effective_year", 0) >= cutoff_year
            if not (is_active or is_recent):
                continue

            page_content = (
                f"Policy ID: {policy.get('id', '')}\n"
                f"Policy Title: {policy.get('title', '')}\n"
                f"Category: {policy.get('category', '')}\n"
                f"Effective Year: {policy.get('effective_year', '')}\n"
                f"Description: {policy.get('description', '')}\n"
                f"Active: {policy.get('active', '')}"
            )

            metadata = {
                "policy_id": policy.get("id", ""),
                "title": policy.get("title", ""),
                "category": policy.get("category", ""),
                "effective_year": policy.get("effective_year", 0),
                "active": policy.get("active", False),
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        # === HRIS DATA ===
        for emp in self.load_hris_data():
            pto_details = emp.get("pto_details", [])
            pto_formatted = ", ".join(
                f"{item.get('type', '')}: {item.get('days', 0)} days"
                for item in pto_details
            )

            page_content = (
                f"Employee Name: {emp.get('name', '')}\n"
                f"Employee ID: {emp.get('employee_id', '')}\n"
                f"Employment Status: {emp.get('employment_status', '')}\n"
                f"Location: {emp.get('location', '')}\n"
                f"PTO Balance: {emp.get('pto_balance', 0)} days\n"
                f"PTO Details: {pto_formatted}"
            )

            metadata = {
                "name": emp.get("name", ""),
                "employee_id": str(emp.get("employee_id", "")),
                "employment_status": emp.get("employment_status", ""),
                "location": emp.get("location", ""),
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents
