"""LangGraph state — the data that flows between nodes.

We use a TypedDict (not Pydantic) for the state container because LangGraph's
reducer semantics work best with plain dicts. Inside the state, individual
fields are Pydantic models for type safety.
"""

from typing import Annotated, TypedDict

from .models import (
    BenchmarkChunk,
    CriticReport,
    MetricsSet,
    NarrativeReport,
    SchemaInspection,
)


def add_messages(left: list, right: list) -> list:
    """Append-only reducer for log messages."""
    return left + right


class CopilotState(TypedDict, total=False):
    """State that flows through the LangGraph workflow.

    `total=False` means every field is optional; nodes fill them in over time.
    """

    # Input
    excel_path: str
    industry: str  # e.g. 'banking', 'retail', 'healthcare'

    # Schema phase
    schema_inspection: SchemaInspection | None

    # Extraction phase
    metrics: MetricsSet | None

    # Retrieval phase
    benchmark_chunks: list[BenchmarkChunk]

    # Reporting phase
    narrative: NarrativeReport | None
    critic_report: CriticReport | None
    attempts: int
    feedback_history: list[str]

    # Guardrails phase
    guardrails_passed: bool
    pii_redactions: list[str]

    # Logging (append-only)
    log: Annotated[list[str], add_messages]

    # Final state
    requires_human_review: bool
