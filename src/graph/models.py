"""Pydantic models shared across the pipeline.

These types are the contract between agents. Every agent input/output is
typed via Pydantic so we can detect schema drift, validate LLM outputs,
and avoid stringly-typed bugs.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ─── Schema inspection ──────────────────────────────────────────────────────


class SheetSummary(BaseModel):
    """Lightweight summary of a single Excel sheet."""

    name: str
    columns: list[str]
    row_count: int
    sample_rows: list[dict] = Field(default_factory=list)


class SchemaInspection(BaseModel):
    """Output of the Schema Inspector agent.

    Identifies which sheets are relevant to call-center KPIs and which to skip.
    """

    relevant_sheets: list[str] = Field(
        default_factory=list,
        description="Sheets the analyst should focus on.",
    )
    skipped_sheets: list[str] = Field(
        default_factory=list,
        description="Sheets deemed irrelevant (HR costs, etc).",
    )
    rationale: str = Field(
        default="",
        description="Short explanation of the selection.",
    )


# ─── Metrics extraction ─────────────────────────────────────────────────────


class MetricType(str, Enum):
    """Common contact-center KPI types."""

    FCR = "first_call_resolution"
    AHT = "average_handle_time"
    CSAT = "customer_satisfaction"
    NPS = "net_promoter_score"
    AGENT_UTILIZATION = "agent_utilization"
    ABANDON_RATE = "abandon_rate"
    SERVICE_LEVEL = "service_level"
    OCCUPANCY = "occupancy"
    OTHER = "other"


class Metric(BaseModel):
    """A single deterministically extracted KPI."""

    name: str
    metric_type: MetricType = MetricType.OTHER
    value: float
    unit: str = Field(default="", description="%, seconds, count, etc.")
    period: str = Field(default="", description="e.g. 'Q3 2025', 'last 30 days'.")
    source_sheet: str = ""

    def formatted(self) -> str:
        """Human-friendly representation, e.g. `FCR: 68.0%`."""
        return f"{self.name}: {self.value}{self.unit}".strip()


class MetricsSet(BaseModel):
    """All metrics extracted from the input Excel.

    This is the deterministic ground truth. LLMs MUST cite values from here
    rather than generating their own numbers.
    """

    metrics: list[Metric] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    source_file: str = ""

    def to_dict(self) -> dict[str, float]:
        """Return a flat name→value map for prompt templating."""
        return {m.name: m.value for m in self.metrics}


# ─── Benchmarks (RAG output) ────────────────────────────────────────────────


class BenchmarkChunk(BaseModel):
    """A single retrieved industry-benchmark passage."""

    content: str
    source: str = Field(default="", description="Document title or filename.")
    industry: str = Field(default="cross_industry")
    is_cross_industry: bool = Field(
        default=False,
        description="True if industry-specific benchmark unavailable.",
    )
    relevance_score: float = 0.0


class BenchmarkComparison(BaseModel):
    """One metric compared against its retrieved benchmark."""

    metric_name: str
    client_value: float
    benchmark_value: float
    benchmark_source: str
    is_cross_industry: bool = False
    gap_percentage: float = Field(
        default=0.0,
        description="(client - benchmark) / benchmark * 100",
    )
    direction: str = Field(
        default="neutral",
        description="One of: 'above', 'below', 'on_par'.",
    )


# ─── ROI ────────────────────────────────────────────────────────────────────


class ROIScenario(BaseModel):
    """Single ROI projection scenario."""

    name: str
    description: str
    estimated_annual_savings_usd: float
    confidence: str = Field(
        default="medium",
        description="One of: 'low', 'medium', 'high'.",
    )


# ─── Narrative report ───────────────────────────────────────────────────────


class Citation(BaseModel):
    """In-narrative source attribution."""

    claim: str
    source: str


class NarrativeReport(BaseModel):
    """Final narrative output of the pipeline."""

    executive_summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    benchmark_analysis: list[BenchmarkComparison] = Field(default_factory=list)
    roi_scenarios: list[ROIScenario] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    requires_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


# ─── Critic ─────────────────────────────────────────────────────────────────


class CriticReport(BaseModel):
    """Output of the Critic agent on a generated narrative."""

    faithfulness_score: float = Field(ge=0.0, le=1.0)
    numeric_accuracy_score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    passes: bool = False
