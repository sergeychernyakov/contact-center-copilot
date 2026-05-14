"""Schema Inspector agent — picks relevant Excel sheets via a fast LLM."""

from __future__ import annotations

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from ..config import get_settings
from ..graph.models import SchemaInspection
from ..graph.state import CopilotState

PROMPT = ChatPromptTemplate.from_template(
    """You are inspecting an Excel file from a call-center diagnostic.

Identify which sheets are relevant to call-center KPIs (handle time, FCR,
CSAT, agent stats, abandon rate, service level). Skip sheets about HR costs,
office supplies, payroll, real estate.

Sheets:

{sheets_summary}

Return JSON with keys: relevant_sheets (list[str]), skipped_sheets (list[str]),
rationale (str, one sentence). Be conservative: when uncertain, mark relevant.
"""
)


def _summarise(path: str) -> str:
    xls = pd.ExcelFile(path)
    parts = []
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name, nrows=3)
        cols = [str(c) for c in df.columns]
        line = f"Sheet '{name}' — columns: {', '.join(cols[:15])}"
        if len(cols) > 15:
            line += f" (+{len(cols) - 15} more)"
        parts.append(line)
    return "\n".join(parts)


async def inspect_schema(state: CopilotState) -> dict:
    """LangGraph node: inspect Excel structure."""
    settings = get_settings()
    summary = _summarise(state["excel_path"])

    llm = ChatAnthropic(
        model=settings.model_fast,
        temperature=0,
        api_key=settings.anthropic_api_key,
    ).with_structured_output(SchemaInspection)

    inspection: SchemaInspection = await (PROMPT | llm).ainvoke(  # type: ignore[assignment]
        {"sheets_summary": summary}
    )

    return {
        "schema_inspection": inspection,
        "log": [
            f"[Schema] {len(inspection.relevant_sheets)} relevant, "
            f"{len(inspection.skipped_sheets)} skipped"
        ],
    }
