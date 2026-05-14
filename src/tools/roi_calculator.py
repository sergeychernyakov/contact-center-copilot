"""ROI calculator — pure-Python (no LLM) financial projections.

Used by the Reporter agent (or directly via MCP) to generate ROI scenarios
without hallucinated numbers.
"""

from __future__ import annotations

from ..graph.models import ROIScenario


def fcr_improvement_savings(
    current_fcr_pct: float,
    target_fcr_pct: float,
    daily_call_volume: int,
    avg_cost_per_call_usd: float = 5.40,
    repeat_call_multiplier: float = 1.4,
) -> ROIScenario:
    """Estimate annual savings from improving FCR.

    Logic: every percentage point increase in FCR removes a fraction of repeat
    calls. We assume failed-FCR calls trigger `repeat_call_multiplier`x
    additional contacts on average.
    """
    if target_fcr_pct <= current_fcr_pct:
        return ROIScenario(
            name="FCR improvement",
            description="Target FCR not higher than current — no savings projected.",
            estimated_annual_savings_usd=0.0,
            confidence="high",
        )

    delta_pp = (target_fcr_pct - current_fcr_pct) / 100.0
    annual_volume = daily_call_volume * 365
    repeat_calls_avoided = annual_volume * delta_pp * (repeat_call_multiplier - 1)
    savings = repeat_calls_avoided * avg_cost_per_call_usd

    return ROIScenario(
        name="FCR improvement",
        description=(
            f"Increase FCR from {current_fcr_pct:.1f}% to {target_fcr_pct:.1f}% "
            f"reduces ~{repeat_calls_avoided:,.0f} repeat calls/year."
        ),
        estimated_annual_savings_usd=round(savings, 2),
        confidence="medium",
    )


def aht_reduction_savings(
    current_aht_seconds: float,
    target_aht_seconds: float,
    num_agents: int,
    fully_loaded_agent_cost_usd: float = 50_000,
) -> ROIScenario:
    """Estimate annual savings from reducing AHT.

    Lower AHT means the same volume can be handled by fewer agents (or the
    same agents handle more value-added activities).
    """
    if target_aht_seconds >= current_aht_seconds:
        return ROIScenario(
            name="AHT reduction",
            description="Target AHT not lower than current — no savings projected.",
            estimated_annual_savings_usd=0.0,
            confidence="high",
        )

    reduction_pct = (current_aht_seconds - target_aht_seconds) / current_aht_seconds
    fte_savings = num_agents * reduction_pct
    savings = fte_savings * fully_loaded_agent_cost_usd

    return ROIScenario(
        name="AHT reduction",
        description=(
            f"Reduce AHT from {current_aht_seconds:.0f}s to {target_aht_seconds:.0f}s "
            f"frees up ~{fte_savings:.1f} FTE."
        ),
        estimated_annual_savings_usd=round(savings, 2),
        confidence="medium",
    )
