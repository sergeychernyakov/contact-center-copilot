"""Tests for the pure-Python ROI calculator (no LLM)."""

from __future__ import annotations

from src.graph.models import ROIScenario
from src.tools.roi_calculator import aht_reduction_savings, fcr_improvement_savings


def test_fcr_improvement_savings_positive():
    scenario = fcr_improvement_savings(
        current_fcr_pct=65.0,
        target_fcr_pct=75.0,
        daily_call_volume=1000,
        avg_cost_per_call_usd=5.0,
        repeat_call_multiplier=1.5,
    )
    assert isinstance(scenario, ROIScenario)
    assert scenario.name == "FCR improvement"
    # delta_pp=0.10, annual=365_000, repeats avoided=365_000*0.10*0.5=18_250
    # savings = 18_250 * 5.0
    assert scenario.estimated_annual_savings_usd == 91_250.0
    assert scenario.confidence == "medium"


def test_fcr_improvement_no_savings_when_target_not_higher():
    scenario = fcr_improvement_savings(
        current_fcr_pct=70.0,
        target_fcr_pct=70.0,
        daily_call_volume=1000,
    )
    assert scenario.estimated_annual_savings_usd == 0.0
    assert scenario.confidence == "high"


def test_aht_reduction_savings_positive():
    scenario = aht_reduction_savings(
        current_aht_seconds=400.0,
        target_aht_seconds=300.0,
        num_agents=50,
        fully_loaded_agent_cost_usd=60_000,
    )
    assert scenario.name == "AHT reduction"
    # reduction_pct=0.25, fte_savings=12.5, savings=12.5*60_000
    assert scenario.estimated_annual_savings_usd == 750_000.0
    assert "FTE" in scenario.description


def test_aht_reduction_no_savings_when_target_not_lower():
    scenario = aht_reduction_savings(
        current_aht_seconds=300.0,
        target_aht_seconds=350.0,
        num_agents=50,
    )
    assert scenario.estimated_annual_savings_usd == 0.0
    assert scenario.confidence == "high"
