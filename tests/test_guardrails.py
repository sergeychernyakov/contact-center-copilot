"""Tests for the Guardrails agent — PII redaction and tone checks (no LLM)."""

from __future__ import annotations

import pytest

from src.agents.guardrails import _redact, _tone_violations, apply_guardrails
from src.graph.models import CriticReport, NarrativeReport


def test_redact_email():
    cleaned, redactions = _redact("Email maria@acme.io for the full report.")
    assert "[REDACTED_EMAIL]" in cleaned
    assert "maria@acme.io" in redactions


def test_redact_phone():
    cleaned, redactions = _redact("Call 555-867-5309 to reach the desk.")
    assert "[REDACTED_PHONE]" in cleaned
    assert redactions == ["555-867-5309"]


def test_redact_ssn():
    cleaned, redactions = _redact("On file: SSN 123-45-6789 for the record.")
    assert "[REDACTED_SSN]" in cleaned
    assert "123-45-6789" in redactions


def test_redact_agent_name():
    cleaned, redactions = _redact("Escalated by agent Jane Smith on Monday.")
    assert "agent [REDACTED_NAME]" in cleaned
    assert "Jane Smith" in redactions


def test_redact_leaves_clean_text_untouched():
    text = "FCR improved by five points this quarter."
    cleaned, redactions = _redact(text)
    assert cleaned == text
    assert redactions == []


def test_tone_violations_detected_and_sorted():
    assert _tone_violations("The team was lazy and incompetent.") == [
        "incompetent",
        "lazy",
    ]


def test_tone_violations_none_for_neutral_text():
    assert _tone_violations("The team handled the volume well.") == []


@pytest.mark.asyncio
async def test_apply_guardrails_no_report_does_not_pass():
    result = await apply_guardrails({})
    assert result["guardrails_passed"] is False


@pytest.mark.asyncio
async def test_apply_guardrails_redacts_pii_and_passes_clean_report():
    report = NarrativeReport(
        executive_summary="Reach us at help@acme.com for details.",
        key_findings=["FCR is strong this quarter."],
    )
    result = await apply_guardrails({"narrative": report})

    assert result["guardrails_passed"] is True
    assert result["requires_human_review"] is False
    assert "help@acme.com" in result["pii_redactions"]
    assert "[REDACTED_EMAIL]" in result["narrative"].executive_summary


@pytest.mark.asyncio
async def test_apply_guardrails_escalates_on_tone_issue():
    report = NarrativeReport(executive_summary="The agents were lazy.")
    result = await apply_guardrails({"narrative": report})

    assert result["guardrails_passed"] is False
    assert result["requires_human_review"] is True
    assert any("Tone issues" in reason for reason in result["narrative"].review_reasons)


@pytest.mark.asyncio
async def test_apply_guardrails_escalates_when_critic_fails():
    report = NarrativeReport(executive_summary="A clean executive summary.")
    critic = CriticReport(faithfulness_score=0.4, numeric_accuracy_score=0.9, passes=False)
    result = await apply_guardrails({"narrative": report, "critic_report": critic})

    assert result["guardrails_passed"] is True
    assert result["requires_human_review"] is True
    assert any("Critic flagged" in reason for reason in result["narrative"].review_reasons)
