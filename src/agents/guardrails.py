"""Guardrails agent — PII redaction and tone check before final delivery."""

from __future__ import annotations

import re

from ..graph.models import NarrativeReport
from ..graph.state import CopilotState

# Naive but effective patterns for PII in this domain
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# "Agent name" patterns — proper-cased word pairs like "John Smith" mentioned as employees
AGENT_RE = re.compile(r"\bagent\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")

AGGRESSIVE_WORDS = {
    "lazy",
    "incompetent",
    "stupid",
    "useless",
    "terrible",
    "awful",
    "horrible",
}


def _redact(text: str) -> tuple[str, list[str]]:
    redactions: list[str] = []

    def sub_email(m: re.Match) -> str:
        redactions.append(m.group(0))
        return "[REDACTED_EMAIL]"

    def sub_phone(m: re.Match) -> str:
        redactions.append(m.group(0))
        return "[REDACTED_PHONE]"

    def sub_ssn(m: re.Match) -> str:
        redactions.append(m.group(0))
        return "[REDACTED_SSN]"

    def sub_agent(m: re.Match) -> str:
        redactions.append(m.group(1))
        return "agent [REDACTED_NAME]"

    # SSN before PHONE: the phone pattern also matches SSN-shaped digit runs,
    # so the more specific pattern must run first to label PII correctly.
    text = EMAIL_RE.sub(sub_email, text)
    text = SSN_RE.sub(sub_ssn, text)
    text = PHONE_RE.sub(sub_phone, text)
    text = AGENT_RE.sub(sub_agent, text)
    return text, redactions


def _tone_violations(text: str) -> list[str]:
    lower = text.lower()
    return sorted(w for w in AGGRESSIVE_WORDS if w in lower)


def _sanitise(report: NarrativeReport) -> tuple[NarrativeReport, list[str], list[str]]:
    all_redactions: list[str] = []
    all_tone: list[str] = []

    def clean(text: str) -> str:
        cleaned, red = _redact(text)
        all_redactions.extend(red)
        all_tone.extend(_tone_violations(cleaned))
        return cleaned

    return (
        NarrativeReport(
            executive_summary=clean(report.executive_summary),
            key_findings=[clean(f) for f in report.key_findings],
            benchmark_analysis=report.benchmark_analysis,
            roi_scenarios=report.roi_scenarios,
            recommendations=[clean(r) for r in report.recommendations],
            citations=report.citations,
            requires_human_review=report.requires_human_review,
            review_reasons=report.review_reasons,
        ),
        all_redactions,
        all_tone,
    )


async def apply_guardrails(state: CopilotState) -> dict:
    """LangGraph node: PII redaction + tone validation."""
    report = state.get("narrative")
    if report is None:
        return {"guardrails_passed": False, "log": ["[Guardrails] No report"]}

    cleaned, redactions, tone_issues = _sanitise(report)

    # Escalate to HITL if Critic didn't pass OR tone issues found
    critic = state.get("critic_report")
    needs_review = (critic is not None and not critic.passes) or bool(tone_issues)
    if needs_review:
        cleaned.requires_human_review = True
        # pylint: disable=no-member  # Pydantic v2 FieldInfo confuses pylint about list type
        if tone_issues:
            cleaned.review_reasons.append(f"Tone issues detected: {', '.join(tone_issues)}")
        if critic and not critic.passes:
            cleaned.review_reasons.append(
                f"Critic flagged: faithfulness={critic.faithfulness_score:.2f}"
            )

    return {
        "narrative": cleaned,
        "pii_redactions": redactions,
        "guardrails_passed": not tone_issues,
        "requires_human_review": needs_review,
        "log": [f"[Guardrails] {len(redactions)} PII redacted, {len(tone_issues)} tone issues"],
    }
