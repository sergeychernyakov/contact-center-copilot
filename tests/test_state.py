"""Tests for the LangGraph state container and its reducers."""

from __future__ import annotations

from src.graph.state import CopilotState, add_messages


def test_add_messages_appends_right_to_left():
    assert add_messages(["a"], ["b", "c"]) == ["a", "b", "c"]


def test_add_messages_with_empty_left():
    assert add_messages([], ["only"]) == ["only"]


def test_copilot_state_accepts_partial_dict():
    # total=False — every field is optional; nodes fill them in over time.
    state: CopilotState = {"excel_path": "/tmp/x.xlsx", "attempts": 0}
    assert state["excel_path"] == "/tmp/x.xlsx"
    assert state.get("metrics") is None
