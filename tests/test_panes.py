"""Tests for the pure formatting functions in agentic/watch/panes.py."""
from __future__ import annotations

import pytest

from agentic.watch.models import AgentState, TranscriptEntry
from agentic.watch.panes import _fmt_elapsed, _render_entry, render_agent_row


# --- _fmt_elapsed ---------------------------------------------------------

def test_fmt_elapsed_none():
    assert _fmt_elapsed(None) == "  —  "


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00"),
        (2.9, "00:02"),
        (59, "00:59"),
        (60, "01:00"),
        (65, "01:05"),
        (3661, "61:01"),
    ],
)
def test_fmt_elapsed_values(seconds, expected):
    assert _fmt_elapsed(seconds) == expected


# --- render_agent_row ------------------------------------------------------

@pytest.mark.parametrize(
    "status, icon, style",
    [
        ("pending", " ", "dim"),
        ("running", "►", "yellow"),
        ("success", "✓", "green"),
        ("failed", "✗", "red"),
    ],
)
def test_render_agent_row_status_icon_and_style(status, icon, style):
    a = AgentState(id="spec", status=status, elapsed_seconds=5)
    row = render_agent_row(a)
    assert row == f"[{style}]{icon}[/] spec         00:05"


def test_render_agent_row_unknown_status_falls_back():
    a = AgentState(id="spec", status="mystery", elapsed_seconds=None)
    row = render_agent_row(a)
    assert row == "[white] [/] spec         " + _fmt_elapsed(None)


def test_render_agent_row_pads_id_to_12_chars():
    a = AgentState(id="x", status="pending", elapsed_seconds=None)
    row = render_agent_row(a)
    assert "x           " in row  # "x" padded to width 12


# --- _render_entry -----------------------------------------------------

def test_render_entry_text():
    e = TranscriptEntry(ts="t1", kind="text", text="hello world")
    assert _render_entry(e) == "[blue]assistant[/] hello world"


def test_render_entry_tool_use_strips_newlines():
    e = TranscriptEntry(ts="t1", kind="tool_use", tool_name="Read",
                         tool_input='{"file_path":\n"x.py"}')
    assert _render_entry(e) == '[yellow]tool: Read[/] {"file_path": "x.py"}'


def test_render_entry_tool_result_success():
    e = TranscriptEntry(ts="t1", kind="tool_result", text="ok content", success=True)
    assert _render_entry(e) == "[green]result: ok[/] ok content"


def test_render_entry_tool_result_failure():
    e = TranscriptEntry(ts="t1", kind="tool_result", text="boom", success=False)
    assert _render_entry(e) == "[red]result: err[/] boom"


def test_render_entry_unknown_kind_returns_raw_text():
    e = TranscriptEntry(ts="t1", kind="mystery", text="raw fallback text")
    assert _render_entry(e) == "raw fallback text"
