import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic.context import RunContext
from agentic.memory import _fallback_summary, write_memory


def _make_ctx(tmp_path: Path, *, stub_mode: bool = False) -> RunContext:
    return RunContext.create(
        workflow_name="test",
        target_repo_path=tmp_path,
        stub_mode=stub_mode,
    )


def test_generate_summary_fallback_on_sdk_failure(tmp_path: Path, caplog):
    """When _query_summary raises, write_memory uses _fallback_summary and logs a warning."""
    ctx = _make_ctx(tmp_path, stub_mode=False)
    content = "This is the document that needs a summary."

    with patch(
        "agentic.memory._query_summary",
        side_effect=RuntimeError("network timeout"),
    ):
        with caplog.at_level(logging.WARNING, logger="agentic.memory"):
            dest = write_memory("NOTES.md", content, ctx)

    written = dest.read_text(encoding="utf-8")
    expected_summary = _fallback_summary(content)

    assert expected_summary in written, (
        f"expected fallback summary {expected_summary!r} in file contents:\n{written}"
    )

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "expected at least one WARNING log from the fallback path"
    assert any("fallback" in r.getMessage().lower() for r in warning_records), (
        "expected the warning to mention 'fallback'"
    )
