from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agentic.context import RunContext
from agentic.logging import setup_run_logging, teardown_run_logging


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    return RunContext.create(workflow_name="test", target_repo_path=tmp_path, inputs={})


@pytest.fixture(autouse=True)
def clean_root_handlers():
    root = logging.getLogger()
    before = {id(h) for h in root.handlers}
    yield
    for h in list(root.handlers):
        if id(h) not in before:
            h.close()
            root.removeHandler(h)


def test_setup_creates_log_file_at_correct_path(ctx: RunContext):
    log_path = setup_run_logging(ctx)
    assert log_path == ctx.working_dir / "run.log"
    assert log_path.exists()


def test_setup_attaches_file_handler(ctx: RunContext):
    setup_run_logging(ctx)
    root = logging.getLogger()
    names = [h.get_name() for h in root.handlers]
    assert f"agentic-file-{ctx.run_id}" in names


def test_setup_attaches_console_handler(ctx: RunContext):
    setup_run_logging(ctx)
    root = logging.getLogger()
    assert any(getattr(h, "_agentic_console", False) for h in root.handlers)


def test_setup_console_handler_not_duplicated(ctx: RunContext, tmp_path: Path):
    ctx2 = RunContext.create(workflow_name="test", target_repo_path=tmp_path, inputs={})
    setup_run_logging(ctx)
    setup_run_logging(ctx2)
    root = logging.getLogger()
    console_count = sum(1 for h in root.handlers if getattr(h, "_agentic_console", False))
    assert console_count == 1


def test_setup_file_handler_format(ctx: RunContext):
    setup_run_logging(ctx)
    root = logging.getLogger()
    file_handler = next(
        h for h in root.handlers if h.get_name() == f"agentic-file-{ctx.run_id}"
    )
    fmt = file_handler.formatter._fmt
    assert "%(asctime)s" in fmt
    assert "%(levelname)s" in fmt
    assert "%(name)s" in fmt
    assert "%(message)s" in fmt


def test_teardown_removes_file_handler(ctx: RunContext):
    setup_run_logging(ctx)
    teardown_run_logging(ctx)
    root = logging.getLogger()
    names = [h.get_name() for h in root.handlers]
    assert f"agentic-file-{ctx.run_id}" not in names


def test_teardown_leaves_console_handler(ctx: RunContext):
    setup_run_logging(ctx)
    teardown_run_logging(ctx)
    root = logging.getLogger()
    assert any(getattr(h, "_agentic_console", False) for h in root.handlers)


def test_teardown_unknown_run_id_is_noop(ctx: RunContext):
    teardown_run_logging(ctx)  # never set up -- must not raise


def test_logs_write_to_file(ctx: RunContext):
    log_path = setup_run_logging(ctx)
    logger = logging.getLogger("agentic.test")
    logger.info("hello from logging test")
    root = logging.getLogger()
    for h in root.handlers:
        if h.get_name() == f"agentic-file-{ctx.run_id}":
            h.flush()
            break
    assert "hello from logging test" in log_path.read_text()
