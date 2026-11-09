"""A truncated reply books its own exit and its own bounded budget, with the real message kept.

The model's reply hit its output cap, so Hermes did not run the incomplete action: nothing about
the work was shown wrong. A Kanban worker used to exit 1, which the dispatcher books as a plain
crash and charges to ``consecutive_failures``, so two cut-off replies parked a card at
``kanban.failure_limit``. The turn result stamps ``failure_reason="truncated"``; the worker now
exits ``KANBAN_TRUNCATED_REPLY_EXIT_CODE`` and the sweep counts it on a streak of its own.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_dispatch_streaks import KANBAN_TRUNCATED_REPLY_EXIT_CODE, TRUNCATED_REPLY_FAILURE_LIMIT
from hermes_cli.turn_exit import turn_exit_code

COPY = ("The model's reply was cut off before it finished (it hit its output length limit), so "
        "Hermes didn't run the incomplete action.")


@pytest.mark.parametrize("result", [
    {"failed": True, "failure_reason": "truncated"},
    {"partial": True, "completed": False, "failure_reason": "truncated"},
])
def test_a_truncated_turn_exits_its_own_code_for_a_worker_only(result):
    assert turn_exit_code(result, kanban_worker=True) == KANBAN_TRUNCATED_REPLY_EXIT_CODE
    assert turn_exit_code(result, kanban_worker=False) == 1


@pytest.mark.parametrize("result, expected", [
    ({"partial": True, "failure_reason": "context_overflow"}, 1),     # not a cut-off reply
    ({"partial": True}, 1),                                           # no verdict: still a plain failure
    ({"completed": True, "failure_reason": "truncated"}, 0),          # completion outranks a stale stamp
    ({"interrupted": True, "failure_reason": "truncated"}, 130),
])
def test_only_a_failed_turn_stamped_truncated_gets_the_code(result, expected):
    assert turn_exit_code(result, kanban_worker=True) == expected


def test_a_managed_worker_books_the_code_the_dispatcher_reads(tmp_path, monkeypatch):
    """Where the exit is decided for a managed run: the authority writes it to ``worker_result``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    path = kb.kanban_db_path()
    with kbc.connect_closing(path) as conn:
        tid = kb.create_task(conn, title="fixture", assignee="default")
        task = kb.claim_task(conn, tid, claimer="private-host:owner")
        context = {"db": str(path), "task_id": tid, "run_id": task.current_run_id, "claim_lock": task.claim_lock}
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "worker_bound", {"pid": os.getpid(), "claim_lock": task.claim_lock},
                             run_id=task.current_run_id)
    cut_off = {"failed": True, "failure_reason": "truncated", "final_response": COPY}
    monkeypatch.setattr(session_kanban, "_run_task_turns", lambda *_a: cut_off)
    frame = {"policy": {"kanban_json": json.dumps(context)}}

    session_kanban.run_worker_turns(SimpleNamespace(), frame, [])

    booked = session_kanban.worker_result(path, context)
    assert booked["exit_code"] == KANBAN_TRUNCATED_REPLY_EXIT_CODE
    assert booked["last_output"] == COPY


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _attempt(conn, tid, exit_code, output=COPY):
    """One dispatched attempt whose worker died after booking ``exit_code``; run the dead-worker sweep."""
    host = kb._claimer_id().split(":", 1)[0]
    task = kb.claim_task(conn, tid, claimer=f"{host}:owner")
    assert task is not None
    submitter, interpreter = _dead_pid(), _dead_pid()
    kbd._set_worker_pid(conn, tid, submitter)
    with kb.write_txn(conn):
        kb._append_event(conn, tid, "worker_bound", {"pid": interpreter, "claim_lock": task.claim_lock, "started_at": "x"},
                         run_id=task.current_run_id)
        kb._append_event(conn, tid, "worker_result", {"pid": interpreter, "claim_lock": task.claim_lock,
                                                      "exit_code": exit_code, "last_output": output},
                         run_id=task.current_run_id)
    kbd.detect_crashed_workers(conn)
    return task.current_run_id


def _state(conn, tid):
    row = conn.execute("SELECT status, consecutive_failures, last_failure_error FROM tasks WHERE id=?", (tid,)).fetchone()
    return row["status"], row["consecutive_failures"], row["last_failure_error"]


def _events(conn, tid, kind):
    return [json.loads(r["payload"]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id", (tid, kind))]


def test_a_truncated_attempt_is_a_crash_that_says_why_and_costs_no_failure(conn):
    tid = kb.create_task(conn, title="t", assignee="w")
    run = _attempt(conn, tid, KANBAN_TRUNCATED_REPLY_EXIT_CODE)

    status, failures, error = _state(conn, tid)
    assert status == "ready" and failures == 0           # re-queued, plain budget untouched
    run_row = conn.execute("SELECT outcome, error FROM task_runs WHERE id=?", (run,)).fetchone()
    assert run_row["outcome"] == "crashed"               # still on the board as a crash, not hidden
    (event,) = _events(conn, tid, "crashed")
    assert event["exit_code"] == KANBAN_TRUNCATED_REPLY_EXIT_CODE and event["exit_kind"] == "truncated_reply"
    assert "cut off" in run_row["error"] and "retained by owner" not in run_row["error"]
    assert "cut off" in error                            # the stamped reason the retry worker reads


def test_truncation_cannot_loop_forever(conn):
    tid = kb.create_task(conn, title="t", assignee="w")
    for n in range(1, TRUNCATED_REPLY_FAILURE_LIMIT):
        _attempt(conn, tid, KANBAN_TRUNCATED_REPLY_EXIT_CODE)
        assert _state(conn, tid)[0] == "ready", f"parked early after {n}"
        assert _events(conn, tid, "gave_up") == []
    _attempt(conn, tid, KANBAN_TRUNCATED_REPLY_EXIT_CODE)

    status, failures, _ = _state(conn, tid)
    assert status == "blocked"
    (gave_up,) = _events(conn, tid, "gave_up")
    assert gave_up["truncated_replies"] == gave_up["truncated_reply_limit"] == TRUNCATED_REPLY_FAILURE_LIMIT
    assert gave_up["sticky"] is True                     # held for an operator, not auto-resumed
    assert failures == 1                                 # only the park itself touches the plain counter


def test_a_different_crash_breaks_the_streak_and_still_spends_the_plain_budget(conn):
    tid = kb.create_task(conn, title="t", assignee="w")
    for _ in range(TRUNCATED_REPLY_FAILURE_LIMIT - 1):
        _attempt(conn, tid, KANBAN_TRUNCATED_REPLY_EXIT_CODE)
    _attempt(conn, tid, 1, output="boom")                # an ordinary crash in between
    assert _state(conn, tid)[1] == 1                     # spent consecutive_failures, as before
    _attempt(conn, tid, KANBAN_TRUNCATED_REPLY_EXIT_CODE)
    assert _state(conn, tid)[0] == "ready"               # the streak restarted at 1, not at the limit
    assert _events(conn, tid, "gave_up") == []


def test_a_task_max_retries_overrides_the_truncation_limit(conn):
    tid = kb.create_task(conn, title="t", assignee="w", max_retries=1)
    _attempt(conn, tid, KANBAN_TRUNCATED_REPLY_EXIT_CODE)
    assert _state(conn, tid)[0] == "blocked"


def test_a_plain_exit_one_is_unchanged(conn):
    """The old behaviour for every other failure: charged to consecutive_failures, parked at the limit."""
    tid = kb.create_task(conn, title="t", assignee="w")
    _attempt(conn, tid, 1, output="boom")
    assert _state(conn, tid)[:2] == ("ready", 1)
    _attempt(conn, tid, 1, output="boom")
    assert _state(conn, tid)[0] == "blocked"


def test_truncation_alternating_with_a_protocol_violation_still_parks(conn):
    """Neither kind charges ``consecutive_failures``; counted separately each reset the other forever."""
    tid = kb.create_task(conn, title="t", assignee="w")
    parked_after = None
    for n in range(1, 11):
        _attempt(conn, tid, KANBAN_TRUNCATED_REPLY_EXIT_CODE if n % 2 else 0, output="done")
        if _state(conn, tid)[0] == "blocked":
            parked_after = n
            break
    assert parked_after is not None, "10 alternating attempts and the card was never parked"
    assert parked_after <= TRUNCATED_REPLY_FAILURE_LIMIT + 1
    kinds = [json.loads(r["metadata"] or "{}") for r in conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id", (tid,))]
    assert any(m.get("protocol_violation") for m in kinds)   # the fixture really produced both kinds
    assert any(m.get("truncated_reply") for m in kinds)
