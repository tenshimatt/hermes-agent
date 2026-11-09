"""Per-cause retry streaks for the Kanban dead-worker sweep.

A card's plain budget is ``consecutive_failures`` vs ``kanban.failure_limit``. Two causes say nothing
about the work and get a bounded streak of their own, counted over the card's closed runs:

* a clean exit with no terminal board call (protocol violation), and
* a truncated reply: the model's reply hit its output cap (or a repetition stop, or a stream closed
  mid tool call), so Hermes did not run the incomplete action and the worker exited
  ``KANBAN_TRUNCATED_REPLY_EXIT_CODE``. It used to exit 1, which booked a plain crash and charged
  ``consecutive_failures``: two cut-off replies parked a card at ``failure_limit`` though nothing
  about the work was shown wrong. It still books ``crashed`` with the worker's real message (the card
  says why), but ``TRUNCATED_REPLY_FAILURE_LIMIT`` in a row park it, sticky, as a protocol-violation
  streak does, and a per-task ``max_retries`` overrides it. The truncation streak also counts
  protocol-violation runs that sit in the same trailing run, so a card that alternates the two
  (both spend no plain budget) still parks after ``TRUNCATED_REPLY_FAILURE_LIMIT`` of them. Any
  other closed run breaks the streak, and a different crash still spends the plain budget.
"""

from __future__ import annotations

import sqlite3

# Worker exit "the model's reply was cut off": 76 == BSD EX_PROTOCOL. Beside 75 (quota wall) and 78
# (terminal provider error) in kanban_db.py; it lives here so that file does not grow.
KANBAN_TRUNCATED_REPLY_EXIT_CODE = 76

# Observed on the jwm board (63 charged truncation crashes on 54 cards): 48 streaks of 1, 6 of 2, 1 of 3;
# the attempt after a truncation completed or reached review 48 times of 63.
TRUNCATED_REPLY_FAILURE_LIMIT = 3

# Closed runs to walk when counting a streak; it trips at a handful anyway.
_SCAN_LIMIT = 50


def protocol_violation_streak(conn: sqlite3.Connection, task_id: str) -> int:
    """Count the task's trailing run of clean-exit protocol violations.

    Walks closed runs newest-first (including the one ``detect_crashed_workers``
    just closed). ``rate_limited`` runs are neutral and skipped (a quota wall
    says nothing about the task); any other closed run breaks the streak, so
    the budget counts ONLY protocol violations. Violations are recognized by the
    ``protocol_violation`` run-metadata marker, with the error text as fallback
    for runs recorded before the marker existed.
    """
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_dispatch import WORKER_OUTPUT_SEP
    streak = 0
    rows = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (task_id, _SCAN_LIMIT),
    ).fetchall()
    for row in rows:
        outcome = row["outcome"] or ""
        if outcome == "rate_limited":
            continue
        head = (row["error"] or "").partition(WORKER_OUTPUT_SEP)[0]  # never the worker's own words
        if outcome == "crashed" and (
                _kb._json_dict(row["metadata"]).get("protocol_violation") or "protocol violation" in head):
            streak += 1
            continue
        break
    return streak


def truncated_reply_streak(conn: sqlite3.Connection, task_id: str) -> int:
    """The task's trailing run of truncated-reply crashes, counting the run just closed.

    Protocol-violation crashes inside the run count too: neither kind charges
    ``consecutive_failures``, so each streak counted alone is reset by the other and a card
    alternating the two would never park. Any other closed run breaks the streak.
    """
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_dispatch import WORKER_OUTPUT_SEP
    streak = 0
    rows = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?", (task_id, _SCAN_LIMIT)).fetchall()
    for row in rows:
        if row["outcome"] == "rate_limited":
            continue
        meta = _kb._json_dict(row["metadata"])
        head = (row["error"] or "").partition(WORKER_OUTPUT_SEP)[0]
        if row["outcome"] == "crashed" and (
                meta.get("truncated_reply") or meta.get("protocol_violation") or "protocol violation" in head):
            streak += 1
            continue
        break
    return streak


def dead_worker(pid, claimer, code):
    """The booking for a worker that exited ``KANBAN_TRUNCATED_REPLY_EXIT_CODE``."""
    from hermes_cli.kanban_db_dispatch import _DeadWorker
    return _DeadWorker(
        "truncated_reply", code,
        f"pid {pid} exited {code}: the model's reply was cut off at its output limit and the "
        "incomplete action did not run. Retried on a bounded budget of its own.",
        "crashed",
        {"pid": pid, "claimer": claimer, "exit_kind": "truncated_reply", "exit_code": code,
         "truncated_reply": True},
        truncated_reply=True,
    )


def account_truncated_reply(conn, task_id, pid, claimer, dead) -> bool:
    """Count one truncated-reply crash on its own streak; True when the card was parked.

    Below the limit the run is already back at ``ready`` with the error stamped and
    ``consecutive_failures`` untouched. At the limit the card is parked sticky (``force_trip``), as
    for a protocol-violation streak: the decision, including a per-task ``max_retries``, was made
    here against the truncation streak, not the plain counter.
    """
    from hermes_cli import kanban_db as _kb
    from hermes_cli.kanban_db_dispatch import _record_task_failure
    row = conn.execute("SELECT max_retries FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return False
    override = _kb._row_get(row, "max_retries")
    limit = int(override) if override is not None else TRUNCATED_REPLY_FAILURE_LIMIT
    streak = truncated_reply_streak(conn, task_id)
    if streak < limit:
        return False
    return _record_task_failure(
        conn, task_id, error=dead.error_text, outcome="crashed", failure_limit=limit,
        force_trip=True, release_claim=False, end_run=False,
        event_payload_extra={"pid": pid, "claimer": claimer, "truncated_replies": streak,
                             "truncated_reply_limit": limit},
    )
