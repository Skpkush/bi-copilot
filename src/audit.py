"""Append-only audit trail for every copilot question.

One JSON object per line: who asked what, how it was routed, the exact SQL
that ran, which documents were cited, latency, and any error. This is what a
governance review or a "why did the bot say that?" investigation reads.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.config import AUDIT_LOG_PATH

logger = logging.getLogger("bi_copilot")


def record(event: dict[str, Any]) -> None:
    """Write one audit event. Logging must never break answering a question."""
    entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:
        logger.warning("Audit log write failed: %s", exc)


def read_recent(limit: int = 50) -> list[dict[str, Any]]:
    if not AUDIT_LOG_PATH.exists():
        return []
    lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    events = []
    for line in reversed(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
