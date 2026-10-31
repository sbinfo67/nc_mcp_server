"""Audit log of tool calls, written as JSON lines. Argument values are never recorded, only their names."""

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TextIO

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_stream: TextIO | None = None
_owns_stream = False


def configure_audit(target: str) -> None:
    """Enable the audit log, or disable it when `target` is empty.

    `target` is 'stderr' or a file path. A new file is created with mode 600 and always opened in append mode.
    An unwritable path raises OSError so that a requested audit log can never be silently missing.
    """
    global _stream, _owns_stream
    new_stream: TextIO | None = None
    owns = False
    if target.lower() == "stderr":
        new_stream = sys.stderr
    elif target:
        fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            new_stream = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
        except BaseException:
            os.close(fd)
            raise
        owns = True
    with _lock:
        if _stream is not None and _owns_stream:
            _stream.close()
        _stream = new_stream
        _owns_stream = owns


def close_audit() -> None:
    """Stop auditing and close the file if one was opened."""
    configure_audit("")


def audit_enabled() -> bool:
    return _stream is not None


def record_call(tool: str, level: str, arg_names: Iterable[str], outcome: str, started: float) -> None:
    """Append one entry. `outcome` is 'ok', 'denied', 'cancelled' or 'error:<ExceptionName>'.

    A failing write is reported through logging and never propagates: it must not turn a successful tool call
    into an error, nor hide the exception of a failed one.
    """
    with _lock:
        # Read under the lock so that a concurrent configure_audit() cannot close the stream mid-write.
        stream = _stream
        if stream is None:
            return
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "tool": tool,
            "level": level,
            "args": sorted(arg_names),
            "outcome": outcome,
            "ms": round((time.monotonic() - started) * 1000),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            stream.write(line)
            stream.flush()
        except (OSError, ValueError) as exc:
            logger.warning("Audit log write failed: %s", type(exc).__name__)
