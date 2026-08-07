"""Per-conversation transcript recorder for kiosk-core (offline analysis).

Controlled entirely by ``config.CONVERSATION_LOGGING_ENABLED``. When off
(the default), ``record_turn`` is a no-op and no file I/O happens. When on,
every completed voice turn is appended as one JSON line to
``<config.CONVERSATION_LOG_DIR>/<conversation_id>.jsonl`` -- one file per
conversation (``agent_session_id``, persistent across a customer's voice
turns), so a whole ordering session can be replayed/analyzed as a unit.

Recording is best-effort: any failure (disk full, permissions, bad path) is
logged and swallowed, never raised, so it can never break a live voice turn.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

from kiosk_core import config

logger = logging.getLogger(__name__)

# Conversation ids ultimately come from request.conversation_id (client
# supplied) -- sanitize before using it in a filename to rule out path
# traversal or invalid characters.
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")

# One process-wide lock: two sessions could in principle share a
# conversation_id (e.g. a client reconnect) and must not interleave partial
# writes to the same file.
_lock = threading.Lock()


def _sanitize_conversation_id(conversation_id: str) -> str:
    safe = _SAFE_ID_RE.sub("_", conversation_id or "unknown")
    # "." survives the char-class substitution above (it's a legal filename
    # character), but a run of them ("..") is exactly what a traversal
    # attempt collapses to once "/" is replaced -- collapse it too so no
    # residual ".." can ever appear in the resulting filename.
    safe = re.sub(r"\.\.+", "_", safe)
    return safe or "unknown"


def _log_dir() -> Path:
    path = Path(config.CONVERSATION_LOG_DIR)
    if not path.is_absolute():
        # kiosk_core/conversation_recorder.py -> project root is the parent
        # of the kiosk_core package (mirrors KIOSK_DB_PATH's convention).
        path = Path(__file__).resolve().parent.parent / path
    return path


def record_turn(
    *,
    conversation_id: str,
    turn_id: str,
    user_text: str,
    assistant_text: str,
    end_reason: str | None = None,
) -> None:
    """Append one voice turn to its conversation's JSONL file.

    Args:
        conversation_id: Stable id for the whole conversation
            (``agent_session_id``) -- determines which file the turn is
            appended to.
        turn_id: Id of this individual voice turn (``session_id``).
        user_text: The customer's transcribed utterance (may be empty, e.g.
            a greeting turn with no speech).
        assistant_text: The full assembled reply spoken back to the customer.
        end_reason: Why the underlying audio session ended (``completed``,
            ``error``, ``stopped_by_api``, ...), for later debugging.

    No-op when ``config.CONVERSATION_LOGGING_ENABLED`` is false -- the flag
    is the single switch controlling whether this ever touches disk.
    """
    if not config.CONVERSATION_LOGGING_ENABLED:
        return

    record = {
        "turn_id": turn_id,
        "conversation_id": conversation_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "user": user_text,
        "assistant": assistant_text,
        "end_reason": end_reason,
    }

    try:
        directory = _log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / f"{_sanitize_conversation_id(conversation_id)}.jsonl"
        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:  # noqa: BLE001 - recording must never break a live turn
        logger.exception(
            "Failed to record conversation turn (conversation_id=%s, turn_id=%s)",
            conversation_id, turn_id,
        )
