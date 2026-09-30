"""MCP integration for kiosk-core — the Central QSR Agent's Kiosk service
(qsr-design §4/§7; retail-use-cases#100/#103).

Same architecture as Order Accuracy's ``mcp_service.py``
(``order-accuracy/dine-in/src/mcp_service.py``): this is the *only* place
that talks to ``mcp_service_sdk``. Kiosk is a **sensor + actuator** (unlike
Order Accuracy, which is sensor-only):

  * Sensor: queue depth (via the standalone queue-service) and order
    activity (via the existing ``kiosk_core.ordering`` DB).
  * Actuator: ``set_board_mode`` — simplify/expand the on-screen menu board,
    the concrete action behind qsr-design's autonomous action #6 ("Kiosk
    queue deep/short -> Kiosk board simplify").

Read tools answer the qsr-design §7-A queries: "How deep is kiosk 4's
queue?", "Is kiosk 4 always slower?", "What's on the board now?". Queue
samples are recorded to the durable log by a background poller
(``start_queue_poller`` in ``mcp_server.py``) so trend/history queries work
across restarts, mirroring how Order Accuracy durably logs every
validation outcome instead of relying on in-memory state.
"""
from __future__ import annotations

import calendar
import logging
import os
import sqlite3
import time
from typing import Any, Optional

from mcp_service_sdk import ServiceConfig, ServiceServer
from mcp_service_sdk.policy import GateLevel

from kiosk_core import config as cfg
from kiosk_core.queue_client import QueueClient
from kiosk_core.qsr_mcp import board_state

logger = logging.getLogger(__name__)

MCP_SERVICE_ENABLED = cfg.QSR_MCP_ENABLED
STORE_ID = cfg.QSR_MCP_STORE_ID
MCP_TRANSPORT = cfg.QSR_MCP_TRANSPORT
MCP_HOST = cfg.QSR_MCP_HOST
MCP_PORT = cfg.QSR_MCP_PORT

_EVENT_TYPES = ("queue_depth_sample", "board_mode_changed")


def _build_service() -> ServiceServer:
    cfg_obj = ServiceConfig(
        service="smart_kiosk",
        store_id=STORE_ID,
        log_backend=cfg.QSR_MCP_LOG_BACKEND if MCP_SERVICE_ENABLED else "memory",
        log_path=cfg.QSR_MCP_LOG_PATH,
        delivery="webhook" if cfg.QSR_MCP_WEBHOOK_URL else "off",
        webhook_url=cfg.QSR_MCP_WEBHOOK_URL,
    )
    return ServiceServer.from_config(cfg_obj)


svc = _build_service()
board_state.configure(cfg.QSR_MCP_BOARD_STATE_PATH)
_queue_client = QueueClient() if cfg.QUEUE_SERVICE_ENABLED else None

# -- 1. declare event types (feeds `describe`) -------------------------------

svc.register_event_type(
    "queue_depth_sample",
    schema={"count": "int", "nearby": "int", "status": "str"},
)

svc.register_event_type(
    "board_mode_changed",
    schema={"mode": "str", "reason": "str|None"},
)


# ---------------------------------------------------------------------------
# Emission — queue samples (called from the background poller) and board
# mode changes (called from the set_board_mode act tool below)
# ---------------------------------------------------------------------------


def emit_queue_sample(snapshot: dict[str, Any]) -> None:
    """Durably log one queue-depth sample. Best-effort: never raises."""
    if not MCP_SERVICE_ENABLED:
        return
    try:
        svc.emit(
            "queue_depth_sample",
            {
                "count": snapshot.get("count"),
                "nearby": snapshot.get("nearby"),
                "status": snapshot.get("status"),
            },
        )
    except Exception:
        logger.exception("[QSR-MCP] Failed to emit queue_depth_sample")


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

_LOG_READ_PAGE = 5000


def _read_entire_log() -> list:
    """Read the full durable log, oldest -> newest (see Order Accuracy's
    identically-named helper for the pagination rationale: both durable log
    backends have a gapless, contiguous ``seq``, so unfiltered paging never
    skips or re-reads rows regardless of how large the log grows)."""
    events: list = []
    since_seq = 0
    while True:
        page = svc.log.read(since_seq=since_seq, limit=_LOG_READ_PAGE)
        if not page:
            break
        events.extend(page)
        since_seq += len(page)
        if len(page) < _LOG_READ_PAGE:
            break
    return events


def _day_bounds_ms(period: str) -> tuple[int, int]:
    """Return ``[start_ms, end_ms)`` epoch-millisecond bounds for a period.

    ``period`` is one of ``"today"``, ``"yesterday"``, ``"all"``, or an
    explicit ``YYYY-MM-DD`` date (UTC calendar day).
    """
    now = time.time()
    day_s = 86400
    today_start = int(now // day_s) * day_s
    if period == "all":
        return 0, int(now * 1000) + 1
    if period == "today":
        start = today_start
    elif period == "yesterday":
        start = today_start - day_s
    else:
        struct = time.strptime(period, "%Y-%m-%d")
        start = calendar.timegm(struct)
    return start * 1000, (start + day_s) * 1000


@svc.read_tool(
    "get_queue_depth",
    description="Current kiosk queue depth (people waiting) and status (LOW/MEDIUM/HIGH).",
    schema={},
)
def get_queue_depth() -> dict[str, Any]:
    if _queue_client is None:
        return {"available": False, "reason": "queue-service disabled"}
    snapshot = _queue_client.get_queue_snapshot()
    if snapshot is None:
        return {"available": False, "reason": "queue-service unreachable"}
    return {"available": True, **snapshot}


@svc.read_tool(
    "get_queue_history",
    description=(
        "Queue-depth trend for a period (today|yesterday|all|YYYY-MM-DD): "
        "sample count, average, and max depth — answers 'is the queue always "
        "deep at this time?'."
    ),
    schema={"period": "str (default 'today')"},
)
def get_queue_history(period: str = "today") -> dict[str, Any]:
    start_ms, end_ms = _day_bounds_ms(period)
    samples = [
        e
        for e in _read_entire_log()
        if e.event_type == "queue_depth_sample" and start_ms <= e.ts_ms < end_ms
    ]
    counts = [e.payload.get("count", 0) for e in samples]
    return {
        "period": period,
        "samples": len(counts),
        "avg_count": round(sum(counts) / len(counts), 2) if counts else 0.0,
        "max_count": max(counts) if counts else 0,
    }


@svc.read_tool(
    "get_order_stats",
    description=(
        "Order activity for a period (today|yesterday|all|YYYY-MM-DD): "
        "confirmed order count, revenue, and top items."
    ),
    schema={"period": "str (default 'today')", "top_n": "int (default 5)"},
)
def get_order_stats(period: str = "today", top_n: int = 5) -> dict[str, Any]:
    date_filter, params = _order_date_filter(period)
    # Raw sqlite3 (sync) rather than the ordering module's aiosqlite
    # repository: this MCP server runs on its own thread/loop, and a
    # short-lived, read-only sync connection is the simplest way to answer a
    # reporting query without mixing async and sync tool-call contexts. WAL
    # mode (set by kiosk_core.ordering.db.init_db) allows this to read
    # concurrently with the async ordering writers.
    if not os.path.exists(cfg.KIOSK_DB_PATH):
        return {"period": period, "orders": 0, "revenue": 0.0, "top_items": []}
    conn = sqlite3.connect(cfg.KIOSK_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        orders_row = conn.execute(
            f"SELECT COUNT(*) AS n, COALESCE(SUM(total), 0) AS revenue "
            f"FROM orders WHERE status = 'confirmed' {date_filter}",
            params,
        ).fetchone()
        top_rows = conn.execute(
            f"SELECT oi.product_id, p.name, SUM(oi.quantity) AS qty "
            f"FROM order_items oi "
            f"JOIN orders o ON o.order_id = oi.order_id "
            f"LEFT JOIN products p ON p.product_id = oi.product_id "
            f"WHERE o.status = 'confirmed' {date_filter} "
            f"GROUP BY oi.product_id ORDER BY qty DESC LIMIT ?",
            (*params, top_n),
        ).fetchall()
    finally:
        conn.close()
    return {
        "period": period,
        "orders": orders_row["n"],
        "revenue": round(orders_row["revenue"], 2),
        "top_items": [
            {"product_id": r["product_id"], "name": r["name"], "quantity": r["qty"]}
            for r in top_rows
        ],
    }


def _order_date_filter(period: str) -> tuple[str, tuple]:
    """Build a ``(sql_fragment, params)`` pair filtering ``orders.created_at``
    (a ``CURRENT_TIMESTAMP`` UTC text column) to the given period."""
    if period == "all":
        return "", ()
    if period == "today":
        return "AND date(created_at) = date('now')", ()
    if period == "yesterday":
        return "AND date(created_at) = date('now', '-1 day')", ()
    # explicit YYYY-MM-DD
    time.strptime(period, "%Y-%m-%d")  # validate, raises ValueError if malformed
    return "AND date(created_at) = date(?)", (period,)


@svc.read_tool(
    "get_board_state",
    description="Current kiosk menu-board mode (full|simplified) and when/why it last changed.",
    schema={},
)
def get_board_state() -> dict[str, Any]:
    return board_state.get()


# ---------------------------------------------------------------------------
# Act tool — the only actuator this service exposes
# ---------------------------------------------------------------------------


@svc.act_tool(
    "set_board_mode",
    level=GateLevel.AUTOMATIC,
    description=(
        "Simplify or restore the kiosk's on-screen menu board "
        "(qsr-design action #6: deep queue -> simplify board)."
    ),
    schema={"mode": "str ('full'|'simplified')", "reason": "str|None"},
    max_calls=20,
    per_seconds=60.0,
)
def set_board_mode(mode: str, reason: Optional[str] = None) -> dict[str, Any]:
    snapshot = board_state.set_mode(mode, reason)
    if MCP_SERVICE_ENABLED:
        try:
            svc.emit("board_mode_changed", {"mode": mode, "reason": reason})
        except Exception:
            logger.exception("[QSR-MCP] Failed to emit board_mode_changed")
    return snapshot
