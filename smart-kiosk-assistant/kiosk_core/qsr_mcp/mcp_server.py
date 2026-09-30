#!/usr/bin/env python3
"""Smart Kiosk MCP server entrypoint (Central QSR Agent surface).

Exposes ``describe``, the read tools (``get_queue_depth``,
``get_queue_history``, ``get_order_stats``, ``get_board_state``), and the one
gated act tool (``set_board_mode``) over MCP — see ``mcp_service.py`` for the
tool implementations. Unlike Order Accuracy (a sensor-only service that must
strip the SDK's default ``subscribe`` tool), Kiosk has no such restriction in
qsr-design, so ``subscribe`` is left in place.

Also starts a background poller that periodically samples queue-service and
durably logs a ``queue_depth_sample`` event, so ``get_queue_history`` has
data to answer trend questions across restarts.

Run directly:
    python -m kiosk_core.qsr_mcp.mcp_server

Or import ``run_mcp_server``/``start_mcp_server`` to run it on a background
thread from the main kiosk-core process (see ``main.py``).
"""
from __future__ import annotations

import logging
import threading
import time

from kiosk_core import config as cfg
from kiosk_core.qsr_mcp.mcp_service import (
    MCP_HOST,
    MCP_PORT,
    MCP_SERVICE_ENABLED,
    MCP_TRANSPORT,
    _queue_client,
    emit_queue_sample,
    svc,
)

logger = logging.getLogger(__name__)


def run_mcp_server() -> None:
    """Start the MCP server (blocking). No-op if disabled via env flag."""
    if not MCP_SERVICE_ENABLED:
        logger.info("[QSR-MCP] MCP_SERVICE_ENABLED=false, MCP server not started")
        return
    logger.info(
        "[QSR-MCP] Starting MCP server: transport=%s host=%s port=%s",
        MCP_TRANSPORT,
        MCP_HOST,
        MCP_PORT,
    )
    app = svc.to_mcp()
    if MCP_TRANSPORT == "stdio":
        app.run("stdio")
    else:
        app.run(MCP_TRANSPORT, host=MCP_HOST, port=MCP_PORT)


def _poll_queue_forever(interval_seconds: float) -> None:
    if _queue_client is None:
        logger.info("[QSR-MCP] Queue-service disabled; queue poller not started")
        return
    logger.info("[QSR-MCP] Queue poller started (interval=%.1fs)", interval_seconds)
    while True:
        snapshot = _queue_client.get_queue_snapshot()
        if snapshot is not None:
            emit_queue_sample(snapshot)
        time.sleep(interval_seconds)


def start_queue_poller(interval_seconds: float | None = None) -> threading.Thread | None:
    """Start the queue-depth sampling loop on a daemon thread."""
    if not MCP_SERVICE_ENABLED:
        return None
    interval = interval_seconds or cfg.QSR_MCP_QUEUE_POLL_SECONDS
    thread = threading.Thread(
        target=_poll_queue_forever,
        args=(interval,),
        daemon=True,
        name="QSRQueuePoller",
    )
    thread.start()
    return thread


if __name__ == "__main__":
    import os

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    start_queue_poller()
    run_mcp_server()
