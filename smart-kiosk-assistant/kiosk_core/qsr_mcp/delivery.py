"""Event delivery fan-out for the Smart Kiosk QSR MCP surface.

Self-contained reimplementation of ``mcp_service_sdk.delivery`` (the only
piece of the SDK ``mcp_service.py``'s ``emit`` depended on beyond the log
itself). ``emit`` always saves to the durable log first; this module then
pushes to whichever sink is enabled — ``off`` (default, matches the old
``Delivery(sinks=[DisabledSink()])``) or a webhook callback, with retries.
Failed callbacks are retried; nothing is dropped silently.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request

from kiosk_core.qsr_mcp.event_log import EventEnvelope


class WebhookSink:
    """POSTs the event envelope to a callback URL (stdlib, no extra deps)."""

    name = "webhook"

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self._url = url
        self._timeout_s = timeout_s

    def push(self, event: EventEnvelope) -> bool:
        data = event.to_json().encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError):
            return False


class Delivery:
    """Dispatches an event to the configured sink (none, or one webhook),
    with bounded retries."""

    def __init__(
        self,
        webhook_url: str | None = None,
        max_retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._sink = WebhookSink(webhook_url) if webhook_url else None
        self._max_retries = max_retries
        self._backoff_s = backoff_s

    def dispatch(self, event: EventEnvelope) -> bool:
        """Deliver to the enabled sink, if any. Returns final success flag
        (always ``True`` when delivery is off, matching the old
        ``DisabledSink``)."""
        if self._sink is None:
            return True
        return self._push_with_retry(event)

    def _push_with_retry(self, event: EventEnvelope) -> bool:
        for attempt in range(self._max_retries):
            if self._sink.push(event):
                return True
            if attempt < self._max_retries - 1:
                time.sleep(self._backoff_s * (2**attempt))
        return False
