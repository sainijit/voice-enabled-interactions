"""AgentClient — HTTP client for the rag-service agent endpoint.

Sends kiosk turns to ``POST /api/v1/agent/chat`` and returns the reply.
The reply is delivered as a single string (non-streaming) since the agent
orchestrates multiple tools before forming a final response.

For streaming TTS compatibility the client yields the response text as a
single chunk — callers handle it identically to the streaming RAG path.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import httpx

from kiosk_core import config
from kiosk_core.ordering.order_claim_guard import validate_reply

logger = logging.getLogger(__name__)


class AgentClient:
    """HTTP client for the ordering agent endpoint on rag-service."""

    def __init__(self, agent_url: str, timeout_seconds: float | None = None):
        self.agent_url = agent_url
        self.timeout_seconds = timeout_seconds or config.DEFAULT_HTTP_TIMEOUT_SECONDS

    def get_reply(
        self,
        transcription: str,
        session_id: str,
        user_id: str = "anonymous",
        history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """Call the agent and yield the reply text.

        The entire reply is returned in a single yield so downstream TTS
        logic receives the full response (the agent needs all tool calls to
        complete before it can compose the final answer).

        Args:
            transcription: User's spoken input (transcribed).
            session_id:    Conversation session identifier.
            user_id:       Customer identifier (default "anonymous").
            history:       Prior turns [{role, content}, ...].

        Yields:
            Reply text (one or more chunks — currently one whole reply).

        Raises:
            httpx.HTTPStatusError: On non-2xx response from the agent.
        """
        payload: dict[str, object] = {
            "transcription": transcription,
            "session_id": session_id,
            "user_id": user_id,
        }
        if history:
            cleaned = [
                {"role": str(t.get("role", "")), "content": str(t.get("content", ""))}
                for t in history
                if t.get("content")
            ]
            if cleaned:
                payload["history"] = cleaned

        logger.info(
            "[AGENT-CLIENT] session=%s user=%s message=%r",
            session_id,
            user_id,
            transcription[:100],
        )

        with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
            response = client.post(self.agent_url, json=payload)
            response.raise_for_status()
            data = response.json()

        reply = data.get("reply", "")
        tool_calls = data.get("tool_calls", [])
        llm_ms = data.get("llm_ms")
        llm_calls = data.get("llm_calls", 0)
        retrieval_ms = data.get("retrieval_ms")

        logger.info(
            "[AGENT-CLIENT] session=%s reply_len=%d tool_calls=%s llm_ms=%s llm_calls=%s "
            "retrieval_ms=%s",
            session_id,
            len(reply),
            tool_calls,
            llm_ms,
            llm_calls,
            retrieval_ms,
        )

        # Reconcile the reply against the tools that actually ran before any of
        # it reaches TTS. The agent has been observed announcing a placed and
        # confirmed order without invoking a single ordering tool, so this
        # claim is verified here rather than trusted.
        reply, corrected = validate_reply(reply, tool_calls)
        if corrected:
            logger.error(
                "[AGENT-CLIENT] session=%s reply failed order-claim validation "
                "and was corrected (tool_calls=%s)",
                session_id, tool_calls,
            )

        if reply:
            yield reply
        # Yield tool_calls and LLM timings as metadata so callers can record
        # pipeline traces with genuine LLM time (not whole-agent round-trip).
        yield {
            "_tool_calls": tool_calls,
            "_llm_ms": llm_ms,
            "_llm_calls": llm_calls,
            "_retrieval_ms": retrieval_ms,
        }
