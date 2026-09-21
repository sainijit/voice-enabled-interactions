"""Direct-to-OVMS LLM TTFT benchmark -- isolates ADK/agent-framework overhead.

Purpose (demo comparison, NOT a production code path): measure how much of
our agent.ttft_ms figure is genuine model prefill time vs. Google ADK's
Runner/session/event-stream machinery, by hitting the exact same OVMS
endpoint + model with a plain OpenAI-compatible streaming chat completion --
no ADK, no LiteLLM adapter, no tool-calling decision round-trip.

This does NOT touch the live agent code path (rag-service's real
ordering_agent.py / adk_runtime.py are untouched); it is a side-by-side
"component isolation" experiment meant to run once for a demo comparison,
the same way the diarization on/off A/B was done earlier this session.

Run from inside the rag-service container (same docker network as
ovms-llm, no port published to the host):

    docker compose exec rag-service python3 tests/benchmarks/llm_direct_ttft_bench.py

Uses the real production system prompt (rag-service/agentic/ordering_agent.py
_AGENT_INSTRUCTION) and the real model/sampling config
(rag-service/agentic/config.py), so the ONLY variable changed is the
ADK/framework layer -- everything else (model, device, prompt, decoding
params) is identical to the live path.
"""

import json
import re
import statistics
import sys
import time

import httpx

sys.path.insert(0, "/app/rag-service")
from agentic import config as agent_cfg  # noqa: E402


def _load_system_prompt() -> str:
    text = open("/app/rag-service/agentic/ordering_agent.py").read()
    m = re.search(r'_AGENT_INSTRUCTION = """(.*?)"""', text, re.S)
    return m.group(1).strip()


def run_once(client: httpx.Client, system_prompt: str, user_message: str, tools=None) -> float:
    """Returns TTFT in ms for one streamed /chat/completions call."""
    payload = {
        "model": agent_cfg.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": agent_cfg.TEMPERATURE,
        "top_p": agent_cfg.TOP_P,
        "seed": agent_cfg.SEED,
        "max_tokens": agent_cfg.MAX_TOKENS,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": agent_cfg.ENABLE_THINKING},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    url = f"{agent_cfg.LLM_URL}/chat/completions"
    started = time.perf_counter()
    ttft_ms = None
    with client.stream("POST", url, json=payload, timeout=30.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            # First real output token can arrive as plain text (delta.content)
            # OR as the start of a tool call (delta.tool_calls) -- a
            # tool-calling turn (e.g. "order a burger" -> place_order) often
            # emits ONLY tool_calls with content staying null for the whole
            # response, so checking content alone silently waits through the
            # entire tool-call generation and reports a hugely inflated TTFT.
            # reasoning_content (thinking-mode chain-of-thought) is
            # deliberately NOT treated as the first real token here --
            # enable_thinking=False should suppress it, and if it leaks
            # through anyway that reasoning time is a real, separate cost
            # that should not be silently absorbed into "TTFT".
            if delta.get("content") or delta.get("tool_calls"):
                ttft_ms = (time.perf_counter() - started) * 1000
                break
    return ttft_ms if ttft_ms is not None else (time.perf_counter() - started) * 1000


def _load_all_tools():
    try:
        with open("/tmp/tools_compacted.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _bench(client, system_prompt, user_message, tools, runs, label):
    results = []
    payload_chars = len(json.dumps(tools)) if tools else 0
    print(f"\n[llm-direct] {label}: tools={len(tools) if tools else 0} "
          f"tool_schema_chars={payload_chars} runs={runs}")
    for i in range(runs):
        ttft = run_once(client, system_prompt, user_message, tools=tools or None)
        results.append(ttft)
        print(f"  run {i + 1}/{runs}  ttft={ttft:.1f}ms")
    results.sort()
    median = statistics.median(results)
    mean = statistics.mean(results)
    p95 = results[max(0, int(len(results) * 0.95) - 1)]
    print(f"  -> mean={mean:.1f}ms median={median:.1f}ms p95={p95:.1f}ms "
          f"min={min(results):.1f}ms max={max(results):.1f}ms")
    return {
        "label": label,
        "tool_count": len(tools) if tools else 0,
        "tool_schema_chars": payload_chars,
        "mean": mean,
        "median": median,
        "p95": p95,
        "min": min(results),
        "max": max(results),
        "results": results,
    }


def main() -> None:
    system_prompt = _load_system_prompt()
    user_message = "I would like to order one classic chicken burger."
    runs = 8

    all_tools = _load_all_tools()
    # Sweep matrix: 0 / 3 / 6 / 12 tools attached, same prompt+message each
    # time, to isolate how much TTFT cost comes from the tool-schema JSON
    # payload size itself vs. everything else (ADK, model, prompt).
    # place_order must always be included past 0 tools so the model still
    # has a genuine reason to emit a tool_calls delta for an order message.
    priority = ["place_order", "list_products", "get_current_order",
                "confirm_order", "get_upsell_suggestions", "update_order",
                "cancel_order", "remove_from_order", "get_order",
                "list_categories", "get_popular_products", "confirm_active_order"]
    by_name = {(t.get("function", {}).get("name") or t.get("name")): t for t in all_tools}
    ordered_tools = [by_name[n] for n in priority if n in by_name]

    sweep = {
        "0 tools (no function-calling, like the lab)": [],
        "3 tools": ordered_tools[:3],
        "6 tools": ordered_tools[:6],
        f"{len(ordered_tools)} tools (full production set)": ordered_tools,
    }

    print(f"[llm-direct] model={agent_cfg.LLM_MODEL} url={agent_cfg.LLM_URL} "
          f"system_prompt_chars={len(system_prompt)}")
    print("[llm-direct] NO ADK, NO LiteLLM adapter, NO tool-calling DECISION round-trip -- "
          "raw OpenAI-compatible streaming HTTP call straight to OVMS.\n")
    print("[llm-direct] TTFT = first delta with content OR tool_calls (fixes earlier "
          "bug that only checked content and silently waited through tool-call generation).")

    summary = []
    with httpx.Client() as client:
        for label, tools in sweep.items():
            summary.append(_bench(client, system_prompt, user_message, tools, runs, label))

    print("\n" + "=" * 72)
    print("TOOL-SCHEMA PAYLOAD SIZE vs TTFT SWEEP (direct-to-OVMS, no ADK)")
    print("=" * 72)
    print(f"{'label':45s} {'tools':>6s} {'chars':>7s} {'median_ms':>10s}")
    for row in summary:
        print(f"{row['label']:45s} {row['tool_count']:6d} {row['tool_schema_chars']:7d} "
              f"{row['median']:10.1f}")


if __name__ == "__main__":
    main()
