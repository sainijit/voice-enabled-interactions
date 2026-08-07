#!/usr/bin/env bash
# Benchmark a matrix of OVMS LLM models/precisions through the full voice pipeline.
#
# Why a driver script: swapping the served LLM is not a code change, it is a
# redeploy. Each candidate needs .env rewritten, ovms-llm recreated (which may
# pull ~1-3 GB from HuggingFace and recompile the IR for the GPU), rag-service
# recreated so LiteLlm re-reads the model name, and only then a replay. Doing
# that by hand for 5 candidates is where transcription errors creep in.
#
# Usage:
#   ./model_matrix_benchmark.sh <conversation.jsonl>
#
# Results land in tests/benchmarks/results/replay-<label>.json

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONV="${1:?usage: model_matrix_benchmark.sh <conversation.jsonl>}"
ENV_FILE="$ROOT/.env"
BACKUP="$ROOT/.env.matrix-backup"

# label|model|tool_parser
CANDIDATES=(
  # Gemma-4 E2B (int4 and int8) is excluded: it loads and lists as AVAILABLE,
  # then fails every inference on this OVMS build with an eltwise shape-broadcast
  # error. The MatFormer / per-layer-embedding architecture is not supported by
  # the bundled GenAI runtime. Re-test when OVMS bumps its OpenVINO version.
  "qwen35-2b-int4|OpenVINO/Qwen3.5-2B-int4-ov|qwen3coder"
  "qwen35-2b-int8|OpenVINO/Qwen3.5-2B-int8-ov|qwen3coder"
)

[[ -f "$BACKUP" ]] || cp "$ENV_FILE" "$BACKUP"

# OVMS pull-mode writes a graph.pbtxt next to the weights, and refuses to serve
# a model directory without one. Any directory populated another way (a manual
# huggingface_hub download, an rsync from another box) therefore looks present
# but crash-loops with "Unable to open file: .../graph.pbtxt". Synthesise the
# graph so a pre-staged model behaves the same as a pulled one.
ensure_graph() {  # ensure_graph MODEL_NAME TOOL_PARSER
  local model="$1" parser="$2" dir="$ROOT/models/$model"
  [[ -d "$dir" ]] || return 0
  # OVMS runs as uid 5000 and *rewrites* graph.pbtxt from the CLI flags on every
  # start, so it needs write access to the model directory — not just read. A
  # directory staged by root fails with a misleading "Unable to open file:
  # .../graph.pbtxt" even when that file exists and is world-readable.
  # Seed the graph and hand the whole directory to uid 5000, matching the
  # ownership OVMS gives directories it pulled itself.
  if [[ ! -f "$dir/graph.pbtxt" ]]; then
    echo "  [..] generating graph.pbtxt for pre-staged $model"
    graph_pbtxt "$parser" | docker run --rm -i --user root \
        -v "$ROOT/models:/models" --entrypoint sh busybox:latest \
        -c "cat > '/models/$model/graph.pbtxt'"
  fi
  docker run --rm --user root -v "$ROOT/models:/models" \
      --entrypoint sh busybox:latest -c "chown -R 5000:5000 '/models/$model'"
}

graph_pbtxt() {  # graph_pbtxt TOOL_PARSER -> stdout
  cat <<PBTXT
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
    [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
      max_num_seqs: 4,
      device: "GPU",
      models_path: "./",
      plugin_config: '{"PERFORMANCE_HINT":"LATENCY","CACHE_DIR":"/tmp/ov_cache"}',
      enable_prefix_caching: true,
      cache_size: 4,
      max_num_batched_tokens: 4096,
      tool_parser: "$1",
      enable_tool_guided_generation: true,
    }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}
PBTXT
}

set_env() {  # set_env KEY VALUE
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    printf '\n%s=%s\n' "$key" "$val" >> "$ENV_FILE"
  fi
}

wait_healthy() {  # wait_healthy CONTAINER TIMEOUT_S
  local name="$1" timeout="$2" waited=0
  while (( waited < timeout )); do
    local st
    st="$(docker inspect "$name" --format '{{.State.Health.Status}}' 2>/dev/null || echo missing)"
    [[ "$st" == "healthy" ]] && { echo "  [ok] $name healthy after ${waited}s"; return 0; }
    if [[ "$st" == "missing" ]]; then echo "  [!!] $name missing"; return 1; fi
    sleep 15; waited=$((waited + 15))
  done
  echo "  [!!] $name not healthy within ${timeout}s (last=$st)"
  return 1
}

# OVMS answers /v3/models with 200 and an empty list while it is still pulling
# weights from HuggingFace and compiling the IR, so the container healthcheck
# flips to "healthy" long before the model can serve a request. Gate on the
# model actually appearing in the served list instead.
wait_model_served() {  # wait_model_served MODEL_NAME TIMEOUT_S
  local model="$1" timeout="$2" waited=0
  while (( waited < timeout )); do
    if curl -sf http://127.0.0.1:8000/v3/models 2>/dev/null | grep -qF "\"$model\""; then
      echo "  [ok] ovms serving $model after ${waited}s"; return 0
    fi
    if [[ "$(docker inspect ovms-llm --format '{{.State.Running}}' 2>/dev/null)" != "true" ]]; then
      echo "  [!!] ovms-llm exited while loading $model"; return 1
    fi
    sleep 20; waited=$((waited + 20))
  done
  echo "  [!!] ovms never served $model within ${timeout}s"
  return 1
}

# Appearing in /v3/models only proves the graph loaded, not that it can infer.
# Gemma-4 E2B, for example, lists as AVAILABLE and then dies on the first token
# with a shape-broadcast error from the eltwise shape inference pass. Send one
# real completion and require actual content back before spending 10 minutes on
# a replay that would only produce error strings.
smoke_test() {  # smoke_test MODEL_NAME
  local model="$1" body
  body="$(curl -s --max-time 180 -X POST http://127.0.0.1:8000/v3/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":40}" 2>&1)"
  if printf '%s' "$body" | grep -q '"content"'; then
    echo "  [ok] smoke test passed: $(printf '%s' "$body" | head -c 200)"
    return 0
  fi
  echo "  [!!] smoke test FAILED: $(printf '%s' "$body" | head -c 300)"
  return 1
}

cd "$ROOT"
for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r label model parser <<< "$entry"
  echo "================================================================"
  echo "### $label  ($model, tool_parser=$parser)"
  echo "================================================================"

  set_env OVMS_MODEL_NAME "$model"
  set_env OVMS_TOOL_PARSER "$parser"
  set_env OVMS_TOOL_GUIDED "true"
  ensure_graph "$model" "$parser"

  docker compose up -d --force-recreate ovms-llm >/dev/null 2>&1
  if ! wait_model_served "$model" 3600; then
    echo "  [SKIP] $label — ovms-llm failed to serve the model"
    docker logs ovms-llm 2>&1 | tr '\r' '\n' | grep -v "^Progress:" | tail -25
    continue
  fi

  if ! smoke_test "$model"; then
    echo "  [SKIP] $label — model cannot serve inference on this OVMS build"
    docker logs ovms-llm 2>&1 | tr '\r' '\n' | grep -i "error" | tail -12
    continue
  fi

  docker compose up -d --force-recreate rag-service >/dev/null 2>&1
  if ! wait_healthy rag-service 600; then
    echo "  [SKIP] $label — rag-service unhealthy"
    continue
  fi
  sleep 10  # let the agent warm its first OVMS connection

  echo "  [..] replaying conversation"
  python3 tests/benchmarks/conversation_replay_benchmark.py \
      --conversation "$CONV" --label "$label" \
      --out "tests/benchmarks/results/replay-${label}.json" 2>&1 | tail -25
done

echo "================================================================"
echo "Matrix complete. Restoring .env from $BACKUP"
cp "$BACKUP" "$ENV_FILE"
BASE_MODEL="$(grep '^OVMS_MODEL_NAME=' "$ENV_FILE" | cut -d= -f2)"
docker compose up -d --force-recreate ovms-llm rag-service >/dev/null 2>&1
wait_model_served "$BASE_MODEL" 3600 && wait_healthy rag-service 600
