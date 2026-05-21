# Configuration

## Config Files And Override Order

Primary settings live in [../config.yaml](../config.yaml).

When the service runs inside the top-level kiosk compose stack, [../config.container.yaml](../config.container.yaml) is mounted and applied through `SMART_KIOSK_RAG_CONFIG_OVERRIDE_PATHS`. That gives this effective precedence order:

1. `config.yaml`
2. YAML files listed in `SMART_KIOSK_RAG_CONFIG_OVERRIDE_PATHS`
3. `SMART_KIOSK_RAG__...` environment variable overrides

## `server`

| Key | Default | Description |
|---|---|---|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8020` | Service port |

## `api`

| Key | Default | Description |
|---|---|---|
| `cors_allow_origins` | `http://127.0.0.1`, `http://localhost` | Allowed browser origins |
| `openai_model_name` | `smart-kiosk-rag` | Model name returned from OpenAI-compatible APIs |

## `models.llm`

| Key | Default | Description |
|---|---|---|
| `hf_id` | `Qwen/Qwen3-4B-Instruct-2507` | Hugging Face model identifier used for generation |
| `device` | `GPU` | OpenVINO target device for the LLM |
| `weight_format` | `int8` | Export precision when the model must be converted locally |
| `models_base_path` | `./models/llm` | Local LLM cache root |
| `cache_dir` | `./storage/ov_cache` | OpenVINO compilation cache directory |

## `models.embedding`

| Key | Default | Description |
|---|---|---|
| `hf_id` | `BAAI/bge-large-en-v1.5` | Embedding model identifier |
| `device` | `CPU` | Target device for embedding inference |
| `models_base_path` | `./models/embeddings` | Local embedding cache root |
| `normalize_embeddings` | `true` | Normalizes vectors before persistence and search |

## `storage`

| Key | Default | Description |
|---|---|---|
| `persist_directory` | `./storage/vector_db` | Chroma persistence directory |
| `collection_name` | `smart-kiosk-assistant-bge-large` | Active Chroma collection name |

## `retrieval`

| Key | Default | Description |
|---|---|---|
| `top_k` | `3` | Retrieved chunks inserted into the prompt |
| `fetch_k` | `6` | Candidate chunks requested from Chroma before filtering |
| `max_context_chars` | `8000` | Hard cap for retrieved context, runtime context, and short history combined |
| `score_threshold` | `null` | Optional numeric cutoff for Chroma scores |

## `chunking`

| Key | Default | Description |
|---|---|---|
| `max_chunk_chars` | `1500` | Target upper bound per stored chunk |
| `min_chunk_chars` | `200` | Merge threshold for very small chunks |
| `overlap_chars` | `200` | Character overlap between adjacent chunks |
| `semantic_similarity_threshold` | `0.72` | Semantic boundary threshold |
| `llm_passage_chars` | `12000` | Max passage size passed into one semantic splitting pass |
| `llm_passage_tokens` | `4000` | Token cap used when splitting long passages |
| `llm_passage_overlap_tokens` | `300` | Token overlap between adjacent LLM chunking passages |
| `save_chunks_debug` | `./storage/chunks_debug` | Directory for saving chunk-debug output; set to `null` to disable |

## `answering`

| Key | Default | Description |
|---|---|---|
| `system_prompt` | Smart Kiosk Assistant domain prompt | Base instruction that adapts to retrieved retail or QSR context |
| `fallback_to_general_knowledge` | `true` | Allows a general answer when store-specific context is weak |
| `include_source_markers` | `false` | Inserts source markers into prompt context when enabled |
| `history_turns` | `2` | Number of prior user and assistant exchanges retained for follow-ups |
| `max_tokens` | `192` | Default answer-generation cap |
| `generation_timeout_secs` | `90` | Cancels and retries once if a generation call hangs or deadlocks |
| `max_generations_before_reload` | `0` | Disabled by default; set above `0` to force periodic pipeline reloads |

## Environment Overrides

Use double underscores to target nested keys:

```bash
SMART_KIOSK_RAG__MODELS__LLM__DEVICE=CPU
SMART_KIOSK_RAG__RETRIEVAL__TOP_K=4
SMART_KIOSK_RAG__ANSWERING__HISTORY_TURNS=1
```

Use `SMART_KIOSK_RAG_CONFIG_OVERRIDE_PATHS` for one or more comma-separated YAML override files.
