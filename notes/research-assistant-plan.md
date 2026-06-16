# Personal research assistant — project plan

## Overview

A locally-hosted multi-agent research assistant. The user submits a research query; a planner model decomposes it into sub-tasks, dispatches specialist agents in parallel, and synthesizes a final answer with citations — all without any cloud API calls.

Designed to run on a GMKtec K16 (Ryzen 7 7735HS, 32GB LPDDR5) as a primary target, but packaged for deployment on Linux and macOS as well. The tool layer is registry-based and user-configurable: each deployment wires in whichever tools are relevant to it.

---

## Goals

- Answer multi-hop research questions that require combining information from multiple sources
- Run fully offline, no data leaving the machine
- Complete a moderately complex query in under 90 seconds
- Produce answers with citations traceable back to source documents
- Be usable as an async tool (fire and check back) rather than requiring instant responses

---

## Hardware & model allocation

| Role | Model | Rationale |
|---|---|---|
| Planner / orchestrator | `qwen2.5:14b` | Best reasoning on available hardware; called once per query |
| Researcher agent (×N) | `llama3.1:8b` | Each instance handles one planner sub-task; runs in parallel with other researcher workers |
| Synthesizer | `qwen2.5:7b` | Single call after all researchers complete; merges results, adds citations |
| Critic / checker | `qwen2.5:3b` | Minimal task (is this correct?), keep it cheap |
| Embeddings | `nomic-embed-text` | Stay resident in RAM throughout, near-instant retrieval |

**Hardware notes:**
- Enable iGPU offloading in Ollama (`OLLAMA_GPU_LAYERS`) to leverage the Radeon 680M — expect ~30–50% latency improvement on the 14B model
- Run two Ollama instances on different ports to allow true parallel agent execution; both read from the same model directory (`%USERPROFILE%\.ollama\models`) — models are downloaded once, not twice
- **Runtime memory policy:** Each instance can independently load model weights and KV cache into RAM. Set `OLLAMA_MAX_LOADED_MODELS=1` per instance so each holds exactly one model at a time. Set `keep_alive=5m` so idle models are evicted before the next pipeline stage loads. Set `OLLAMA_NUM_CTX` to a fixed limit per model (4096 recommended) to cap KV-cache growth — at ctx=4096 a 7B model adds ~300–500 MB of KV cache per loaded context. With these settings: two running instances hold at most two models in RAM simultaneously, which matches the budget below.
- The iGPU shares the same 32GB LPDDR5 pool as system RAM; offloading GPU layers does not add memory, it trades inference speed for headroom
- Keep the critic (`qwen2.5:3b`) off the always-loaded set — load it on demand to save ~2 GB during the research pipeline

**Memory budget (approximate, 4-bit quantization):**

| Component | RAM |
|---|---|
| `qwen2.5:14b` | ~8–9 GB |
| `llama3.1:8b` | ~5 GB |
| `qwen2.5:7b` | ~4 GB |
| `qwen2.5:3b` | ~2 GB (on demand) |
| `nomic-embed-text` | ~270 MB |
| ChromaDB vector index | ~100 MB–1 GB |
| NetworkX graph (in-memory) | ~100 MB–2 GB |
| Python processes + app | ~500 MB |
| Windows 11 OS | ~4–5 GB |
| **Total (critic resident)** | **~24–29 GB** |
| **Total (critic on demand)** | **~22–27 GB** |

The table above covers fixed model weights only. Per-query variable costs add on top:

| Variable cost | Per-query estimate |
|---|---|
| KV cache (7B model, ctx=4096) | ~300–500 MB per loaded context |
| KV cache (14B model, ctx=4096) | ~600–900 MB per loaded context |
| PDF ingestion buffer | ~50–200 MB per document batch |
| Chroma query overhead | ~50–100 MB during active query |
| HTTP response buffers (async) | ~20–50 MB |
| WSL2 / Docker overhead | ~500 MB–1 GB |

The "Total (critic on demand)" row is the minimum fixed footprint; add 1–2 GB per active context window during a live query. 32 GB is sufficient for a personal document corpus, but leaves limited headroom. If total usage exceeds ~28 GB, Windows will begin paging and query latency will degrade significantly. Monitor with `ollama ps` + Task Manager during initial runs.

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────┐
│   Planner  (qwen2.5:14b)        │  ← single call; produces structured task list
│   - decompose query             │
│   - assign agents + tools       │
│   - specify output format       │
└──────────┬──────────────────────┘
           │
   ┌───────┼───────┐
   │    parallel   │
   ▼               ▼
┌──────────┐  ┌──────────┐
│Researcher│  │Researcher│   ← N independent workers, each handling one planner sub-task;
│  (sub-q  │  │  (sub-q  │     pay latency of slowest, not sum; no shared data dependency
│   1)     │  │   2..N)  │
└────┬─────┘  └────┬─────┘
     │              │
     ▼              ▼
┌─────────────────────────────────┐
│         Tool layer              │
│       (ToolRegistry)            │  ← planner receives schemas dynamically;
│  - Vector RAG                   │    which tools are loaded is config-driven,
│  - Graph RAG                    │    not hardcoded
│  - Coding assistant             │
│  - File reader / web fetch      │
│  - … any Tool implementation    │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Synthesizer  (qwen2.5:7b)      │  ← single call; combines agent results
│  - merge findings               │
│  - add citations                │
│  - produce final answer         │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Critic  (qwen2.5:3b)           │  ← optional pass; flag gaps or contradictions
└─────────────────────────────────┘
           │
           ▼
      Final answer
```

**Shared memory store:** `nomic-embed-text` + ChromaDB for durable corpus storage only — embeddings of indexed documents. Per-query agent outputs are not written to ChromaDB. Instead, the orchestrator collects each researcher's result as an in-process structured object (source IDs, chunk IDs, short excerpts, token counts) and passes them directly to the synthesizer. This avoids per-query embedding, index writes, and compaction churn.

**Tool registry:** Tools are not hardcoded — they register into a `ToolRegistry` at startup. Each tool implements a standard `Tool` protocol:

```python
class Tool(Protocol):
    name: str
    description: str
    parameters: dict  # JSON schema
    def run(self, **kwargs) -> str: ...
```

Which tools are loaded is controlled by `tools.yaml`:

```yaml
tools:
  - module: tools.vector_rag
    class: VectorRAGTool
    config:
      collection: my_docs
  - module: tools.graph_rag
    class: GraphRAGTool
  - module: tools.file_reader
    class: FileReaderTool
```

The planner receives the registry's tool schemas at runtime and selects from whatever is available. Adding a new tool means implementing the protocol and adding one entry to the config — no changes to the planner or agent code.

---

## Phase 1 — foundation (week 1–2)

**Goal:** planner + researcher + RAG tool, end-to-end, no parallelism yet.

- [ ] Set up two Ollama instances (ports 11434 and 11435) with iGPU offloading configured
- [ ] Define the `Tool` protocol and `ToolRegistry` in `tools/base.py` and `tools/registry.py`
- [ ] Implement `tools.yaml` config loading with `yaml.safe_load()` — registry validates that every `module` path starts with `tools.` before importing; hard error at startup if not
- [ ] Define `ALLOWED_MODELS` in `config.py`; validate all configured model names against it at startup with a hard error if any are missing
- [ ] Wrap existing vector RAG as a `Tool` implementation (`VectorRAGTool`)
- [ ] Wrap existing Graph RAG as a `Tool` implementation (`GraphRAGTool`)
- [ ] Implement the planner loop: prompt `qwen2.5:14b` with the user query + tool schemas from the registry, parse the structured task list it returns
- [ ] Validate planner output against a strict JSON schema before dispatching any agent: tool names must exist in the registry, model names must be in `ALLOWED_MODELS`; reject and retry once with the validation error if invalid
- [ ] Implement the researcher agent: `llama3.1:8b` with access to registered tools
- [ ] Wire planner → researcher → synthesizer as a sequential pipeline (no parallelism yet)
- [ ] Basic CLI interface: `python research.py "your query here"`
- [ ] Log each agent's input/output to a JSON file for debugging

**Success criterion:** A query like *"What are the main differences between RAG and Graph RAG?"* (answerable from your own indexed docs) returns a coherent cited answer end-to-end.

---

## Phase 2 — parallelism + memory (week 3–4)

**Goal:** parallel agent execution, shared memory store, measurable latency improvement.

- [ ] Implement parallel agent dispatch using `asyncio` + two Ollama instances: N independent researcher workers, each handling one planner sub-task; orchestrator collects all results before passing to synthesizer
- [ ] Have each researcher return a structured result object (source IDs, chunk IDs, excerpts, token counts) — no writes to ChromaDB during a query
- [ ] Synthesizer receives the list of result objects directly from the orchestrator
- [ ] Add a resource governor: concurrency limit (max 2 researchers at once), per-call context length cap, memory pressure fallback that reduces parallelism to 1 if RSS exceeds a configurable threshold (default: 26 GB)
- [ ] Benchmark: track per run — total wall-clock time, per-stage latency (planner / each researcher / synthesizer / critic), peak RSS, swap pages faulted, model load time, prompt + context tokens per call, tokens/sec per model, cold vs warm run distinction, corpus size at time of run; minimum 5 queries across sequential and parallel configurations
- [ ] Add a simple retry: if an agent returns an empty or malformed result, rerun it once

**Success criterion:** Two-agent parallel run completes in less time than sequential, confirmed by benchmark.

---

## Phase 3 — critic + quality loop (week 5)

**Goal:** self-correcting output; critic flags gaps before the answer reaches the user.

- [ ] Implement the critic agent (`qwen2.5:3b`): given the synthesized answer + original query, return a JSON object `{ "pass": bool, "issues": [...] }`
- [ ] If critic fails, route back to planner with the issues list for one retry (max 1 re-plan to avoid infinite loops)
- [ ] Add citation linking: each claim in the final answer traces back to a source document chunk
- [ ] Add confidence scoring: synthesizer estimates how well the answer covers the query (1–5)

**Success criterion:** Critic catches at least one genuine gap in a deliberately incomplete test query.

---

## Phase 4 — interface + ingestion (week 6)

**Goal:** usable day-to-day without touching the CLI.

- [ ] Simple web UI using FastAPI + a minimal HTML frontend (or Gradio for speed)
- [ ] Document ingestion endpoint: drag-and-drop PDFs/markdown files → auto-indexed into both RAG stores
- [ ] Query history: store past queries + answers in SQLite, searchable
- [ ] Background indexing: new documents get indexed without blocking the UI
- [ ] Optional: streaming output so partial answers appear as agents complete
- [ ] Docker Compose packaging: containerize the FastAPI app, ChromaDB, and SQLite; Ollama remains a native prerequisite (required for GPU access on all platforms)
- [ ] Security hardening for the web UI and ingestion endpoint:
  - Pydantic request validation (max query 12K chars, max messages 200, model name 128 chars)
  - Bearer token auth (`hmac.compare_digest()`), bcrypt session passwords, `secrets.token_hex(32)` session tokens, 8-hour expiry, HttpOnly + Secure + SameSite=Lax cookies
  - `ALLOW_INSECURE_LOCALONLY=true` flag with startup warning; conflict-check with `CORS_ORIGINS=*`
  - Per-IP rate limiting (30 req/60s general, 10/60s login) via token-bucket middleware
  - Security headers middleware: `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`
  - Document ingestion: symlink rejection, `path.resolve()` + docs-root boundary check, 10 MB size limit before read, extension allowlist (PDF/Markdown/TXT)
  - All SQLite queries use `?` parameterized placeholders
  - Docker: run app as non-root `appuser`, mount docs directory read-only
- [ ] Document the Ollama prerequisite and `tools.yaml` configuration in a setup guide

**Success criterion:** A non-technical user could add a document and query it without touching the terminal. A developer on Linux or macOS could stand up the stack with `docker compose up` after installing Ollama natively.

---

## Tech stack

| Component | Library | Notes |
|---|---|---|
| LLM calls | `ollama` Python SDK | Native streaming support |
| Parallelism | `asyncio` + `httpx` | Two Ollama instances, async calls |
| Vector store | ChromaDB | Lightweight, file-based, no separate server needed |
| Graph store | NetworkX | In-memory, no separate server. Hard cap: 50K nodes / 200K edges; refuse ingestion beyond cap until pruning runs (drop edges below weight threshold, evict oldest nodes). At query time, load only the k-hop neighborhood of relevant nodes — not the full graph. Persist the graph as a serialized file (pickle) on disk; reload on startup. Fallback: if cap cannot be recovered by pruning, export the edge table to SQLite/DuckDB and query with SQL before considering Neo4j. |
| Embeddings | `nomic-embed-text` via Ollama | Already in your stack |
| Web framework | FastAPI | Minimal, fast, good async support |
| UI | Gradio or plain HTML | Gradio for speed, plain HTML for control |
| Persistence | SQLite | Query history, document metadata |
| Tool interface | `Tool` protocol + `ToolRegistry` | Defined in `tools/base.py`; config-driven loading via `tools.yaml` |
| Containerization | Docker + docker-compose | App + ChromaDB containerized; Ollama is a native prerequisite on all platforms |

---

## Key risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Planner outputs malformed JSON | Medium | Strict output schema in system prompt + validation layer with retry |
| 14B model too slow for interactive use | Medium | iGPU offloading + single planner call design; set expectation of 60–90s |
| Agents hallucinate citations | Medium | Critic pass + force agents to quote chunk IDs, not generate citations freehand |
| RAM pressure when all models loaded | Medium | Total budget is ~22–29 GB against a 32 GB pool shared with the iGPU; keep critic off the always-loaded set, monitor with `ollama ps` + Task Manager, evict summarizer if needed |
| Graph RAG and vector RAG return conflicting results | Low | Synthesizer prompt explicitly handles contradiction; critic flags it |
| Tool misconfigured or unavailable at startup | Low | Registry validates all tools on load and fails fast with a clear error before any query is accepted |
| Prompt injection via malicious corpus content | Low | Planner output validated against strict JSON schema before dispatch; model names checked against allowlist; user query and chunks treated as data in static templates, not as instructions |

---

## Security

This project follows the same security patterns established in `local-graph-rag` and `rag-system`. Controls are listed here; implementation tasks are distributed into the relevant phases below.

### Input validation
- All API endpoints validated via Pydantic at the boundary: max query length (12K chars), max message count (200), max model name length (128 chars)
- CLI entry point enforces a bare query-length check before calling the planner

### Authentication (Phase 4)
- Bearer token (`API_KEY`, min 32 chars) validated with `hmac.compare_digest()` — same pattern as `local-graph-rag/web/auth.py`
- Session auth: bcrypt passwords, `secrets.token_hex(32)` session tokens, 8-hour expiry, HttpOnly + Secure + SameSite=Lax cookies
- `ALLOW_INSECURE_LOCALONLY=true` bypasses auth with a startup warning; forbidden in combination with `CORS_ORIGINS=*`

### Rate limiting (Phase 4)
- Per-IP token-bucket on all API endpoints (default: 30 req/60s)
- Tighter limit on login endpoint (default: 10 attempts/60s)
- Important here: each query invokes multiple LLM calls, so unconstrained hammering has significant compute cost

### Security headers (Phase 4)
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Content-Security-Policy: default-src 'self'` via FastAPI middleware

### Document ingestion hardening (Phase 4)
- Symlink rejection before read
- `path.resolve()` + root boundary check (resolved path must remain under configured docs root)
- File size limit enforced before read (10 MB max)
- Extension allowlist: PDF, Markdown, plain text only

### SQL safety (Phase 4)
- All SQLite queries use `?` parameterized placeholders; no string interpolation

### Secrets handling
- All secrets via environment variables; never hardcoded or committed
- API key generation: `openssl rand -hex 32`
- No credential logging at any log level

### YAML safety (Phase 1)
- `tools.yaml` loaded with `yaml.safe_load()` only — no `yaml.load()`

### Container security (Phase 4)
- App container runs as non-root `appuser`
- Docs directory mounted read-only (`:ro`) in docker-compose

### Model allowlist (Phase 1)
- `ALLOWED_MODELS` defined in config; all model names in `config.py` must be in the allowlist
- Validated at startup with a hard error if any configured model is not in the list

### Tool registry safety (Phase 1)
- At registry load time, each `module` path in `tools.yaml` must start with `tools.` (or a configurable prefix); anything else is rejected with a hard startup error
- `tools.yaml` is operator-controlled only — no user input ever reaches the registry loader

### Multi-agent prompt injection (Phase 1, novel to this architecture)
The planner generates a structured task list that becomes instructions dispatched to N researcher agents — a wider attack surface than single-agent RAG. Two mitigations:

1. **Prompt construction**: User queries and retrieved document chunks appear as isolated data values in static prompt templates, never as instruction text — same pattern as existing projects
2. **Planner output validation**: Treat planner output as untrusted. Before dispatching any researcher, validate the task list against a strict JSON schema: allowed tool names must exist in the registry, model names must be in `ALLOWED_MODELS`. Tasks that fail validation are rejected; the planner is retried once with the validation error appended.

---

## Stretch goals

- **Scheduled research:** cron job that runs nightly queries on topics you're following, saves summaries to the knowledge base
- **Multi-modal:** add a vision model for querying PDFs with diagrams (Qwen2.5-VL variants exist)
- **LAN serving:** expose the API over your 2.5GbE network so other devices on your homelab can query it
- **Incremental learning:** when the critic flags a gap, automatically queue a follow-up indexing job for the missing topic

---

## File structure

```
research-assistant/
├── agents/
│   ├── planner.py          # task decomposition + routing
│   ├── researcher.py       # RAG-augmented research agent (spawned N times in parallel)
│   ├── synthesizer.py      # merges researcher results, adds citations
│   └── critic.py           # output quality checker
├── tools/
│   ├── base.py             # Tool protocol definition
│   ├── registry.py         # ToolRegistry: loads tools from tools.yaml, exposes schemas
│   ├── vector_rag.py       # VectorRAGTool — wraps existing vector RAG
│   ├── graph_rag.py        # GraphRAGTool — wraps existing Graph RAG
│   └── file_reader.py      # FileReaderTool — PDF/markdown ingestion
├── memory/
│   └── store.py            # ChromaDB shared context store
├── orchestrator.py         # main pipeline: plan → dispatch → synthesize
├── api.py                  # FastAPI endpoints
├── ui/
│   └── index.html          # minimal frontend
├── tools.yaml              # declares which tools to load and their config
├── config.py               # model names, Ollama URLs, paths
├── Dockerfile              # app container (FastAPI + agents)
├── docker-compose.yml      # app + ChromaDB; Ollama runs on host
├── benchmark.py            # latency testing
└── research.py             # CLI entrypoint
```

---

*Generated: June 2026*
