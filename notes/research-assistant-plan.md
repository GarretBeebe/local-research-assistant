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
| Embeddings | `nomic-embed-text` | Stays resident in a dedicated embedding-only Ollama instance (port 11434); never shares an instance with generation models |

**Hardware notes:**
- Enable iGPU offloading in Ollama (`OLLAMA_GPU_LAYERS`) to leverage the Radeon 680M — expect ~30–50% latency improvement on the 14B model
- Run **three** Ollama instances, all reading from the same model directory (`%USERPROFILE%\.ollama\models`) — models are downloaded once, not duplicated:
  - **Instance A (port 11434) — embeddings only:** `nomic-embed-text`, `OLLAMA_MAX_LOADED_MODELS=1`, `keep_alive=-1` (never evict). Embedding calls throughout the pipeline always route here.
  - **Instance B (port 11435) — generation, primary:** planner (`qwen2.5:14b`) and synthesizer (`qwen2.5:7b`). `OLLAMA_MAX_LOADED_MODELS=1`, `keep_alive=5m`.
  - **Instance C (port 11436) — generation, researchers:** researcher workers (`llama3.1:8b`) and critic (`qwen2.5:3b`, on demand). `OLLAMA_MAX_LOADED_MODELS=1`, `keep_alive=5m`.
- **Runtime memory policy:** `OLLAMA_MAX_LOADED_MODELS=1` per instance ensures each holds at most one model at a time. `OLLAMA_NUM_CTX=4096` caps KV-cache growth — at ctx=4096 a 7B model adds ~300–500 MB per loaded context. Maximum simultaneous RAM footprint: nomic-embed-text (270 MB) + one generation model from B + one generation model from C.
- **Model-load churn budget:** Instance B sequences planner → synthesizer (one 14B→7B swap). Instance C holds `llama3.1:8b` for all researcher workers, then swaps to `qwen2.5:3b` for the critic if used — one swap. Measure cold model-load time for each model in Phase 2 benchmarks; if 14B cold-load exceeds ~30s, pre-warm instance B before accepting queries.
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
| Graph store (SQLite on disk; per-query NetworkX subgraph) | ~10–50 MB (subgraph only; full graph not resident) |
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
│  - File reader                  │
│  - Web fetch (opt-in, off)      │
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

The planner receives the registry's tool schemas at runtime and selects from whatever is available. Adding a new tool means implementing the protocol, adding one entry to the config, and updating the compile-time tool allowlist — no changes to the planner or agent code.

---

## Phase 1 — foundation (week 1–2)

**Goal:** planner + researcher + RAG tool, end-to-end, no parallelism yet.

- [ ] Set up a single Ollama instance (default port 11434) with iGPU offloading configured — three-instance routing (11434/11435/11436) is introduced in Phase 2 alongside parallel dispatch
- [ ] Define the `Tool` protocol and `ToolRegistry` in `tools/base.py` and `tools/registry.py`
- [ ] Implement `tools.yaml` config loading with `yaml.safe_load()` — registry validates each `module` + `class` pair against an explicit compile-time allowlist of known tool implementations; reject relative imports, duplicate names, and unlisted pairs with a hard startup error (a prefix check alone is insufficient)
- [ ] Define `ALLOWED_MODELS` in `config.py`; validate all configured model names against it at startup with a hard error if any are missing
- [ ] Wrap existing vector RAG as a `Tool` implementation (`VectorRAGTool`)
- [ ] Implement the planner loop: prompt `qwen2.5:14b` with the user query + tool schemas from the registry, parse the structured task list it returns
- [ ] Validate planner output against a strict JSON schema before dispatching any agent: tool names must exist in the registry, model names must be in `ALLOWED_MODELS`; reject and retry once with the validation error if invalid
- [ ] Implement the researcher agent: `llama3.1:8b` with access to registered tools
- [ ] Wire planner → researcher → synthesizer as a sequential pipeline (no parallelism yet)
- [ ] Basic CLI interface: `python research.py "your query here"`
- [ ] Log each agent's input/output to a JSON file for debugging — with guards: log rotation (max 10 MB per file, keep 5), max payload size per entry (truncate at 2K chars), excerpt-only for retrieved chunks (first 200 chars + chunk ID, not full text), and a `DEBUG_LOG_FULL_PAYLOADS` flag to opt into verbose logging; defaults to off

**Success criterion:** A query like *"What are the main differences between RAG and Graph RAG?"* (answerable from your own indexed docs) returns a coherent cited answer end-to-end.

---

## Phase 2 — parallelism + memory (week 3–4)

**Goal:** parallel agent execution, shared memory store, measurable latency improvement.

- [ ] Implement parallel agent dispatch using `asyncio` + three Ollama instances (11434 embeddings, 11435 planner/synthesizer, 11436 researchers/critic): N independent researcher workers, each handling one planner sub-task; orchestrator collects all results before passing to synthesizer
- [ ] Have each researcher return a structured result object (source IDs, chunk IDs, excerpts, token counts) — no writes to ChromaDB during a query
- [ ] Synthesizer receives the list of result objects directly from the orchestrator
- [ ] Add a resource governor: concurrency limit (max 2 researchers at once), per-call context length cap, memory pressure fallback that reduces parallelism to 1 when available system memory drops below a configurable threshold (default: 6 GB free). Measure pressure via `psutil.virtual_memory().available` (captures OS, WSL2, Docker, and Ollama process footprint) plus swap activity (`psutil.swap_memory().sin > 0` as a hard signal); app RSS alone is not sufficient as it excludes native Ollama and GPU shared memory
- [ ] Benchmark: track per run — total wall-clock time, per-stage latency (planner / each researcher / synthesizer / critic), peak RSS, swap pages faulted, model load time, prompt + context tokens per call, tokens/sec per model, cold vs warm run distinction, corpus size at time of run; minimum 5 queries across sequential and parallel configurations
- [ ] Add a simple retry: if an agent returns an empty or malformed result, rerun it once

**Success criterion:** Two-agent parallel run completes in less time than sequential, confirmed by benchmark, without exceeding the memory pressure threshold or materially increasing swap/page faults relative to the sequential baseline.

---

## Phase 3 — critic + quality loop (week 5)

**Goal:** self-correcting output; critic flags gaps before the answer reaches the user.

- [ ] Wrap existing Graph RAG as a `Tool` implementation (`GraphRAGTool`); add it to the compile-time allowlist and `tools.yaml`
- [ ] Implement the critic agent (`qwen2.5:3b`): given the synthesized answer + original query, return a JSON object `{ "pass": bool, "issues": [...] }`
- [ ] If critic fails, route back to planner with the issues list for one retry (max 1 re-plan to avoid infinite loops)
- [ ] Add citation linking: each claim in the final answer traces back to a source document chunk
- [ ] Add confidence scoring: synthesizer estimates how well the answer covers the query (1–5)

**Success criterion:** Critic catches at least one genuine gap in a deliberately incomplete test query.

---

## Phase 4 — interface + ingestion (week 6)

**Goal:** usable day-to-day without touching the CLI.

- [ ] Simple web UI using FastAPI + a minimal HTML frontend (Gradio is permitted as a local dev prototype only — it is incompatible with the strict CSP required for networked deployment and must be replaced with plain HTML before Phase 4 is considered complete)
- [ ] Document ingestion endpoint: drag-and-drop PDFs/markdown files → auto-indexed into both RAG stores
- [ ] Query history: store past queries + answers in SQLite, searchable
- [ ] Background indexing: new documents get indexed without blocking the UI
- [ ] Optional: streaming output so partial answers appear as agents complete
- [ ] Docker Compose packaging: containerize the FastAPI app, ChromaDB, and SQLite; Ollama remains a native prerequisite (required for GPU access on all platforms)
- [ ] Security hardening for the web UI and ingestion endpoint:
  - Pydantic request validation per pipeline caps (query 12K chars, messages 200, model name 128 chars, chunks per query 20, planner tasks 10)
  - Local exposure invariant at startup: `ALLOW_INSECURE_LOCALONLY=true` hard-binds to 127.0.0.1/::1 and fails closed if `HOST` is 0.0.0.0 or any LAN address; ambiguous config (no API_KEY and no insecure flag) also fails closed
  - Bearer token auth (`hmac.compare_digest()`) on `/v1/*` API endpoints only; session cookies (bcrypt, `secrets.token_hex(32)`, 8-hour expiry, HttpOnly + SameSite=Lax; `Secure=True` only when HTTPS is confirmed via trusted proxy or direct TLS) on browser UI endpoints only — no endpoint accepts both
  - Per-IP rate limiting (30 req/60s general, 10/60s login) via token-bucket middleware
  - Security headers middleware: `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy: default-src 'self'` (plain HTML only; Gradio is not permitted in networked deployments)
  - Document ingestion: symlink rejection, `path.resolve()` + docs-root boundary check, 10 MB size limit before read, extension allowlist, MIME/magic-byte validation (`python-magic`), PDF extraction in subprocess with 30s timeout and 500-page limit, 50 MB extracted-text limit
  - All SQLite queries use `?` parameterized placeholders
  - Docker: run app as non-root `appuser`, mount existing docs corpus read-only; uploads land in a separate writable staging volume (`UPLOAD_STAGING_DIR`) with size quota and post-indexing cleanup
- [ ] Document the Ollama prerequisite and `tools.yaml` configuration in a setup guide

**Success criterion:** A non-technical user could add a document and query it without touching the terminal. A developer on Linux or macOS could stand up the stack with `docker compose up` after installing Ollama natively.

---

## Tech stack

| Component | Library | Notes |
|---|---|---|
| LLM calls | `ollama` Python SDK | Native streaming support |
| Parallelism | `asyncio` + `httpx` | Three Ollama instances: port 11434 (embeddings), 11435 (planner/synthesizer), 11436 (researchers/critic); async calls |
| Vector store | ChromaDB | Lightweight, file-based, no separate server needed |
| Graph store | SQLite (edges/nodes) + NetworkX (per-query subgraph) | Edges and nodes are stored durably in SQLite tables; no full graph is held in RAM. At query time, fetch only the k-hop neighborhood of relevant nodes from SQLite and materialize a small NetworkX subgraph for traversal — then discard it. Hard cap: 50K nodes / 200K edges in SQLite; refuse ingestion beyond cap until pruning runs (drop edges below weight threshold, oldest nodes first). Pruning operates on the SQLite tables, not an in-memory graph. Fallback to DuckDB or Neo4j if SQLite query latency becomes a bottleneck. This replaces the earlier "pickle the full graph and reload on startup" approach, which would have kept the entire graph resident. |
| Embeddings | `nomic-embed-text` via Ollama | Already in your stack |
| Web framework | FastAPI | Minimal, fast, good async support |
| UI | Plain HTML | Required for Phase 4 networked deployment and strict CSP. Gradio is allowed only as a local development prototype and must be replaced before Phase 4 is complete. |
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
| RAM pressure when all models loaded | Medium | Three Ollama instances with MAX_LOADED_MODELS=1 each; resource governor triggers parallelism reduction when available system memory drops below 6 GB free; keep critic off the always-loaded set; monitor with `ollama ps` + `psutil.virtual_memory()` |
| Graph RAG and vector RAG return conflicting results | Low | Synthesizer prompt explicitly handles contradiction; critic flags it |
| Tool misconfigured or unavailable at startup | Low | Registry validates all tools on load and fails fast with a clear error before any query is accepted |
| Prompt injection via malicious corpus content | Low | Planner output validated against strict JSON schema before dispatch; model names checked against allowlist; user query and chunks treated as data in static templates, not as instructions |

---

## Security

This project follows the same security patterns established in `local-graph-rag` and `rag-system`. Controls are listed here; implementation tasks are distributed into the relevant phases below.

### Input validation and pipeline caps
API boundary (Pydantic):
- Max query length: 12K chars
- Max message count per conversation: 200
- Max model name length: 128 chars

Ingestion caps:
- Max file size before read: 10 MB
- Max extracted text per document: 50 MB
- Max pages per PDF: 500

Pipeline caps (enforced by orchestrator, not by LLM):
- Max chunks returned per query: 20
- Max tool calls per agent per turn: 5
- Max tasks in planner output: 10
- Max concurrent researcher workers: 2 (resource governor may reduce to 1)
- Max final context tokens passed to synthesizer: 8K

CLI entry point enforces a bare query-length check before calling the planner.

### Local exposure invariant
The server enforces a hard binding rule at startup, checked before any request is accepted:
- **Insecure mode** (`ALLOW_INSECURE_LOCALONLY=true`): server must bind to `127.0.0.1` or `::1` only. If `HOST` is `0.0.0.0`, any LAN address, or unset while insecure mode is on, startup fails with a hard error — no warning, no fallback. `CORS_ORIGINS=*` is also rejected in combination with insecure mode.
- **Authenticated mode**: any bind address is allowed; LAN exposure requires auth to be configured.
- **Ambiguous config** (e.g., no `API_KEY` set and `ALLOW_INSECURE_LOCALONLY` not explicitly set): startup fails closed with a clear error directing the operator to set one or the other.

### Authentication (Phase 4)
- Bearer token (`API_KEY`, min 32 chars) validated with `hmac.compare_digest()` — accepted on `/v1/*` API endpoints only
- Session auth: bcrypt passwords, `secrets.token_hex(32)` session tokens, 8-hour expiry, HttpOnly + SameSite=Lax cookies — accepted on browser UI endpoints only. `Secure=True` only when HTTPS is confirmed via `X-Forwarded-Proto: https` from a trusted proxy IP or when the app itself terminates TLS; `Secure=False` on plain-HTTP local dev and in `ALLOW_INSECURE_LOCALONLY` mode.
- API endpoints do not accept session cookies; UI endpoints do not accept bearer tokens. No endpoint accepts both.
- **CSRF protection** on all cookie-authenticated state-changing UI routes (POST/PUT/DELETE): require a per-session CSRF token in a custom request header (`X-CSRF-Token`) or double-submit cookie. SameSite=Lax is not sufficient alone — it does not protect against same-site requests or direct navigation POSTs. Read-only GET routes are exempt.
- **TLS requirement**: the `Secure` cookie flag requires HTTPS. For LAN/networked deployments, TLS must be provided by a reverse proxy (Caddy or nginx) that terminates TLS and forwards to the app over localhost. The app detects HTTPS via `X-Forwarded-Proto: https` from a trusted proxy IP and sets `Secure=True` on cookies; without it (plain HTTP dev), `Secure=False`. `ALLOW_INSECURE_LOCALONLY=true` always sets `Secure=False` and is incompatible with LAN exposure.

### Rate limiting (Phase 4)
- Per-IP token-bucket on all API endpoints (default: 30 req/60s)
- Tighter limit on login endpoint (default: 10 attempts/60s)
- Important here: each query invokes multiple LLM calls, so unconstrained hammering has significant compute cost
- **Reverse proxy IP trust**: `X-Forwarded-For` is only trusted when the request arrives from a configured `TRUSTED_PROXY_IPS` list (same pattern as `local-graph-rag/web/security.py`). If the list is empty (direct bind, no proxy), the connecting IP is used directly. Without this, all users behind a proxy collapse to the same rate-limit bucket, and a client can spoof its IP by injecting the header.

### Security headers (Phase 4)
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Content-Security-Policy: default-src 'self'` via FastAPI middleware
- **Gradio and CSP are incompatible**: Gradio loads scripts and styles from external CDNs and injects inline scripts, requiring `unsafe-inline` and external origins in the CSP — which negates the header. Plain HTML is the only UI option that supports a strict `default-src 'self'` policy. If Gradio is chosen for speed during development, document that it must be replaced before any networked deployment.

### Document ingestion hardening (Phase 4)
- Symlink rejection before read
- `path.resolve()` + root boundary check (resolved path must remain under configured docs root)
- File size limit enforced before read (10 MB max)
- Extension allowlist: PDF, Markdown, plain text only
- MIME / magic-byte check: read the first 512 bytes and validate actual file type against the declared extension using `python-magic`; reject mismatches before passing to any parser
- PDF extraction runs in a subprocess with a hard timeout (30s) and page limit (500 pages); the calling process treats any timeout or crash as a rejected file, not a retry
- Decompression-bomb protection: reject any file that expands beyond 50 MB of extracted text regardless of compressed size

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
- Existing docs corpus directory mounted read-only (`:ro`) in docker-compose — this directory is never written to by the app
- Uploads use a separate writable staging directory (`UPLOAD_STAGING_DIR`) with a configured size quota and automatic cleanup (files removed after indexing completes or on failure). Originals are not persisted beyond the staging window unless explicitly configured. The staging directory is a named Docker volume, not a host-path mount.

### Model allowlist (Phase 1)
- `ALLOWED_MODELS` defined in config; all model names in `config.py` must be in the allowlist
- Validated at startup with a hard error if any configured model is not in the list

### Tool registry safety (Phase 1)
- `tools.yaml` is operator-controlled only — no user input ever reaches the registry loader
- At registry load time, each `module` + `class` pair is validated against an explicit compile-time allowlist of known tool implementations (e.g., `{"tools.vector_rag": ["VectorRAGTool"], "tools.graph_rag": ["GraphRAGTool"], ...}`); any unlisted combination is rejected with a hard startup error. A prefix check alone (e.g., `starts with "tools."`) is insufficient because it still permits loading arbitrary code within the namespace.
- Reject relative imports (module paths starting with `.`)
- Reject duplicate tool names in the loaded registry
- Reject class names that do not match the expected name for their module (no aliasing tricks)

### Web fetch tool — SSRF controls (optional tool, disabled by default)
The web fetch tool conflicts with the "fully offline" project goal and is disabled by default (`enabled: false` in `tools.yaml`). When enabled by the operator, the following controls apply:
- **URL scheme allowlist**: `https://` only; reject `http://`, `file://`, `ftp://`, `gopher://`, and all other schemes
- **DNS/IP resolution check**: resolve the hostname before connecting; reject any result that is a loopback address (127.0.0.0/8, ::1), link-local (169.254.0.0/16, fe80::/10), private range (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, fc00::/7), or unroutable (0.0.0.0, ::0). Connect using the resolved IP directly (do not re-resolve the hostname at connect time), while still sending the original hostname for TLS SNI and verifying the certificate against the original hostname; never disable certificate verification. Verify the peer address after connect matches the checked IP — this prevents DNS rebinding, where a TTL=0 record returns a public IP for the check but a private IP for the actual connection. Re-check after any redirect.
- **Redirect limit**: max 3 redirects; re-validate each redirect destination against the IP blocklist
- **Response limits**: max 5 MB response body, 10s total timeout
- **No cookies or auth headers forwarded** from the app to external hosts
- Document explicitly in `tools.yaml` that enabling this tool breaks the offline guarantee

### Multi-agent prompt injection (Phase 1, novel to this architecture)
The planner generates a structured task list that becomes instructions dispatched to N researcher agents — a wider attack surface than single-agent RAG. Two mitigations:

1. **Prompt construction**: User queries and retrieved document chunks appear as isolated data values in static prompt templates, never as instruction text. System prompts explicitly state that retrieved content cannot change: tool selection, model selection, output schema, citation format, task routing, or any system-level behavior. The model is instructed to treat anything in the context block as passive data only.
2. **Citation integrity**: Citations in the final answer must reference chunk IDs from the retrieved result objects only. The synthesizer is instructed never to generate or invent citations from model knowledge; the critic checks that every citation ID exists in the result set.
3. **Planner output validation**: Treat planner output as untrusted. Before dispatching any researcher, validate the task list against a strict JSON schema: allowed tool names must exist in the registry, model names must be in `ALLOWED_MODELS`. Tasks that fail validation are rejected; the planner is retried once with the validation error appended.

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
