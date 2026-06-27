# Local Research Assistant

A locally-hosted multi-agent research assistant. Submit a research query, and a planner model decomposes it into sub-tasks, dispatches specialist agents in parallel, and synthesizes a final answer with citations — all without any cloud API calls.

Builds on existing vector RAG, Graph RAG, and coding assistant work, exposing them as tools in the agent's tool layer.

## Status

**Phase 4 complete** — FastAPI web UI with session-cookie auth (bcrypt, CSRF), bearer-token `/v1/*` API, per-IP rate limiting, security headers, document upload with full validation, SQLite query history, and Docker Compose packaging. Two items remain wired-but-stubbed pending final configuration: RAG ingest endpoint signatures and three-instance Ollama routing.

Phase 3: `GraphRAGTool`, critic agent (`qwen2.5:3b`), one re-plan cycle on failure, inline citations, 1–5 confidence scoring.

Phase 2: parallel researcher dispatch via `asyncio.TaskGroup`, `ResourceGovernor`, benchmark logging.

See [`notes/research-assistant-plan.md`](notes/research-assistant-plan.md) for the full project plan, architecture diagram, and phased roadmap.

**CLI:**
```
python research.py "your query here"
```

**Web UI:**
```bash
# Generate a bcrypt password hash
python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"

# Add to .env:
#   ADMIN_PASSWORD_HASH=<hash from above>
#   ALLOW_INSECURE_LOCALONLY=true  (loopback only, no API bearer token needed)

uv run uvicorn api:app --host 127.0.0.1 --port 8080
# Then open http://localhost:8080
```

**Docker Compose:**
```bash
# Copy and fill in .env (ADMIN_PASSWORD_HASH, API_KEY, RAG_BASE_URL, etc.)
cp .env.example .env

docker compose up
```

`docker-compose.yml` always sets `HOST=0.0.0.0` inside the container so Docker's port
publishing (`8080:8080`) is reachable. This means `ALLOW_INSECURE_LOCALONLY=true` is
**incompatible with Docker** — `validate_server()` rejects a non-loopback `HOST` in insecure
mode. Set `API_KEY` instead. To expose the published port only on the host's loopback
interface, change the port binding in `docker-compose.yml` to `127.0.0.1:8080:8080`.

`ALLOW_INSECURE_LOCALONLY=true` is for bare-metal or VM deployments where `HOST=127.0.0.1`
actually controls the network binding.

## Setup

**Prerequisites:** Ollama running locally with the required models pulled; `rag-system` and `local-graph-rag` running and accessible.

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env:
#   RAG_BASE_URL / RAG_INTERNAL_TOKEN  — match your rag-system .env
#   GRAPH_RAG_BASE_URL                 — default http://localhost:8001
#   GRAPH_RAG_API_KEY                  — match API_KEY in local-graph-rag .env
#                                        (omit only if ALLOW_INSECURE_LOCALONLY=true)
# Note: local-graph-rag defaults to port 8000; set API_PORT=8001 in its .env
# to avoid conflicting with rag-system.

# Run a query
uv run python research.py "What is retrieval-augmented generation?"
```

Pipeline logs are written to `logs/pipeline.jsonl`; per-run benchmarks to `logs/benchmark.jsonl` (both rotating, 10 MB max). Benchmark entries include `critic_passed`, `re_planned`, and `confidence` fields from Phase 3. Set `DEBUG_LOG_FULL_PAYLOADS=true` in `.env` to log full chunk text. Set `MAX_CONCURRENT_RESEARCHERS` and `MEMORY_PRESSURE_THRESHOLD_MB` to tune the resource governor.

## Goals

- Answer multi-hop research questions that combine information from multiple sources
- Run fully offline — no data leaves the machine
- Complete a moderately complex query in under 90 seconds
- Produce answers with citations traceable back to source documents
- Work as an async tool (fire and check back) rather than requiring instant responses

## Architecture (target)

```
User query
    │
    ▼
Planner (qwen2.5:14b) — decomposes query into independent sub-tasks
    │
    ├──► Researcher · sub-task 1 (llama3.1:8b) ─┐
    ├──► Researcher · sub-task 2 (llama3.1:8b) ─┤  N independent workers, each with a
    └──► Researcher · sub-task N ...            ─┤  distinct sub-question; tool layer:
                                                 │  Vector RAG · Graph RAG · file tools
    ┌────────────────────────────────────────────┘
    ▼
Synthesizer (qwen2.5:7b) — merges findings, adds citations
    │
    ▼
Critic (qwen2.5:3b) — flags gaps or contradictions
    │
    ▼
Final answer
```

Full diagram, shared memory design, and rationale in the [project plan](notes/research-assistant-plan.md).

## Hardware & models

Primary target is a GMKtec K16 (Ryzen 7 7735HS, 32GB LPDDR5) with iGPU offloading via Ollama. Phase 2 supports two dedicated Ollama instances to prevent model-load churn under parallel dispatch: one for planner/synthesizer (`OLLAMA_PLANNER_URL`) and one for researcher workers (`OLLAMA_RESEARCHER_URL`). Both default to port 11434; point them at separate ports once you have dedicated instances running. All instances share a single model directory.

The iGPU draws from the same 32GB LPDDR5 pool as system RAM. Total memory budget for all models + RAG stores + OS is approximately 22–27 GB with the critic loaded on demand (recommended) or 24–29 GB with all models resident. Expect paging and degraded latency if usage exceeds ~28 GB. Also deployable on Linux and macOS via Docker Compose (Ollama runs natively on all platforms for GPU access).

| Role | Model |
|---|---|
| Planner / orchestrator | `qwen2.5:14b` |
| Researcher agent (×N) | `llama3.1:8b` |
| Synthesizer | `qwen2.5:7b` |
| Critic / checker | `qwen2.5:3b` |
| Embeddings | `nomic-embed-text` |

## Tech stack

| Component | Library |
|---|---|
| LLM calls | `ollama` Python SDK |
| Parallelism | `asyncio` + `asyncio.to_thread` |
| Vector store | ChromaDB |
| Graph store | SQLite (edges/nodes) + NetworkX (per-query subgraph) |
| Web framework | FastAPI |
| UI | Plain HTML (Gradio for local dev only — incompatible with strict CSP) |
| Persistence | SQLite |
| Tool interface | `Tool` protocol + `ToolRegistry` (config-driven via `tools.yaml`) |
| Containerization | Docker + docker-compose (app + ChromaDB; Ollama is a native prerequisite) |

## Security

The project follows the same security patterns established in the sibling `local-graph-rag` and `rag-system` projects, extended for the multi-agent architecture:

- **Startup fails closed**: server requires either a configured `API_KEY` (min 32 chars) or `ALLOW_INSECURE_LOCALONLY=true` (loopback-only binding). `ADMIN_PASSWORD_HASH` must be a valid bcrypt hash — format is validated at startup so a misconfigured value produces a clear error rather than silently locking out all logins.
- **Tool registry allowlist**: `tools.yaml` entries are validated against an explicit compile-time allowlist of known module+class pairs; arbitrary code loading is rejected at startup.
- **Planner output treated as untrusted**: task list is validated against a strict JSON schema (tool names, model names) before any agent is dispatched.
- **Three-way auth separation**: bearer tokens for `/v1/*` API, session cookies for browser UI (`/ui/*`), no overlap. `/ui/*` is browser-only — non-browser clients should use `/v1/*`. CSRF tokens on all state-changing cookie-authenticated routes.
- **Document ingestion hardening**: symlink rejection, path boundary checks, MIME/magic-byte validation (`magic.MagicException` handled), PDF extraction in a sandboxed subprocess with timeout and page limit. Staging file cleaned up on any validation or write failure.
- **Rate limiting**: token-bucket per real IP (30 req/60s general, 10 req/60s on POST `/login`), capped at 10,000 tracked IPs to bound memory under rotating/spoofed source addresses.

Full security model with all controls and phase assignments: [`notes/research-assistant-plan.md § Security`](notes/research-assistant-plan.md).

## Roadmap

- **Phase 1 — Foundation:** planner + researcher + RAG tool wired as a sequential pipeline, basic CLI ✓
- **Phase 2 — Parallelism + memory:** concurrent agent dispatch, resource governor, two-instance Ollama split, benchmarking (wall-clock, RSS, tokens/sec, cold vs warm) ✓
- **Phase 3 — Critic + quality loop:** GraphRAG tool, critic agent, one re-plan cycle on failure, inline citations, confidence scoring ✓
- **Phase 4 — Interface + ingestion:** web UI, session auth (bcrypt + CSRF), bearer-token API, document upload, query history, Docker Compose ✓

Success criteria, risks/mitigations, and stretch goals are detailed in [`notes/research-assistant-plan.md`](notes/research-assistant-plan.md).
