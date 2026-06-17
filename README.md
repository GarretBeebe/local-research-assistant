# Local Research Assistant

A locally-hosted multi-agent research assistant. Submit a research query, and a planner model decomposes it into sub-tasks, dispatches specialist agents in parallel, and synthesizes a final answer with citations — all without any cloud API calls.

Builds on existing vector RAG, Graph RAG, and coding assistant work, exposing them as tools in the agent's tool layer.

## Status

**Phase 1 complete** — the sequential pipeline is implemented and runnable. See [`notes/research-assistant-plan.md`](notes/research-assistant-plan.md) for the full project plan, architecture diagram, and phased roadmap.

```
python research.py "your query here"
```

## Setup

**Prerequisites:** Ollama running locally with the required models pulled; `rag-system` running and accessible.

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env: set RAG_BASE_URL and RAG_INTERNAL_TOKEN to match your rag-system

# Run a query
uv run python research.py "What is retrieval-augmented generation?"
```

Logs are written to `logs/pipeline.jsonl` (rotating, 10 MB max). Set `DEBUG_LOG_FULL_PAYLOADS=true` in `.env` to log full chunk text.

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

Primary target is a GMKtec K16 (Ryzen 7 7735HS, 32GB LPDDR5) with iGPU offloading via Ollama. Phase 1 uses a single Ollama instance (default port 11434, configured via `OLLAMA_URL`). Phase 2 introduces three dedicated instances — one for embeddings, one for the planner/synthesizer, one for researcher workers — to prevent model-load churn under parallel dispatch. All instances share a single model directory.

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
| Parallelism | `asyncio` + `httpx` |
| Vector store | ChromaDB |
| Graph store | SQLite (edges/nodes) + NetworkX (per-query subgraph) |
| Web framework | FastAPI |
| UI | Plain HTML (Gradio for local dev only — incompatible with strict CSP) |
| Persistence | SQLite |
| Tool interface | `Tool` protocol + `ToolRegistry` (config-driven via `tools.yaml`) |
| Containerization | Docker + docker-compose (app + ChromaDB; Ollama is a native prerequisite) |

## Security

The project follows the same security patterns established in the sibling `local-graph-rag` and `rag-system` projects, extended for the multi-agent architecture:

- **Startup fails closed**: server requires either a configured `API_KEY` or `ALLOW_INSECURE_LOCALONLY=true` (loopback-only). Ambiguous config is a hard startup error.
- **Tool registry allowlist**: `tools.yaml` entries are validated against an explicit compile-time allowlist of known module+class pairs; arbitrary code loading is rejected at startup.
- **Planner output treated as untrusted**: task list is validated against a strict JSON schema (tool names, model names) before any agent is dispatched.
- **Three-way auth separation**: bearer tokens for `/v1/*` API, session cookies for browser UI, no overlap. CSRF tokens on state-changing cookie-authenticated routes.
- **Document ingestion hardening**: symlink rejection, path boundary checks, MIME/magic-byte validation, PDF extraction in a sandboxed subprocess with timeout and page limit.

Full security model with all controls and phase assignments: [`notes/research-assistant-plan.md § Security`](notes/research-assistant-plan.md).

## Roadmap

- **Phase 1 — Foundation:** planner + researcher + RAG tool wired as a sequential pipeline, basic CLI ✓
- **Phase 2 — Parallelism + memory:** concurrent agent dispatch, resource governor, three-instance Ollama split, benchmarking (wall-clock, RSS, tokens/sec, cold vs warm)
- **Phase 3 — Critic + quality loop:** self-correction pass, GraphRAG tool, citation linking, confidence scoring
- **Phase 4 — Interface + ingestion:** web UI, drag-and-drop document ingestion, query history

Success criteria, risks/mitigations, and stretch goals are detailed in [`notes/research-assistant-plan.md`](notes/research-assistant-plan.md).
