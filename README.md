# Local Research Assistant

A locally-hosted multi-agent research assistant. Submit a research query, and a planner model decomposes it into sub-tasks, dispatches specialist agents in parallel, and synthesizes a final answer with citations — all without any cloud API calls.

Builds on existing vector RAG, Graph RAG, and coding assistant work, exposing them as tools in the agent's tool layer.

## Status

Early planning stage — no code yet. See [`notes/research-assistant-plan.md`](notes/research-assistant-plan.md) for the full project plan, architecture diagram, and phased roadmap.

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
Planner (qwen2.5:14b) — decomposes query, assigns agents + tools
    │
    ├──► Researcher (llama3.1:8b)  ─┐
    └──► Summarizer (qwen2.5:7b)   ─┤  run in parallel, share a tool layer:
                                     │  Vector RAG · Graph RAG · file/web tools
    ┌────────────────────────────────┘
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

Designed to run on a GMKtec K16 (Ryzen 7 7735HS, 32GB LPDDR5) with iGPU offloading via Ollama, using two Ollama instances for parallel agent execution.

| Role | Model |
|---|---|
| Planner / orchestrator | `qwen2.5:14b` |
| Researcher agent | `llama3.1:8b` |
| Summarizer agent | `qwen2.5:7b` |
| Critic / checker | `qwen2.5:3b` |
| Embeddings | `nomic-embed-text` |

## Tech stack

| Component | Library |
|---|---|
| LLM calls | `ollama` Python SDK |
| Parallelism | `asyncio` + `httpx` |
| Vector store | ChromaDB |
| Graph store | Neo4j or NetworkX |
| Web framework | FastAPI |
| UI | Gradio or plain HTML |
| Persistence | SQLite |

## Roadmap

- **Phase 1 — Foundation:** planner + researcher + RAG tool wired as a sequential pipeline, basic CLI
- **Phase 2 — Parallelism + memory:** concurrent agent dispatch, shared ChromaDB context store, benchmarking
- **Phase 3 — Critic + quality loop:** self-correction pass, citation linking, confidence scoring
- **Phase 4 — Interface + ingestion:** web UI, drag-and-drop document ingestion, query history

Success criteria, risks/mitigations, and stretch goals are detailed in [`notes/research-assistant-plan.md`](notes/research-assistant-plan.md).
