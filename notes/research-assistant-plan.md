# Personal research assistant — project plan

## Overview

A locally-hosted multi-agent research assistant running entirely on a GMKtec K16 (Ryzen 7 7735HS, 32GB LPDDR5). The user submits a research query; a planner model decomposes it into sub-tasks, dispatches specialist agents in parallel, and synthesizes a final answer with citations — all without any cloud API calls.

Builds directly on existing work: the vector RAG, Graph RAG, and coding assistant become callable tools in the agent tool layer.

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
| Researcher agent | `llama3.1:8b` | Good at synthesis, runs concurrently with other agents |
| Summarizer agent | `qwen2.5:7b` | Lightweight, fast, sufficient for summarization |
| Critic / checker | `qwen2.5:3b` | Minimal task (is this correct?), keep it cheap |
| Embeddings | `nomic-embed-text` | Stay resident in RAM throughout, near-instant retrieval |

**Hardware notes:**
- Enable iGPU offloading in Ollama (`OLLAMA_GPU_LAYERS`) to leverage the Radeon 680M — expect ~30–50% latency improvement on the 14B model
- Run two Ollama instances on different ports to allow true parallel agent execution; both read from the same model directory (`%USERPROFILE%\.ollama\models`) — models are downloaded once, not twice
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

32 GB is sufficient for a personal document corpus, but leaves limited headroom. If total usage exceeds ~28 GB, Windows will begin paging and query latency will degrade significantly. Monitor with `ollama ps` + Task Manager during initial runs.

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
     ┌─────┴──────┐
     │  parallel  │
     ▼            ▼
┌──────────┐  ┌──────────┐
│Researcher│  │Summarizer│   ← run concurrently; pay latency of slowest, not sum
│llama3.1  │  │qwen2.5:7b│
└────┬─────┘  └────┬─────┘
     │              │
     ▼              ▼
┌─────────────────────────────────┐
│         Tool layer              │
│  - Vector RAG                   │  ← existing work, promoted to callable tools
│  - Graph RAG                    │
│  - Coding assistant             │
│  - File reader / web fetch      │
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

**Shared memory store:** `nomic-embed-text` + ChromaDB (or Qdrant). Agents read and write to a shared context store so the summarizer can see what the researcher found without manual passing.

---

## Phase 1 — foundation (week 1–2)

**Goal:** planner + researcher + RAG tool, end-to-end, no parallelism yet.

- [ ] Set up two Ollama instances (ports 11434 and 11435) with iGPU offloading configured
- [ ] Wrap existing vector RAG as a callable Python function with a JSON schema descriptor
- [ ] Wrap existing Graph RAG as a callable Python function with a JSON schema descriptor
- [ ] Implement the planner loop: prompt `qwen2.5:14b` with the user query + tool schemas, parse the structured task list it returns
- [ ] Implement the researcher agent: `llama3.1:8b` with access to both RAG tools
- [ ] Wire planner → researcher → synthesizer as a sequential pipeline (no parallelism yet)
- [ ] Basic CLI interface: `python research.py "your query here"`
- [ ] Log each agent's input/output to a JSON file for debugging

**Success criterion:** A query like *"What are the main differences between RAG and Graph RAG?"* (answerable from your own indexed docs) returns a coherent cited answer end-to-end.

---

## Phase 2 — parallelism + memory (week 3–4)

**Goal:** parallel agent execution, shared memory store, measurable latency improvement.

- [ ] Implement parallel agent dispatch using `asyncio` + two Ollama instances
- [ ] Set up ChromaDB as the shared memory/context store
- [ ] Have agents write their intermediate results to the store with metadata (agent name, timestamp, source docs)
- [ ] Synthesizer reads from the store rather than receiving results directly
- [ ] Add the summarizer agent as a second parallel worker alongside the researcher
- [ ] Benchmark: measure wall-clock time for 5 test queries before and after parallelism
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

**Success criterion:** A non-technical user could add a document and query it without touching the terminal.

---

## Tech stack

| Component | Library | Notes |
|---|---|---|
| LLM calls | `ollama` Python SDK | Native streaming support |
| Parallelism | `asyncio` + `httpx` | Two Ollama instances, async calls |
| Vector store | ChromaDB | Lightweight, file-based, no separate server needed |
| Graph store | NetworkX | In-memory, no separate server; sufficient for personal corpus scale. Neo4j only if graph exceeds available RAM |
| Embeddings | `nomic-embed-text` via Ollama | Already in your stack |
| Web framework | FastAPI | Minimal, fast, good async support |
| UI | Gradio or plain HTML | Gradio for speed, plain HTML for control |
| Persistence | SQLite | Query history, document metadata |

---

## Key risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Planner outputs malformed JSON | Medium | Strict output schema in system prompt + validation layer with retry |
| 14B model too slow for interactive use | Medium | iGPU offloading + single planner call design; set expectation of 60–90s |
| Agents hallucinate citations | Medium | Critic pass + force agents to quote chunk IDs, not generate citations freehand |
| RAM pressure when all models loaded | Medium | Total budget is ~22–29 GB against a 32 GB pool shared with the iGPU; keep critic off the always-loaded set, monitor with `ollama ps` + Task Manager, evict summarizer if needed |
| Graph RAG and vector RAG return conflicting results | Low | Synthesizer prompt explicitly handles contradiction; critic flags it |

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
│   ├── researcher.py       # RAG-augmented research agent
│   ├── summarizer.py       # lightweight summarization
│   └── critic.py           # output quality checker
├── tools/
│   ├── vector_rag.py       # wraps your existing vector RAG
│   ├── graph_rag.py        # wraps your existing Graph RAG
│   └── file_reader.py      # PDF/markdown ingestion
├── memory/
│   └── store.py            # ChromaDB shared context store
├── orchestrator.py         # main pipeline: plan → dispatch → synthesize
├── api.py                  # FastAPI endpoints
├── ui/
│   └── index.html          # minimal frontend
├── config.py               # model names, Ollama URLs, paths
├── benchmark.py            # latency testing
└── research.py             # CLI entrypoint
```

---

*Generated: June 2026*
