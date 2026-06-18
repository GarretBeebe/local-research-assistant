from __future__ import annotations

import re
import time

import ollama

import config
from models import ResearchResult, Task
from tools.registry import ToolRegistry

_SOURCE_PATTERN = re.compile(r"\[Source: ([^\]]+)\]")

_client = ollama.Client(host=config.OLLAMA_RESEARCHER_URL)


def _extract_sources(context: str) -> list[str]:
    return list(dict.fromkeys(_SOURCE_PATTERN.findall(context)))  # deduplicated, order-preserving


def research(task: Task, registry: ToolRegistry) -> ResearchResult:
    t0 = time.monotonic()

    tool = registry.get(task.tool)
    context = tool.run(query=task.sub_question)

    sources = _extract_sources(context) if context else []
    context_block = context if context else "No relevant documents found."

    response = _client.chat(
        model=task.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research agent. Answer the question using ONLY the provided "
                    "context. Be specific and accurate. If the context does not contain enough "
                    "information to answer, say so explicitly. Do not make up information."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {task.sub_question}\n\nContext:\n{context_block}",
            },
        ],
    )

    return ResearchResult(
        task_id=task.id,
        sub_question=task.sub_question,
        tool_used=task.tool,
        finding=response.message.content,
        sources=sources,
        prompt_tokens=response.prompt_eval_count or 0,
        completion_tokens=response.eval_count or 0,
        eval_duration_ns=response.eval_duration or 0,
        load_duration_ns=response.load_duration or 0,
        wall_clock_sec=time.monotonic() - t0,
    )
