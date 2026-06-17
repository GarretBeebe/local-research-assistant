from __future__ import annotations

import dataclasses

from agents.planner import plan
from agents.researcher import research
from agents.synthesizer import synthesize
from logging_utils import log_stage
from models import ResearchResult
from tools.registry import ToolRegistry


def run_pipeline(question: str) -> str:
    registry = ToolRegistry()
    tool_schemas = registry.schemas()

    tasks = plan(question, tool_schemas)
    log_stage("plan", {"question": question}, {"tasks": [dataclasses.asdict(t) for t in tasks]})

    results: list[ResearchResult] = []
    for task in tasks:
        result = research(task, registry)
        results.append(result)
        log_stage(
            "research",
            {"task": dataclasses.asdict(task)},
            {"finding": result.finding, "sources": result.sources},
        )

    answer = synthesize(question, results)
    log_stage(
        "synthesize",
        {"question": question, "result_count": len(results)},
        {"answer": answer},
    )

    return answer
