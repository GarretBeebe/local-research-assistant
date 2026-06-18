from __future__ import annotations

import asyncio
import dataclasses
import sys
import time

import psutil

import config
from agents.planner import plan
from agents.researcher import research
from agents.synthesizer import synthesize
from governor import ResourceGovernor
from logging_utils import log_benchmark, log_stage
from models import BenchmarkResult, ResearchResult, Task
from tools.registry import ToolRegistry

_PROC = psutil.Process()


def _rss_mb() -> float:
    return _PROC.memory_info().rss >> 20


async def run_pipeline(question: str) -> str:
    registry = ToolRegistry()
    governor = ResourceGovernor(
        max_concurrent=config.MAX_CONCURRENT_RESEARCHERS,
        threshold_mb=config.MEMORY_PRESSURE_THRESHOLD_MB,
    )

    t0 = time.monotonic()
    rss0 = _rss_mb()
    swap0 = psutil.swap_memory().sin

    # Plan (sync — single call, no benefit from async)
    t_plan = time.monotonic()
    tasks = plan(question, registry.schemas())
    planner_sec = time.monotonic() - t_plan
    log_stage("plan", {"question": question}, {"tasks": [dataclasses.asdict(t) for t in tasks]})

    # Parallel research
    rss_samples: list[float] = []

    async def dispatch(task: Task) -> ResearchResult:
        async with governor.slot():
            result = await asyncio.to_thread(research, task, registry)
        if not result.finding.strip():
            print(f"Warning: empty finding for task {task.id!r}, retrying", file=sys.stderr)
            first = result
            async with governor.slot():
                result = await asyncio.to_thread(research, task, registry)
            result = dataclasses.replace(
                result,
                prompt_tokens=result.prompt_tokens + first.prompt_tokens,
                completion_tokens=result.completion_tokens + first.completion_tokens,
                wall_clock_sec=result.wall_clock_sec + first.wall_clock_sec,
            )
        rss_samples.append(_rss_mb())
        return result

    # futures pre-initialized so the results line below can reference it after except*
    futures: list = []
    try:
        async with asyncio.TaskGroup() as tg:
            futures = [tg.create_task(dispatch(task)) for task in tasks]
    except* Exception as eg:
        for err in eg.exceptions:
            print(f"Warning: researcher task failed: {err}", file=sys.stderr)

    results = [f.result() for f in futures if not f.cancelled() and f.exception() is None]
    for result in results:
        log_stage(
            "research",
            {"task_id": result.task_id, "sub_question": result.sub_question},
            {"finding": result.finding, "sources": result.sources},
        )

    # Synthesize (sync — single call, depends on all results)
    t_synth = time.monotonic()
    answer = synthesize(question, results)
    synth_sec = time.monotonic() - t_synth
    log_stage(
        "synthesize",
        {"question": question, "result_count": len(results)},
        {"answer": answer},
    )

    total_sec = time.monotonic() - t0
    rss_samples.append(_rss_mb())
    peak_rss = max(rss_samples, default=rss0)

    bench = BenchmarkResult(
        query=question,
        total_wall_clock_sec=total_sec,
        planner_wall_clock_sec=planner_sec,
        synthesizer_wall_clock_sec=synth_sec,
        researcher_wall_clock_sec=[r.wall_clock_sec for r in results],
        peak_rss_mb=peak_rss,
        swap_faults_in_start=swap0,
        swap_faults_in_end=psutil.swap_memory().sin,
        concurrent_researchers=config.MAX_CONCURRENT_RESEARCHERS,
        cold_loads=[r.task_id for r in results if r.load_duration_ns > 1_000_000_000],
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
    )
    log_benchmark(bench)

    return answer
