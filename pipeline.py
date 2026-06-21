from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
from typing import NamedTuple

import psutil

import config
from agents.critic import critique
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


class _CycleOut(NamedTuple):
    results: list[ResearchResult]
    answer_text: str
    confidence: int
    planner_sec: float
    synth_sec: float
    peak_rss_mb: float


async def _run_cycle(
    question: str,
    registry: ToolRegistry,
    governor: ResourceGovernor,
    issues: list[str] | None = None,
) -> _CycleOut:
    plan_question = question
    if issues:
        issues_text = "\n".join(f"- {i}" for i in issues)
        plan_question = (
            f"{question}\n\n"
            f"A previous answer had the following issues — address them:\n{issues_text}"
        )

    if len(plan_question) > config.MAX_QUERY_LENGTH:
        print(
            f"Warning: plan_question truncated from {len(plan_question)} "
            f"to {config.MAX_QUERY_LENGTH} chars",
            file=sys.stderr,
        )
        plan_question = plan_question[:config.MAX_QUERY_LENGTH]

    t_plan = time.monotonic()
    tasks = plan(plan_question, registry.schemas())
    planner_sec = time.monotonic() - t_plan
    log_stage(
        "plan",
        {"question": plan_question},
        {"tasks": [dataclasses.asdict(t) for t in tasks]},
    )

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
                load_duration_ns=result.load_duration_ns + first.load_duration_ns,
                eval_duration_ns=result.eval_duration_ns + first.eval_duration_ns,
            )
        rss_samples.append(_rss_mb())
        return result

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

    t_synth = time.monotonic()
    answer_text, confidence = synthesize(plan_question, results)
    synth_sec = time.monotonic() - t_synth
    log_stage(
        "synthesize",
        {"question": plan_question, "result_count": len(results)},
        {"answer": answer_text, "confidence": confidence},
    )

    rss_samples.append(_rss_mb())
    return _CycleOut(
        results=results,
        answer_text=answer_text,
        confidence=confidence,
        planner_sec=planner_sec,
        synth_sec=synth_sec,
        peak_rss_mb=max(rss_samples),
    )


async def run_pipeline(question: str) -> str:
    registry = ToolRegistry()
    governor = ResourceGovernor(
        max_concurrent=config.MAX_CONCURRENT_RESEARCHERS,
        threshold_mb=config.MEMORY_PRESSURE_THRESHOLD_MB,
    )

    t0 = time.monotonic()
    swap0 = psutil.swap_memory().sin

    c1 = await _run_cycle(question, registry, governor)

    t_critic = time.monotonic()
    critic = await asyncio.to_thread(critique, question, c1.answer_text)
    critic_sec = time.monotonic() - t_critic
    log_stage("critic", {"question": question}, {"passed": critic.passed, "issues": critic.issues})

    # Accumulated benchmark values — updated if a second cycle runs.
    final = c1
    all_results = c1.results
    planner_sec = c1.planner_sec
    synth_sec = c1.synth_sec
    peak_rss = c1.peak_rss_mb

    re_planned = False
    if not critic.passed:
        re_planned = True
        c2 = await _run_cycle(question, registry, governor, issues=critic.issues)
        t_critic2 = time.monotonic()
        critic = await asyncio.to_thread(critique, question, c2.answer_text)
        critic_sec += time.monotonic() - t_critic2
        log_stage(
            "critic",
            {"question": question, "re_plan": True},
            {"passed": critic.passed, "issues": critic.issues},
        )
        final = c2
        all_results = c1.results + c2.results
        planner_sec += c2.planner_sec
        synth_sec += c2.synth_sec
        peak_rss = max(peak_rss, c2.peak_rss_mb)

    bench = BenchmarkResult(
        query=question,
        total_wall_clock_sec=time.monotonic() - t0,
        planner_wall_clock_sec=planner_sec,
        synthesizer_wall_clock_sec=synth_sec,
        researcher_wall_clock_sec=[r.wall_clock_sec for r in final.results],
        peak_rss_mb=peak_rss,
        swap_faults_in_start=swap0,
        swap_faults_in_end=psutil.swap_memory().sin,
        concurrent_researchers=config.MAX_CONCURRENT_RESEARCHERS,
        cold_loads=[r.task_id for r in all_results if r.load_duration_ns > 1_000_000_000],
        total_prompt_tokens=sum(r.prompt_tokens for r in all_results),
        total_completion_tokens=sum(r.completion_tokens for r in all_results),
        critic_wall_clock_sec=critic_sec,
        critic_passed=critic.passed,
        re_planned=re_planned,
        confidence=final.confidence,
    )
    log_benchmark(bench)

    return final.answer_text
