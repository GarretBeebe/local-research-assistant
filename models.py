from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    sub_question: str
    tool: str
    model: str


@dataclass
class ResearchResult:
    task_id: str
    sub_question: str
    tool_used: str
    finding: str
    sources: list[str] = field(default_factory=list)
    # Phase 2 benchmarking — populated by researcher, defaults keep Phase 1 callers valid
    prompt_tokens: int = 0
    completion_tokens: int = 0
    eval_duration_ns: int = 0   # nanoseconds; from ollama response
    load_duration_ns: int = 0   # nanoseconds; > 1e9 signals a cold model load
    wall_clock_sec: float = 0.0


@dataclass
class BenchmarkResult:
    query: str
    total_wall_clock_sec: float
    planner_wall_clock_sec: float
    synthesizer_wall_clock_sec: float
    researcher_wall_clock_sec: list[float]  # one entry per task, in submission order
    peak_rss_mb: float
    swap_faults_in_start: int
    swap_faults_in_end: int
    concurrent_researchers: int             # MAX_CONCURRENT_RESEARCHERS at run time
    cold_loads: list[str]                   # task IDs where load_duration_ns > 1s
    total_prompt_tokens: int
    total_completion_tokens: int
