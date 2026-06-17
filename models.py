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
