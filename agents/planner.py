from __future__ import annotations

import json

import jsonschema
import ollama

import config
from models import Task

_TASK_LIST_SCHEMA = {
    "type": "object",
    "required": ["tasks"],
    "additionalProperties": False,
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": ["id", "sub_question", "tool", "model"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "sub_question": {"type": "string", "minLength": 1},
                    "tool": {"type": "string"},
                    "model": {"type": "string"},
                },
            },
        }
    },
}

_client = ollama.Client(host=config.OLLAMA_URL)


class PlannerError(Exception):
    pass


def _system_prompt(tool_schemas: list[dict]) -> str:
    tools_text = "\n".join(
        f"- {t['name']}: {t['description']}" for t in tool_schemas
    )
    allowed = ", ".join(sorted(config.CHAT_MODELS))
    return (
        "You are a research planning agent. Break the user's research question into "
        "independent sub-tasks that can be investigated separately.\n\n"
        f"Available tools:\n{tools_text}\n\n"
        f"Allowed models: {allowed}\n\n"
        'Respond ONLY with a JSON object in this exact format:\n'
        '{"tasks": [{"id": "1", "sub_question": "...", '
        '"tool": "VectorRAGTool", "model": "llama3.1:8b"}]}\n\n'
        "Rules:\n"
        "- Each task must be fully independent (no task depends on another)\n"
        "- Use only tools and models from the lists above\n"
        "- Generate 1-5 focused tasks; prefer fewer, sharper questions"
    )


def _parse_and_validate(raw: str, tool_names: set[str]) -> list[Task]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PlannerError(f"Invalid JSON from planner: {e}") from e

    try:
        jsonschema.validate(data, _TASK_LIST_SCHEMA)
    except jsonschema.ValidationError as e:
        raise PlannerError(f"Schema validation failed: {e.message}") from e

    seen_ids: set[str] = set()
    tasks = []
    for item in data["tasks"]:
        if item["id"] in seen_ids:
            raise PlannerError(f"Duplicate task ID in planner output: {item['id']!r}")
        seen_ids.add(item["id"])

        if item["tool"] not in tool_names:
            raise PlannerError(f"Unknown tool in planner output: {item['tool']!r}")
        if item["model"] not in config.CHAT_MODELS:
            raise PlannerError(f"Model not in CHAT_MODELS: {item['model']!r}")

        tasks.append(Task(
            id=item["id"],
            sub_question=item["sub_question"],
            tool=item["tool"],
            model=item["model"],
        ))
    return tasks


def plan(question: str, tool_schemas: list[dict]) -> list[Task]:
    tool_names = {t["name"] for t in tool_schemas}
    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(tool_schemas)},
        {"role": "user", "content": question},
    ]

    last_error: PlannerError | None = None
    for attempt in range(2):
        response = _client.chat(
            model=config.PLANNER_MODEL,
            messages=messages,
            format="json",
        )
        raw = response.message.content
        try:
            return _parse_and_validate(raw, tool_names)
        except PlannerError as e:
            last_error = e
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your output was invalid: {e}\n\n"
                        "Please correct it and respond with valid JSON only."
                    ),
                })

    raise PlannerError(f"Planner failed after 2 attempts: {last_error}")
