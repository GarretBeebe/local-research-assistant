from __future__ import annotations

import json
import time

import jsonschema
import ollama

import config
from models import CriticResult

_CRITIC_SCHEMA = {
    "type": "object",
    "required": ["passed", "issues"],
    "additionalProperties": False,
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

_client = ollama.Client(host=config.OLLAMA_PLANNER_URL)


class CriticError(Exception):
    pass


def critique(question: str, answer: str) -> CriticResult:
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a quality critic. Given a research question and a synthesized answer, "
                "identify factual gaps, unsupported claims, or contradictions. "
                "Set passed=true if the answer adequately addresses the question. "
                "Set passed=false and list specific issues if it does not. "
                "Respond ONLY with valid JSON: "
                '{"passed": true, "issues": []} or '
                '{"passed": false, "issues": ["issue 1", "issue 2"]}'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nAnswer:\n{answer}",
        },
    ]

    t0 = time.monotonic()
    last_error: CriticError | None = None
    for attempt in range(2):
        try:
            response = _client.chat(
                model=config.CRITIC_MODEL,
                messages=messages,
                format="json",
            )
        except Exception as e:
            raise CriticError(f"Ollama call failed: {e}") from e
        raw = response.message.content
        try:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise CriticError(f"Invalid JSON from critic: {e}") from e
            try:
                jsonschema.validate(data, _CRITIC_SCHEMA)
            except jsonschema.ValidationError as e:
                raise CriticError(f"Schema validation failed: {e.message}") from e
            non_empty_issues = [i for i in data["issues"] if i.strip()]
            if data["passed"] and non_empty_issues:
                raise CriticError("Critic returned passed=true with non-empty issues")
            if not data["passed"] and not non_empty_issues:
                raise CriticError("Critic returned passed=false without actionable issues")
            return CriticResult(
                passed=data["passed"],
                issues=non_empty_issues,
                wall_clock_sec=time.monotonic() - t0,
                prompt_tokens=response.prompt_eval_count or 0,
                completion_tokens=response.eval_count or 0,
            )
        except CriticError as e:
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

    raise CriticError(f"Critic failed after 2 attempts: {last_error}")
