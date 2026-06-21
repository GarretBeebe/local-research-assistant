from __future__ import annotations

import json

import jsonschema
import ollama

import config
from models import ResearchResult

_SYNTHESIS_SCHEMA = {
    "type": "object",
    "required": ["answer", "confidence"],
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
}

_client = ollama.Client(host=config.OLLAMA_PLANNER_URL)


class SynthesizerError(Exception):
    pass


def synthesize(question: str, results: list[ResearchResult]) -> tuple[str, int]:
    all_sources = list(dict.fromkeys(s for r in results for s in r.sources))
    source_index = {s: i + 1 for i, s in enumerate(all_sources)}

    findings_parts = []
    for r in results:
        refs = " ".join(f"[{source_index[s]}]" for s in r.sources)
        src_line = f"Sources: {refs}" if refs else "Sources: none"
        findings_parts.append(
            f"Sub-question: {r.sub_question}\nFinding: {r.finding}\n{src_line}"
        )
    findings_text = "\n\n".join(findings_parts)

    sources_text = (
        "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(all_sources)) or "None"
    )

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a synthesis agent. Combine the research findings below into a single "
                "coherent answer. Use inline citation markers like [1] where a claim comes from "
                "a specific source. End the answer text with a 'References:' section listing "
                "[n] source_path for each cited source. "
                "Set confidence to 1-5: 5 = fully covered with direct evidence, "
                "1 = speculation or major gaps. "
                'Respond ONLY with valid JSON: {"answer": "...", "confidence": 4}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question: {question}\n\n"
                f"Research findings:\n{findings_text}\n\n"
                f"Numbered source list:\n{sources_text}"
            ),
        },
    ]

    last_error: SynthesizerError | None = None
    for attempt in range(2):
        try:
            response = _client.chat(
                model=config.SYNTHESIZER_MODEL,
                messages=messages,
                format="json",
            )
        except Exception as e:
            raise SynthesizerError(f"Ollama call failed: {e}") from e
        raw = response.message.content
        try:
            data = json.loads(raw)
            jsonschema.validate(data, _SYNTHESIS_SCHEMA)
            return data["answer"], data["confidence"]
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            last_error = SynthesizerError(str(e))
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your output was invalid: {e}\n\n"
                        "Please correct it and respond with valid JSON only."
                    ),
                })

    raise SynthesizerError(f"Synthesizer failed after 2 attempts: {last_error}")
