from __future__ import annotations

import ollama

import config
from models import ResearchResult

_client = ollama.Client(host=config.OLLAMA_PLANNER_URL)


def synthesize(question: str, results: list[ResearchResult]) -> str:
    findings_text = "\n\n".join(
        f"Sub-question: {r.sub_question}\nFinding: {r.finding}"
        for r in results
    )

    all_sources = list(dict.fromkeys(s for r in results for s in r.sources))

    sources_text = "\n".join(f"- {s}" for s in all_sources) if all_sources else "None"

    response = _client.chat(
        model=config.SYNTHESIZER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a synthesis agent. Combine the research findings below into a single "
                    "coherent answer to the original question. Be clear and concise. "
                    "End your response with a 'Sources:' section listing the source files."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Research findings:\n{findings_text}\n\n"
                    f"Available sources:\n{sources_text}"
                ),
            },
        ],
    )

    return response.message.content
