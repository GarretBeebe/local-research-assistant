from __future__ import annotations

import sys

import requests

import config

_session = requests.Session()


class VectorRAGTool:
    name = "VectorRAGTool"
    description = (
        "Searches the local document corpus using vector similarity. "
        "Use this to retrieve relevant passages from indexed documents."
    )
    parameters = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
    }

    def run(self, query: str) -> str:
        try:
            resp = _session.post(
                f"{config.RAG_BASE_URL}/v1/retrieve",
                json={"query": query, "limit": 4},
                headers={"Authorization": f"Bearer {config.RAG_INTERNAL_TOKEN}"},
                timeout=30,
            )
            resp.raise_for_status()
            chunks = resp.json().get("chunks", [])
            if not isinstance(chunks, list):
                raise ValueError(
                    f"Unexpected response type for 'chunks': {type(chunks).__name__}"
                )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                raise  # 4xx = auth/config problem, do not suppress
            print(f"Warning: VectorRAGTool HTTP error: {e}", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"Warning: VectorRAGTool failed: {e}", file=sys.stderr)
            return ""

        if not chunks:
            return ""

        parts = []
        for chunk in chunks:
            filepath = chunk.get("filepath", "unknown")
            text = chunk.get("text", "")
            parts.append(f"[Source: {filepath}]\n{text}")

        return "\n\n".join(parts)
