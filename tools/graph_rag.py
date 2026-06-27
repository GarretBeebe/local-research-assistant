from __future__ import annotations

import json
import logging

import requests

import config

_logger = logging.getLogger(__name__)


class GraphRAGTool:
    name = "GraphRAGTool"
    description = (
        "Queries the knowledge graph for entity relationships and thematic summaries. "
        "Use this for questions about connections between concepts, entities, or themes."
    )
    parameters = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
    }

    def run(self, query: str) -> str:
        headers: dict[str, str] = {}
        if config.GRAPH_RAG_API_KEY:
            headers["Authorization"] = f"Bearer {config.GRAPH_RAG_API_KEY}"

        try:
            resp = requests.post(
                f"{config.GRAPH_RAG_BASE_URL}/v1/chat/completions",
                json={
                    # Required by schema; graph-rag ignores it and uses its own GEN_MODEL.
                    "model": config.GRAPH_RAG_MODEL,
                    "messages": [{"role": "user", "content": query}],
                    "stream": False,
                    "graph_mode": "auto",
                },
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            try:
                return resp.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                _logger.warning("GraphRAGTool unexpected response: %s", e)
                return ""
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                raise  # 4xx = auth/config problem, do not suppress
            _logger.warning("GraphRAGTool HTTP error: %s", e)
            return ""
        except requests.exceptions.RequestException as e:
            _logger.warning("GraphRAGTool failed: %s", e)
            return ""
