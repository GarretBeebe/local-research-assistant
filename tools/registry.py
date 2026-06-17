from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from tools.base import Tool

# Compile-time allowlist — any module+class pair not listed here is a hard startup error.
# A prefix check alone (e.g. "starts with tools.") is insufficient; it would permit loading
# any code in that namespace.
TOOL_ALLOWLIST: dict[str, list[str]] = {
    "tools.vector_rag": ["VectorRAGTool"],
}

_DEFAULT_TOOLS_YAML = Path(__file__).parent.parent / "tools.yaml"


class ToolRegistry:
    def __init__(self, config_path: str | Path = _DEFAULT_TOOLS_YAML) -> None:
        self._tools: dict[str, Tool] = {}
        self._load(Path(config_path))

    def _load(self, path: Path) -> None:
        with open(path) as f:
            config = yaml.safe_load(f)

        seen_pairs: set[str] = set()
        for entry in config.get("tools") or []:
            module_path: str = entry.get("module", "")
            class_name: str = entry.get("class", "")

            if module_path.startswith("."):
                raise SystemExit(
                    f"Relative import rejected in tools.yaml: {module_path!r}"
                )

            pair_key = f"{module_path}.{class_name}"
            if pair_key in seen_pairs:
                raise SystemExit(f"Duplicate tool entry in tools.yaml: {pair_key!r}")
            seen_pairs.add(pair_key)

            allowed_classes = TOOL_ALLOWLIST.get(module_path)
            if allowed_classes is None or class_name not in allowed_classes:
                raise SystemExit(
                    f"Tool not in allowlist: module={module_path!r} class={class_name!r}"
                )

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            tool: Tool = cls()

            if tool.name in self._tools:
                raise SystemExit(f"Duplicate tool name: {tool.name!r}")

            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool not found: {name!r}")
        return tool

    def schemas(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]
