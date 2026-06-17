from typing import Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict  # JSON schema for this tool's run() arguments

    def run(self, query: str) -> str: ...
