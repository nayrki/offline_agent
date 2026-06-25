"""Test doubles for the agent loop: fake ACP connection, backend, and MCP."""

from __future__ import annotations

from typing import AsyncIterator

from offline_agent.backends.base import (
    Capabilities,
    ModelBackend,
    OutputConstraint,
    SamplingParams,
    StreamDelta,
)
from offline_agent.mcp.client import MCPToolError


class FakeConn:
    """Records ACP session/update calls and services fs callbacks."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.updates: list[tuple[str, object]] = []
        self.writes: dict[str, str] = {}
        self._files = files or {}

    async def session_update(self, session_id: str, update) -> None:
        self.updates.append((session_id, update))

    async def read_text_file(self, path, session_id, line=None, limit=None):
        class _R:
            content = self._files.get(path, "")

        return _R()

    async def write_text_file(self, content, path, session_id):
        self.writes[path] = content
        return None


class StubBackend(ModelBackend):
    """Yields pre-scripted StreamDelta sequences, one list per turn."""

    def __init__(self, turns: list[list[StreamDelta]]) -> None:
        self._turns = list(turns)
        self.calls: list[dict] = []
        self.prepared = False

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(True, True, True, False, 8192)

    @property
    def chat_format(self):
        from offline_agent.chat_formats import get_chat_format

        return get_chat_format("default")

    async def prepare_session(self, system_prompt: str, tools: list[dict]) -> None:
        self.prepared = True
        self.system_prompt = system_prompt
        self.tools = tools

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        constraint: OutputConstraint,
        sampling: SamplingParams,
    ) -> AsyncIterator[StreamDelta]:
        # Snapshot the request for prefix-stability assertions.
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        script = self._turns.pop(0) if self._turns else [StreamDelta(finished=True, stop_reason="end_turn")]
        for delta in script:
            yield delta

    async def aclose(self) -> None:
        pass


class StubMCP:
    def __init__(
        self,
        tools: list[dict],
        results: dict[str, str] | None = None,
        errors: dict[str, str] | None = None,
    ) -> None:
        self._tools = tools
        self._results = results or {}
        # tool name -> error text; raises like the real client does on isError.
        self._errors = errors or {}
        self.calls: list[tuple[str, dict]] = []

    def openai_tools(self) -> list[dict]:
        return list(self._tools)

    def has_tool(self, name: str) -> bool:
        return any(t["function"]["name"] == name for t in self._tools)

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        if name in self._errors:
            raise MCPToolError(self._errors[name])
        return self._results.get(name, f"ran {name}")

    async def aclose(self) -> None:
        pass


def mcp_tool(name: str, props: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": props or {}},
        },
    }
