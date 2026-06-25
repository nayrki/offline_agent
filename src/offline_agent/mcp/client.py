"""An MCP *client* that connects to the servers declared in ACP ``session/new``.

The primary server is Jupyter's own notebook-tools endpoint (``jupyter-server-mcp``,
streamable HTTP at ``http://localhost:3001/mcp``); stdio servers are handled
generically for anything else a user configures. Tools are aggregated and frozen
once per session, then exposed in OpenAI ``tools=`` format to the backend.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

log = logging.getLogger(__name__)


class MCPToolError(RuntimeError):
    """An MCP tool returned ``isError=True``.

    Raised (rather than returned as a string) so the caller can mark the tool
    call ``failed`` instead of silently reporting a failed call as completed.
    The message is the tool's rendered error text.
    """


def _coerce_headers(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    if isinstance(headers, dict):
        return {str(k): str(v) for k, v in headers.items()}
    # ACP may model headers as a list of {name, value} objects.
    out: dict[str, str] = {}
    for h in headers:
        name = getattr(h, "name", None)
        value = getattr(h, "value", None)
        if name is not None:
            out[str(name)] = str(value)
    return out


def _coerce_env(env: Any) -> dict[str, str]:
    if not env:
        return {}
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    out: dict[str, str] = {}
    for e in env:
        name = getattr(e, "name", None)
        value = getattr(e, "value", None)
        if name is not None:
            out[str(name)] = str(value)
    return out


class MCPClients:
    """Holds one or more MCP sessions open for the lifetime of an ACP session."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._sessions: list[ClientSession] = []
        self._route: dict[str, ClientSession] = {}  # tool name -> owning session
        self._tools: list[dict] = []

    @classmethod
    async def connect(cls, mcp_servers: list[Any]) -> "MCPClients":
        self = cls()
        try:
            for server in mcp_servers or []:
                await self._connect_one(server)
        except Exception:
            await self.aclose()
            raise
        return self

    async def _connect_one(self, server: Any) -> None:
        name = getattr(server, "name", "<unnamed>")
        if getattr(server, "url", None):  # HTTP / SSE
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(
                    url=server.url,
                    headers=_coerce_headers(getattr(server, "headers", None)),
                )
            )
        elif getattr(server, "command", None):  # stdio
            params = StdioServerParameters(
                command=server.command,
                args=list(getattr(server, "args", []) or []),
                env=_coerce_env(getattr(server, "env", None)) or None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        else:
            log.warning("skipping MCP server %r: unrecognized transport", name)
            return

        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions.append(session)
        await self._register_tools(session, name)

    async def _register_tools(self, session: ClientSession, server_name: str) -> None:
        result = await session.list_tools()
        for tool in result.tools:
            if tool.name in self._route:
                log.warning(
                    "duplicate MCP tool %r (from %s) shadows an earlier one; skipping",
                    tool.name,
                    server_name,
                )
                continue
            self._route[tool.name] = session
            self._tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )

    def openai_tools(self) -> list[dict]:
        """Frozen, ordered tool list in OpenAI ``tools=`` format (the cache key)."""
        return list(self._tools)

    def has_tool(self, name: str) -> bool:
        return name in self._route

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        session = self._route.get(name)
        if session is None:
            raise KeyError(f"unknown MCP tool: {name!r}")
        result = await session.call_tool(name, arguments or {})
        text = _render_content(result.content)
        if getattr(result, "isError", False):
            raise MCPToolError(text)
        return text

    async def aclose(self) -> None:
        await self._stack.aclose()
        self._sessions.clear()
        self._route.clear()


def _render_content(content: Any) -> str:
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(str(getattr(block, "data", block)))
    return "\n".join(parts)
