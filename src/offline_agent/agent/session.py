"""Per-session state and its construction (the prompt-cache freeze point)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import ModelBackend
from ..chat_formats import ChatFormat
from ..config import Config
from ..mcp.client import MCPClients
from ..tools.fs import fs_tools
from .project_prompt import load_project_instructions
from .prompts import build_system_prompt


@dataclass
class Session:
    session_id: str
    cwd: str
    backend: ModelBackend
    mcp: MCPClients
    tools: list[dict]               # FROZEN once here; never re-pulled in-session
    system_prompt: str
    chat_format: ChatFormat         # per-model tool envelope (backend.chat_handler)
    messages: list[dict] = field(default_factory=list)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)

    async def aclose(self) -> None:
        await self.mcp.aclose()
        await self.backend.aclose()


def _fs_caps(client_caps: Any, config: Config) -> tuple[bool, bool]:
    fs = getattr(client_caps, "fs", None) if client_caps is not None else None
    can_read = bool(getattr(fs, "read_text_file", False))
    can_write = bool(getattr(fs, "write_text_file", False)) and config.fs.allow_write
    return can_read, can_write


async def build_session(
    *,
    session_id: str,
    cwd: str,
    mcp_servers: list[Any],
    backend: ModelBackend,
    config: Config,
    client_caps: Any,
) -> Session:
    # The active backend already resolved its own model's format (auto-detected
    # or explicit); use it as the single source of truth so the local fallback
    # and remote model never share one mismatched handler.
    chat_format = backend.chat_format

    mcp = await MCPClients.connect(mcp_servers)

    can_read, can_write = _fs_caps(client_caps, config)
    # Freeze the tool list ONCE: MCP (notebook) tools + ACP-serviced fs tools.
    tools = mcp.openai_tools() + fs_tools(can_read=can_read, can_write=can_write)

    # If the active notebook's directory has a notebook_agent.md, it replaces the
    # default system prompt (frozen here, once, to keep the cached prefix stable).
    project_instructions = await load_project_instructions(mcp)
    system_prompt = build_system_prompt(cwd, project_instructions)
    await backend.prepare_session(system_prompt, tools)

    return Session(
        session_id=session_id,
        cwd=cwd,
        backend=backend,
        mcp=mcp,
        tools=tools,
        system_prompt=system_prompt,
        chat_format=chat_format,
        messages=[{"role": "system", "content": system_prompt}],
    )
