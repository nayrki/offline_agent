"""The ``acp.Agent`` implementation: ACP lifecycle wired to the agent loop."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PROTOCOL_VERSION,
    PromptResponse,
)
from acp import schema

from ..backends import make_backend
from ..config import Config, load_config
from .loop import run_agent_loop
from .session import Session, build_session

log = logging.getLogger(__name__)


class OfflineAgent(Agent):
    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()
        self._conn = None
        self._client_caps: Any = None
        self._sessions: dict[str, Session] = {}

    # -- ACP lifecycle ----------------------------------------------------

    def on_connect(self, conn) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities=None,
        client_info=None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self._client_caps = client_capabilities
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            # Must advertise HTTP MCP support, or the Jupyter AI client filters
            # out HTTP MCP servers (e.g. the notebook-tools server at :3001) from
            # `.jupyter/mcp_settings.json` and never forwards them to new_session
            # -- leaving the agent with no notebook tools at all.
            agent_capabilities=schema.AgentCapabilities(
                mcp_capabilities=schema.McpCapabilities(http=True)
            ),
            auth_methods=[],
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories=None,
        mcp_servers=None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = uuid.uuid4().hex
        backend = await make_backend(self._config)
        session = await build_session(
            session_id=session_id,
            cwd=cwd,
            mcp_servers=mcp_servers or [],
            backend=backend,
            config=self._config,
            client_caps=self._client_caps,
        )
        self._sessions[session_id] = session
        log.info("new session %s (cwd=%s, %d tools)", session_id, cwd, len(session.tools))
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        prompt: list,
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id!r}")
        return await run_agent_loop(
            self._conn, session, prompt, message_id, self._config
        )

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.cancel.set()

    async def close_session(self, session_id: str, **kwargs: Any):
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.aclose()
        return None
