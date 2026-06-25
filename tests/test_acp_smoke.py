"""End-to-end smoke tests for the ACP agent.

``test_lifecycle_in_process`` drives OfflineAgent's methods directly with a fake
connection. ``test_stdio_roundtrip`` spawns the real ``offline-agent`` process and
talks to it over stdio with the SDK's client, using the guarded echo backend so no
model is needed.
"""

from __future__ import annotations

import os
import sys

import pytest

import acp
from acp import PROTOCOL_VERSION, schema, text_block

from offline_agent.agent.acp_agent import OfflineAgent
from offline_agent.config import Config

from _stubs import FakeConn


def _agent_message_texts(updates):
    out = []
    for _sid, update in updates:
        content = getattr(update, "content", None)
        text = getattr(content, "text", None)
        if text is not None:
            out.append(text)
    return out


@pytest.mark.asyncio
async def test_lifecycle_in_process(monkeypatch):
    monkeypatch.setenv("OFFLINE_AGENT_STUB_BACKEND", "1")
    agent = OfflineAgent(Config())
    conn = FakeConn()
    agent.on_connect(conn)

    init = await agent.initialize(PROTOCOL_VERSION, client_capabilities=schema.ClientCapabilities())
    assert init.protocol_version == PROTOCOL_VERSION

    sess = await agent.new_session(cwd="/work", mcp_servers=[])
    assert sess.session_id

    resp = await agent.prompt([text_block("hello")], sess.session_id, message_id="m1")
    assert resp.stop_reason == "end_turn"
    assert "echo: hello" in _agent_message_texts(conn.updates)

    await agent.close_session(sess.session_id)


class RecordingClient(acp.Client):
    def __init__(self) -> None:
        self.updates: list = []

    def on_connect(self, conn) -> None:  # noqa: D401
        self._conn = conn

    async def session_update(self, session_id, update, **kwargs) -> None:
        self.updates.append(update)

    async def request_permission(self, *a, **k):
        raise NotImplementedError

    async def read_text_file(self, *a, **k):
        raise NotImplementedError

    async def write_text_file(self, *a, **k):
        raise NotImplementedError

    async def create_terminal(self, *a, **k):
        raise NotImplementedError

    async def terminal_output(self, *a, **k):
        raise NotImplementedError

    async def release_terminal(self, *a, **k):
        raise NotImplementedError

    async def wait_for_terminal_exit(self, *a, **k):
        raise NotImplementedError

    async def kill_terminal(self, *a, **k):
        raise NotImplementedError

    async def ext_method(self, *a, **k):
        raise NotImplementedError

    async def ext_notification(self, *a, **k):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_stdio_roundtrip():
    client = RecordingClient()
    env = dict(os.environ)
    env["OFFLINE_AGENT_STUB_BACKEND"] = "1"

    async with acp.spawn_agent_process(
        client,
        sys.executable,
        "-m",
        "offline_agent.agent.main",
        env=env,
    ) as (conn, proc):
        await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=schema.ClientCapabilities(),
        )
        sess = await conn.new_session(cwd=os.getcwd(), mcp_servers=[])
        await conn.prompt(prompt=[text_block("hello")], session_id=sess.session_id)

    texts = [getattr(getattr(u, "content", None), "text", None) for u in client.updates]
    assert "echo: hello" in texts
