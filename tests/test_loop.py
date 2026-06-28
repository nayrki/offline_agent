"""Unit tests for the backend-agnostic agent loop."""

from __future__ import annotations

import pytest

from offline_agent.agent.loop import run_agent_loop
from offline_agent.agent.session import Session
from offline_agent.backends.base import StreamDelta, ToolCall
from offline_agent.chat_formats import get_chat_format
from offline_agent.config import Config

from _stubs import FakeConn, StubBackend, StubMCP, mcp_tool


def make_session(backend, mcp, tools=None):
    sys_prompt = "SYSTEM"
    return Session(
        session_id="s1",
        cwd="/work",
        backend=backend,
        mcp=mcp,
        tools=tools if tools is not None else mcp.openai_tools(),
        system_prompt=sys_prompt,
        chat_format=get_chat_format("default"),
        messages=[{"role": "system", "content": sys_prompt}],
    )


def text_block_value(update):
    # AgentMessageChunk -> .content (TextContentBlock) -> .text
    content = getattr(update, "content", None)
    return getattr(content, "text", None)


@pytest.mark.asyncio
async def test_native_tool_then_finish():
    tools = [mcp_tool("run_cell", {"index": {"type": "integer"}})]
    mcp = StubMCP(tools, results={"run_cell": "4"})
    backend = StubBackend(
        turns=[
            # Turn 1: call run_cell
            [
                StreamDelta(tool_call=ToolCall(id="t1", name="run_cell", arguments={"index": 1})),
                StreamDelta(finished=True, stop_reason="tool_use"),
            ],
            # Turn 2: final text answer
            [
                StreamDelta(text="The result is 4."),
                StreamDelta(finished=True, stop_reason="end_turn"),
            ],
        ]
    )
    session = make_session(backend, mcp)
    conn = FakeConn()

    resp = await run_agent_loop(conn, session, [{"text": "run cell 1"}], "m1", Config())

    assert resp.stop_reason == "end_turn"
    assert mcp.calls == [("run_cell", {"index": 1})]
    # The tool result was fed back to the model on turn 2.
    assert any(m.get("role") == "tool" and m["content"] == "4" for m in session.messages)
    # Streamed the final answer text.
    assert any(text_block_value(u) == "The result is 4." for _, u in conn.updates)


@pytest.mark.asyncio
async def test_tool_call_turn_records_string_content():
    # A pure tool-call turn must record content="" (a string), never None.
    # The gpt-oss/Harmony GGUF template guards with `if "content" in message`
    # then does `"<|channel|>..." in message.content`; content=None raises
    # "argument of type 'NoneType' is not iterable" on the NEXT turn's render.
    tools = [mcp_tool("run_cell", {"index": {"type": "integer"}})]
    mcp = StubMCP(tools, results={"run_cell": "ok"})
    backend = StubBackend(
        turns=[
            [StreamDelta(tool_call=ToolCall(id="t1", name="run_cell", arguments={"index": 1})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(text="done"), StreamDelta(finished=True, stop_reason="end_turn")],
        ]
    )
    session = make_session(backend, mcp)
    await run_agent_loop(FakeConn(), session, [{"text": "run cell 1"}], "m1", Config())

    assistants = [m for m in session.messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert assistants, "expected an assistant message carrying tool_calls"
    for m in assistants:
        assert m["content"] == ""
        assert m["content"] is not None


@pytest.mark.asyncio
async def test_prefix_stability_across_turns():
    """System+tools prefix must be byte-identical on every model call."""
    tools = [mcp_tool("run_cell", {"index": {"type": "integer"}})]
    mcp = StubMCP(tools)
    backend = StubBackend(
        turns=[
            [StreamDelta(tool_call=ToolCall(id="t1", name="run_cell", arguments={"index": 1})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(text="done"), StreamDelta(finished=True, stop_reason="end_turn")],
        ]
    )
    session = make_session(backend, mcp)
    await run_agent_loop(FakeConn(), session, [{"text": "go"}], "m1", Config())

    assert len(backend.calls) == 2
    # First message (system) identical; tools list identical object content.
    assert backend.calls[0]["messages"][0] == backend.calls[1]["messages"][0]
    assert backend.calls[0]["tools"] == backend.calls[1]["tools"]
    # Second call's messages must be a strict prefix-extension of the first.
    first = backend.calls[0]["messages"]
    second = backend.calls[1]["messages"]
    assert second[: len(first)] == first


@pytest.mark.asyncio
async def test_fs_write_dispatch():
    mcp = StubMCP([])
    write_tool = {
        "type": "function",
        "function": {"name": "write_file", "parameters": {"type": "object", "properties": {}}},
    }
    backend = StubBackend(
        turns=[
            [StreamDelta(tool_call=ToolCall(id="w1", name="write_file",
                                            arguments={"path": "lib.py", "content": "x=1"})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(text="exported"), StreamDelta(finished=True, stop_reason="end_turn")],
        ]
    )
    session = make_session(backend, mcp, tools=[write_tool])
    conn = FakeConn()

    resp = await run_agent_loop(conn, session, [{"text": "export"}], "m1", Config())
    assert resp.stop_reason == "end_turn"
    assert conn.writes == {"lib.py": "x=1"}


def tool_call_status(update):
    # update_tool_call(...) -> object carrying .status for the tool call.
    return getattr(update, "status", None)


@pytest.mark.asyncio
async def test_remote_backend_connection_error_reported_to_user():
    class APIConnectionError(Exception):
        pass

    class ExplodingBackend(StubBackend):
        async def stream(self, messages, tools, constraint, sampling):
            self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
            raise APIConnectionError("Connection error.")
            yield  # pragma: no cover - keeps this an async generator

    mcp = StubMCP([])
    backend = ExplodingBackend(turns=[])
    session = make_session(backend, mcp)
    conn = FakeConn()
    cfg = Config()
    cfg.backend.mode = "remote"
    cfg.remote.base_url = "http://localhost:1234/v1"

    resp = await run_agent_loop(conn, session, [{"text": "hello"}], "m1", cfg)

    assert resp.stop_reason == "end_turn"
    streamed = " ".join(text_block_value(u) or "" for _, u in conn.updates)
    assert "couldn't reach the configured remote model endpoint" in streamed
    assert cfg.remote.base_url in streamed


@pytest.mark.asyncio
async def test_mcp_error_marks_tool_call_failed():
    # An MCP tool returning isError must be reported as a FAILED tool call, not
    # silently completed, and its error text must reach the model.
    tools = [mcp_tool("insert_cell", {"file_path": {"type": "string"}})]
    mcp = StubMCP(tools, errors={"insert_cell": "boom: NoneType >= int"})
    backend = StubBackend(
        turns=[
            [StreamDelta(tool_call=ToolCall(id="t1", name="insert_cell", arguments={"file_path": "n.ipynb"})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(text="giving up"), StreamDelta(finished=True, stop_reason="end_turn")],
        ]
    )
    session = make_session(backend, mcp)
    conn = FakeConn()

    await run_agent_loop(conn, session, [{"text": "insert"}], "m1", Config())

    # Model saw the error text in the tool result.
    assert any(
        m.get("role") == "tool" and "boom: NoneType >= int" in m["content"]
        for m in session.messages
    )
    # The ACP client was told the tool FAILED (status no longer masked as completed).
    assert any(tool_call_status(u) == "failed" for _, u in conn.updates)


@pytest.mark.asyncio
async def test_breaks_on_three_identical_tool_calls():
    # The model wedges itself re-issuing the same failing call; the loop must
    # break after 3 identical consecutive calls and report a meaningful message.
    tools = [mcp_tool("insert_cell", {"file_path": {"type": "string"}})]
    mcp = StubMCP(tools, errors={"insert_cell": "ERR: insert_index is None"})
    same = lambda: [  # noqa: E731 - identical call, re-emitted every turn
        StreamDelta(tool_call=ToolCall(id="t", name="insert_cell", arguments={"file_path": "n.ipynb"})),
        StreamDelta(finished=True, stop_reason="tool_use"),
    ]
    backend = StubBackend(turns=[same(), same(), same(), same(), same()])
    session = make_session(backend, mcp)
    conn = FakeConn()

    resp = await run_agent_loop(conn, session, [{"text": "insert"}], "m1", Config())

    # Stopped via the loop-breaker, after exactly 3 identical calls (not all turns).
    assert resp.stop_reason == "refusal"
    assert len(mcp.calls) == 3
    # The user got a meaningful message carrying the offending call + error packet.
    streamed = " ".join(text_block_value(u) or "" for _, u in conn.updates)
    assert "insert_cell" in streamed
    assert "ERR: insert_index is None" in streamed


@pytest.mark.asyncio
async def test_empty_args_caught_before_dispatch():
    # A degenerate/truncated Harmony call parses to args={}. A tool with a
    # required field must NOT be dispatched with {}; the model gets a clear,
    # tool-agnostic error naming the missing field instead of a backend dump.
    tools = [mcp_tool("insert_cell", {"file_path": {"type": "string"}})]
    tools[0]["function"]["parameters"]["required"] = ["file_path"]
    mcp = StubMCP(tools, results={"insert_cell": "ok"})
    backend = StubBackend(
        turns=[
            [StreamDelta(tool_call=ToolCall(id="i", name="insert_cell", arguments={})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(text="ok"), StreamDelta(finished=True, stop_reason="end_turn")],
        ]
    )
    session = make_session(backend, mcp)
    conn = FakeConn()

    await run_agent_loop(conn, session, [{"text": "insert"}], "m1", Config())

    # The empty call never reached the MCP server.
    assert mcp.calls == []
    # The model got an actionable error naming the missing required argument.
    tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
    assert tool_msgs and "file_path" in tool_msgs[0]["content"]
    assert "required" in tool_msgs[0]["content"]
    # The failure was reported as a failed (not completed) tool call.
    assert any(tool_call_status(u) == "failed" for _, u in conn.updates)


@pytest.mark.asyncio
async def test_silent_giveup_after_failure_reports_error():
    # The model calls a tool that fails, then ends its turn saying nothing (no
    # text, no tool call). The agent must NOT return an empty message; it must
    # surface the failure so it is never swallowed ("did not report its failure").
    tools = [mcp_tool("add_cell", {"file_path": {"type": "string"}})]
    tools[0]["function"]["parameters"]["required"] = ["file_path"]
    mcp = StubMCP(tools, errors={"add_cell": "No such file or directory: 'x.ipynb'"})
    backend = StubBackend(
        turns=[
            [StreamDelta(tool_call=ToolCall(id="a", name="add_cell", arguments={"file_path": "x.ipynb"})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(finished=True, stop_reason="end_turn")],  # silent give-up
        ]
    )
    session = make_session(backend, mcp)
    conn = FakeConn()

    resp = await run_agent_loop(conn, session, [{"text": "add a cell"}], "m1", Config())

    assert resp.stop_reason == "end_turn"
    streamed = " ".join(text_block_value(u) or "" for _, u in conn.updates)
    assert "add_cell" in streamed
    assert "No such file or directory" in streamed


@pytest.mark.asyncio
async def test_empty_turn_from_truncation_reports():
    # An empty turn caused by hitting max_tokens is reported, not returned blank.
    mcp = StubMCP([])
    backend = StubBackend(turns=[[StreamDelta(finished=True, stop_reason="max_tokens")]])
    session = make_session(backend, mcp)
    conn = FakeConn()

    resp = await run_agent_loop(conn, session, [{"text": "hi"}], "m1", Config())

    assert resp.stop_reason == "max_tokens"
    streamed = " ".join(text_block_value(u) or "" for _, u in conn.updates).lower()
    assert "token" in streamed


@pytest.mark.asyncio
async def test_reasoning_streamed_as_thought_not_answer():
    # A reasoning delta (gpt-oss analysis / remote reasoning_content) is surfaced
    # as an ACP thought, never as the user-visible answer.
    mcp = StubMCP([])
    backend = StubBackend(
        turns=[[
            StreamDelta(reasoning="let me think..."),
            StreamDelta(text="The answer is 42."),
            StreamDelta(finished=True, stop_reason="end_turn"),
        ]]
    )
    session = make_session(backend, mcp)
    conn = FakeConn()

    await run_agent_loop(conn, session, [{"text": "go"}], "m1", Config())

    kinds = [type(u).__name__ for _, u in conn.updates]
    assert "AgentThoughtChunk" in kinds          # reasoning went to a thought
    answer = [
        text_block_value(u)
        for _, u in conn.updates
        if type(u).__name__ == "AgentMessageChunk"
    ]
    assert "The answer is 42." in answer
    assert "let me think..." not in answer        # CoT never leaks into the answer


@pytest.mark.asyncio
async def test_reasoning_preserved_in_history_on_tool_turns():
    # For preserve_thinking templates (e.g. Gemma-4 coding), the turn's reasoning
    # must be carried back on the tool-call assistant message and resent next turn.
    tools = [mcp_tool("run_cell", {"index": {"type": "integer"}})]
    mcp = StubMCP(tools, results={"run_cell": "ok"})
    backend = StubBackend(
        turns=[
            [StreamDelta(reasoning="I will run the cell."),
             StreamDelta(tool_call=ToolCall(id="t1", name="run_cell", arguments={"index": 1})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(text="done"), StreamDelta(finished=True, stop_reason="end_turn")],
        ]
    )
    session = make_session(backend, mcp)
    await run_agent_loop(FakeConn(), session, [{"text": "go"}], "m1", Config())

    tool_turn = next(m for m in session.messages if m.get("role") == "assistant" and m.get("tool_calls"))
    assert tool_turn.get("reasoning_content") == "I will run the cell."
    assert tool_turn["tool_calls"][0]["function"]["arguments"] == {"index": 1}
    # The reasoning was resent in the next model call's history.
    assert any(m.get("reasoning_content") == "I will run the cell." for m in backend.calls[1]["messages"])


@pytest.mark.asyncio
async def test_no_reasoning_content_key_without_reasoning():
    # A tool-call turn with no reasoning must not add an empty reasoning_content.
    tools = [mcp_tool("run_cell", {"index": {"type": "integer"}})]
    mcp = StubMCP(tools, results={"run_cell": "ok"})
    backend = StubBackend(
        turns=[
            [StreamDelta(tool_call=ToolCall(id="t1", name="run_cell", arguments={"index": 1})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(text="done"), StreamDelta(finished=True, stop_reason="end_turn")],
        ]
    )
    session = make_session(backend, mcp)
    await run_agent_loop(FakeConn(), session, [{"text": "go"}], "m1", Config())

    tool_turn = next(m for m in session.messages if m.get("role") == "assistant" and m.get("tool_calls"))
    assert "reasoning_content" not in tool_turn


@pytest.mark.asyncio
async def test_normal_text_answer_has_no_fallback():
    # A turn that DID produce visible text must not get a fallback appended.
    mcp = StubMCP([])
    backend = StubBackend(
        turns=[[StreamDelta(text="All done."), StreamDelta(finished=True, stop_reason="end_turn")]]
    )
    session = make_session(backend, mcp)
    conn = FakeConn()

    await run_agent_loop(conn, session, [{"text": "go"}], "m1", Config())

    streamed = [text_block_value(u) for _, u in conn.updates if text_block_value(u)]
    assert streamed == ["All done."]


@pytest.mark.asyncio
async def test_invalid_tool_args_surface_as_error():
    tools = [mcp_tool("run_cell")]
    mcp = StubMCP(tools)
    backend = StubBackend(
        turns=[
            [StreamDelta(tool_call=ToolCall(id="t1", name="run_cell", arguments={"__raw__": "{bad"})),
             StreamDelta(finished=True, stop_reason="tool_use")],
            [StreamDelta(text="sorry"), StreamDelta(finished=True, stop_reason="end_turn")],
        ]
    )
    session = make_session(backend, mcp)
    await run_agent_loop(FakeConn(), session, [{"text": "go"}], "m1", Config())
    # The bad call must NOT have reached MCP; an error was fed back instead.
    assert mcp.calls == []
    assert any(m.get("role") == "tool" and m["content"].startswith("ERROR:") for m in session.messages)
