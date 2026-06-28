"""The single, backend-agnostic agent loop.

Each turn: stream the model, surface text + tool calls as ACP ``session/update``
notifications, dispatch tool calls (MCP notebook tools or ACP-serviced fs tools),
append results, and repeat until the model stops calling tools (or a bound/cancel
is hit). The frozen tool list is passed verbatim every turn (single source of
truth) to keep the cached prefix stable.
"""

from __future__ import annotations

import json
import logging

from acp import (
    PromptResponse,
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)

from ..backends.base import OutputConstraint, SamplingParams, ToolCall
from ..config import Config
from ..tools.fs import dispatch_fs, is_fs_tool
from .session import Session

log = logging.getLogger(__name__)

FINAL_ANSWER = "final_answer"

# Injected only in forced (json_schema/grammar) modes so the model has a way to
# stop calling tools and answer. In native mode the model emits plain text.
_FINAL_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": FINAL_ANSWER,
        "description": "Provide the final answer to the user and stop.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}

# A model can wedge itself re-issuing the same call when a tool keeps returning
# the same (often unhelpful) error -- e.g. a tool whose schema marks an argument
# optional but whose implementation requires it. Break out after this many
# identical, consecutive calls instead of burning every remaining turn.
_MAX_IDENTICAL_TOOL_CALLS = 3

# Map tool name hints to ACP ToolKind for nicer client rendering.
_EXECUTE_HINTS = ("run", "execute", "exec")
_EDIT_HINTS = ("edit", "write", "add", "insert", "overwrite", "delete", "create")


def _tool_kind(name: str) -> str:
    low = name.lower()
    if any(h in low for h in _EXECUTE_HINTS):
        return "execute"
    if name == "read_file" or "read" in low or "get" in low:
        return "read"
    if any(h in low for h in _EDIT_HINTS):
        return "edit"
    return "other"


def _sampling(config: Config) -> SamplingParams:
    s = config.sampling
    return SamplingParams(
        temperature=s.temperature,
        top_p=s.top_p,
        top_k=s.top_k,
        max_tokens=s.max_tokens,
        seed=s.seed,
        stop=s.stop,
    )


def _build_constraint(session: Session, config: Config) -> OutputConstraint:
    mode = config.constraints.tool_call_mode
    if mode == "native":
        return OutputConstraint.none()
    # Forced modes: constrain output to one tool call OR a final answer, using
    # the session's chat-format envelope (parsed back with the same field names).
    schema = session.chat_format.envelope_schema(session.tools + [_FINAL_ANSWER_TOOL])
    # Both backends enforce a JSON schema (vLLM guided_json / llama from_json_schema).
    return OutputConstraint(kind="json_schema", json_schema=schema)


def _request_tools(session: Session, config: Config) -> list[dict]:
    if config.constraints.tool_call_mode == "native":
        return session.tools
    return session.tools + [_FINAL_ANSWER_TOOL]


def _format_backend_error(exc: Exception, config: Config) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    if config.backend.mode == "remote":
        url = config.remote.base_url
        model = config.remote.model or "<unset>"
        low = detail.lower()
        if exc.__class__.__name__ == "APIConnectionError" or detail == "Connection error.":
            return (
                f"I couldn't reach the configured remote model endpoint at `{url}`.\n\n"
                "Check that the server is running and that `[remote].base_url` is correct. "
                "If you want the built-in llama.cpp backend instead, switch `[backend] mode` "
                "to `\"local\"` (or enable `fallback_to_local` with a valid `local.model_path`)."
            )
        if exc.__class__.__name__ == "NotFoundError" or (
            "model" in low and "not found" in low
        ):
            return (
                f"The remote endpoint at `{url}` could not find model `{model}`.\n\n"
                "Check `[remote].model` against the ids exposed by the server's `/v1/models` endpoint."
            )
        return (
            f"The remote model request to `{url}` failed: {detail}\n\n"
            "Check the remote server and the `[remote]` settings in `offline_agent.toml`."
        )
    if config.backend.mode == "local":
        model_path = config.local.model_path or "<unset>"
        return (
            f"I couldn't use the local model at `{model_path}`: {detail}\n\n"
            "Check `[local].model_path` and confirm the GGUF is present and readable."
        )
    return f"I couldn't complete that because the model backend failed: {detail}"


async def _report_backend_error(
    conn,
    session: Session,
    config: Config,
    exc: Exception,
    message_id: str | None,
) -> PromptResponse:
    msg = _format_backend_error(exc, config)
    log.warning("backend request failed", exc_info=exc)
    await conn.session_update(session.session_id, update_agent_message(text_block(msg)))
    session.messages.append({"role": "assistant", "content": msg})
    return PromptResponse(stop_reason="end_turn", user_message_id=message_id)


async def run_agent_loop(
    conn,
    session: Session,
    prompt_blocks: list,
    message_id: str | None,
    config: Config,
) -> PromptResponse:
    user_text = _blocks_to_text(prompt_blocks)
    session.messages.append({"role": "user", "content": user_text})

    sampling = _sampling(config)
    last_call_sig: tuple[str, str] | None = None
    repeat_count = 0
    recent_failures: list[tuple[str, str]] = []
    for _turn in range(config.constraints.max_tool_turns):
        if session.cancel.is_set():
            return PromptResponse(stop_reason="cancelled", user_message_id=message_id)

        constraint = _build_constraint(session, config)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason = "end_turn"

        try:
            async for delta in session.backend.stream(
                session.messages, _request_tools(session, config), constraint, sampling
            ):
                if delta.text:
                    text_parts.append(delta.text)
                    await conn.session_update(
                        session.session_id, update_agent_message(text_block(delta.text))
                    )
                if delta.reasoning:
                    # Chain-of-thought: shown as a distinct ACP thought, never folded
                    # into the answer. Also means a reasoning-only turn isn't mistaken
                    # for the model producing nothing at all.
                    reasoning_parts.append(delta.reasoning)
                    await conn.session_update(
                        session.session_id, update_agent_thought(text_block(delta.reasoning))
                    )
                if delta.tool_call is not None:
                    tool_calls.append(delta.tool_call)
                if delta.finished and delta.stop_reason:
                    stop_reason = delta.stop_reason
        except Exception as exc:  # noqa: BLE001 - convert backend failures into user-visible replies
            return await _report_backend_error(conn, session, config, exc, message_id)

        # In forced mode a final_answer "tool call" is really the assistant text.
        final = _take_final_answer(tool_calls)
        if final is not None:
            await conn.session_update(
                session.session_id, update_agent_message(text_block(final))
            )
            session.messages.append({"role": "assistant", "content": final})
            return PromptResponse(stop_reason="end_turn", user_message_id=message_id)

        session.messages.append(_assistant_message(text_parts, tool_calls, reasoning_parts))

        if not tool_calls:
            # The model ended its turn. If it said nothing visible (common with
            # gpt-oss/Gemma when the answer lands in an elided reasoning channel,
            # or the model gives up silently after a tool failure, or it runs out
            # of tokens), don't return an empty message -- explain what happened
            # so a failure is never swallowed.
            if not "".join(text_parts).strip():
                await _report_no_answer(conn, session, recent_failures, stop_reason, message_id)
            return PromptResponse(stop_reason=stop_reason, user_message_id=message_id)

        recent_failures = []
        for tc in tool_calls:
            result, status = await _dispatch_tool(conn, session, tc, config)
            session.messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
            if status == "failed":
                recent_failures.append((tc.name, result))

            # Detect a tight loop: the same tool name + arguments, back to back.
            sig = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
            repeat_count = repeat_count + 1 if sig == last_call_sig else 1
            last_call_sig = sig
            if repeat_count >= _MAX_IDENTICAL_TOOL_CALLS:
                return await _break_tool_loop(conn, session, tc, result, message_id)

    msg = (
        "I reached the tool-call limit for this request without finishing."
        + _failure_suffix(recent_failures)
    )
    await conn.session_update(session.session_id, update_agent_message(text_block(msg)))
    session.messages.append({"role": "assistant", "content": msg})
    return PromptResponse(stop_reason="max_turn_requests", user_message_id=message_id)


def _failure_suffix(failures: list[tuple[str, str]]) -> str:
    if not failures:
        return ""
    lines = "\n".join(f"- `{name}`: {result}" for name, result in failures)
    return f" The last tool call(s) failed:\n\n{lines}"


async def _report_no_answer(
    conn,
    session: Session,
    failures: list[tuple[str, str]],
    stop_reason: str,
    message_id: str | None,
) -> None:
    """Stream a fallback message when a turn produced no user-visible text.

    Prefer surfacing the concrete tool failures that preceded the silence; fall
    back to a truncation note or a generic line. Without this the user would see
    an empty agent reply and a failure would go unreported.
    """
    if failures:
        msg = "I couldn't complete that." + _failure_suffix(failures)
    elif stop_reason == "max_tokens":
        msg = (
            "I ran out of output tokens before finishing my response. "
            "Try a smaller step, or raise sampling.max_tokens."
        )
    else:
        msg = "I don't have a response for that. Please try rephrasing your request."
    log.info("reporting empty-turn fallback (stop_reason=%s, failures=%d)", stop_reason, len(failures))
    await conn.session_update(session.session_id, update_agent_message(text_block(msg)))
    session.messages.append({"role": "assistant", "content": msg})


async def _break_tool_loop(
    conn, session: Session, tc: ToolCall, result: str, message_id: str | None
) -> PromptResponse:
    """Abort the turn and tell the user, when the model is stuck on one call.

    Surfaces the offending call and its last tool response (e.g. the error
    packet) so the user gets a meaningful explanation instead of silence or a
    bare ``max_turn_requests``.
    """
    log.warning(
        "breaking tool loop: %r called %d times with identical arguments",
        tc.name,
        _MAX_IDENTICAL_TOOL_CALLS,
    )
    args = json.dumps(tc.arguments, indent=2, default=str)
    message = (
        f"I stopped because the tool `{tc.name}` was called "
        f"{_MAX_IDENTICAL_TOOL_CALLS} times in a row with identical arguments and "
        f"kept returning the same result, so I was stuck in a loop.\n\n"
        f"Arguments:\n```json\n{args}\n```\n\n"
        f"Last tool response:\n```\n{result}\n```\n\n"
        f"This usually means the call is being rejected — for example a required "
        f"value is missing or malformed. Please check the arguments above or try a "
        f"different approach."
    )
    await conn.session_update(
        session.session_id, update_agent_message(text_block(message))
    )
    return PromptResponse(stop_reason="refusal", user_message_id=message_id)


def _take_final_answer(tool_calls: list[ToolCall]) -> str | None:
    for tc in tool_calls:
        if tc.name == FINAL_ANSWER:
            return str(tc.arguments.get("text", ""))
    return None


def _assistant_message(
    text_parts: list[str],
    tool_calls: list[ToolCall],
    reasoning_parts: list[str] | None = None,
) -> dict:
    # content stays a string ("" when the turn was a pure tool call). It must NOT
    # be None: the gpt-oss/Harmony GGUF chat template guards with `if "content" in
    # message` and then does `"<|channel|>..." in message.content`, which raises
    # "argument of type 'NoneType' is not iterable" when content is None.
    msg: dict = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]
        # Carry this turn's chain-of-thought back on tool-call turns so a
        # preserve-thinking chat template (e.g. the Gemma-4 coding template) can
        # re-emit prior reasoning -- the model card's recommended setup for
        # multi-step tool use. Only on tool-call turns: that's the sole case such
        # templates consume it, and templates that don't know the field ignore it.
        reasoning = "".join(reasoning_parts or [])
        if reasoning:
            msg["reasoning_content"] = reasoning
    return msg


def _missing_required_args(session: Session, name: str, arguments: dict) -> list[str]:
    """Required schema fields absent from a tool call's arguments.

    Catches degenerate calls -- most often an empty ``{}`` from a Harmony tool
    call emitted without a body or truncated mid-stream -- so we can hand the
    model a clear, tool-agnostic error instead of forwarding the call and
    surfacing a backend-specific validation traceback (e.g. a pydantic
    "Missing required argument" dump from the MCP server).
    """
    for tool in session.tools:
        fn = tool.get("function") or {}
        if fn.get("name") == name:
            params = fn.get("parameters") or {}
            required = params.get("required") or []
            return [r for r in required if r not in arguments]
    return []


async def _dispatch_tool(conn, session: Session, tc: ToolCall, config: Config):
    await conn.session_update(
        session.session_id,
        start_tool_call(tc.id, tc.name, kind=_tool_kind(tc.name), status="in_progress"),
    )
    try:
        if "__raw__" in tc.arguments:
            raise ValueError(
                "tool arguments were not valid JSON; re-emit a single valid JSON object"
            )
        missing = _missing_required_args(session, tc.name, tc.arguments)
        if missing:
            hint = " (the previous call may have been cut off)" if not tc.arguments else ""
            raise ValueError(
                f"missing required argument(s) for {tc.name!r}: {', '.join(missing)}; "
                f"re-emit the call with a complete JSON arguments object" + hint
            )
        if is_fs_tool(tc.name):
            result = await dispatch_fs(conn, session.session_id, tc.name, tc.arguments)
        elif session.mcp.has_tool(tc.name):
            result = await session.mcp.call_tool(tc.name, tc.arguments)
        else:
            raise KeyError(f"unknown tool: {tc.name!r}")
        status = "completed"
    except Exception as exc:  # noqa: BLE001 - surface tool errors back to the model
        log.info("tool %s failed: %s", tc.name, exc)
        result = f"ERROR: {exc}"
        status = "failed"

    await conn.session_update(
        session.session_id,
        update_tool_call(tc.id, status=status, content=[tool_content(text_block(result))]),
    )
    return result, status


def _blocks_to_text(blocks: list) -> str:
    parts: list[str] = []
    for block in blocks or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts)
