"""Shared helpers for normalizing model output across backends."""

from __future__ import annotations

import json

from .base import ToolCall

_FINISH_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}


def map_finish_reason(reason: str) -> str:
    return _FINISH_REASONS.get(reason, "end_turn")


def to_tool_call(slot: dict[str, str]) -> ToolCall:
    """Build a ToolCall from accumulated native tool-call fragments."""
    raw = slot.get("args", "")
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        args = {"__raw__": raw}
    name = slot.get("name", "")
    return ToolCall(id=slot.get("id") or name, name=name, arguments=args)


def parse_envelope(
    text: str, *, name_key: str = "name", args_key: str = "arguments"
) -> ToolCall | None:
    """Parse a ``{<name_key>: ..., <args_key>: {...}}`` envelope from guided output.

    ``name_key``/``args_key`` mirror the schema a chat format produced via
    ``tool_call_schema``, so parsing stays in lockstep with the forced envelope.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    name = obj.get(name_key)
    if not name:
        return None
    return ToolCall(id=name, name=name, arguments=obj.get(args_key) or {})
