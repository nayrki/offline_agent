"""Helpers for normalizing assistant tool-call arguments in chat history."""

from __future__ import annotations

import json
from typing import Any


def _normalize_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def tool_call_arguments_as_objects(messages: list[dict]) -> list[dict]:
    """Return a shallow-copied history with tool-call arguments as mappings."""
    out: list[dict] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            out.append(message)
            continue
        out.append(
            {
                **message,
                "tool_calls": [
                    {
                        **tool_call,
                        "function": {
                            **(tool_call.get("function") or {}),
                            "arguments": _normalize_arguments(
                                (tool_call.get("function") or {}).get("arguments")
                            ),
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )
    return out


def tool_call_arguments_as_json_strings(messages: list[dict]) -> list[dict]:
    """Return a shallow-copied history with tool-call arguments as JSON strings."""
    out: list[dict] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            out.append(message)
            continue
        out.append(
            {
                **message,
                "tool_calls": [
                    {
                        **tool_call,
                        "function": {
                            **(tool_call.get("function") or {}),
                            "arguments": json.dumps(
                                _normalize_arguments(
                                    (tool_call.get("function") or {}).get("arguments")
                                )
                            ),
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )
    return out
