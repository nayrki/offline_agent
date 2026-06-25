"""Build constrained-decoding artifacts from a frozen tool list.

The portable representation is a JSON Schema for the tool-call *envelope*
(``{"name": <tool>, "arguments": <that tool's params>}``). Each backend enforces
it natively: the remote backend passes it as vLLM ``guided_json``; the local
backend converts it to GBNF via ``llama_cpp``'s built-in JSON-schema converter.
"""

from __future__ import annotations

from typing import Any


def tool_names(tools: list[dict]) -> list[str]:
    return [t["function"]["name"] for t in tools]


def _tool_branch(tool: dict, name_key: str, args_key: str) -> dict[str, Any]:
    fn = tool["function"]
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    return {
        "type": "object",
        "properties": {
            name_key: {"const": fn["name"]},
            args_key: params,
        },
        "required": [name_key, args_key],
        "additionalProperties": False,
    }


def tool_call_schema(
    tools: list[dict], *, name_key: str = "name", args_key: str = "arguments"
) -> dict[str, Any]:
    """A JSON Schema constraining output to one valid tool call.

    Uses ``anyOf`` over per-tool branches so each tool's arguments are validated
    against that specific tool's parameter schema. ``name_key``/``args_key`` let a
    chat format select the envelope field names (the registry's per-model knob).
    """
    if not tools:
        raise ValueError("tool_call_schema requires at least one tool")
    branches = [_tool_branch(t, name_key, args_key) for t in tools]
    if len(branches) == 1:
        return branches[0]
    return {"anyOf": branches}
