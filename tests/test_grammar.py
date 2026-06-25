"""Tool-call schema validity and prefix-stability guardrails."""

from __future__ import annotations

import jsonschema
import pytest

from offline_agent.backends.grammars import tool_call_schema, tool_names

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_cell",
            "description": "add",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "index": {"type": "integer"}},
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cell",
            "description": "run",
            "parameters": {
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
            },
        },
    },
]


def test_tool_names():
    assert tool_names(TOOLS) == ["add_cell", "run_cell"]


def test_schema_accepts_valid_calls():
    schema = tool_call_schema(TOOLS)
    jsonschema.validate({"name": "add_cell", "arguments": {"source": "print(1)"}}, schema)
    jsonschema.validate({"name": "run_cell", "arguments": {"index": 2}}, schema)


def test_schema_rejects_unknown_tool_and_bad_args():
    schema = tool_call_schema(TOOLS)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"name": "delete_everything", "arguments": {}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        # run_cell requires integer index
        jsonschema.validate({"name": "run_cell", "arguments": {"index": "two"}}, schema)


def test_single_tool_schema_is_branch_not_anyof():
    schema = tool_call_schema([TOOLS[0]])
    assert "anyOf" not in schema
    assert schema["properties"]["name"]["const"] == "add_cell"


def test_empty_tools_raises():
    with pytest.raises(ValueError):
        tool_call_schema([])
