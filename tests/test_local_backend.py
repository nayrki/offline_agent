"""Tests for local-backend message normalization (no model load required)."""

from __future__ import annotations

from offline_agent.backends._messages import (
    tool_call_arguments_as_json_strings,
    tool_call_arguments_as_objects,
)


def _msgs(arguments):
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "run_cell", "arguments": arguments}},
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]


def test_json_string_args_become_objects():
    out = tool_call_arguments_as_objects(_msgs('{"code": "print(1)"}'))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"code": "print(1)"}


def test_empty_and_malformed_args_normalize_to_empty_dict():
    assert tool_call_arguments_as_objects(_msgs(""))[1]["tool_calls"][0]["function"]["arguments"] == {}
    assert tool_call_arguments_as_objects(_msgs("{bad"))[1]["tool_calls"][0]["function"]["arguments"] == {}


def test_does_not_mutate_input_messages():
    original = _msgs('{"code": "x"}')
    tool_call_arguments_as_objects(original)
    assert original[1]["tool_calls"][0]["function"]["arguments"] == '{"code": "x"}'


def test_messages_without_tool_calls_pass_through():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert tool_call_arguments_as_objects(msgs) == msgs


def test_mapping_args_become_json_strings_for_builtin_handlers():
    out = tool_call_arguments_as_json_strings(_msgs({"code": "print(1)"}))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == '{"code": "print(1)"}'


def test_stringify_normalizes_non_mapping_args_to_empty_object():
    out = tool_call_arguments_as_json_strings(_msgs('["x"]'))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == "{}"
