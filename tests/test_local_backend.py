"""Tests for local-backend message normalization (no model load required)."""

from __future__ import annotations

from offline_agent.backends.local_llama import _objectify_tool_args


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
    # OpenAI-canonical string args must be parsed to a dict so the GGUF jinja
    # template's `| tojson` emits {"code": ...}, not a double-encoded string.
    out = _objectify_tool_args(_msgs('{"code": "print(1)"}'))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"code": "print(1)"}


def test_empty_and_malformed_args_normalize_to_empty_dict():
    assert _objectify_tool_args(_msgs(""))[1]["tool_calls"][0]["function"]["arguments"] == {}
    assert _objectify_tool_args(_msgs("{bad"))[1]["tool_calls"][0]["function"]["arguments"] == {}


def test_does_not_mutate_input_messages():
    original = _msgs('{"code": "x"}')
    _objectify_tool_args(original)
    # the canonical history still holds the OpenAI string form
    assert original[1]["tool_calls"][0]["function"]["arguments"] == '{"code": "x"}'


def test_messages_without_tool_calls_pass_through():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert _objectify_tool_args(msgs) == msgs
