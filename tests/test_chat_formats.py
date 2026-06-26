"""Tests for the chat_handler registry."""

from __future__ import annotations

import jsonschema
import pytest

from offline_agent.chat_formats import (
    ChatFormat,
    available,
    get_chat_format,
    register,
    resolve_chat_format,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_cell",
            "description": "add",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string"}},
                "required": ["source"],
            },
        },
    }
]


def test_default_is_passthrough():
    fmt = get_chat_format("default")
    assert fmt.llama_chat_format is None       # use the GGUF's embedded template
    assert (fmt.name_key, fmt.args_key) == ("name", "arguments")


def test_builtin_handlers_present_and_named():
    for name in ("llama3", "mistral", "gemma", "chatml-function-calling"):
        assert name in available()
    assert get_chat_format("llama3").llama_chat_format == "llama-3"
    assert get_chat_format("mistral").llama_chat_format == "mistral-instruct"


def test_unknown_handler_lists_options():
    with pytest.raises(KeyError) as exc:
        get_chat_format("nope")
    assert "llama3" in str(exc.value)          # error surfaces available names


def test_explicit_handler_overrides_auto_detection():
    # A non-"auto" handler is returned as-is, ignoring model_arch/model_id.
    fmt = resolve_chat_format("llama3", model_arch="gpt-oss", model_id="gpt-oss")
    assert fmt.name == "llama3"


def test_auto_selects_harmony_for_gpt_oss():
    # gpt-oss (local GGUF arch or remote model id) => the Harmony native parser.
    assert resolve_chat_format("auto", model_arch="gpt-oss").name == "gpt-oss"
    assert resolve_chat_format("auto", model_id="openai/gpt-oss-20b").name == "gpt-oss"
    assert resolve_chat_format("auto", model_arch="gpt-oss").native_parser is not None


def test_auto_falls_through_to_default_for_other_models():
    # Llama / unknown => "default": trust the server/GGUF template's own native
    # tool calling. Gemma is handled separately because some stacks expose its
    # native tool DSL as plain text rather than structured tool_calls.
    for ident in ("meta-llama/Llama-3.1-8B", "", "mistralai/Mistral-7B-Instruct"):
        fmt = resolve_chat_format("auto", model_id=ident)
        assert fmt.name == "default"
        assert fmt.native_parser is None


def test_auto_with_no_identifier_is_default():
    assert resolve_chat_format("auto").name == "default"


def test_explicit_bad_handler_still_raises():
    with pytest.raises(KeyError):
        resolve_chat_format("nope")


def test_envelope_and_parse_round_trip_default():
    fmt = get_chat_format("default")
    schema = fmt.envelope_schema(TOOLS)
    payload = '{"name": "add_cell", "arguments": {"source": "print(1)"}}'
    jsonschema.validate(__import__("json").loads(payload), schema)
    tc = fmt.parse(payload)
    assert tc.name == "add_cell"
    assert tc.arguments == {"source": "print(1)"}


def test_custom_envelope_keys_stay_in_lockstep():
    # A format with non-default field names: the schema it constrains and the
    # parser that reads it must agree.
    fmt = ChatFormat("hermes-style", name_key="tool", args_key="tool_input")
    schema = fmt.envelope_schema(TOOLS)
    branch = schema  # single tool => a bare branch, not anyOf
    assert "tool" in branch["properties"]
    assert "tool_input" in branch["properties"]

    good = '{"tool": "add_cell", "tool_input": {"source": "x"}}'
    jsonschema.validate(__import__("json").loads(good), schema)
    tc = fmt.parse(good)
    assert tc.name == "add_cell" and tc.arguments == {"source": "x"}

    # The default name/arguments envelope must NOT satisfy this format.
    assert fmt.parse('{"name": "add_cell", "arguments": {}}') is None


def test_register_rejects_duplicate_without_replace():
    register(ChatFormat("temp-fmt"))
    with pytest.raises(ValueError):
        register(ChatFormat("temp-fmt"))
    register(ChatFormat("temp-fmt", llama_chat_format="chatml"), replace=True)
    assert get_chat_format("temp-fmt").llama_chat_format == "chatml"
