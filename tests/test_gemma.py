"""Tests for the Gemma native-output parser."""

from __future__ import annotations

from offline_agent.chat_formats import available, get_chat_format, resolve_chat_format
from offline_agent.chat_formats.gemma import parse_gemma

TOOL_OUTPUT = (
    "<|channel>thought\n"
    "Need the current notebook before editing.\n"
    "<channel|><|tool_call>call:get_active_notebook{}<tool_call|>"
)

ARG_OUTPUT = (
    "<|tool_call>call:add_cell{"
    'after:null,cell_type:<|"|>markdown<|"|>,'
    'lines:[1,2],meta:{trusted:true,score:1.5}}<tool_call|>'
)


def test_gemma_handler_registered():
    assert "gemma" in available()
    fmt = get_chat_format("gemma")
    assert fmt.llama_chat_format == "gemma"
    assert fmt.native_parser is not None


def test_auto_selects_gemma_for_gemma_models():
    assert resolve_chat_format("auto", model_id="google/gemma-4-12b-qat").name == "gemma"


def test_parse_tool_call():
    turn = parse_gemma(TOOL_OUTPUT)
    assert turn.tool_call is not None
    assert turn.tool_call.name == "get_active_notebook"
    assert turn.tool_call.arguments == {}
    assert turn.reasoning == "Need the current notebook before editing."
    assert turn.content == ""


def test_parse_arguments_dsl():
    turn = parse_gemma(ARG_OUTPUT)
    assert turn.tool_call is not None
    assert turn.tool_call.arguments == {
        "after": None,
        "cell_type": "markdown",
        "lines": [1, 2],
        "meta": {"trusted": True, "score": 1.5},
    }


def test_plain_answer_strips_thought_and_turn_markers():
    turn = parse_gemma("<|turn>model\n<|channel>thought\nbriefly think\n<channel|>All done.<turn|>")
    assert turn.tool_call is None
    assert turn.reasoning == "briefly think"
    assert turn.content == "All done."
