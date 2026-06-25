"""Tests for the Harmony (gpt-oss) output parser.

The fixtures are verbatim ``create_chat_completion`` output captured from
gpt-oss-20b-mxfp4 on llama_cpp 0.3.26 (which returns Harmony text in ``content``
with ``tool_calls=None``).
"""

from __future__ import annotations

from offline_agent.chat_formats import available, get_chat_format
from offline_agent.chat_formats.harmony import parse_harmony

# Live capture: a tool call (generation stopped at <|call|>, which detokenizes to "").
TOOL_OUTPUT = (
    '<|channel|>analysis<|message|>We need to call the get_weather function with '
    'city "Paris".<|end|><|start|>assistant<|channel|>commentary '
    'to=functions.get_weather <|constrain|>json<|message|>{"city":"Paris"}'
)

# Live capture: a plain answer (generation stopped at <|return|>).
FINAL_OUTPUT = (
    '<|channel|>analysis<|message|>User wants a short greeting.<|end|>'
    '<|start|>assistant<|channel|>final<|message|>Hello!'
)


def test_gpt_oss_handler_registered():
    assert "gpt-oss" in available()
    fmt = get_chat_format("gpt-oss")
    assert fmt.llama_chat_format is None        # render with the GGUF's Harmony template
    assert fmt.native_parser is not None        # parse channel output ourselves


def test_parse_tool_call():
    turn = parse_harmony(TOOL_OUTPUT)
    assert turn.tool_call is not None
    assert turn.tool_call.name == "get_weather"
    assert turn.tool_call.arguments == {"city": "Paris"}
    assert turn.content == ""                    # no user-visible text on a tool turn
    assert turn.reasoning and "Paris" in turn.reasoning


def test_parse_final_answer():
    turn = parse_harmony(FINAL_OUTPUT)
    assert turn.tool_call is None
    assert turn.content == "Hello!"              # markers stripped, final channel only
    assert turn.reasoning == "User wants a short greeting."


def test_tool_call_with_explicit_call_terminator():
    # If the <|call|> terminator survives (special=True detok), still parse cleanly.
    text = (
        '<|start|>assistant<|channel|>commentary to=functions.add '
        '<|constrain|>json<|message|>{"a": 1, "b": 2}<|call|>'
    )
    turn = parse_harmony(text)
    assert turn.tool_call.name == "add"
    assert turn.tool_call.arguments == {"a": 1, "b": 2}


def test_empty_args_object():
    text = (
        '<|start|>assistant<|channel|>commentary to=functions.now '
        '<|constrain|>json<|message|>{}<|call|>'
    )
    turn = parse_harmony(text)
    assert turn.tool_call.name == "now"
    assert turn.tool_call.arguments == {}


def test_no_arg_tool_call_null_args():
    # gpt-oss renders a no-argument call as `null`; json.loads("null") is None.
    # ToolCall must normalize it to {} so the loop's `"__raw__" in arguments`
    # membership test doesn't crash with "argument of type 'NoneType'...".
    text = (
        '<|start|>assistant<|channel|>commentary to=functions.list_cells '
        '<|constrain|>json<|message|>null<|call|>'
    )
    turn = parse_harmony(text)
    assert turn.tool_call.name == "list_cells"
    assert turn.tool_call.arguments == {}
    assert "__raw__" in turn.tool_call.arguments or True  # must not raise


def test_malformed_json_falls_back_to_raw():
    text = (
        '<|channel|>commentary to=functions.broken '
        '<|constrain|>json<|message|>{not json'
    )
    turn = parse_harmony(text)
    assert turn.tool_call.name == "broken"
    assert turn.tool_call.arguments == {"__raw__": "{not json"}


def test_non_harmony_text_falls_back_to_content():
    turn = parse_harmony("just a plain string with no channels")
    assert turn.tool_call is None
    assert turn.content == "just a plain string with no channels"
