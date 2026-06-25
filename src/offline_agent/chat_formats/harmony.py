"""Parser for OpenAI **Harmony** output (gpt-oss family).

gpt-oss does not emit OpenAI-style ``tool_calls``; it speaks Harmony, where the
assistant turn is a sequence of *channel* segments delimited by special tokens:

    <|channel|>analysis<|message|>{chain-of-thought}<|end|>
    <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{json args}<|call|>
    <|start|>assistant<|channel|>final<|message|>{user-visible answer}<|return|>

``llama_cpp`` (0.3.26) has **no** Harmony handler, so ``create_chat_completion``
returns this raw text in ``content`` with ``tool_calls=None`` -- the channel
markers (``<|channel|>``/``<|message|>``/...) survive detokenization as literal
text, while the terminators ``<|call|>``/``<|return|>`` are end-of-generation
control tokens that stop the stream and detokenize to "". We parse the raw text
ourselves. (``llama-server`` and Ollama work only because they ship the
equivalent parser in C++, which the Python binding does not expose.)

The shapes here are taken from live gpt-oss-20b output, not just the GGUF's
input-side template -- the model puts ``to=functions.NAME`` *after*
``<|channel|>commentary``, which the template's render path does not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..backends.base import ToolCall


@dataclass(frozen=True)
class NativeTurn:
    """A parsed assistant turn from a native (non-enveloped) format.

    ``content`` is the user-visible answer (Harmony ``final`` channel);
    ``reasoning`` is the model's chain-of-thought (``analysis`` channel), kept
    separate so it never leaks into the message text.
    """

    tool_call: ToolCall | None = None
    content: str = ""
    reasoning: str | None = None


# All Harmony control/structural tokens, for defensive stripping of captured text.
_CONTROL = re.compile(r"<\|(?:start|end|call|return|channel|message|constrain)\|>")
_ANALYSIS = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", re.DOTALL
)
_FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.DOTALL
)
# A tool call routes to the `functions` namespace: `to=functions.<name>`.
_TOOL_NAME = re.compile(r"to=functions\.([A-Za-z0-9_.\-]+)")
# Args are the first message body following the recipient, up to a terminator.
_TOOL_ARGS = re.compile(
    r"<\|message\|>(.*?)(?:<\|call\|>|<\|return\|>|<\|end\|>|$)", re.DOTALL
)


def _clean(text: str) -> str:
    return _CONTROL.sub("", text).strip()


def parse_harmony(text: str) -> NativeTurn:
    """Parse one Harmony assistant turn into a :class:`NativeTurn`.

    Falls back to returning the cleaned text as ``content`` if no recognizable
    channels are present (a model that ignored the Harmony format).
    """
    analysis = _ANALYSIS.search(text)
    reasoning = _clean(analysis.group(1)) if analysis else None

    name_match = _TOOL_NAME.search(text)
    if name_match:
        name = name_match.group(1)
        args_match = _TOOL_ARGS.search(text, name_match.end())
        raw = args_match.group(1).strip() if args_match else ""
        try:
            arguments = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            arguments = {"__raw__": raw}
        return NativeTurn(
            tool_call=ToolCall(id=name, name=name, arguments=arguments),
            reasoning=reasoning,
        )

    final = _FINAL.search(text)
    if final:
        return NativeTurn(content=_clean(final.group(1)), reasoning=reasoning)

    return NativeTurn(content=_clean(text), reasoning=reasoning)
