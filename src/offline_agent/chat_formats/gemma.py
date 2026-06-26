"""Parser for Gemma native tool-call output.

Some Gemma stacks emit tool calls as plain assistant text instead of structured
OpenAI ``tool_calls`` objects, for example::

    <|channel>thought
    I should inspect the notebook first.
    <channel|><|tool_call>call:get_active_notebook{}<tool_call|>

This parser turns that DSL back into a normalized ``ToolCall`` so the agent loop
can execute it instead of showing the raw markup to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..backends.base import ToolCall
from .harmony import NativeTurn

_THOUGHT = re.compile(r"<\|channel>thought\s*(.*?)\s*<channel\|>", re.DOTALL)
_TURN_MARKERS = re.compile(r"<\|turn>model\s*|<turn\|>")
_TOOL_CALL = re.compile(r"<\|tool_call>call:([A-Za-z0-9_.\-]+)\{")
_STRING_TOKEN = '<|"|>'


class _ParseError(ValueError):
    pass


@dataclass
class _DslParser:
    text: str
    pos: int = 0

    def parse(self):
        value = self._parse_value()
        self._skip_ws()
        if self.pos != len(self.text):
            raise _ParseError("trailing characters")
        return value

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _parse_value(self):
        self._skip_ws()
        if self.text.startswith(_STRING_TOKEN, self.pos):
            return self._parse_string()
        if self.pos >= len(self.text):
            raise _ParseError("unexpected end of input")
        ch = self.text[self.pos]
        if ch == "{":
            return self._parse_object()
        if ch == "[":
            return self._parse_array()
        for literal, value in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(literal, self.pos):
                self.pos += len(literal)
                return value
        return self._parse_number()

    def _parse_string(self) -> str:
        self.pos += len(_STRING_TOKEN)
        end = self.text.find(_STRING_TOKEN, self.pos)
        if end < 0:
            raise _ParseError("unterminated string")
        value = self.text[self.pos:end]
        self.pos = end + len(_STRING_TOKEN)
        return value

    def _parse_key(self) -> str:
        self._skip_ws()
        if self.text.startswith(_STRING_TOKEN, self.pos):
            return self._parse_string()
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ":,{}[] \t\r\n":
            self.pos += 1
        if self.pos == start:
            raise _ParseError("expected object key")
        return self.text[start:self.pos]

    def _parse_object(self) -> dict:
        obj: dict = {}
        self.pos += 1
        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == "}":
            self.pos += 1
            return obj
        while True:
            key = self._parse_key()
            self._skip_ws()
            if self.pos >= len(self.text) or self.text[self.pos] != ":":
                raise _ParseError("expected ':' after key")
            self.pos += 1
            obj[key] = self._parse_value()
            self._skip_ws()
            if self.pos >= len(self.text):
                raise _ParseError("unterminated object")
            if self.text[self.pos] == "}":
                self.pos += 1
                return obj
            if self.text[self.pos] != ",":
                raise _ParseError("expected ',' or '}'")
            self.pos += 1

    def _parse_array(self) -> list:
        items: list = []
        self.pos += 1
        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == "]":
            self.pos += 1
            return items
        while True:
            items.append(self._parse_value())
            self._skip_ws()
            if self.pos >= len(self.text):
                raise _ParseError("unterminated array")
            if self.text[self.pos] == "]":
                self.pos += 1
                return items
            if self.text[self.pos] != ",":
                raise _ParseError("expected ',' or ']'")
            self.pos += 1

    def _parse_number(self) -> int | float:
        start = self.pos
        if self.text[self.pos] == "-":
            self.pos += 1
        digits_start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == digits_start:
            raise _ParseError("expected number")
        if self.pos < len(self.text) and self.text[self.pos] == ".":
            self.pos += 1
            frac_start = self.pos
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            if self.pos == frac_start:
                raise _ParseError("malformed float")
        if self.pos < len(self.text) and self.text[self.pos] in "eE":
            self.pos += 1
            if self.pos < len(self.text) and self.text[self.pos] in "+-":
                self.pos += 1
            exp_start = self.pos
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            if self.pos == exp_start:
                raise _ParseError("malformed exponent")
        raw = self.text[start:self.pos]
        return float(raw) if any(ch in raw for ch in ".eE") else int(raw)


def _extract_balanced_object(text: str, start: int) -> tuple[str, int]:
    depth = 0
    i = start
    while i < len(text):
        if text.startswith(_STRING_TOKEN, i):
            j = text.find(_STRING_TOKEN, i + len(_STRING_TOKEN))
            if j < 0:
                raise _ParseError("unterminated string token")
            i = j + len(_STRING_TOKEN)
            continue
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
        i += 1
    raise _ParseError("unterminated object")


def _clean(text: str) -> str:
    text = _THOUGHT.sub("", text)
    text = _TURN_MARKERS.sub("", text)
    return text.strip()


def parse_gemma(text: str) -> NativeTurn:
    """Parse one Gemma assistant turn into a :class:`NativeTurn`."""
    thought = _THOUGHT.search(text)
    reasoning = thought.group(1).strip() if thought else None

    tool = _TOOL_CALL.search(text)
    if tool:
        name = tool.group(1)
        start = tool.end() - 1  # include the opening "{"
        try:
            raw_args, _ = _extract_balanced_object(text, start)
            arguments = _DslParser(raw_args).parse()
        except _ParseError:
            arguments = {"__raw__": text[start:]}
        return NativeTurn(
            tool_call=ToolCall(id=name, name=name, arguments=arguments),
            reasoning=reasoning,
        )

    return NativeTurn(content=_clean(text), reasoning=reasoning)

