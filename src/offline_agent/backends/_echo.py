"""A trivial backend for smoke-testing the ACP/agent wiring without a model.

Enabled by setting ``OFFLINE_AGENT_STUB_BACKEND=1`` (or ``=tool`` to make it emit
one tool call to the first available tool before answering). Never used in
normal operation.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from .base import (
    Capabilities,
    ModelBackend,
    OutputConstraint,
    SamplingParams,
    StreamDelta,
    ToolCall,
)


class EchoBackend(ModelBackend):
    def __init__(self, mode: str = "echo") -> None:
        self._mode = mode
        self._called_tool = False

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(False, False, True, False, 4096)

    @property
    def chat_format(self):
        from ..chat_formats import get_chat_format

        return get_chat_format("default")

    async def prepare_session(self, system_prompt: str, tools: list[dict]) -> None:
        pass

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        constraint: OutputConstraint,
        sampling: SamplingParams,
    ) -> AsyncIterator[StreamDelta]:
        last_user = next(
            (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if self._mode == "tool" and tools and not self._called_tool:
            self._called_tool = True
            fn = tools[0]["function"]
            yield StreamDelta(
                tool_call=ToolCall(id="echo-1", name=fn["name"], arguments={})
            )
            yield StreamDelta(finished=True, stop_reason="tool_use")
            return
        yield StreamDelta(text=f"echo: {last_user}")
        yield StreamDelta(finished=True, stop_reason="end_turn")

    async def aclose(self) -> None:
        pass


def echo_backend_from_env() -> "EchoBackend | None":
    flag = os.environ.get("OFFLINE_AGENT_STUB_BACKEND")
    if not flag:
        return None
    return EchoBackend(mode="tool" if flag == "tool" else "echo")
