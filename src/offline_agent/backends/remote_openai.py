"""Remote backend: the ``openai`` SDK against any OpenAI-compatible endpoint.

Primary path is native function calling (``tools=``), which on vLLM/llama.cpp
already constrains tool-call arguments to the schema. When the config requests
explicit forcing (``json_schema``/``grammar``), we use vLLM's ``guided_json`` /
``guided_grammar`` via ``extra_body`` -- gated behind a one-time capability probe
that falls back to native + validate-and-retry when the server rejects them.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ..chat_formats import resolve_chat_format
from ..config import Config
from ._parsing import map_finish_reason, to_tool_call
from .base import (
    Capabilities,
    ModelBackend,
    OutputConstraint,
    SamplingParams,
    StreamDelta,
)

log = logging.getLogger(__name__)


def _reasoning_of(delta) -> str | None:
    """Extract a reasoning/thinking fragment from a streamed delta, if any.

    Reasoning models served over the OpenAI-compatible API (LM Studio, vLLM,
    DeepSeek, ...) put chain-of-thought in a ``reasoning_content`` field that the
    SDK keeps in ``model_extra`` rather than ``content``. Ignoring it makes a
    pure-reasoning chunk look empty -- and a reasoning-only turn (it ran out of
    tokens before answering) indistinguishable from a silent failure."""
    if delta is None:
        return None
    direct = getattr(delta, "reasoning_content", None)
    if direct:
        return direct
    extra = getattr(delta, "model_extra", None) or {}
    return extra.get("reasoning_content") or extra.get("reasoning")


class RemoteOpenAIBackend(ModelBackend):
    def __init__(self, config: Config) -> None:
        from openai import AsyncOpenAI

        self._cfg = config
        self._rcfg = config.remote
        # Used only to parse the forced envelope; the server owns its own native
        # template, so llama_chat_format is irrelevant here. Resolved from the
        # served model id so "auto" doesn't force the remote model onto the local
        # fallback's handler (e.g. a Gemma endpoint getting the gpt-oss parser).
        self._fmt = resolve_chat_format(config.backend.chat_handler, model_id=self._rcfg.model)
        log.info("remote model %s -> chat_format=%s", self._rcfg.model, self._fmt.name)
        self._client = AsyncOpenAI(
            base_url=self._rcfg.base_url,
            api_key=self._rcfg.api_key,
            timeout=self._rcfg.request_timeout,
        )
        # Optimistic until probed; force_native_tools pins them off.
        guided = not self._rcfg.force_native_tools
        self._caps = Capabilities(
            supports_grammar=guided,
            supports_guided_json=guided,
            supports_native_tools=True,
            supports_kv_snapshot=False,
            max_context=0,
        )
        self._probed = self._rcfg.force_native_tools

    @property
    def chat_format(self):
        return self._fmt

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    async def is_reachable(self) -> bool:
        """Lightweight liveness probe used to decide remote-vs-local failover.

        Hits ``GET /v1/models`` -- the most universally supported endpoint across
        vLLM, LM Studio, and Ollama's OpenAI-compat shim, and one that needs no
        model name -- with a short timeout and no retries, so an offline server
        fails fast instead of stalling the session on ``request_timeout``.
        """
        try:
            await self._client.with_options(
                timeout=self._rcfg.connect_timeout, max_retries=0
            ).models.list()
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            log.info("remote endpoint %s not reachable: %s", self._rcfg.base_url, exc)
            return False

    async def prepare_session(self, system_prompt: str, tools: list[dict]) -> None:
        # Server-side automatic prefix caching handles reuse; nothing to prefill.
        # Probe guided-decoding support once if we might use it.
        if not self._probed and self._cfg.constraints.tool_call_mode != "native":
            await self._probe()

    async def _probe(self) -> None:
        self._probed = True
        try:
            await self._client.chat.completions.create(
                model=self._rcfg.model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
                extra_body={"guided_choice": ["ok"]},
            )
            supported = True
        except Exception as exc:  # noqa: BLE001 - any rejection means "unsupported"
            log.info("guided decoding probe failed (%s); using native tools", exc)
            supported = False
        self._caps = Capabilities(
            supports_grammar=supported,
            supports_guided_json=supported,
            supports_native_tools=True,
            supports_kv_snapshot=False,
            max_context=self._caps.max_context,
        )

    def _sampling_kwargs(self, sampling: SamplingParams) -> tuple[dict[str, Any], dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        if sampling.stop:
            kwargs["stop"] = sampling.stop
        extra: dict[str, Any] = {}
        if sampling.top_k is not None:
            extra["top_k"] = sampling.top_k
        return kwargs, extra

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        constraint: OutputConstraint,
        sampling: SamplingParams,
    ) -> AsyncIterator[StreamDelta]:
        kwargs, extra_body = self._sampling_kwargs(sampling)
        # Turn the model's reasoning budget down (or off) so simple asks don't
        # exhaust max_tokens thinking. Sent in extra_body since "none" isn't a
        # standard OpenAI value; servers that don't know the field ignore it.
        if self._rcfg.reasoning_effort:
            extra_body["reasoning_effort"] = self._rcfg.reasoning_effort
        use_guided = constraint.kind != "none" and (
            (constraint.kind == "json_schema" and self._caps.supports_guided_json)
            or (constraint.kind == "grammar" and self._caps.supports_grammar)
        )

        if use_guided:
            if constraint.kind == "json_schema":
                extra_body["guided_json"] = constraint.json_schema
            else:
                extra_body["guided_grammar"] = constraint.grammar
            if self._rcfg.guided_decoding_backend != "auto":
                extra_body["guided_decoding_backend"] = self._rcfg.guided_decoding_backend
        elif tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if extra_body:
            kwargs["extra_body"] = extra_body

        stream = await self._client.chat.completions.create(
            model=self._rcfg.model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        # Accumulators for native tool_calls (keyed by index) and guided JSON text.
        partial_tools: dict[int, dict[str, str]] = {}
        guided_text: list[str] = []
        stop_reason = "end_turn"

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            reasoning = _reasoning_of(delta)
            if reasoning:
                yield StreamDelta(reasoning=reasoning)

            if use_guided:
                if delta and delta.content:
                    guided_text.append(delta.content)
            else:
                if delta and delta.content:
                    yield StreamDelta(text=delta.content)
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = partial_tools.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments

            if choice.finish_reason:
                stop_reason = map_finish_reason(choice.finish_reason)

        # Emit any tool calls collected from the stream.
        if use_guided and guided_text:
            tc = self._fmt.parse("".join(guided_text))
            if tc is not None:
                yield StreamDelta(tool_call=tc)
                stop_reason = "tool_use"
        else:
            for slot in partial_tools.values():
                if not slot["name"]:
                    continue
                yield StreamDelta(tool_call=to_tool_call(slot))
                stop_reason = "tool_use"

        yield StreamDelta(finished=True, stop_reason=stop_reason)

    async def aclose(self) -> None:
        await self._client.close()
