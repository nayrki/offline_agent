"""Local backend: in-process ``llama_cpp.Llama``.

llama.cpp inference is blocking C, so each generation runs in a thread executor
and is bridged to the event loop via an ``asyncio.Queue``. Prompt caching relies
on llama.cpp's *implicit* prefix reuse (identical prefix => cached KV); explicit
``save_state``/``load_state`` is opt-in and gated (see ``LocalConfig``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
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


def read_gguf_architecture(path: str) -> str | None:
    """Best-effort read of ``general.architecture`` from a GGUF header.

    Used to auto-select the chat format (e.g. ``gpt-oss`` -> Harmony) without
    loading the whole model. Parses only the key/value metadata block, skipping
    over values until it finds the architecture string. Returns ``None`` on any
    error -- a genuinely corrupt/missing file surfaces later at model load.
    """
    # GGUF value-type tag -> fixed byte width (string=8 and array=9 handled below).
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.read(4)  # version
            f.read(8)  # tensor count
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def read_str() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", "replace")

            def skip(vt: int) -> None:
                if vt == 8:
                    read_str()
                elif vt == 9:  # array: element type + count, then each element
                    et = struct.unpack("<I", f.read(4))[0]
                    for _ in range(struct.unpack("<Q", f.read(8))[0]):
                        skip(et)
                else:
                    f.read(fixed[vt])

            for _ in range(n_kv):
                key = read_str()
                vt = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture":
                    return read_str() if vt == 8 else None
                skip(vt)
    except Exception as exc:  # noqa: BLE001 - detection is best-effort
        log.debug("could not read GGUF architecture from %s: %s", path, exc)
    return None


def _objectify_tool_args(messages: list[dict]) -> list[dict]:
    """Parse OpenAI-style JSON-string tool-call arguments into objects.

    Canonical history stores ``tool_calls[].function.arguments`` as a JSON
    *string* (OpenAI spec). The GGUF jinja chat template renders them with
    ``| tojson``, so a string would be re-encoded into a double-quoted literal
    (``"{\\"k\\": 1}"``); the template needs a dict to emit a JSON object.
    Returns a shallow copy -- the canonical ``messages`` list is never mutated.
    """
    out: list[dict] = []
    for m in messages:
        tool_calls = m.get("tool_calls")
        if not tool_calls:
            out.append(m)
            continue
        new_calls = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
                tc = {**tc, "function": {**fn, "arguments": args}}
            new_calls.append(tc)
        out.append({**m, "tool_calls": new_calls})
    return out


class LocalLlamaBackend(ModelBackend):
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._lcfg = config.local
        arch = read_gguf_architecture(self._lcfg.model_path) if self._lcfg.model_path else None
        self._fmt = resolve_chat_format(config.backend.chat_handler, model_arch=arch)
        log.info("local model arch=%s -> chat_format=%s", arch, self._fmt.name)
        self._llama = self._load_model()
        self._system_prompt = ""
        self._tools: list[dict] = []
        self._warned_snapshot = False

    def _load_model(self):
        from llama_cpp import Llama

        if not self._lcfg.model_path:
            raise ValueError("local.model_path is not set")
        # chat_format=None lets llama_cpp auto-detect from the GGUF's template;
        # a registered handler (e.g. "chatml-function-calling") overrides it.
        chat_format = self._fmt.llama_chat_format
        try:
            return Llama(
                model_path=self._lcfg.model_path,
                n_gpu_layers=self._lcfg.n_gpu_layers,
                n_ctx=self._lcfg.n_ctx,
                n_batch=self._lcfg.n_batch,
                chat_format=chat_format,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            if self._lcfg.n_gpu_layers == 0:
                raise
            log.warning("GPU model load failed (%s); falling back to CPU", exc)
            return Llama(
                model_path=self._lcfg.model_path,
                n_gpu_layers=0,
                n_ctx=self._lcfg.n_ctx,
                n_batch=self._lcfg.n_batch,
                chat_format=chat_format,
                verbose=False,
            )

    @property
    def chat_format(self):
        return self._fmt

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_grammar=True,
            supports_guided_json=True,
            supports_native_tools=True,
            supports_kv_snapshot=self._lcfg.use_kv_snapshot,
            max_context=self._lcfg.n_ctx,
        )

    async def prepare_session(self, system_prompt: str, tools: list[dict]) -> None:
        self._system_prompt = system_prompt
        self._tools = tools
        if self._lcfg.use_kv_snapshot and not self._warned_snapshot:
            # Explicit save_state/load_state is opt-in and known-suspect on GPU
            # (#743). Until the release-gate test validates it on the pinned
            # version, we rely on implicit prefix caching and log the request.
            log.warning(
                "use_kv_snapshot is set but explicit snapshotting is not enabled in "
                "this build; relying on implicit prefix caching."
            )
            self._warned_snapshot = True

    def _build_grammar(self, constraint: OutputConstraint):
        from llama_cpp import LlamaGrammar

        if constraint.kind == "json_schema" and constraint.json_schema is not None:
            return LlamaGrammar.from_json_schema(json.dumps(constraint.json_schema))
        if constraint.kind == "grammar" and constraint.grammar:
            return LlamaGrammar.from_string(constraint.grammar)
        return None

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        constraint: OutputConstraint,
        sampling: SamplingParams,
    ) -> AsyncIterator[StreamDelta]:
        grammar = self._build_grammar(constraint)
        use_guided = grammar is not None
        # Some native templates (notably Harmony/gpt-oss and some Gemma stacks)
        # emit a model-family DSL in content instead of structured tool_calls.
        # Buffer that content and parse it ourselves only if no real tool_calls arrive.
        native_parser = None if use_guided else self._fmt.native_parser

        # The GGUF jinja template (llama_chat_format is None) tojson's tool-call
        # args, so it needs objects; built-in llama_cpp handlers expect the
        # OpenAI JSON-string form, so leave those untouched.
        msgs = _objectify_tool_args(messages) if self._fmt.llama_chat_format is None else messages

        kwargs: dict[str, Any] = {
            "messages": msgs,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "stream": True,
        }
        if sampling.top_k is not None:
            kwargs["top_k"] = sampling.top_k
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        if sampling.stop:
            kwargs["stop"] = sampling.stop
        if use_guided:
            kwargs["grammar"] = grammar
        elif tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def produce() -> None:
            try:
                for chunk in self._llama.create_chat_completion(**kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        loop.run_in_executor(None, produce)

        partial_tools: dict[int, dict[str, str]] = {}
        guided_text: list[str] = []
        native_text: list[str] = []
        stop_reason = "end_turn"

        while True:
            kind, payload = await queue.get()
            if kind == "error":
                raise payload
            if kind == "done":
                break
            choice = (payload.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason")

            if use_guided:
                if delta.get("content"):
                    guided_text.append(delta["content"])
            else:
                if delta.get("content"):
                    if native_parser is not None:
                        native_text.append(delta["content"])
                    else:
                        yield StreamDelta(text=delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = partial_tools.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

            if finish:
                stop_reason = map_finish_reason(finish)

        if use_guided and guided_text:
            tc = self._fmt.parse("".join(guided_text))
            if tc is not None:
                yield StreamDelta(tool_call=tc)
                stop_reason = "tool_use"
        else:
            emitted_tools = False
            for slot in partial_tools.values():
                if slot["name"]:
                    yield StreamDelta(tool_call=to_tool_call(slot))
                    stop_reason = "tool_use"
                    emitted_tools = True
            if not emitted_tools and native_parser is not None and native_text:
                turn = native_parser("".join(native_text))
                if turn.reasoning:
                    yield StreamDelta(reasoning=turn.reasoning)
                if turn.content:
                    yield StreamDelta(text=turn.content)
                if turn.tool_call is not None:
                    yield StreamDelta(tool_call=turn.tool_call)
                    stop_reason = "tool_use"

        yield StreamDelta(finished=True, stop_reason=stop_reason)

    async def aclose(self) -> None:
        close = getattr(self._llama, "close", None)
        if close is not None:
            close()
