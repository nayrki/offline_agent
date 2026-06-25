"""Per-model tool-format registry, keyed by ``backend.chat_handler``.

A ``ChatFormat`` captures the two model-family-specific knobs the agent needs:

  * ``llama_chat_format`` -- the ``llama_cpp`` chat format used for **native**
    tool calling (``tools=``). ``None`` means "use the template embedded in the
    GGUF". Ignored by the remote backend (the vLLM server owns its template).
  * ``name_key`` / ``args_key`` -- the field names of the **forced** tool-call
    envelope (``json_schema``/``grammar`` modes). The schema we constrain the
    model to and the parser that reads its output are built from the same pair,
    so they never drift.

Register a custom format with :func:`register`; select one via
``[backend] chat_handler = "<name>"`` in config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..backends._parsing import parse_envelope
from ..backends.base import ToolCall
from ..backends.grammars import tool_call_schema
from .harmony import NativeTurn, parse_harmony


@dataclass(frozen=True)
class ChatFormat:
    name: str
    # llama_cpp chat_format for native tool calling; None => GGUF's own template.
    llama_chat_format: str | None = None
    # Forced-envelope field names (json_schema / grammar modes).
    name_key: str = "name"
    args_key: str = "arguments"
    # Optional parser for models whose native output is NOT OpenAI tool_calls
    # (e.g. Harmony/gpt-oss). When set, the local backend buffers the raw
    # assistant text and runs this instead of trusting llama_cpp's tool_calls.
    native_parser: Callable[[str], NativeTurn] | None = None

    def envelope_schema(self, tools: list[dict]) -> dict:
        return tool_call_schema(tools, name_key=self.name_key, args_key=self.args_key)

    def parse(self, text: str) -> ToolCall | None:
        return parse_envelope(text, name_key=self.name_key, args_key=self.args_key)


_REGISTRY: dict[str, ChatFormat] = {}


def register(fmt: ChatFormat, *, replace: bool = False) -> ChatFormat:
    """Add a chat format. Raises on a duplicate name unless ``replace=True``."""
    if fmt.name in _REGISTRY and not replace:
        raise ValueError(f"chat_handler {fmt.name!r} is already registered")
    _REGISTRY[fmt.name] = fmt
    return fmt


def get_chat_format(name: str) -> ChatFormat:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown chat_handler {name!r}; available: {', '.join(available())}"
        ) from None


def available() -> list[str]:
    return sorted(_REGISTRY)


# Auto-detection rules for ``chat_handler = "auto"``: ordered
# (substring-in-identifier -> registered handler name). The identifier is the
# GGUF ``general.architecture`` (local backend) or the served model id (remote
# backend), lowercased. First match wins; anything unmatched falls through to
# ``default`` -- i.e. trust the server's / GGUF template's own native tool
# calling. Only families whose native output is NOT OpenAI tool_calls need a
# rule here; today that is just gpt-oss (Harmony).
_AUTO_RULES: tuple[tuple[str, str], ...] = (
    ("gpt-oss", "gpt-oss"),
    ("gpt_oss", "gpt-oss"),
)


def resolve_chat_format(
    handler: str, *, model_arch: str | None = None, model_id: str | None = None
) -> ChatFormat:
    """Resolve the :class:`ChatFormat` for the *active* model.

    A ``handler`` other than ``"auto"`` is an explicit override, returned (and
    validated) as-is. ``"auto"`` derives the format from the model identifier --
    the GGUF ``general.architecture`` for the local backend, or the served model
    id for the remote backend -- so a gpt-oss model gets the Harmony parser while
    every other family falls through to ``default`` (server-/template-native tool
    calling). This lets the local fallback and the remote model each resolve
    their own format instead of being forced onto one shared, mismatched handler.
    """
    if handler != "auto":
        return get_chat_format(handler)
    ident = (model_arch or model_id or "").lower()
    for needle, name in _AUTO_RULES:
        if needle in ident:
            return get_chat_format(name)
    return get_chat_format("default")


# --- Built-in formats -------------------------------------------------------
# All use the portable {"name", "arguments"} forced envelope; they differ only
# in the native llama_cpp chat_format. Names are validated against the formats
# llama_cpp registers (chatml-function-calling, llama-3, mistral-instruct, ...).

# Pass-through: rely on the GGUF's embedded template for native tool calling.
# Byte-for-byte identical to the pre-registry behavior.
register(ChatFormat("default"))

# Generic OpenAI-style function calling for any ChatML model whose GGUF template
# lacks tool support -- the most useful non-default for local native mode.
register(ChatFormat("chatml-function-calling", llama_chat_format="chatml-function-calling"))

# Model-family templates.
register(ChatFormat("chatml", llama_chat_format="chatml"))
register(ChatFormat("llama3", llama_chat_format="llama-3"))
register(ChatFormat("mistral", llama_chat_format="mistral-instruct"))
register(ChatFormat("gemma", llama_chat_format="gemma"))
register(ChatFormat("functionary", llama_chat_format="functionary-v2"))

# gpt-oss speaks Harmony, not OpenAI tool_calls. Render with the GGUF's own
# template (llama_chat_format=None) and parse the channel output ourselves --
# llama_cpp 0.3.26 has no Harmony handler. See chat_formats/harmony.py.
register(ChatFormat("gpt-oss", native_parser=parse_harmony))


__all__ = [
    "ChatFormat",
    "NativeTurn",
    "register",
    "get_chat_format",
    "resolve_chat_format",
    "available",
]
