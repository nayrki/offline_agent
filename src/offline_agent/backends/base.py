"""The ``ModelBackend`` interface the agent loop is written against.

Both the local (in-process llama.cpp) and remote (openai SDK) backends normalize
to the same async ``StreamDelta`` stream so ``agent/loop.py`` is backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal

if TYPE_CHECKING:
    from ..chat_formats import ChatFormat


@dataclass(frozen=True)
class Capabilities:
    supports_grammar: bool          # GBNF / vLLM guided_grammar
    supports_guided_json: bool      # JSON-schema-constrained decoding
    supports_native_tools: bool     # OpenAI-style tools= function calling
    supports_kv_snapshot: bool      # save_state/load_state prefix reuse (local only)
    max_context: int


@dataclass(frozen=True)
class OutputConstraint:
    """A normalized constraint; each backend maps it to its native mechanism."""

    kind: Literal["none", "grammar", "json_schema"] = "none"
    grammar: str | None = None          # GBNF text          (kind == "grammar")
    json_schema: dict | None = None     # JSON Schema object  (kind == "json_schema")

    @classmethod
    def none(cls) -> "OutputConstraint":
        return cls(kind="none")


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 2048
    seed: int | None = None
    stop: list[str] | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        # The agent loop treats arguments as a dict (membership tests, ``.get``,
        # ``json.dumps`` to re-render the call). Some models emit non-object JSON
        # for a no-argument call -- notably gpt-oss/Harmony renders ``null`` --
        # which ``json.loads`` turns into ``None``/a scalar/a list. Coerce any
        # non-dict to an empty dict so a valid tool call never crashes the loop.
        # (Parsers that detect *malformed* JSON wrap it as ``{"__raw__": ...}``,
        # which is a dict and is preserved here.)
        if not isinstance(self.arguments, dict):
            self.arguments = {}


@dataclass
class StreamDelta:
    """A single increment from a backend stream.

    Exactly one of ``text``/``reasoning``/``tool_call`` is typically set; the
    final delta sets ``finished=True`` and ``stop_reason``. ``reasoning`` is the
    model's chain-of-thought (gpt-oss Harmony ``analysis`` channel, or a remote
    reasoning model's ``reasoning_content``) -- surfaced as an ACP *thought*, not
    folded into the user-visible answer.
    """

    text: str | None = None
    reasoning: str | None = None
    tool_call: ToolCall | None = None
    finished: bool = False
    stop_reason: str | None = None


class ModelBackend(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @property
    @abstractmethod
    def chat_format(self) -> "ChatFormat":
        """The tool-call format resolved for THIS backend's active model.

        Single source of truth for the session: when ``chat_handler = "auto"``
        the local and remote backends resolve their own model's format, so the
        agent loop must read it from the active backend rather than re-deriving
        it from the (single, possibly mismatched) config string."""

    @abstractmethod
    async def prepare_session(self, system_prompt: str, tools: list[dict]) -> None:
        """Freeze the stable prefix for prompt caching.

        Local: optionally prefill ``[system + tools]`` once and snapshot the KV
        state. Remote: a no-op beyond caching the frozen tool list (the server's
        automatic prefix caching handles reuse)."""

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        tools: list[dict],
        constraint: OutputConstraint,
        sampling: SamplingParams,
    ) -> AsyncIterator[StreamDelta]:
        """Stream the model's response for ``messages``.

        ``tools`` MUST be the same frozen list every turn (single source of truth;
        never also embedded in the system prompt) so the cacheable prefix stays
        byte-identical."""

    @abstractmethod
    async def aclose(self) -> None: ...
