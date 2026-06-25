"""Configuration schema and loader.

Config is read (in order of increasing precedence) from defaults, an
``offline_agent.toml`` file, and ``OFFLINE_AGENT_*`` environment variables.
The TOML path may be overridden with ``OFFLINE_AGENT_CONFIG``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _config_path() -> Path:
    return Path(os.environ.get("OFFLINE_AGENT_CONFIG", "offline_agent.toml"))


class LocalConfig(BaseModel):
    """In-process ``llama_cpp.Llama`` backend settings."""

    model_path: str = ""
    n_gpu_layers: int = -1  # -1 = offload all layers; runtime falls back to 0 on failure
    n_ctx: int = 32768
    n_batch: int = 512
    # Explicit save_state/load_state prefix snapshotting. OFF by default because it
    # can produce garbage with n_gpu_layers > 0 (llama-cpp-python #743). Enable only
    # after the GPU-correctness release test passes on the pinned version.
    use_kv_snapshot: bool = False


class RemoteConfig(BaseModel):
    """OpenAI-compatible remote endpoint (vLLM, LM Studio, Ollama, ...)."""

    # Default targets a local OpenAI-compatible server. LM Studio's default port
    # is 1234; Ollama's OpenAI-compat shim is on 11434. Plain chat-completions
    # only -- no server-specific extensions are assumed.
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "EMPTY"  # ignored by local servers, but the openai SDK requires a value
    model: str = ""
    guided_decoding_backend: str = "auto"  # vLLM: auto | xgrammar | guidance
    # Reasoning/thinking budget for reasoning models (gpt-oss, Gemma-qat,
    # DeepSeek, ...), sent as ``reasoning_effort`` in the request body. Empty
    # leaves the server default untouched. "none" disables thinking entirely --
    # the practical fix when a model burns the whole token budget reasoning about
    # otherwise-simple asks. Graduated values ("minimal"/"low"/"medium"/"high")
    # are server-dependent. Ignored by servers that don't recognize the field.
    reasoning_effort: str = ""
    # Skip the guided-decoding probe and go straight to native tools= + validate-retry.
    force_native_tools: bool = False
    request_timeout: float = 120.0
    # Short timeout for the liveness probe that decides remote-vs-local failover,
    # so an offline server fails fast instead of stalling on request_timeout.
    connect_timeout: float = 5.0


class SamplingConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 2048
    seed: int | None = None
    stop: list[str] | None = None


class ConstraintsConfig(BaseModel):
    # How tool calls are produced and enforced.
    #   "native"      -- model's built-in tool calling (tools= param). On vLLM and
    #                    llama.cpp this already constrains arg JSON to the schema;
    #                    a validate-and-retry wrapper covers servers that don't.
    #   "json_schema" -- force a single tool-call envelope via guided_json /
    #                    llama from_json_schema (no free-text turn).
    #   "grammar"     -- force via raw GBNF / vLLM guided_grammar.
    tool_call_mode: Literal["native", "json_schema", "grammar"] = "native"
    max_tool_retries: int = 2
    max_tool_turns: int = 20  # safety bound on the agent loop


class FsConfig(BaseModel):
    allow_write: bool = True


class BackendConfig(BaseModel):
    # Default to the remote OpenAI-compatible endpoint; if it is unreachable at
    # session start and fallback_to_local is set, make_backend transparently
    # loads the in-process llama.cpp model instead.
    mode: Literal["local", "remote"] = "remote"
    # When mode == "remote" and the endpoint fails a liveness probe, fall back to
    # the local llama.cpp backend instead of erroring. No effect in local mode.
    fallback_to_local: bool = True
    # Keys into the chat_formats registry (offline_agent.chat_formats). Selects
    # the llama_cpp chat_format for native tool calling and the forced tool-call
    # envelope. "auto" (default) derives it from the ACTIVE model -- the GGUF
    # general.architecture (local) or the served model id (remote) -- so the
    # local fallback and remote model each get their own handler instead of one
    # shared string. Explicit overrides: "default" (GGUF's own template),
    # "llama3", "mistral", "gemma", "functionary", "chatml",
    # "chatml-function-calling", "gpt-oss" (Harmony parsed in-process; see
    # chat_formats/harmony.py).
    chat_handler: str = "auto"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OFFLINE_AGENT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    backend: BackendConfig = Field(default_factory=BackendConfig)
    local: LocalConfig = Field(default_factory=LocalConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    fs: FsConfig = Field(default_factory=FsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: init kwargs > env > TOML file > defaults.
        toml = TomlConfigSettingsSource(settings_cls, toml_file=_config_path())
        return (init_settings, env_settings, toml)


def load_config() -> Config:
    return Config()
