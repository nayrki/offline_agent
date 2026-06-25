"""Model backends: one agent loop, swappable local/remote inference."""

from __future__ import annotations

import logging

from ..config import Config
from .base import (
    Capabilities,
    ModelBackend,
    OutputConstraint,
    SamplingParams,
    StreamDelta,
    ToolCall,
)

log = logging.getLogger(__name__)


def _local_backend(config: Config) -> ModelBackend:
    from .local_llama import LocalLlamaBackend

    return LocalLlamaBackend(config)


async def make_backend(config: Config) -> ModelBackend:
    """Select and construct the configured backend.

    In ``remote`` mode the OpenAI-compatible endpoint is probed for liveness;
    if it is unreachable and ``backend.fallback_to_local`` is set, the in-process
    llama.cpp backend is loaded instead. This is the default path: a configured
    network server is preferred, with the local model as an offline safety net.
    """
    from ._echo import echo_backend_from_env

    stub = echo_backend_from_env()
    if stub is not None:  # test/debug affordance, opt-in via env var
        return stub
    if config.backend.mode == "local":
        return _local_backend(config)
    elif config.backend.mode == "remote":
        from .remote_openai import RemoteOpenAIBackend

        remote = RemoteOpenAIBackend(config)
        if await remote.is_reachable() or not config.backend.fallback_to_local:
            return remote
        log.warning(
            "remote endpoint %s unreachable; falling back to local model %s",
            config.remote.base_url,
            config.local.model_path,
        )
        await remote.aclose()
        return _local_backend(config)
    raise ValueError(f"unknown backend mode: {config.backend.mode!r}")


__all__ = [
    "Capabilities",
    "ModelBackend",
    "OutputConstraint",
    "SamplingParams",
    "StreamDelta",
    "ToolCall",
    "make_backend",
]
