"""Unit tests for RemoteOpenAIBackend capability handling (no network)."""

from __future__ import annotations

import types

import pytest

from offline_agent.backends.remote_openai import RemoteOpenAIBackend
from offline_agent.config import Config


def _backend(**remote_over):
    cfg = Config()
    for k, v in remote_over.items():
        setattr(cfg.remote, k, v)
    return RemoteOpenAIBackend(cfg)


def _fake_client(create_impl):
    completions = types.SimpleNamespace(create=create_impl)
    chat = types.SimpleNamespace(completions=completions)
    return types.SimpleNamespace(chat=chat, close=_noop)


async def _noop(*a, **k):
    return None


async def _empty_stream():
    return
    yield  # pragma: no cover - makes this an async generator


@pytest.mark.asyncio
async def test_reasoning_effort_forwarded_in_extra_body():
    from offline_agent.backends.base import OutputConstraint, SamplingParams

    b = _backend(reasoning_effort="none")
    captured = {}

    async def create(*a, **k):
        captured.update(k)
        return _empty_stream()

    b._client = _fake_client(create)
    async for _ in b.stream([{"role": "user", "content": "hi"}], [], OutputConstraint.none(), SamplingParams()):
        pass
    assert captured.get("extra_body", {}).get("reasoning_effort") == "none"


@pytest.mark.asyncio
async def test_reasoning_effort_omitted_when_unset():
    from offline_agent.backends.base import OutputConstraint, SamplingParams

    b = _backend(reasoning_effort="")  # explicitly unset (default is "")
    captured = {}

    async def create(*a, **k):
        captured.update(k)
        return _empty_stream()

    b._client = _fake_client(create)
    async for _ in b.stream([{"role": "user", "content": "hi"}], [], OutputConstraint.none(), SamplingParams()):
        pass
    assert "reasoning_effort" not in (captured.get("extra_body") or {})


def test_force_native_disables_guided_and_skips_probe():
    b = _backend(force_native_tools=True)
    assert b.capabilities.supports_guided_json is False
    assert b.capabilities.supports_grammar is False
    assert b.capabilities.supports_native_tools is True
    assert b._probed is True


@pytest.mark.asyncio
async def test_probe_success_keeps_guided():
    b = _backend()

    async def ok_create(*a, **k):
        return object()

    b._client = _fake_client(ok_create)
    await b.prepare_session("sys", [])  # tool_call_mode default is native -> skips probe
    # Force a probe directly:
    await b._probe()
    assert b.capabilities.supports_guided_json is True
    assert b.capabilities.supports_grammar is True


@pytest.mark.asyncio
async def test_probe_failure_falls_back_to_native():
    b = _backend()

    async def bad_create(*a, **k):
        raise RuntimeError("guided_choice not supported")

    b._client = _fake_client(bad_create)
    await b._probe()
    assert b.capabilities.supports_guided_json is False
    assert b.capabilities.supports_grammar is False
    assert b.capabilities.supports_native_tools is True


def _reachable_client(list_impl):
    """A client whose ``with_options(...).models.list()`` runs ``list_impl``."""
    models = types.SimpleNamespace(list=list_impl)
    inner = types.SimpleNamespace(models=models)
    return types.SimpleNamespace(with_options=lambda **k: inner, close=_noop)


@pytest.mark.asyncio
async def test_is_reachable_true_on_models_list_ok():
    b = _backend()

    async def ok_list(*a, **k):
        return object()

    b._client = _reachable_client(ok_list)
    assert await b.is_reachable() is True


@pytest.mark.asyncio
async def test_is_reachable_false_on_connection_error():
    b = _backend()

    async def boom(*a, **k):
        raise ConnectionError("refused")

    b._client = _reachable_client(boom)
    assert await b.is_reachable() is False


@pytest.mark.asyncio
async def test_make_backend_uses_remote_when_reachable(monkeypatch):
    from offline_agent import backends

    cfg = Config()
    cfg.backend.mode = "remote"

    async def yes(self):
        return True

    monkeypatch.setattr(RemoteOpenAIBackend, "is_reachable", yes)
    monkeypatch.setattr(
        backends, "_local_backend", lambda c: pytest.fail("should not load local")
    )
    backend = await backends.make_backend(cfg)
    assert isinstance(backend, RemoteOpenAIBackend)


@pytest.mark.asyncio
async def test_make_backend_falls_back_to_local_when_unreachable(monkeypatch):
    from offline_agent import backends

    cfg = Config()
    cfg.backend.mode = "remote"
    cfg.backend.fallback_to_local = True

    async def no(self):
        return False

    sentinel = object()
    monkeypatch.setattr(RemoteOpenAIBackend, "is_reachable", no)
    monkeypatch.setattr(backends, "_local_backend", lambda c: sentinel)
    backend = await backends.make_backend(cfg)
    assert backend is sentinel


@pytest.mark.asyncio
async def test_make_backend_keeps_remote_when_fallback_disabled(monkeypatch):
    from offline_agent import backends

    cfg = Config()
    cfg.backend.mode = "remote"
    cfg.backend.fallback_to_local = False

    async def no(self):
        return False

    monkeypatch.setattr(RemoteOpenAIBackend, "is_reachable", no)
    monkeypatch.setattr(
        backends, "_local_backend", lambda c: pytest.fail("should not load local")
    )
    backend = await backends.make_backend(cfg)
    assert isinstance(backend, RemoteOpenAIBackend)


def test_sampling_kwargs_splits_top_k_into_extra_body():
    from offline_agent.backends.base import SamplingParams

    b = _backend()
    kwargs, extra = b._sampling_kwargs(SamplingParams(temperature=0.3, top_k=40, seed=7))
    assert kwargs["temperature"] == 0.3
    assert kwargs["seed"] == 7
    assert extra["top_k"] == 40
