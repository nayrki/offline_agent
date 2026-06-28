from __future__ import annotations

from pathlib import Path

from offline_agent.config import load_config


def test_packaged_defaults_apply_without_cwd_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OFFLINE_AGENT_CONFIG", raising=False)

    cfg = load_config()

    assert cfg.backend.mode == "remote"
    assert cfg.backend.fallback_to_local is False
    assert cfg.remote.base_url == "http://localhost:10101/v1"
    assert cfg.remote.model == "gemma4"
    assert cfg.remote.reasoning_effort == "none"
    assert cfg.local.model_path == ""
    assert cfg.sampling.max_tokens == 4096


def test_cwd_toml_overrides_packaged_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OFFLINE_AGENT_CONFIG", raising=False)
    (tmp_path / "offline_agent.toml").write_text(
        '[remote]\nmodel = "cwd-model"\n[local]\nmodel_path = "deps/models/local.gguf"\n',
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.remote.model == "cwd-model"
    assert cfg.local.model_path == "deps/models/local.gguf"


def test_env_config_path_overrides_cwd_toml(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cwd_cfg = tmp_path / "offline_agent.toml"
    env_cfg = tmp_path / "env_offline_agent.toml"
    cwd_cfg.write_text('[remote]\nmodel = "cwd-model"\n', encoding="utf-8")
    env_cfg.write_text('[remote]\nmodel = "env-model"\n', encoding="utf-8")
    monkeypatch.setenv("OFFLINE_AGENT_CONFIG", str(Path(env_cfg)))

    cfg = load_config()

    assert cfg.remote.model == "env-model"
