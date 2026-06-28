"""Tests for the tiny llama-cpp-python server wrapper script."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from contextlib import redirect_stderr

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "model_server"
    / "run_llama_cpp_server.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_llama_cpp_server", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_template_kwargs_accepts_json():
    mod = _load_module()
    assert mod._parse_template_kwargs('{"enable_thinking": false}') == {
        "enable_thinking": False
    }


def test_parse_template_kwargs_accepts_python_literal():
    mod = _load_module()
    assert mod._parse_template_kwargs("{'preserve_thinking': True}") == {
        "preserve_thinking": True
    }


def test_parse_template_kwargs_rejects_non_mapping():
    mod = _load_module()
    with pytest.raises(ValueError):
        mod._parse_template_kwargs("[1, 2, 3]")


def test_model_settings_include_exposed_knobs():
    mod = _load_module()
    mod.SPLIT_MODE = 0
    mod.MAIN_GPU = 1
    mod.TYPE_K = 8
    mod.TYPE_V = 9
    mod.FLASH_ATTN = True
    mod.LOGITS_ALL = False
    settings = mod._build_model_settings_kwargs("model.gguf")
    assert settings["split_mode"] == 0
    assert settings["main_gpu"] == 1
    assert settings["type_k"] == 8
    assert settings["type_v"] == 9
    assert settings["flash_attn"] is True
    assert settings["logits_all"] is False


def test_n_parallel_warning_only_for_non_default():
    mod = _load_module()
    mod.N_PARALLEL = 4
    err = io.StringIO()
    with redirect_stderr(err):
        mod._warn_unsupported_n_parallel()
    assert "ignored" in err.getvalue()
