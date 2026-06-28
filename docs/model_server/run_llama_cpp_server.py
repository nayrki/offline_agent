#!/usr/bin/env python3
"""Tiny llama-cpp-python server wrapper with an external Jinja chat template.

Edit the configuration block below, then run:

    python docs/model_server/run_llama_cpp_server.py

This starts the built-in OpenAI-compatible llama_cpp server, but overrides the
model's chat template at load time so you can test a local `.jinja` file
without patching the GGUF metadata first.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

# --- Configuration ---------------------------------------------------------
# Required: point this at the GGUF you want to serve.
MODEL_PATH = r""

# Optional: point this at a Jinja chat template file. Leave blank to use the
# template embedded in the GGUF.
TEMPLATE_PATH = r""

# The loaded context window.
N_CTX = 16384

# Server listen port.
PORT = 10101

# Optional knobs you may want to adjust.
HOST = "127.0.0.1"
MODEL_ALIAS = "local-model"
N_GPU_LAYERS = -1
# GPU placement: 0 = LLAMA_SPLIT_MODE_NONE (single GPU), 1 = LAYER, 2 = ROW.
SPLIT_MODE = 0
MAIN_GPU = 0
N_PARALLEL = 1
TYPE_K = None
TYPE_V = None
FLASH_ATTN = False
# The upstream Python server defaults this to True for compatibility, but that
# allocates a large fp32 logits buffer. For normal chat serving, keep it False.
LOGITS_ALL = False
VERBOSE = True

# Optional kwargs exposed to the Jinja template at render time. Use JSON (or a
# Python dict literal) if your template expects flags like enable_thinking.
CHAT_TEMPLATE_KWARGS = ""


def _parse_template_kwargs(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    loaders = (json.loads, ast.literal_eval)
    last_error: Exception | None = None
    for load in loaders:
        try:
            value = load(raw)
        except Exception as exc:  # noqa: BLE001 - convert into one clearer error
            last_error = exc
            continue
        if not isinstance(value, dict):
            raise ValueError("CHAT_TEMPLATE_KWARGS must be a JSON/Python object")
        return value
    raise ValueError(f"could not parse CHAT_TEMPLATE_KWARGS: {last_error}") from last_error


def _detokenize_special_token(llm, token_id: int) -> str:
    if token_id < 0:
        return ""
    return llm.detokenize([token_id], special=True).decode("utf-8")


def _build_model_settings_kwargs(model_path: str) -> dict[str, Any]:
    return {
        "model": model_path,
        "model_alias": MODEL_ALIAS,
        "n_ctx": N_CTX,
        "n_gpu_layers": N_GPU_LAYERS,
        "split_mode": SPLIT_MODE,
        "main_gpu": MAIN_GPU,
        "type_k": TYPE_K,
        "type_v": TYPE_V,
        "flash_attn": FLASH_ATTN,
        "logits_all": LOGITS_ALL,
        "verbose": VERBOSE,
    }


def _warn_unsupported_n_parallel() -> None:
    if N_PARALLEL != 1:
        sys.stderr.write(
            "Warning: N_PARALLEL is currently ignored. llama-cpp-python's Python "
            "server does not expose llama.cpp-style parallel request configuration "
            "yet.\n"
        )


def _install_template_patch(
    template_path: str, template_kwargs: dict[str, Any] | None
) -> None:
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    from llama_cpp.server import model as server_model

    original = server_model.LlamaProxy.load_llama_from_model_settings
    template = Path(template_path).read_text(encoding="utf-8")

    def patched(settings):
        llm = original(settings)
        eos_token_id = llm.token_eos()
        formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=_detokenize_special_token(llm, eos_token_id),
            bos_token=_detokenize_special_token(llm, llm.token_bos()),
            stop_token_ids=[eos_token_id] if eos_token_id >= 0 else None,
        ).to_chat_handler()

        if template_kwargs:
            base_handler = formatter

            def chat_handler_with_kwargs(*args, **kwargs):
                return base_handler(*args, **{**template_kwargs, **kwargs})

            formatter = chat_handler_with_kwargs

        llm.chat_handler = formatter
        llm.chat_format = None
        return llm

    server_model.LlamaProxy.load_llama_from_model_settings = staticmethod(patched)


def main() -> int:
    if not MODEL_PATH:
        sys.stderr.write(
            "Set MODEL_PATH in docs/model_server/run_llama_cpp_server.py first.\n"
        )
        return 2

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        sys.stderr.write(f"Model path does not exist: {model_path}\n")
        return 2

    template_path = TEMPLATE_PATH.strip()
    if template_path and not Path(template_path).exists():
        sys.stderr.write(f"Template path does not exist: {template_path}\n")
        return 2

    try:
        template_kwargs = _parse_template_kwargs(CHAT_TEMPLATE_KWARGS)
        import uvicorn
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import ModelSettings, ServerSettings
    except ModuleNotFoundError as exc:
        sys.stderr.write(
            f"Missing dependency: {exc.name}. Install llama-cpp-python[server] in this environment.\n"
        )
        return 1
    except ValueError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2

    if template_path:
        _install_template_patch(template_path, template_kwargs)
    _warn_unsupported_n_parallel()

    app = create_app(
        server_settings=ServerSettings(host=HOST, port=PORT),
        model_settings=[ModelSettings(**_build_model_settings_kwargs(str(model_path)))],
    )
    uvicorn.run(app, host=HOST, port=PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
