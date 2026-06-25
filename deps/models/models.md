# Models

The `local` backend needs a **GGUF** weights file on disk. These are **not
vendored in git** — they run to many GB and exceed hosting file-size limits
(GitHub caps blobs/LFS objects at 2 GB; the default below is ~11.5 GB).

Download the GGUF you want **into this directory** (`deps/models/`) and point
`local.model_path` at it. `*.gguf` here is git-ignored, so your local weights
stay put and never get committed.

```toml
[local]
model_path = "deps/models/gpt-oss-20b-mxfp4.gguf"
```

## Vendored default (what this repo was developed/tested against)

- **gpt-oss-20b** (MXFP4 GGUF) — `gpt-oss-20b-mxfp4.gguf` (~11.5 GB)
  - GGUF: <https://huggingface.co/ggml-org/gpt-oss-20b-GGUF> (file
    `gpt-oss-20b-mxfp4.gguf`)
  - Base model: <https://huggingface.co/openai/gpt-oss-20b>
  - Harmony format — set `chat_handler = "gpt-oss"` (see README §2,
    "gpt-oss / Harmony").

## Other known-good GGUFs (instruct + tool-calling)

Pick an **instruct** GGUF whose chat template supports **tool calling** (see
README §2). Known-good families:

- **Llama-3.1-8B-Instruct** (recommended; ~8B fits ~16 GB, strong tool use)
  - GGUF: <https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF>
  - Official weights: <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
- **Mistral-7B-Instruct**
  - GGUF: <https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF>
- **Hermes / Functionary** builds (tool-aware templates)
  - <https://huggingface.co/NousResearch>
  - <https://huggingface.co/meetkai>

## Remote backend (LM Studio) — Gemma-4

The `remote` backend serves the model from your own endpoint instead of loading
a local GGUF. For the tested **LM Studio + Gemma-4** setup — including the
customized chat template that fixes agentic tool calling — see
[`docs/lmstudio-gemma4/`](../../docs/lmstudio-gemma4/README.md). That directory
is the source of truth for the template
(`custom_pub_chat_template_gemma4.jinja`) and ships an installer
(`apply-template.py`).

- Model: <https://huggingface.co/google/gemma-4-12b-qat>
