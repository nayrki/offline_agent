# offline-agent

A locally-hosted, fully-offline-installable agentic coding agent that plugs into
**Jupyter AI 3** over the **Agent Client Protocol (ACP)**.

- **One agent loop, two model backends** behind a `ModelBackend` abstraction:
  - `local` — in-process `llama_cpp.Llama`, built from source for the local GPU
    (CPU fallback inherent).
  - `remote` — the `openai` SDK pointed at any OpenAI-compatible endpoint (e.g.
    vLLM), with capability detection + graceful fallback.
- **Notebook tools** come from Jupyter's own MCP HTTP server (`jupyter-server-mcp`);
  this agent is an MCP *client*. Non-notebook file read/write use ACP client
  capabilities.
- **Offline install:** `offline-agent-install` builds `llama-cpp-python` from a
  vendored sdist against the locally-detected GPU compute capability, then
  verifies the build can offload to the GPU.

See `claude.md` for the original design constraints.

---

## 1. Install

**Quick start** — the root `install.sh` (or `install.bat` on Windows) ties the
steps below together. By default it installs only the lightweight agent package
and prompts for the local model server address (blank keeps the current value);
the `llama-cpp-python` runtime is opt-in:

```bash
./install.sh                  # agent package only (no model runtime) + configure server
./install.sh --llama          # also build llama-cpp-python with CUDA offload
./install.sh --cpu            # also build llama-cpp-python, CPU-only (implies --llama)
./install.sh --jupyter        # also pull the ACP bridge (jupyter extra)
./install.sh --lab            # also pull the full JupyterLab + Jupyter AI host stack
./install.sh --offline        # air-gapped llama build from deps/ (implies --llama)
./install.sh --server-url URL # set the remote server address non-interactively
./install.sh --help
```

On Windows, use the batch equivalent with the same flags:

```bat
install.bat                  :: agent package only + configure server
install.bat --llama          :: also build llama-cpp-python with CUDA offload
install.bat --lab            :: also pull the full JupyterLab + Jupyter AI host stack
install.bat --server-url URL :: set the remote server address non-interactively
install.bat --help
```

`install.bat` also applies a required Windows event-loop fix by default (so the
agent subprocess can spawn under JupyterLab). See **[WINDOWS.md](WINDOWS.md)** for
the Windows setup details, the uv/Python-version gotchas, and troubleshooting.

For a plain pip flow, `requirements.txt` mirrors the runtime deps and includes
the CUDA llama-cpp line (`llama-cpp-python -C cmake.args="-DGGML_CUDA=on"`):

```bash
pip install -r requirements.txt
```

The sections below cover each piece in detail.

### 1a. The agent package

The agent subprocess is intentionally lightweight (no GPU deps). Install it into
whatever environment runs your **Jupyter Server**:

```bash
pip install -e ".[jupyter]"     # pulls jupyter-ai-acp-client so the persona registers
```

`[jupyter]` is required only in the JupyterLab environment — it pulls
`jupyter-ai-acp-client` so the persona can register. A remote-only node that just
launches the agent over stdio needs only the base package.

### 1b. The local model runtime (offline build)

If you want the **`local`** backend, build `llama-cpp-python` from the vendored
sdist. The installer autodetects the GPU and toolkit, compiles against the right
compute capability, and runs a postinstall check:

```bash
offline-agent-install            # detect GPU + nvcc, build from deps/, verify
offline-agent-install --cpu      # force a CPU-only build
offline-agent-install --offline-only   # never touch the network (air-gapped)
offline-agent-install --dry-run        # print the commands without running them
```

What it does, in order:

1. Detect OS/arch, GPU compute capability (`nvidia-smi --query-gpu=compute_cap`),
   and locate `nvcc` — including toolkits **not on `PATH`** (`CUDA_HOME`,
   `/usr/local/cuda*`, Windows `Program Files`).
2. Install the runtime closure from `deps/runtime/` (offline, network fallback).
3. Install the build backend (`scikit-build-core`, `cmake`, `ninja`) from
   `deps/build/`.
4. **Build `llama-cpp-python`** from `deps/sdist/` with platform-targeted
   `CMAKE_ARGS` (`-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=<cc>` for GPU, CPU
   otherwise).
5. **Postinstall verification:** import `llama_cpp` in a fresh interpreter and
   assert a GPU target actually reports `llama_supports_gpu_offload()`. A
   silently-CPU build fails the install loudly.

> **Air-gapped targets:** run `python scripts/populate_deps.py` once on a
> network-connected machine *of the same OS* to fill `deps/{sdist,build,runtime}/`,
> copy the repo across, then `offline-agent-install --offline-only`. Only the
> llama-cpp-python sdist is platform-agnostic; the wheels are per-OS, so populate
> on each target platform (Linux+CUDA, Windows+CUDA, Linux/CPU).

A CUDA build still runs on CPU at runtime (offload is opt-in), and the loader
falls back to `n_gpu_layers=0` if GPU init fails — so a GPU build degrades
gracefully on a machine without a usable GPU.

---

## 2. Get the model

The `local` backend needs a **GGUF** file. The `remote` backend needs a model
served by your endpoint (see §4).

Pick an **instruct** GGUF whose chat template supports **tool calling** — the
agent works exclusively by calling tools, so a base/completion model or a chat
model without a tool-aware template will not drive the loop. Known-good families:

- **Llama-3.1-8B-Instruct** (recommended; 8B fits ~16 GB, strong tool use)
- Hermes / functionary builds
- Mistral-Instruct / Gemma-Instruct
- gpt-oss-20b (Harmony format — set `chat_handler = "gpt-oss"`, see below)

Download the `.gguf` (e.g. `meta-llama-3.1-8b-instruct-q4_k_m.gguf`) into
`deps/models/` and note its path for `local.model_path`. Weights are **not
vendored in git** (too large to host) — see
[`deps/models/models.md`](deps/models/models.md) for download links to the
HuggingFace repositories.

### Model templating (read this if tool calls misbehave)

By default (`chat_handler = "default"`) the local backend passes `tools=` to
`llama_cpp` and relies on the **chat template embedded in the GGUF** to render
those tools into the model's native tool-call format. Most instruct GGUFs ship a
correct template, so nothing extra is needed.

**The `chat_handler` registry** (`offline_agent.chat_formats`) covers models whose
GGUF lacks a tool-aware template, or whose family needs a specific `llama_cpp`
chat format. Set `[backend] chat_handler` to one of:

| `chat_handler` | llama_cpp chat format | use for |
|---|---|---|
| `default` | *(GGUF's own template)* | any instruct GGUF with a tool-capable template |
| `chatml-function-calling` | `chatml-function-calling` | a ChatML model whose template lacks tool support — adds generic OpenAI-style function calling |
| `llama3` | `llama-3` | Llama 3.x Instruct |
| `mistral` | `mistral-instruct` | Mistral Instruct |
| `gemma` | `gemma` | Gemma |
| `functionary` | `functionary-v2` | Functionary |
| `chatml` | `chatml` | plain ChatML (no tool calling) |
| `gpt-oss` | *(GGUF's own template)* | gpt-oss / OpenAI Harmony — see note below |

```toml
[backend]
chat_handler = "chatml-function-calling"   # e.g. a ChatML GGUF without a tool template
```

**gpt-oss / Harmony.** gpt-oss models don't emit OpenAI `tool_calls` — they speak
the [Harmony](https://github.com/openai/harmony) format (reasoning / tool calls /
answer split across `analysis` / `commentary` / `final` channels). The pinned
`llama_cpp` (0.3.26) has no Harmony handler, so `create_chat_completion` returns
the raw channel text with `tool_calls = None` (this is why native tool calling
"silently does nothing" on gpt-oss — `llama-server` and Ollama work only because
they ship the parser in C++). The `gpt-oss` handler closes that gap: it renders
the prompt with the GGUF's own Harmony template and parses the channel output
in-process (`chat_formats/harmony.py`). Two consequences worth knowing:

- The assistant turn is **buffered, not token-streamed** (channel markers can't
  be parsed mid-stream), and the `analysis` chain-of-thought is elided from the
  message. Keep `[sampling] max_tokens` generous — gpt-oss reasons before the
  tool call, and a low cap can truncate the call itself.
- Leave `[constraints] tool_call_mode = "native"`. The forced-envelope modes
  (`json_schema`/`grammar`) would suppress the reasoning channel and fight the
  model's training; the `gpt-oss` handler is the better path on this stack.

```toml
[backend]
chat_handler = "gpt-oss"
[local]
model_path = "/models/gpt-oss-20b-mxfp4.gguf"
```

The handler controls **both** the native `llama_cpp` chat format *and* the field
names of the forced tool-call envelope (so the constrained schema and the parser
that reads it never drift). The remote backend uses only the envelope half — the
vLLM server owns its own template.

Register a custom one (e.g. a `{"tool", "tool_input"}` envelope) at import time:

```python
from offline_agent.chat_formats import ChatFormat, register
register(ChatFormat("my-model", llama_chat_format="chatml",
                    name_key="tool", args_key="tool_input"))
```

If tool calling is still unreliable, **force a structured envelope** instead of
trusting the model's native format — model-agnostic, constrained at the decoder:

```toml
[constraints]
tool_call_mode = "grammar"      # raw GBNF (local: LlamaGrammar; remote: vLLM guided_grammar)
# or
tool_call_mode = "json_schema"  # JSON-schema envelope (local: from_json_schema; remote: guided_json)
```

---

## 3. Configure

Config is read (increasing precedence) from **defaults → `offline_agent.toml` →
`OFFLINE_AGENT_*` env vars**. Override the file path with `OFFLINE_AGENT_CONFIG`.
Copy the example and edit:

```bash
cp offline_agent.toml.example offline_agent.toml
```

```toml
[backend]
mode = "local"              # "local" (in-process llama.cpp) | "remote" (OpenAI-compatible)
chat_handler = "default"    # per-model tool format (see "Model templating" above)

[local]
model_path  = "/models/meta-llama-3.1-8b-instruct-q4_k_m.gguf"
n_gpu_layers = -1           # -1 = offload all; runtime falls back to 0 (CPU) on failure
n_ctx       = 32768
n_batch     = 512
use_kv_snapshot = false     # explicit save_state prefix reuse (opt-in; suspect on GPU, #743)

[remote]
base_url = "http://10.0.0.5:8000/v1"     # your vLLM / OpenAI-compatible server
api_key  = "EMPTY"                        # vLLM ignores it; the openai SDK requires a value
model    = "meta-llama/Llama-3.1-8B-Instruct"
guided_decoding_backend = "auto"          # vLLM: auto | xgrammar | guidance
force_native_tools = false                # skip the guided-decoding probe; use tools= directly
request_timeout = 120.0

[sampling]
temperature = 0.0
top_p = 1.0
max_tokens = 2048
seed = 0

[constraints]
tool_call_mode = "native"   # "native" | "json_schema" | "grammar"  (see §2)
max_tool_retries = 2
max_tool_turns = 20         # safety bound on the agent loop

[fs]
allow_write = true          # allow write_file (e.g. notebook -> .py export); reads always allowed
```

Every key is also an env var, nested with `__`:

```bash
OFFLINE_AGENT_BACKEND__MODE=remote
OFFLINE_AGENT_REMOTE__BASE_URL=http://10.0.0.5:8000/v1
OFFLINE_AGENT_LOCAL__MODEL_PATH=/models/llama.gguf
```

The agent subprocess loads this config from **its working directory** (the
JupyterLab server's cwd), so put `offline_agent.toml` there or point
`OFFLINE_AGENT_CONFIG` at an absolute path via the Jupyter server's environment.

---

## 4. Remote backend (vLLM / OpenAI-compatible)

Use this to offload inference to a beefier shared box (e.g. a 40 GB GPU) while
desktops stay light. Set `[backend] mode = "remote"` and point `[remote] base_url`
at the server.

On the server, launch vLLM with **tool calling enabled** — this is the remote
equivalent of the GGUF template requirement:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json   # match the parser to the model family
    # --chat-template <file>         # only if the model's template lacks tools
```

The backend **probes** the endpoint once per session for guided decoding
(`guided_grammar`/`guided_json`). If present, `tool_call_mode = "grammar"` /
`"json_schema"` constrains output server-side; if the probe 400s (or you set
`force_native_tools = true`), it falls back to native `tools=` with
validate-and-retry. vLLM's automatic prefix caching gives the same
cached-prefix benefit as the local backend's implicit KV reuse.

---

## 5. Use it in JupyterLab

1. **Install** the package with `[jupyter]` into the Jupyter Server env (§1a).
2. **Enable the notebook MCP server.** Install and enable `jupyter-server-mcp` so
   Jupyter exposes its notebook tools over streamable HTTP (default
   `http://localhost:3001/mcp`).
3. **Point Jupyter AI at that MCP server.** Add it to `.jupyter/mcp_settings.json`
   so Jupyter AI passes it to the agent on `new_session`:

   ```json
   {
     "mcpServers": {
       "jupyter": { "type": "http", "url": "http://localhost:3001/mcp" }
     }
   }
   ```

   (The exact file location/schema is owned by Jupyter AI — follow its MCP
   settings docs for your version; the entry above is the `http` server shape it
   passes through to the agent.)

   The agent never starts an MCP server itself — it connects as a client to
   whatever Jupyter hands it on `new_session`, freezes that tool list once per
   session (a stable cache prefix), and adds `read_file`/`write_file` via ACP fs
   capabilities.
4. **Launch JupyterLab.** "Offline Agent" appears as a persona in the Jupyter AI
   chat panel.
5. **Chat.** Ask it to work on the current notebook — it edits at the **cell**
   level through whatever tools `jupyter-server-mcp` exposes (e.g.
   `read_notebook`/`add_cell`/`edit_cell`/`delete_cell`) and uses `write_file`
   only for non-notebook output (e.g. "export this notebook to a `.py` library").

Switch a session between `local` and `remote` by editing `[backend] mode` (or
`OFFLINE_AGENT_BACKEND__MODE`) and restarting the agent — the persona relaunches
the subprocess, which reloads config.

---

## 6. Run / debug standalone (no Jupyter)

The agent speaks ACP over stdio, so you can drive it without JupyterLab:

```bash
offline-agent        # reads offline_agent.toml from the cwd; logs to stderr
```

For a stub run with no model or GPU, the backend can be replaced via
`OFFLINE_AGENT_STUB_BACKEND=1` (used by the test suite). See `tests/` for an ACP
stdio smoke client and backend/grammar/install tests:

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Architecture at a glance

```
JupyterLab ──ACP──> OfflineAgentPersona ──spawns──> offline-agent (this package)
   │                                                      │
   │  .jupyter/mcp_settings.json                          ├─ MCP client ──HTTP──> jupyter-server-mcp (:3001)
   └──────────────────────────────────────────────┐      │                         (notebook cell tools)
                                                   │      ├─ ACP fs ── read_file / write_file
                                                   │      │
                                       new_session(mcp_servers=...)  └─ ModelBackend
                                                                          ├─ local : in-process llama_cpp.Llama (GGUF)
                                                                          └─ remote: openai SDK -> vLLM / OpenAI-compatible
```
