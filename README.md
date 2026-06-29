# offline-agent

`offline-agent` is a locally hosted agent for **Jupyter AI 3** that speaks the
**Agent Client Protocol (ACP)**. It is designed to run in the same Python
environment as JupyterLab, register as a Jupyter AI persona, and talk to either:

- a **remote OpenAI-compatible model server** (the default path), or
- a **local `llama-cpp-python` model** if you explicitly configure the local backend.

The package now ships with built-in default configuration, so a TOML file is
optional.

## Installation from scratch

This install flow uses:

- `requirements.txt` for the agent runtime dependencies
- `supplemental_requirements.txt` for the JupyterLab / Jupyter AI host dependencies
- an editable install of this repository so JupyterLab always sees your local code changes

### 1. Create a virtual environment

**Windows (cmd)**

```bat
cd C:\path\to\offline_agent
py -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

**Linux / macOS**

```bash
cd /path/to/offline_agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use Python **3.11 or newer**.

### 2. Install the runtime dependencies

```bash
pip install -r requirements.txt
```

This installs the lightweight agent runtime. It does **not** install
`llama-cpp-python`; that is only needed if you want the in-process local model
backend.

### 3. Install the Jupyter host dependencies

```bash
pip install -r supplemental_requirements.txt
```

This installs:

- `jupyterlab`
- `jupyter-ai`
- `jupyter-ai-acp-client`

So you do **not** need a separate Jupyter AI install command if you use
`supplemental_requirements.txt`.

### 4. Install this repository as an editable library

```bash
pip install -e .
```

Use editable install so changes under `src\offline_agent\` are picked up from
your working tree without reinstalling.

### 5. Start JupyterLab

```bash
jupyter lab
```

The **Offline Agent** persona should appear in Jupyter AI.

## Jupyter notebook tool access

The agent edits notebooks through Jupyter's MCP server, not by directly parsing
`.ipynb` files itself. Install and enable `jupyter-server-mcp` in the same
environment if you want notebook cell tools exposed to the agent.

Then point Jupyter AI at that MCP server using its MCP settings. A typical entry
looks like:

```json
{
  "mcpServers": {
    "jupyter": {
      "type": "http",
      "url": "http://localhost:3001/mcp"
    }
  }
}
```

The exact settings file location is owned by Jupyter AI / Jupyter Server for
your installation.

## Default behavior

If you do nothing else, the packaged defaults use the **remote** backend and
expect a local OpenAI-compatible server at:

```text
http://localhost:10101/v1
```

with:

- model id `gemma4`
- no required local GGUF path
- no required `offline_agent.toml`

## Optional: use a local `llama-cpp-python` model

`llama-cpp-python` is **not** required for the default remote-server flow.

Install it only if you want:

- `backend.mode = "local"`, or
- remote fallback into a local model

The project no longer uses the setup script as the primary install path in this
README. If you want the local backend, install `llama-cpp-python` separately in
this same environment and then configure `local.model_path`.

## Development workflow

After the initial install:

1. activate the virtual environment
2. make code changes in this repository
3. restart JupyterLab when you change persona or backend Python code

Reinstall is usually only needed if you change packaging metadata or dependency
definitions.

## Optional configuration (`offline_agent.toml`)

Configuration is optional. The agent loads settings in this order:

1. packaged defaults
2. `offline_agent.toml` in the current working directory
3. `OFFLINE_AGENT_*` environment variables

If you want JupyterLab launched from any directory to use a specific config
file, set:

**Windows (cmd)**

```bat
set OFFLINE_AGENT_CONFIG=C:\path\to\offline_agent.toml
jupyter lab
```

**Linux / macOS**

```bash
export OFFLINE_AGENT_CONFIG=/path/to/offline_agent.toml
jupyter lab
```

Example optional config:

```toml
[backend]
mode = "remote"
fallback_to_local = false
chat_handler = "auto"

[remote]
base_url = "http://localhost:10101/v1"
api_key = "EMPTY"
model = "gemma4"
reasoning_effort = "none"

[sampling]
temperature = 0.0
top_p = 1.0
max_tokens = 4096

[constraints]
tool_call_mode = "native"
max_tool_turns = 20
```
