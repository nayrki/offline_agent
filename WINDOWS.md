# Running offline-agent on Windows

This covers the Windows-specific wrinkles for getting the **Offline Agent**
persona working inside JupyterLab. The agent itself is cross-platform; the
friction is all in the environment and in how JupyterLab spawns subprocesses on
Windows.

If you just want it working: from the repo root, in your project's Python
environment, run

```bat
install.bat --jupyter
```

and then launch `jupyter lab`. The sections below explain what that does and how
to fix things by hand if needed.

---

## 1. Use Python 3.11 or newer

The package targets `>=3.11`. Create the environment with a supported
interpreter:

```bat
:: with uv
uv venv .venv
:: or with stock venv
py -m venv .venv
```

> **Gotcha:** if `pip --version` prints a *different* Python than `python -V`
> or points at a different interpreter than the one you intend to use, an
> install will target the wrong environment.
> Make sure both commands point at the same virtual environment before
> installing.

## 2. Installing in a `uv` environment

`uv` venvs do **not** ship `pip`, so `python -m pip ...` fails and a bare `pip`
may resolve to some other Python on `PATH`. Use `uv pip`, which always targets
the active venv:

```bat
uv pip install -e ".[jupyter]"
```

(or `".[lab]"` to also pull JupyterLab + Jupyter AI into this env).

`jupyter-ai-acp-client` currently caps `agent-client-protocol<0.10`, so the
package metadata keeps `agent-client-protocol` broad enough (`>=0.9,<0.11`) for
the Jupyter extra to resolve cleanly under `uv` as well as `pip`.

## 3. The event-loop fix (the important one)

**Symptom** — the persona appears in the chat panel, but the first message dies
with:

```
File ".../jupyter_ai_acp_client/base_acp_persona.py", line 262, in get_client
    return await self.__class__._client_future
NotImplementedError
```

**Cause** — on Windows, `jupyter_server` forces the asyncio **Selector** event
loop ("for tornado + pyzmq", in `ServerApp._init_asyncio_patch`). But
`jupyter-ai-acp-client` launches the agent with
`asyncio.create_subprocess_exec`, and on Windows that works **only on the
Proactor loop** — under Selector it raises `NotImplementedError`. The agent
subprocess can never start.

**Fix** — force the Proactor loop back on in the Jupyter server config. Because
`_init_asyncio_patch()` runs *before* config files load and *before* the loop is
created, a `jupyter_server_config.py` can flip the policy and it sticks. pyzmq
still works on Proactor via tornado's `AddThreadSelectorEventLoop` shim
(tornado ≥ 6.1); you may see a one-time `RuntimeWarning` about an extra thread —
harmless.

`install.bat` applies this automatically (unless you pass `--no-win-patch`). To
apply or re-apply it on its own:

```bat
patch_windows.bat
```

This is **idempotent** — it appends a marked block to the Jupyter config only if
it isn't already there. It writes to the active environment's Jupyter config dir
(honoring `JUPYTER_CONFIG_DIR`); run `jupyter --paths` to see where that is. The
block it adds lives in `scripts\win_proactor_eventloop.py`.

### Doing it by hand

Add this to your `jupyter_server_config.py` (in the dir from `jupyter --paths`):

```python
import sys, asyncio
if sys.platform.startswith("win") and hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

### Undoing it

Delete the block between the
`# >>> offline-agent: Windows Proactor event-loop fix >>>` and
`# <<< ... <<<` markers from your `jupyter_server_config.py`.

## 4. Launch

```bat
jupyter lab
```

Open the Jupyter AI chat panel and pick (or @-mention) **Offline Agent**.

The agent always has packaged defaults. To override them for a specific
JupyterLab launch, either drop an `offline_agent.toml` into the server's working
directory or point it at one explicitly before launching:

```bat
set OFFLINE_AGENT_CONFIG=C:\path\to\offline_agent.toml
jupyter lab
```

---

## Troubleshooting quick reference

| Symptom | Cause | Fix |
| --- | --- | --- |
| `requires a different Python` during install | install is targeting the wrong interpreter | activate the intended env and install into that interpreter explicitly |
| `python -m pip` fails in a uv venv | uv venvs have no `pip` | use `uv pip ...` (or `python -m ensurepip` to add pip) |
| resolver conflict on `agent-client-protocol` | stale package metadata from before the ACP range was widened | update to the current checkout, then `uv pip install -e ".[jupyter]"` |
| `NotImplementedError` in `get_client` on first message | Jupyter forces the Selector loop; ACP needs Proactor for subprocesses | run `patch_windows.bat` (or `install.bat`), then restart `jupyter lab` |
| ACP `Internal error` with `{"details":"Connection error."}` on prompt | the configured OpenAI-compatible endpoint is unreachable or `[remote].base_url` is wrong | check the remote server, fix `[remote].base_url`, or switch `[backend] mode` to `local` / enable `fallback_to_local` |
| persona doesn't appear at all | package not installed in the env running `jupyter lab` (a bare clone isn't enough) | `uv pip install -e ".[jupyter]"` in that env |
