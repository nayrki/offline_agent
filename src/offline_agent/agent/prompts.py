"""Static system-prompt construction.

The system prompt MUST stay byte-identical across every turn in a session so the
cacheable prefix (system + frozen tools) is reused. It therefore contains only
stable, per-session content (e.g. cwd) -- never timestamps, selected-cell text,
or the tool list (tools are the ``tools=`` param, the single source of truth).
"""

from __future__ import annotations

_BASE = """\
You are Offline Agent, a locally-hosted coding assistant working inside a \
JupyterLab notebook. You complete the user's request by calling the available \
tools and then giving a concise final answer.

Your full set of callable tools is provided to you directly each turn; rely on \
that list rather than any assumption about which tools exist, and pick the most \
specific tool for the task. The tools cover notebook operations (reading, \
adding, inserting, editing, deleting, selecting, and running cells; creating \
and inspecting notebooks and open documents; running notebook commands) and \
workspace file access.

Guidelines:
- For work on the current notebook, use the notebook tools and edit at the \
granularity of individual cells.
- Use the file tools for non-notebook files (e.g. exporting a notebook to a \
.py library, reading a data file).
- Inspect before you edit: read the relevant cell or file first.
- After running code, check the output and fix errors before continuing.
- Keep going until the task is done, then summarize what you changed.
"""


def build_system_prompt(cwd: str, project_instructions: str | None = None) -> str:
    # When a notebook supplies its own instructions (notebook_agent.md), they
    # *replace* the default persona/guidance entirely -- such files typically
    # redefine the assistant's role, and nothing in _BASE is required for tool
    # calling (tools are driven by the `tools=` param, not the prompt). The cwd
    # line is always kept as harmless, useful context.
    base = project_instructions.strip() if project_instructions else _BASE
    return f"{base}\nWorking directory: {cwd}\n"
