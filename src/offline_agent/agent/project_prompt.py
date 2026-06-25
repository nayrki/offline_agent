"""Load per-notebook project instructions from a ``notebook_agent.md`` file.

When a chat starts, if the user has a notebook open, we look in *that notebook's*
directory for a ``notebook_agent.md`` (case-insensitive) and, if present, return
its contents to be appended to the system prompt. The open notebook is discovered
via the Jupyter MCP tool ``get_active_notebook``, which returns the notebook's
path relative to the Jupyter server root (or null when there is no active one).

Every failure path -- no tool, no active notebook, no file, read error -- returns
``None`` so the caller falls back to the default system prompt.
"""

from __future__ import annotations

import logging
import os

from ..mcp.client import MCPClients

log = logging.getLogger(__name__)

FILENAME = "notebook_agent.md"
TOOL = "get_active_notebook"
_NULLISH = {"", "null", "none"}


async def load_project_instructions(mcp: MCPClients) -> str | None:
    """Return the text of ``notebook_agent.md`` for the active notebook, or None."""
    try:
        if not mcp.has_tool(TOOL):
            return None
        raw = await mcp.call_tool(TOOL, {})
        nb_path = _parse_notebook_path(raw)
        if not nb_path:
            return None
        return _read_notebook_agent_md(nb_path)
    except Exception:  # never let prompt loading break session setup
        log.warning("failed to load %s", FILENAME, exc_info=True)
        return None


def _parse_notebook_path(raw: str) -> str | None:
    """Normalize the ``get_active_notebook`` result to a path, or None."""
    path = (raw or "").strip().strip('"').strip("'").strip()
    if path.lower() in _NULLISH:
        return None
    return path


def _read_notebook_agent_md(nb_path: str) -> str | None:
    """Find and read ``notebook_agent.md`` next to the given notebook path."""
    nb_dir = os.path.dirname(nb_path)
    # An absolute notebook path resolves directly; a relative one (the common
    # case -- it's relative to the Jupyter server root) is anchored to the agent
    # process's cwd, which is that server root.
    base = nb_dir if os.path.isabs(nb_path) else os.path.join(os.getcwd(), nb_dir)
    if not os.path.isdir(base):
        return None
    for entry in os.listdir(base):
        if entry.lower() == FILENAME:
            path = os.path.join(base, entry)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if not text.strip():
                return None
            log.info("loaded %s from %s", FILENAME, path)
            return text
    return None
