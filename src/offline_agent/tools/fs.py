"""Filesystem tools serviced by the ACP client (not MCP).

``read_file`` / ``write_file`` are exposed to the model only when the connected
client advertises the matching capability (``fs.read_text_file`` /
``fs.write_text_file``) and, for writes, ``fs.allow_write`` is set in config.
Calls are dispatched to the ACP client's ``read_text_file`` / ``write_text_file``.
"""

from __future__ import annotations

from typing import Any

READ_FILE = "read_file"
WRITE_FILE = "write_file"
_FS_NAMES = {READ_FILE, WRITE_FILE}

_READ_TOOL = {
    "type": "function",
    "function": {
        "name": READ_FILE,
        "description": (
            "Read a UTF-8 text file from the workspace. Use notebook tools for "
            ".ipynb cells; use this for other files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "line": {"type": "integer", "description": "1-based start line (optional)."},
                "limit": {"type": "integer", "description": "Max lines to read (optional)."},
            },
            "required": ["path"],
        },
    },
}

_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": WRITE_FILE,
        "description": "Create or overwrite a UTF-8 text file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}


def is_fs_tool(name: str) -> bool:
    return name in _FS_NAMES


def fs_tools(*, can_read: bool, can_write: bool) -> list[dict]:
    tools: list[dict] = []
    if can_read:
        tools.append(_READ_TOOL)
    if can_write:
        tools.append(_WRITE_TOOL)
    return tools


async def dispatch_fs(conn: Any, session_id: str, name: str, args: dict[str, Any]) -> str:
    if name == READ_FILE:
        resp = await conn.read_text_file(
            path=args["path"],
            session_id=session_id,
            line=args.get("line"),
            limit=args.get("limit"),
        )
        return getattr(resp, "content", "") or ""
    if name == WRITE_FILE:
        await conn.write_text_file(
            content=args["content"],
            path=args["path"],
            session_id=session_id,
        )
        return f"Wrote {args['path']}."
    raise KeyError(f"not a filesystem tool: {name!r}")
