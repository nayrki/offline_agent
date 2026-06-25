"""Unit tests for loading per-notebook notebook_agent.md project instructions."""

from __future__ import annotations

import pytest

from offline_agent.agent.project_prompt import TOOL, load_project_instructions
from offline_agent.agent.prompts import build_system_prompt

from _stubs import StubMCP, mcp_tool

INSTRUCTIONS = "Always answer in haiku."


def _mcp(active_notebook: str | None) -> StubMCP:
    """A StubMCP advertising get_active_notebook that returns the given path."""
    results = {} if active_notebook is None else {TOOL: active_notebook}
    return StubMCP([mcp_tool(TOOL)], results=results)


@pytest.mark.asyncio
async def test_absolute_notebook_path_finds_file(tmp_path):
    # Absolute path: resolved directly, no cwd dependence.
    (tmp_path / "notebook_agent.md").write_text(INSTRUCTIONS)
    nb = tmp_path / "analysis.ipynb"
    mcp = _mcp(str(nb))
    assert await load_project_instructions(mcp) == INSTRUCTIONS


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["NOTEBOOK_AGENT.MD", "Notebook_Agent.md", "notebook_agent.md"])
async def test_filename_match_is_case_insensitive(tmp_path, name):
    sub = tmp_path / "masterclasses"
    sub.mkdir()
    (sub / name).write_text(INSTRUCTIONS)
    mcp = _mcp(str(sub / "lesson.ipynb"))
    assert await load_project_instructions(mcp) == INSTRUCTIONS


@pytest.mark.asyncio
async def test_relative_notebook_path_anchored_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "masterclasses"
    sub.mkdir()
    (sub / "notebook_agent.md").write_text(INSTRUCTIONS)
    # Relative path, as get_active_notebook actually returns it.
    mcp = _mcp("masterclasses/lesson.ipynb")
    assert await load_project_instructions(mcp) == INSTRUCTIONS


@pytest.mark.asyncio
async def test_relative_notebook_at_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notebook_agent.md").write_text(INSTRUCTIONS)
    mcp = _mcp("lesson.ipynb")  # dirname == "" -> the root itself
    assert await load_project_instructions(mcp) == INSTRUCTIONS


@pytest.mark.asyncio
async def test_no_tool_returns_none():
    mcp = StubMCP([])  # get_active_notebook not advertised
    assert await load_project_instructions(mcp) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "null", "None", "  null  "])
async def test_no_active_notebook_returns_none(value):
    assert await load_project_instructions(_mcp(value)) is None


@pytest.mark.asyncio
async def test_file_missing_returns_none(tmp_path):
    nb = tmp_path / "analysis.ipynb"  # dir exists, but no notebook_agent.md
    assert await load_project_instructions(_mcp(str(nb))) is None


@pytest.mark.asyncio
async def test_missing_directory_returns_none(tmp_path):
    nb = tmp_path / "does_not_exist" / "analysis.ipynb"
    assert await load_project_instructions(_mcp(str(nb))) is None


@pytest.mark.asyncio
async def test_empty_file_returns_none(tmp_path):
    (tmp_path / "notebook_agent.md").write_text("   \n")
    nb = tmp_path / "analysis.ipynb"
    assert await load_project_instructions(_mcp(str(nb))) is None


def test_build_system_prompt_replaces_default():
    base = build_system_prompt("/work")
    replaced = build_system_prompt("/work", INSTRUCTIONS)
    assert replaced != base
    assert INSTRUCTIONS in replaced
    assert "You are Offline Agent" not in replaced  # default persona dropped
    assert replaced.endswith("Working directory: /work\n")  # cwd line kept


def test_build_system_prompt_default_unchanged():
    # No project instructions -> byte-identical to passing None.
    assert build_system_prompt("/work") == build_system_prompt("/work", None)
    assert build_system_prompt("/work").endswith("Working directory: /work\n")
