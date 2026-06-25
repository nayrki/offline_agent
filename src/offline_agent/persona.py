"""Jupyter AI persona shim.

Registered via the ``jupyter_ai.personas`` entry point. Its only job is to make
the agent appear in Jupyter AI's chat and to launch our ACP agent executable;
``BaseAcpPersona`` handles the ACP client/subprocess plumbing. This module is
imported only inside the JupyterLab environment (the ``jupyter`` extra), where
``jupyter-ai-acp-client`` is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults

_AVATAR = str(Path(__file__).parent / "static" / "icon.svg")

# Launch the agent via the current interpreter + module so resolution does not
# depend on the console-script being on PATH in the Jupyter server's env.
_EXECUTABLE = [sys.executable, "-m", "offline_agent.agent.main"]


class OfflineAgentPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, executable=_EXECUTABLE, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Offline Agent",
            description="Locally-hosted, offline agentic coding agent.",
            avatar_path=_AVATAR,
            system_prompt="",  # the agent subprocess owns its own system prompt
        )
