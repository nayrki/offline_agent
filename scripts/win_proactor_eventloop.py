# >>> offline-agent: Windows Proactor event-loop fix >>>
# Appended to jupyter_server_config.py by patch_windows.bat. See WINDOWS.md.
#
# jupyter_server forces the Selector event loop on Windows (ServerApp.
# _init_asyncio_patch, "for tornado + pyzmq"), but jupyter-ai-acp-client spawns
# the agent via asyncio.create_subprocess_exec, which on Windows works only on
# the Proactor loop -- under Selector it raises NotImplementedError. This block
# runs after _init_asyncio_patch() but before the event loop is created, so it
# flips the policy back to Proactor. pyzmq still works on Proactor via tornado's
# AddThreadSelectorEventLoop shim (tornado >= 6.1).
import sys as _oa_sys
import asyncio as _oa_asyncio

if _oa_sys.platform.startswith("win") and hasattr(_oa_asyncio, "WindowsProactorEventLoopPolicy"):
    _oa_asyncio.set_event_loop_policy(_oa_asyncio.WindowsProactorEventLoopPolicy())
# <<< offline-agent: Windows Proactor event-loop fix <<<
