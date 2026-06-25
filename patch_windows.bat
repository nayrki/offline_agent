@echo off
rem ===========================================================================
rem patch_windows.bat -- apply Windows-specific runtime fixes for the offline
rem agent under JupyterLab. Idempotent and safe to re-run. See WINDOWS.md.
rem
rem Currently applies one fix: forces the asyncio Proactor event loop in the
rem Jupyter server config so jupyter-ai-acp-client can spawn the agent
rem subprocess (Selector loop -> NotImplementedError on Windows).
rem
rem install.bat runs this by default; you can also run it on its own:
rem   patch_windows.bat
rem
rem Override the interpreter with:  set PYTHON=C:\path\to\python.exe
rem ===========================================================================
setlocal EnableExtensions
if not defined PYTHON set "PYTHON=python"

rem The config fragment shipped next to this script (in scripts\).
set "SNIPPET=%~dp0scripts\win_proactor_eventloop.py"
set "MARKER=offline-agent: Windows Proactor"

rem NOTE: paths may contain spaces or parentheses (e.g. "Program Files (x86)"),
rem so we keep %SNIPPET%/%TARGET% out of parenthesized ( ) blocks and branch
rem with goto instead -- a ')' inside an expanded path would break block parsing.
if not exist "%SNIPPET%" goto no_snippet

rem Resolve the Jupyter config dir for the ACTIVE environment (honors
rem JUPYTER_CONFIG_DIR). If jupyter-core isn't installed, Jupyter isn't set up
rem in this env yet -- soft-skip rather than fail the install.
set "CFGDIR="
for /f "usebackq delims=" %%D in (`"%PYTHON%" -c "from jupyter_core.paths import jupyter_config_dir as d; print(d())" 2^>nul`) do set "CFGDIR=%%D"
if not defined CFGDIR goto no_jupyter

set "TARGET=%CFGDIR%\jupyter_server_config.py"
if not exist "%CFGDIR%" mkdir "%CFGDIR%"

if not exist "%TARGET%" goto apply
findstr /c:"%MARKER%" "%TARGET%" >nul 2>&1
if errorlevel 1 goto apply
echo [win-patch] Proactor event-loop fix already present; skipping.
echo [win-patch]   %TARGET%
endlocal & exit /b 0

:apply
echo [win-patch] applying Proactor event-loop fix to:
echo [win-patch]   %TARGET%
rem Blank line then the fragment, so it never glues onto an existing last line.
echo.>>"%TARGET%"
type "%SNIPPET%">>"%TARGET%"
if errorlevel 1 goto write_fail
echo [win-patch] done. Restart 'jupyter lab' for it to take effect.
endlocal & exit /b 0

:no_snippet
echo [win-patch] missing fragment "%SNIPPET%"; nothing to apply. 1>&2
endlocal & exit /b 1

:no_jupyter
echo [win-patch] jupyter-core not found in this environment; skipping.
echo [win-patch] re-run after installing the Jupyter stack, or apply manually per WINDOWS.md.
endlocal & exit /b 0

:write_fail
echo [win-patch] FAILED to write "%TARGET%" 1>&2
endlocal & exit /b 1
