@echo off
rem ===========================================================================
rem install.bat -- Windows entry point for installing the offline agent and its
rem optional local llama-cpp-python runtime. Mirrors install.sh.
rem
rem By default this installs ONLY the lightweight agent package and registers
rem the local model server address. The in-process llama-cpp-python runtime is
rem opt-in (--llama / --cpu / --offline), since the remote (OpenAI-compatible)
rem backend is the common case and the build needs a CUDA/C++ toolchain.
rem
rem When the llama runtime IS requested, two build paths exist:
rem   * online  (default): build llama-cpp-python from PyPI (CUDA, or CPU --cpu)
rem   * offline (--offline): air-gapped build from the vendored deps\ tree
rem
rem Usage:
rem   install.bat                  : agent package only (no model runtime) + configure server
rem   install.bat --llama          : also build llama-cpp-python with CUDA offload
rem   install.bat --cpu            : also build llama-cpp-python, CPU-only (implies --llama)
rem   install.bat --jupyter        : also pull the ACP bridge (jupyter extra)
rem   install.bat --lab            : also pull the full JupyterLab + Jupyter AI host stack
rem   install.bat --offline        : air-gapped llama build from deps\ (implies --llama)
rem   install.bat --server-url URL : set the remote server address non-interactively
rem   install.bat --no-win-patch   : skip the Windows event-loop fix (see WINDOWS.md)
rem   install.bat --help
rem
rem Override the interpreter with:  set PYTHON=C:\path\to\python.exe
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

if not defined PYTHON set "PYTHON=python"

rem CUDA on by default; flipped to off by --cpu (same flag requirements.txt uses).
set "CMAKE_CUDA=on"
rem llama-cpp-python is OFF by default -- opt in with --llama / --cpu / --offline.
set "WITH_LLAMA=0"
set "WITH_JUPYTER=0"
set "WITH_LAB=0"
set "OFFLINE=0"
rem Apply Windows runtime fixes (Proactor event loop) by default; --no-win-patch
rem opts out.
set "WITH_WIN_PATCH=1"
rem Empty => prompt interactively (blank answer keeps the existing value).
rem A value via --server-url skips the prompt entirely.
set "SERVER_URL="
set "SERVER_URL_SET=0"
set "CURRENT="

rem --- argument parsing -----------------------------------------------------
:parse
if "%~1"=="" goto endparse
set "A=%~1"
if /i "!A!"=="--llama"      ( set "WITH_LLAMA=1" & shift & goto parse )
if /i "!A!"=="--cpu"        ( set "CMAKE_CUDA=off" & set "WITH_LLAMA=1" & shift & goto parse )
if /i "!A!"=="--no-llama"   ( set "WITH_LLAMA=0" & shift & goto parse )
if /i "!A!"=="--jupyter"    ( set "WITH_JUPYTER=1" & shift & goto parse )
if /i "!A!"=="--lab"        ( set "WITH_LAB=1" & shift & goto parse )
if /i "!A!"=="--offline"    ( set "OFFLINE=1" & set "WITH_LLAMA=1" & shift & goto parse )
if /i "!A!"=="--no-win-patch" ( set "WITH_WIN_PATCH=0" & shift & goto parse )
if /i "!A!"=="--server-url" ( set "SERVER_URL=%~2" & set "SERVER_URL_SET=1" & shift & shift & goto parse )
if /i "!A:~0,13!"=="--server-url=" ( set "SERVER_URL=!A:~13!" & set "SERVER_URL_SET=1" & shift & goto parse )
if /i "!A!"=="-h"           goto usage
if /i "!A!"=="--help"       goto usage
echo install.bat: unknown option '!A!' 1>&2
goto usage_err
:endparse

rem --- 1. The agent package (lightweight; no GPU deps) ----------------------
rem The `lab` extra already pulls in `jupyter` (the ACP bridge) via a
rem self-reference, but both flags compose cleanly.
set "EXTRAS="
if "%WITH_JUPYTER%"=="1" set "EXTRAS=jupyter"
if "%WITH_LAB%"=="1" if defined EXTRAS (set "EXTRAS=!EXTRAS!,lab") else (set "EXTRAS=lab")

if defined EXTRAS (
    echo == installing agent package ^(editable^) with extras [!EXTRAS!] ==
    set "PKGSPEC=.[!EXTRAS!]"
) else (
    echo == installing agent package ^(editable^) ==
    set "PKGSPEC=."
)
"%PYTHON%" -m pip install -e "!PKGSPEC!"
if errorlevel 1 goto fail

rem --- 2. Register the local model server address ---------------------------
rem Read/edit logic lives in scripts\server_url.py so install.sh and install.bat
rem share one implementation. Bootstraps the config from the example first run.
set "CONFIG=offline_agent.toml"
set "URL_HELPER=scripts\server_url.py"
if not exist "%CONFIG%" if exist "%CONFIG%.example" (
    echo == creating %CONFIG% from %CONFIG%.example ==
    copy /y "%CONFIG%.example" "%CONFIG%" >nul
)

if not exist "%CONFIG%" goto llama

for /f "usebackq delims=" %%U in (`"%PYTHON%" "%URL_HELPER%" --get "%CONFIG%"`) do set "CURRENT=%%U"

if "%SERVER_URL_SET%"=="0" call :prompt_url

if defined SERVER_URL (
    "%PYTHON%" "%URL_HELPER%" --set "%CONFIG%" "!SERVER_URL!"
    if errorlevel 1 goto fail
    echo == updated %CONFIG% ==
) else (
    echo == keeping server address "%CURRENT%" ==
)

rem --- 2b. Windows runtime fixes (Jupyter <-> ACP subprocess) ---------------
rem Forces the Proactor event loop so the agent subprocess can spawn under
rem JupyterLab. Best-effort: soft-skips if the Jupyter stack isn't installed,
rem and a failure here does not abort the install. See WINDOWS.md.
rem Single-line ifs (not a parenthesized block) so a '(' / ')' in the install
rem path -- e.g. "Program Files (x86)" -- can't break block parsing.
if "%WITH_WIN_PATCH%"=="1" call "%~dp0patch_windows.bat"
if "%WITH_WIN_PATCH%"=="0" echo == skipping Windows event-loop patch ^(--no-win-patch^) ==

rem --- 3. The local model runtime (opt-in) ----------------------------------
:llama
if "%WITH_LLAMA%"=="0" goto skip_llama
if "%OFFLINE%"=="1" goto llama_offline
goto llama_online

:skip_llama
echo == skipping llama-cpp-python (enable with --llama / --cpu / --offline) ==
goto done

:llama_offline
rem Air-gapped: build from the vendored sdist with platform-targeted CMAKE args
rem and run the postinstall GPU/CPU verification.
echo == building llama-cpp-python offline from deps\ ==
set "OAARGS=--offline-only"
if "%CMAKE_CUDA%"=="off" set "OAARGS=%OAARGS% --cpu"
"%PYTHON%" -m offline_agent.install.cli %OAARGS%
if errorlevel 1 goto fail
goto done

:llama_online
rem Online: compile from PyPI with CUDA (or CPU) offload enabled. Mirrors the
rem llama-cpp-python line in requirements.txt.
echo == building llama-cpp-python from source (GGML_CUDA=%CMAKE_CUDA%) ==
"%PYTHON%" -m pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=%CMAKE_CUDA%"
if errorlevel 1 goto fail
goto done

:done
echo == install complete ==
popd
endlocal
exit /b 0

:fail
echo install.bat: a step failed; aborting. 1>&2
popd
endlocal
exit /b 1

rem --- subroutines ----------------------------------------------------------
:prompt_url
rem set /p leaves SERVER_URL unchanged (empty) on a blank/EOF answer -> keep.
set /p "SERVER_URL=Local model server address (OpenAI-compatible base URL)  [Enter to keep %CURRENT%]: "
goto :eof

:usage
call :print_usage
popd
endlocal
exit /b 0

:usage_err
call :print_usage
popd
endlocal
exit /b 1

:print_usage
echo install.bat -- install the offline agent (and optional llama-cpp-python runtime)
echo.
echo Usage:
echo   install.bat                  : agent package only (no model runtime) + configure server
echo   install.bat --llama          : also build llama-cpp-python with CUDA offload
echo   install.bat --cpu            : also build llama-cpp-python, CPU-only (implies --llama)
echo   install.bat --jupyter        : also pull the ACP bridge (jupyter extra)
echo   install.bat --lab            : also pull the full JupyterLab + Jupyter AI host stack
echo   install.bat --offline        : air-gapped llama build from deps\ (implies --llama)
echo   install.bat --server-url URL : set the remote server address non-interactively
echo   install.bat --no-win-patch   : skip the Windows event-loop fix (see WINDOWS.md)
echo   install.bat --help
goto :eof
