#!/usr/bin/env bash
#
# install.sh -- visible entry point for installing the offline agent and its
# optional local llama-cpp-python runtime.
#
# By default this installs ONLY the lightweight agent package and registers the
# local model server address. The in-process llama-cpp-python runtime is opt-in
# (--llama / --cpu / --offline), since the remote (OpenAI-compatible) backend is
# the common case and the build needs a CUDA/C++ toolchain.
#
# When the llama runtime IS requested, two build paths exist:
#   * online  (default): build llama-cpp-python from PyPI, compiling against
#                        CUDA (or CPU with --cpu).
#   * offline (--offline): air-gapped build from the vendored deps/ tree, via
#                        the packaged `offline-agent-install` (autodetects the
#                        GPU/toolkit and runs a postinstall verification).
#
# Usage:
#   ./install.sh                 # agent package only (no model runtime) + configure server
#   ./install.sh --llama         # also build llama-cpp-python with CUDA offload
#   ./install.sh --cpu           # also build llama-cpp-python, CPU-only (implies --llama)
#   ./install.sh --jupyter       # also pull the ACP bridge (jupyter extra)
#   ./install.sh --lab           # also pull the full JupyterLab + Jupyter AI host stack
#   ./install.sh --offline       # air-gapped llama build from deps/ (implies --llama)
#   ./install.sh --server-url URL # set the remote server address non-interactively
#   ./install.sh --help
#
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
PIP=("$PYTHON" -m pip)

# CUDA on by default; flipped to off by --cpu. This is the same flag the
# requirements.txt llama-cpp-python line carries.
CMAKE_CUDA="on"
# llama-cpp-python is OFF by default -- opt in with --llama / --cpu / --offline.
WITH_LLAMA=0
WITH_JUPYTER=0
WITH_LAB=0
OFFLINE=0
# Empty => prompt interactively (and keep the existing value on a blank answer).
# A non-empty value (via --server-url) skips the prompt entirely.
SERVER_URL=""
SERVER_URL_SET=0

usage() { sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --llama)    WITH_LLAMA=1 ;;
        --cpu)      CMAKE_CUDA="off"; WITH_LLAMA=1 ;;
        --no-llama) WITH_LLAMA=0 ;;   # deprecated: this is now the default
        --jupyter)  WITH_JUPYTER=1 ;;
        --lab)      WITH_LAB=1 ;;
        --offline)  OFFLINE=1; WITH_LLAMA=1 ;;
        --server-url) shift; SERVER_URL="${1:-}"; SERVER_URL_SET=1 ;;
        --server-url=*) SERVER_URL="${1#*=}"; SERVER_URL_SET=1 ;;
        -h|--help)  usage 0 ;;
        *) echo "install.sh: unknown option '$1'" >&2; usage 1 ;;
    esac
    shift
done

# --- 1. The agent package (lightweight; no GPU deps) ----------------------
# Accumulate optional extras. The `lab` extra already pulls in `jupyter`
# (the ACP bridge) via a self-reference, but both flags compose cleanly.
extras_list=()
[ "$WITH_JUPYTER" -eq 1 ] && extras_list+=("jupyter")
[ "$WITH_LAB" -eq 1 ] && extras_list+=("lab")
extras=""
[ "${#extras_list[@]}" -gt 0 ] && extras="[$(IFS=,; echo "${extras_list[*]}")]"
echo "== installing agent package (editable)${extras:+ with extras $extras} =="
"${PIP[@]}" install -e ".${extras}"

# --- 2. Register the local model server address ---------------------------
# Writes [remote].base_url in offline_agent.toml. With --server-url the value is
# applied non-interactively; otherwise we prompt (a blank answer keeps the
# current value). Bootstraps the config from the example on first run.
CONFIG="offline_agent.toml"
if [ ! -f "$CONFIG" ] && [ -f "${CONFIG}.example" ]; then
    echo "== creating $CONFIG from ${CONFIG}.example =="
    cp "${CONFIG}.example" "$CONFIG"
fi

if [ -f "$CONFIG" ]; then
    # Read/edit logic lives in scripts/server_url.py so install.sh and the
    # Windows install.bat share one implementation.
    URL_HELPER="scripts/server_url.py"
    current="$("$PYTHON" "$URL_HELPER" --get "$CONFIG")"
    if [ "$SERVER_URL_SET" -eq 0 ]; then
        if [ -t 0 ]; then
            printf 'Local model server address (OpenAI-compatible base URL)\n  [Enter to keep "%s"]: ' "$current"
            read -r SERVER_URL || SERVER_URL=""
        else
            echo "== non-interactive shell; keeping server address \"$current\" =="
        fi
    fi

    if [ -n "$SERVER_URL" ]; then
        "$PYTHON" "$URL_HELPER" --set "$CONFIG" "$SERVER_URL"
        echo "== updated $CONFIG =="
    else
        echo "== keeping server address \"$current\" =="
    fi
fi

# --- 3. The local model runtime (opt-in) ----------------------------------
if [ "$WITH_LLAMA" -eq 0 ]; then
    echo "== skipping llama-cpp-python (enable with --llama / --cpu / --offline) =="
elif [ "$OFFLINE" -eq 1 ]; then
    # Air-gapped: build from the vendored sdist with platform-targeted CMAKE
    # args and run the postinstall GPU/CPU verification.
    echo "== building llama-cpp-python offline from deps/ =="
    args=(--offline-only)
    [ "$CMAKE_CUDA" = "off" ] && args+=(--cpu)
    "$PYTHON" -m offline_agent.install.cli "${args[@]}"
else
    # Online: compile from PyPI with CUDA (or CPU) offload enabled. Mirrors the
    # llama-cpp-python line in requirements.txt.
    echo "== building llama-cpp-python from source (GGML_CUDA=${CMAKE_CUDA}) =="
    "${PIP[@]}" install llama-cpp-python -C cmake.args="-DGGML_CUDA=${CMAKE_CUDA}"
fi

echo "== install complete =="
