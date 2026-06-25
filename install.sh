#!/usr/bin/env bash
#
# install.sh -- visible entry point for installing the offline agent and its
# optional local llama-cpp-python CUDA runtime.
#
# Two paths:
#   * online  (default): install the agent + build llama-cpp-python from PyPI,
#                        compiling against CUDA (or CPU with --cpu).
#   * offline (--offline): air-gapped build from the vendored deps/ tree, via
#                        the packaged `offline-agent-install` (autodetects the
#                        GPU/toolkit and runs a postinstall verification).
#
# Usage:
#   ./install.sh                 # editable agent install + CUDA llama-cpp build
#   ./install.sh --cpu           # CPU-only llama-cpp build
#   ./install.sh --no-llama      # agent package only (skip the model runtime)
#   ./install.sh --jupyter       # also pull the Jupyter AI persona stack
#   ./install.sh --offline       # air-gapped build from deps/ (adds --offline-only)
#   ./install.sh --help
#
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
PIP=("$PYTHON" -m pip)

# CUDA on by default; flipped to off by --cpu. This is the same flag the
# requirements.txt llama-cpp-python line carries.
CMAKE_CUDA="on"
WITH_LLAMA=1
WITH_JUPYTER=0
OFFLINE=0

usage() { sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)      CMAKE_CUDA="off" ;;
        --no-llama) WITH_LLAMA=0 ;;
        --jupyter)  WITH_JUPYTER=1 ;;
        --offline)  OFFLINE=1 ;;
        -h|--help)  usage 0 ;;
        *) echo "install.sh: unknown option '$1'" >&2; usage 1 ;;
    esac
    shift
done

# --- 1. The agent package (lightweight; no GPU deps) ----------------------
extras=""
[ "$WITH_JUPYTER" -eq 1 ] && extras="[jupyter]"
echo "== installing agent package (editable)${extras:+ with extras $extras} =="
"${PIP[@]}" install -e ".${extras}"

# --- 2. The local model runtime -------------------------------------------
if [ "$WITH_LLAMA" -eq 0 ]; then
    echo "== skipping llama-cpp-python (--no-llama) =="
    echo "== install complete =="
    exit 0
fi

if [ "$OFFLINE" -eq 1 ]; then
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
