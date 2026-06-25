#!/usr/bin/env python3
"""Populate deps/ for offline install. Run natively on each target OS.

  - deps/sdist/   : the llama-cpp-python SOURCE distribution (built per-machine)
  - deps/build/   : Python build-backend wheels for offline build isolation
  - deps/runtime/ : the agent package + its deps AND llama-cpp-python's runtime
                    deps (numpy, diskcache, jinja2, ...), so the from-source
                    llama build resolves everything under --no-index.

Cross-OS note: only the llama-cpp-python sdist is platform-agnostic. The wheels
resolve to the running interpreter/OS, so run this on each target
(Linux+CUDA, Windows+CUDA, Linux/CPU) -- a single Linux box cannot produce the
Windows wheels.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPS = ROOT / "deps"


def _pip_download(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "download", *args]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _fetch_sdist(version: str | None) -> None:
    """Download the raw sdist tarball directly (pip download --no-binary would
    try to BUILD metadata, which needs the CUDA toolchain)."""
    meta_url = "https://pypi.org/pypi/llama-cpp-python/json"
    data = json.load(urllib.request.urlopen(meta_url, timeout=60))
    version = version or data["info"]["version"]
    urls = data.get("releases", {}).get(version) or data["urls"]
    sdist = next(u for u in urls if u["packagetype"] == "sdist")
    dest = DEPS / "sdist" / sdist["filename"]
    print(f"+ download {sdist['url']}")
    urllib.request.urlretrieve(sdist["url"], dest)
    print(f"  saved {dest} ({dest.stat().st_size // 1024} KB)")


def main() -> int:
    manifest = tomllib.loads((DEPS / "manifest.toml").read_text())
    llama_version = manifest["llama_cpp"].get("version")
    build_pkgs = manifest["build_requires"]["packages"]
    llama_runtime = manifest["llama_cpp"].get("runtime_deps", [])

    for sub in ("sdist", "build", "runtime"):
        (DEPS / sub).mkdir(parents=True, exist_ok=True)

    # 1. llama-cpp-python sdist (raw tarball; no build).
    _fetch_sdist(llama_version)

    # 2. Build-backend wheels for offline build isolation / --no-build-isolation.
    _pip_download(["--dest", str(DEPS / "build"), *build_pkgs])

    # 3. Runtime closure: the agent + its deps, plus llama-cpp-python's deps.
    _pip_download(["--dest", str(DEPS / "runtime"), str(ROOT), *llama_runtime])

    print("== deps populated for this platform ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
