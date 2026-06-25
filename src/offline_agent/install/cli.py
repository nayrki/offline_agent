"""``offline-agent-install``: offline-first install of the runtime + llama-cpp.

Steps:
  1. Detect platform + GPU compute capability + the CUDA toolkit (nvcc).
  2. Install the runtime closure from deps/runtime/ (offline), else from network.
  3. Install build-backend deps from deps/build/ (offline), else network.
  4. BUILD llama-cpp-python from the vendored sdist with ``--no-build-isolation``
     and platform-targeted CMAKE args (GPU arch + located nvcc; CPU otherwise).
     CPU support is baseline in any build, and the runtime additionally falls
     back to n_gpu_layers=0.
  5. POSTINSTALL VERIFICATION: import the freshly built llama_cpp and assert it
     matches the target -- a GPU target must report ``llama_supports_gpu_offload``;
     a CPU target must import and load its shared libs cleanly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from shutil import which

from .detect import Target, detect_cuda_arch, detect_target


def _deps_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path(os.environ.get("OFFLINE_AGENT_DEPS", "deps")).resolve()


def _run(cmd: list[str], env: dict | None = None, dry: bool = False) -> int:
    print("+ " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, env=env).returncode


def _pip(args: list[str]) -> list[str]:
    if which("uv"):
        return ["uv", "pip", "install", *args]
    return [sys.executable, "-m", "pip", "install", *args]


def _cmake_env(target: Target) -> dict:
    env = dict(os.environ)
    if target.gpu:
        nvcc = Path(target.nvcc)
        cuda_bin = nvcc.parent
        cuda_root = cuda_bin.parent
        env["CMAKE_ARGS"] = (
            f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={target.cuda_arch} "
            f"-DCMAKE_CUDA_COMPILER={nvcc}"
        )
        env["CUDACXX"] = str(nvcc)
        # Put nvcc on PATH and its runtime libs on the loader path for the build.
        env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
        lib64 = cuda_root / "lib64"
        if lib64.is_dir():
            env["LD_LIBRARY_PATH"] = str(lib64) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    else:
        env["CMAKE_ARGS"] = "-DGGML_CUDA=off -DGGML_NATIVE=on"
    env["FORCE_CMAKE"] = "1"
    return env


def install(target: Target, deps: Path, *, offline_only: bool, dry: bool) -> int:
    runtime = deps / "runtime"
    sdist = deps / "sdist"
    build = deps / "build"

    print(f"== target: {target.slot} (gpu={target.gpu}, cuda_arch={target.cuda_arch}) ==")
    if target.gpu:
        print(f"   using nvcc: {target.nvcc}")
    elif detect_cuda_arch() is not None:
        print("   WARNING: a CUDA GPU was detected but no nvcc toolkit was found; "
              "building CPU-only. Set CUDA_HOME or add nvcc to PATH for a GPU build.")

    # 1. Runtime closure (agent + deps), offline first.
    if not _install_step(
        _pip(["--no-index", "--find-links", str(runtime), "offline-agent"]),
        fallback=_pip(["offline-agent"]),
        offline_only=offline_only, dry=dry, what="runtime closure",
    ):
        return 1

    # 2. Build-backend deps (needed for --no-build-isolation).
    build_pkgs = ["scikit-build-core", "ninja", "cmake"]
    if not _install_step(
        _pip(["--no-index", "--find-links", str(build), *build_pkgs]),
        fallback=_pip(build_pkgs),
        offline_only=offline_only, dry=dry, what="build backend",
    ):
        return 1

    # 3. Build llama-cpp-python from the vendored sdist, targeted at this GPU.
    #    runtime/ is on the find-links too so llama's own deps (numpy, diskcache,
    #    jinja2, ...) resolve offline during the build install.
    env = _cmake_env(target)
    if not _install_step(
        _pip([
            "--no-index", "--no-build-isolation",
            "--find-links", str(sdist),
            "--find-links", str(build),
            "--find-links", str(runtime),
            "llama-cpp-python",
        ]),
        fallback=_pip(["--no-build-isolation", "llama-cpp-python"]),
        offline_only=offline_only, dry=dry, what="llama-cpp-python", env=env,
    ):
        return 1

    # 4. Postinstall verification: prove the build actually works on this box.
    if not _verify(target, dry=dry):
        return 1

    print("== install complete ==")
    return 0


# A self-contained probe run in the target interpreter: imports the freshly
# built llama_cpp, reports its GPU-offload support, and (for a GPU target)
# fails loudly if offload isn't available -- catching a silently-CPU build.
_VERIFY_SNIPPET = r"""
import sys
try:
    import llama_cpp
except Exception as exc:  # import or shared-lib load failure
    print("VERIFY: failed to import llama_cpp: %r" % (exc,), file=sys.stderr)
    raise SystemExit(2)
ver = getattr(llama_cpp, "__version__", "?")
try:
    offload = bool(llama_cpp.llama_supports_gpu_offload())
except Exception as exc:
    print("VERIFY: llama_supports_gpu_offload() raised: %r" % (exc,), file=sys.stderr)
    raise SystemExit(2)
want_gpu = sys.argv[1] == "gpu"
print("VERIFY: llama_cpp==%s supports_gpu_offload=%s (want_gpu=%s)"
      % (ver, offload, want_gpu))
if want_gpu and not offload:
    print("VERIFY: GPU target but the build reports no GPU offload support",
          file=sys.stderr)
    raise SystemExit(3)
"""


def _verify(target: Target, *, dry: bool) -> bool:
    arg = "gpu" if target.gpu else "cpu"
    cmd = [sys.executable, "-c", _VERIFY_SNIPPET, arg]
    print("+ postinstall verification (" + arg + ")")
    if dry:
        print("+ " + " ".join(cmd[:2]) + " <verify> " + arg)
        return True
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        return True
    print("ERROR: postinstall verification failed (exit %d)" % rc, file=sys.stderr)
    return False


def _install_step(cmd, *, fallback, offline_only, dry, what, env=None) -> bool:
    rc = _run(cmd, env=env, dry=dry)
    if rc == 0:
        return True
    if offline_only:
        print(f"ERROR: {what} unavailable offline", file=sys.stderr)
        return False
    print(f"offline {what} install failed; falling back to network")
    if _run(fallback, env=env, dry=dry) != 0:
        print(f"ERROR: network {what} install failed", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offline-agent-install")
    parser.add_argument("--deps", help="path to the deps/ directory (default: ./deps)")
    parser.add_argument("--cpu", action="store_true", help="force a CPU-only build")
    parser.add_argument("--offline-only", action="store_true", help="never use the network")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args(argv)

    target = detect_target(force_cpu=args.cpu)
    deps = _deps_root(args.deps)
    return install(target, deps, offline_only=args.offline_only, dry=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
