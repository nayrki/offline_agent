"""Platform and GPU-accelerator detection for the installer.

A CUDA build needs both (a) a GPU compute capability (from ``nvidia-smi``) and
(b) the CUDA toolkit's ``nvcc`` -- which is frequently NOT on PATH. We therefore
locate ``nvcc`` explicitly (env vars, PATH, then common install locations) and
treat "GPU present but toolkit missing" as a CPU build.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    os: str                 # "linux" | "windows"
    arch: str               # "x86_64"
    cuda_arch: str | None   # e.g. "120" for compute capability 12.0; None => no GPU
    nvcc: str | None        # absolute path to nvcc, or None if toolkit not found

    @property
    def gpu(self) -> bool:
        # Only a real GPU build if BOTH the device arch and the compiler exist.
        return self.cuda_arch is not None and self.nvcc is not None

    @property
    def slot(self) -> str:
        accel = f"cu-sm{self.cuda_arch}" if self.gpu else "cpu"
        return f"{self.os}_{self.arch}-{accel}"


def detect_platform() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        raise SystemExit("macOS is not a supported target for offline-agent.")
    os_name = {"linux": "linux", "windows": "windows"}.get(system)
    if os_name is None:
        raise SystemExit(f"unsupported OS: {system!r}")
    arch = {"x86_64": "x86_64", "amd64": "x86_64"}.get(machine)
    if arch is None:
        raise SystemExit(f"unsupported architecture: {machine!r} (need x86_64)")
    return os_name, arch


def detect_cuda_arch() -> str | None:
    """Return the local GPU compute capability as a CMake arch (e.g. "120"), or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    match = re.search(r"(\d+)\.(\d+)", out)  # first GPU's "12.0" -> "120"
    return f"{match.group(1)}{match.group(2)}" if match else None


def find_nvcc() -> str | None:
    """Locate nvcc via env vars, PATH, then common toolkit install locations."""
    exe = "nvcc.exe" if platform.system().lower() == "windows" else "nvcc"

    for var in ("CUDACXX", "CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(var)
        if not val:
            continue
        cand = Path(val)
        if cand.name.startswith("nvcc") and cand.is_file():
            return str(cand)
        nvcc = cand / "bin" / exe
        if nvcc.is_file():
            return str(nvcc)

    on_path = shutil.which("nvcc")
    if on_path:
        return on_path

    if platform.system().lower() == "windows":
        patterns = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe",
        ]
    else:
        patterns = [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-*/bin/nvcc",
            "/opt/cuda/bin/nvcc",
        ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    # Prefer the highest versioned toolkit.
    candidates.sort(reverse=True)
    return candidates[0]


def detect_target(force_cpu: bool = False) -> Target:
    os_name, arch = detect_platform()
    if force_cpu:
        return Target(os=os_name, arch=arch, cuda_arch=None, nvcc=None)
    cuda_arch = detect_cuda_arch()
    nvcc = find_nvcc() if cuda_arch is not None else None
    return Target(os=os_name, arch=arch, cuda_arch=cuda_arch, nvcc=nvcc)
