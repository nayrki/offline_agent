"""Tests for installer platform/accelerator detection and build-env logic."""

from __future__ import annotations

from offline_agent.install.cli import _cmake_env, _verify
from offline_agent.install.detect import Target, detect_target, find_nvcc


def _gpu(nvcc="/usr/local/cuda/bin/nvcc"):
    return Target("linux", "x86_64", "120", nvcc)


def _cpu():
    return Target("linux", "x86_64", None, None)


def test_slot_naming():
    assert _gpu().slot == "linux_x86_64-cu-sm120"
    assert _cpu().slot == "linux_x86_64-cpu"
    assert Target("windows", "x86_64", "89", "C:/cuda/bin/nvcc.exe").slot == "windows_x86_64-cu-sm89"


def test_gpu_requires_both_arch_and_nvcc():
    assert _gpu().gpu is True
    assert _cpu().gpu is False
    # GPU device present but toolkit missing => CPU build.
    assert Target("linux", "x86_64", "120", None).gpu is False


def test_cmake_env_gpu_points_at_nvcc():
    env = _cmake_env(_gpu("/usr/local/cuda/bin/nvcc"))
    assert "-DGGML_CUDA=on" in env["CMAKE_ARGS"]
    assert "-DCMAKE_CUDA_ARCHITECTURES=120" in env["CMAKE_ARGS"]
    assert "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" in env["CMAKE_ARGS"]
    assert env["CUDACXX"] == "/usr/local/cuda/bin/nvcc"
    assert env["PATH"].startswith("/usr/local/cuda/bin")
    assert env["FORCE_CMAKE"] == "1"


def test_cmake_env_cpu():
    env = _cmake_env(_cpu())
    assert "-DGGML_CUDA=off" in env["CMAKE_ARGS"]
    assert "-DGGML_NATIVE=on" in env["CMAKE_ARGS"]


def test_verify_dry_run_is_a_noop():
    # Dry run must not spawn the probe; returns True for both target kinds.
    assert _verify(_gpu(), dry=True) is True
    assert _verify(_cpu(), dry=True) is True


def test_detect_target_runs_on_this_host():
    target = detect_target()
    assert target.os in {"linux", "windows"}
    assert target.arch == "x86_64"
    assert target.slot
    # If this host has a GPU build, nvcc must have been located.
    if target.gpu:
        assert target.nvcc and find_nvcc()
