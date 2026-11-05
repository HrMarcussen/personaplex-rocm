# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Runtime defaults for AMD ROCm (and Windows) builds of PyTorch.

Imported by ``moshi/__init__.py`` *before* torch, because several PyTorch
knobs are read from the environment on first use. Never imports torch.

Everything here only sets defaults: any variable the user already set wins.
"""
import importlib.metadata
import importlib.util
import os
from pathlib import Path


def _torch_build() -> str:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return ""


def is_rocm_build() -> bool:
    return "rocm" in _torch_build()


def cache_dir() -> Path:
    base = os.environ.get("PERSONAPLEX_CACHE")
    if base:
        p = Path(base)
    elif os.name == "nt":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "personaplex-rocm"
    else:
        p = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "personaplex-rocm"
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure() -> None:
    # torch.compile needs Triton for GPU kernels. The AMD Windows wheels ship
    # without Triton, so compile would fail at first call: disable it.
    if "NO_TORCH_COMPILE" not in os.environ and importlib.util.find_spec("triton") is None:
        os.environ["NO_TORCH_COMPILE"] = "1"

    if is_rocm_build():
        # Batch-1 GEMVs dominate Moshi inference. The default rocBLAS/hipBLASLt
        # heuristics pick slow kernels for several of its shapes; TunableOp
        # benchmarks all candidates once per shape and caches the winner.
        # Measured on RX 7900 XTX: LM step 63 ms -> 52 ms.
        os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
        os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "1")
        os.environ.setdefault(
            "PYTORCH_TUNABLEOP_FILENAME", str(cache_dir() / "tunableop_results.csv")
        )


def save_tuning_results() -> None:
    """Persist TunableOp results (call after warmup, so later runs start fast)."""
    import torch

    try:
        if torch.cuda.is_available() and torch.cuda.tunable.is_enabled():
            torch.cuda.tunable.write_file()
    except Exception:  # noqa: BLE001 - best effort only
        pass



def enable_high_res_timer() -> None:
    """Windows: raise the system timer resolution to 1 ms for this process.

    The realtime server polls with `asyncio.sleep(0.001)`; at Windows' default
    ~15.6 ms timer resolution each poll really sleeps ~15 ms (measured), adding
    up to ~30 ms of latency and jitter per audio frame. With 1 ms: ~2 ms.
    Released automatically when the process exits.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.WinDLL("winmm").timeBeginPeriod(1)
    except Exception:  # noqa: BLE001 - best effort only
        pass


configure()
