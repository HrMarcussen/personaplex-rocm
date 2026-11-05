# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Custom HIP kernels for AMD GPUs, loaded without a host C++ compiler.

Entry points (all fall back to plain PyTorch when unavailable):

* ``fast_linear(x, w, b)``: drop-in for ``F.linear``. Batch-1 bf16 inputs use the
  GEMV kernels in ``gemv.hip`` (autotuned per shape against ``F.linear``).
* ``rms_norm(x, alpha, eps)``: fused fp32 RMSNorm for bf16 activations.
* ``attention_step(...)``: fused RoPE + ring KV-cache write + flash-decoding
  attention for one streaming token (``fused.hip``).

The kernels are compiled once with the ROCm SDK's bundled clang (device-only
and header-free, so no MSVC is needed on Windows), cached on disk, loaded with
``hipModuleLoadData`` and launched with ``hipModuleLaunchKernel`` on torch's
current stream, so HIP graphs capture them like any other kernel.

Environment switches: ``PERSONAPLEX_FAST_GEMV=0``, ``PERSONAPLEX_FUSED_ATTN=0``,
``PERSONAPLEX_FUSED_NORM=0``, or ``PERSONAPLEX_KERNELS=0`` for all of them.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .._platform_env import cache_dir, is_rocm_build

logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent
_SOURCES = {
    "gemv": ["gemv_bf16_r1", "gemv_bf16_r2", "gemv_bf16_r4", "gemv_bf16_sk2", "gemv_bf16_sk4",
             "gemv_i8_r1", "gemv_i8_r2", "gemv_i8_sk2", "gemv_i8_sk4"],
    "fused": ["rope_kv_write", "attn_decode_d128", "attn_decode_d64", "attn_combine", "rmsnorm_bf16"],
}
_GEMV_VARIANTS = {  # name -> rows per block (all use 256 threads)
    "gemv_bf16_r1": 8,
    "gemv_bf16_r2": 16,
    "gemv_bf16_r4": 32,
    "gemv_bf16_sk2": 2,
    "gemv_bf16_sk4": 4,
}
_I8_VARIANTS = {"gemv_i8_r1": 8, "gemv_i8_r2": 16, "gemv_i8_sk2": 2, "gemv_i8_sk4": 4}
# id(Parameter) -> (int8 weight [N, K], fp32 scale [N], the Parameter itself)
_QUANT: dict[int, tuple[torch.Tensor, torch.Tensor, torch.nn.Parameter]] = {}
_ARCHS = ["gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"]

_lock = threading.Lock()
_state: "_Kernels | None | bool" = None  # None = not tried, False = unavailable


def _env_on(name: str) -> bool:
    if os.environ.get("PERSONAPLEX_KERNELS", "1").lower() in ("0", "false", "no", "off"):
        return False
    return os.environ.get(name, "1").lower() not in ("0", "false", "no", "off")


def _find_clang() -> str | None:
    """The ROCm LLVM clang (device compiler). No host toolchain is required."""
    roots: list[Path] = []
    try:
        import _rocm_sdk_core  # type: ignore  # AMD's Windows/Linux pip SDK

        roots.append(Path(_rocm_sdk_core.__file__).parent)
    except ImportError:
        pass
    for env in ("ROCM_PATH", "HIP_PATH"):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    roots.append(Path("/opt/rocm"))
    for root in roots:
        for name in ("clang.exe", "clang"):
            cand = root / "lib" / "llvm" / "bin" / name
            if cand.exists():
                return str(cand)
    return shutil.which("amdclang") or None


def _compile(stem: str) -> bytes:
    src_path = _DIR / f"{stem}.hip"
    src = src_path.read_bytes()
    key = hashlib.sha256(src + ",".join(_ARCHS).encode()).hexdigest()[:16]
    out = cache_dir() / f"{stem}-{key}.hsaco"
    if out.exists():
        return out.read_bytes()
    clang = _find_clang()
    if clang is None:
        raise RuntimeError("ROCm clang not found (install the ROCm SDK wheels or set ROCM_PATH)")
    # Device-only, header-free: no HIP runtime headers, no device libs, no host compiler.
    tmp = out.with_suffix(".tmp")
    cmd = [clang, "-x", "hip", "--cuda-device-only", "-nogpuinc", "-nogpulib",
           "-std=c++17", "-O3", "-o", str(tmp), str(src_path)]
    cmd += [f"--offload-arch={a}" for a in _ARCHS]
    logger.info("compiling %s: %s", src_path.name, " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"clang failed on {src_path.name} ({res.returncode}):\n{res.stdout}\n{res.stderr}")
    tmp.replace(out)
    return out.read_bytes()


def _load_hip_runtime() -> ctypes.CDLL:
    names = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"] if os.name == "nt" else \
        ["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"]
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL  # type: ignore[attr-defined]
    last = None
    for n in names:
        try:
            # torch has already loaded the runtime; this returns the same instance.
            return loader(n)
        except OSError as e:  # noqa: PERF203
            last = e
    raise RuntimeError(f"could not load the HIP runtime: {last}")


_P = ctypes.c_void_p


class _Kernels:
    def __init__(self) -> None:
        torch.cuda.init()
        self.hip = hip = _load_hip_runtime()
        hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        hip.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ]
        self._images = []
        self.funcs: dict[str, ctypes.c_void_p] = {}
        for stem, names in _SOURCES.items():
            image = ctypes.create_string_buffer(_compile(stem))  # must outlive the module
            self._images.append(image)
            module = ctypes.c_void_p()
            self._check(hip.hipModuleLoadData(ctypes.byref(module), image), f"hipModuleLoadData({stem})")
            for name in names:
                f = ctypes.c_void_p()
                self._check(hip.hipModuleGetFunction(ctypes.byref(f), module, name.encode()), name)
                self.funcs[name] = f
        self.choice: dict[tuple[int, int, int], str] = {}
        self._tune_file = cache_dir() / "fast_gemv_tuning.json"
        try:
            self.choice = {tuple(map(int, k.split(","))): v
                           for k, v in json.loads(self._tune_file.read_text()).items()}
        except (OSError, ValueError):
            pass

    @staticmethod
    def _check(err: int, what: str) -> None:
        if err != 0:
            raise RuntimeError(f"{what} failed with hipError {err}")

    def launch(self, name: str, grid: tuple[int, int], block: int, args: list) -> None:
        """args: list of ctypes scalars (c_void_p / c_int / c_float)."""
        params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._check(self.hip.hipModuleLaunchKernel(
            self.funcs[name], grid[0], grid[1], 1, block, 1, 1, 0, stream, params, None), name)

    # ---------------------------------------------------------------- GEMV
    def gemv(self, name: str, w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        N, K = w.shape
        grid = (N + _GEMV_VARIANTS[name] - 1) // _GEMV_VARIANTS[name]
        self.launch(name, (grid, 1), 256, [_P(w.data_ptr()), _P(x.data_ptr()), _P(y.data_ptr()),
                                           ctypes.c_int(N), ctypes.c_int(K // 8)])

    def gemv_i8(self, name: str, q: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        N, K = q.shape
        grid = (N + _I8_VARIANTS[name] - 1) // _I8_VARIANTS[name]
        self.launch(name, (grid, 1), 256, [_P(q.data_ptr()), _P(scale.data_ptr()), _P(x.data_ptr()),
                                           _P(y.data_ptr()), ctypes.c_int(N), ctypes.c_int(K // 16)])

    def pick_i8(self, q: torch.Tensor, scale: torch.Tensor) -> str:
        N, K = q.shape
        key = (N, K, q.device.index or 0, 8)
        name = self.choice.get(key)
        if name is not None and (name == "dequant" or name in self.funcs):
            return name
        if torch.cuda.is_current_stream_capturing():
            return "dequant"  # never benchmark inside graph capture
        x = torch.randn(1, K, device=q.device, dtype=torch.bfloat16)
        y = torch.empty(1, N, device=q.device, dtype=torch.bfloat16)
        ref = F.linear(x.float(), q.float() * scale[:, None])
        best, best_t = "dequant", float("inf")
        for cand in _I8_VARIANTS:
            self.gemv_i8(cand, q, scale, x, y)
            err = (y.float() - ref).abs().max().item()
            if err > 2e-2 * max(1.0, ref.abs().max().item()):
                logger.warning("int8 gemv %s wrong for %sx%s (err %.3g), skipping", cand, N, K, err)
                continue
            dt = _time_in_graph(lambda c=cand: self.gemv_i8(c, q, scale, x, y))
            if dt < best_t:
                best, best_t = cand, dt
        logger.info("int8 gemv %dx%d -> %s (%.1f us)", N, K, best, best_t * 1e6)
        self.choice[key] = best
        self._save()
        return best

    def _save(self) -> None:
        try:
            self._tune_file.write_text(json.dumps({",".join(map(str, k)): v for k, v in self.choice.items()}))
        except OSError:
            pass

    def pick(self, w: torch.Tensor) -> str:
        N, K = w.shape
        key = (N, K, w.device.index or 0)
        name = self.choice.get(key)
        if name is not None and (name == "torch" or name in self.funcs):
            return name
        if torch.cuda.is_current_stream_capturing():
            return "torch"  # never benchmark inside graph capture
        name = self._autotune(w)
        self.choice[key] = name
        self._save()
        return name

    def _autotune(self, w: torch.Tensor) -> str:
        N, K = w.shape
        x = torch.randn(1, K, device=w.device, dtype=w.dtype)
        y = torch.empty(1, N, device=w.device, dtype=w.dtype)
        ref = F.linear(x, w).float()
        cands = {"torch": lambda: F.linear(x, w)}
        for name in _GEMV_VARIANTS:
            self.gemv(name, w, x, y)
            err = (y.float() - ref).abs().max().item()
            tol = 2e-2 * max(1.0, ref.abs().max().item())
            if err > tol:
                logger.warning("fast gemv %s wrong for %sx%s (err %.3g), skipping", name, N, K, err)
                continue
            cands[name] = (lambda n=name: self.gemv(n, w, x, y))
        # Time inside a HIP graph: that is how the model runs, and it removes
        # the Python/ctypes launch overhead that would otherwise dominate.
        best, best_t = "torch", float("inf")
        for name, fn in cands.items():
            try:
                dt = _time_in_graph(fn)
            except Exception:  # noqa: BLE001
                continue
            if dt < best_t:
                best, best_t = name, dt
        logger.info("fast gemv %dx%d -> %s (%.1f us)", N, K, best, best_t * 1e6)
        return best


def _time_in_graph(fn, reps: int = 10, iters: int = 20) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (iters * reps)


def _get() -> "_Kernels | None":
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                if not (_env_on("PERSONAPLEX_KERNELS") and is_rocm_build() and torch.cuda.is_available()):
                    _state = False
                else:
                    try:
                        _state = _Kernels()
                    except Exception as e:  # noqa: BLE001
                        logger.warning("custom HIP kernels unavailable, using PyTorch ops: %s", e)
                        print(f"[personaplex-rocm] custom HIP kernels unavailable: {e}", file=sys.stderr)
                        _state = False
    return _state or None


def available() -> bool:
    return _get() is not None


def _tracing() -> bool:
    return torch.compiler.is_compiling()


# --------------------------------------------------------------------- linear
def fast_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    if _QUANT:
        qe = _QUANT.get(id(w))
        if qe is not None:
            return _int8_linear(x, qe[0], qe[1], b)
    K = x.shape[-1]
    if (
        b is None
        and not _tracing()
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and w.dim() == 2
        and w.shape[1] == K
        and K % 8 == 0
        and x.numel() == K
        and w.is_contiguous()
        and w.data_ptr() % 16 == 0
        and _env_on("PERSONAPLEX_FAST_GEMV")
    ):
        k = _get()
        if k is not None:
            name = k.pick(w)
            if name != "torch":
                xc = x.contiguous()
                if xc.data_ptr() % 16 == 0:
                    y = torch.empty(x.shape[:-1] + (w.shape[0],), device=x.device, dtype=x.dtype)
                    k.gemv(name, w, xc, y)
                    return y
    return F.linear(x, w, b)


# ------------------------------------------------------------------- rms norm
def rms_norm_supported(x: torch.Tensor, alpha: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and alpha.dtype == torch.bfloat16
        and x.is_contiguous()
        and alpha.is_contiguous()
        and alpha.numel() == x.shape[-1]
        and x.shape[-1] % 2 == 0
        and not _tracing()
        and _env_on("PERSONAPLEX_FUSED_NORM")
        and _get() is not None
    )


def rms_norm(x: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    k = _get()
    assert k is not None
    D = x.shape[-1]
    rows = x.numel() // D
    y = torch.empty_like(x)
    k.launch("rmsnorm_bf16", (rows, 1), 256, [_P(x.data_ptr()), _P(alpha.data_ptr()), _P(y.data_ptr()),
                                              ctypes.c_int(D), ctypes.c_float(eps)])
    return y


# ------------------------------------------------------------------ attention
def attention_step_supported(projected: torch.Tensor, cache: torch.Tensor, head_dim: int) -> bool:
    return (
        projected.is_cuda
        and projected.dtype == torch.bfloat16
        and cache.dtype == torch.bfloat16
        and cache.shape[1] == 1          # batch 1
        and projected.numel() == 3 * cache.shape[2] * head_dim
        and head_dim in (64, 128)
        and cache.is_contiguous()
        and not _tracing()
        and _env_on("PERSONAPLEX_FUSED_ATTN")
        and _get() is not None
    )


def attention_step(projected: torch.Tensor, cache: torch.Tensor, end_offset: torch.Tensor,
                   offset: torch.Tensor, rope_freqs: torch.Tensor | None,
                   context: int | None) -> torch.Tensor:
    """One streaming token through RoPE + ring KV cache + causal attention.

    projected: [1, 1, 3*H*D] (q, k, v), cache: [2, 1, H, C, D], end_offset and
    offset: int64 [1] device tensors. Mutates the cache and bumps end_offset.
    Returns [1, 1, H*D].
    """
    k = _get()
    assert k is not None
    _, _, H, C, D = cache.shape
    proj = projected.contiguous()
    q_rot = torch.empty(H * D, device=proj.device, dtype=proj.dtype)
    k.launch("rope_kv_write", (1, 1), 256, [
        _P(proj.data_ptr()), _P(rope_freqs.data_ptr() if rope_freqs is not None else 0),
        _P(offset.data_ptr()), _P(end_offset.data_ptr()), _P(cache.data_ptr()), _P(q_rot.data_ptr()),
        ctypes.c_int(H), ctypes.c_int(D), ctypes.c_int(C)])
    # Split the cache so that ~H*S blocks fill the GPU; chunk must be <= 256.
    S = max(1, math.ceil(C / 192)) if C > 256 else 1
    chunk = math.ceil(C / S)
    assert chunk <= 256
    out = torch.empty(1, 1, H * D, device=proj.device, dtype=proj.dtype)
    if S > 1:
        part_o = torch.empty(H * S * D, device=proj.device, dtype=torch.float32)
        part_ml = torch.empty(H * S * 2, device=proj.device, dtype=torch.float32)
    else:
        part_o = part_ml = out  # unused by the kernel when S == 1
    name = "attn_decode_d128" if D == 128 else "attn_decode_d64"
    k.launch(name, (H, S), 256, [
        _P(q_rot.data_ptr()), _P(cache.data_ptr()), _P(end_offset.data_ptr()), _P(offset.data_ptr()),
        _P(part_o.data_ptr()), _P(part_ml.data_ptr()), _P(out.data_ptr()),
        ctypes.c_int(H), ctypes.c_int(C), ctypes.c_int(S), ctypes.c_int(chunk),
        ctypes.c_int(-1 if context is None else int(context)), ctypes.c_float(1.0 / math.sqrt(D))])
    if S > 1:
        k.launch("attn_combine", (H, 1), D, [_P(part_o.data_ptr()), _P(part_ml.data_ptr()),
                                             _P(out.data_ptr()), ctypes.c_int(S), ctypes.c_int(D)])
    return out


# ----------------------------------------------------------------- int8 weights
def _int8_linear(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
                 b: torch.Tensor | None) -> torch.Tensor:
    N, K = q.shape
    k = _get()
    if (k is not None and b is None and x.dtype == torch.bfloat16 and x.numel() == K
            and K % 16 == 0 and not _tracing()):
        xc = x.contiguous()
        y = torch.empty(x.shape[:-1] + (N,), device=x.device, dtype=x.dtype)
        name = k.pick_i8(q, scale)
        if name != "dequant" and xc.data_ptr() % 16 == 0:
            k.gemv_i8(name, q, scale, xc, y)
            return y
    # Generic (slow) path, e.g. several tokens at once: dequantize on the fly.
    w = (q.to(torch.float32) * scale[:, None]).to(x.dtype)
    return F.linear(x, w, b)


@torch.no_grad()
def quantize_tensor_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row absmax int8 quantization: w ~= q * scale[:, None]."""
    scale = (w.abs().amax(dim=1).float() / 127.0).clamp_min(1e-12)
    q = torch.empty(w.shape, device=w.device, dtype=torch.int8)
    for i in range(0, w.shape[0], 4096):  # chunked to bound temporary memory
        q[i:i + 4096] = torch.round(w[i:i + 4096].float() / scale[i:i + 4096, None]).clamp_(-127, 127)
    return q, scale


def register_int8(p: torch.nn.Parameter, q: torch.Tensor, scale: torch.Tensor) -> None:
    """Make `fast_linear(x, p)` use the int8 weight `q` with per-row `scale`."""
    _QUANT[id(p)] = (q, scale, p)


@torch.no_grad()
def quantize_int8_(params: list[torch.nn.Parameter]) -> int:
    """Replace 2D bf16 weights by int8 + per-row scale (symmetric absmax).

    The Parameter object stays (same identity, now an empty tensor on the same
    device/dtype) so `fast_linear(x, param)` finds its quantized version.
    Returns the number of bytes saved.
    """
    saved = 0
    for p in params:
        w = p.data
        if w.dim() != 2 or w.shape[1] % 16 != 0 or id(p) in _QUANT:
            continue
        q, scale = quantize_tensor_int8(w)
        register_int8(p, q, scale)
        saved += w.numel() * w.element_size() - q.numel() - scale.numel() * 4
        p.data = torch.empty(0, device=w.device, dtype=w.dtype)
    if saved:
        torch.cuda.empty_cache()
    return saved


def quantize_lm_int8_(lm: torch.nn.Module) -> int:
    """int8 weight-only quantization of the 7B temporal transformer + text head.

    The depformer stays bf16 (its per-codebook weights are sliced views, and it
    is latency- rather than bandwidth-bound anyway).
    """
    params: list[torch.nn.Parameter] = []
    for layer in lm.transformer.layers:
        attn = layer.self_attn
        if not attn.weights_per_step:
            params += [attn.in_proj_weight, attn.out_proj.weight]
        gating = getattr(layer, "gating", None)
        if gating is not None and hasattr(gating, "linear_in"):
            params += [gating.linear_in.weight, gating.linear_out.weight]
    params.append(lm.text_linear.weight)
    return quantize_int8_(params)
