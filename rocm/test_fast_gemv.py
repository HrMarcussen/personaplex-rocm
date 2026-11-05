"""Correctness + speed of the custom HIP GEMV vs F.linear for Moshi's shapes."""
import logging
import time

import moshi  # noqa: F401  (sets ROCm env defaults before torch use)
import torch
import torch.nn.functional as F

from moshi import kernels

logging.basicConfig(level=logging.INFO)
SHAPES = [(4096, 12288), (4096, 4096), (4096, 22528), (11264, 4096), (1024, 3072),
          (1024, 1024), (1024, 8448), (4224, 1024), (4096, 1024), (1024, 2048), (4096, 32000)]


def bench(fn, n=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    assert kernels.available(), "kernels failed to load"
    k = kernels._get()
    for K, N in SHAPES:
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
        x = torch.randn(1, 1, K, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x.float(), w.float())
        y = kernels.fast_linear(x, w)
        err = (y.float() - ref).abs().max().item()
        err_t = (F.linear(x, w).float() - ref).abs().max().item()
        gb = w.numel() * 2 / 1e9
        t_torch = bench(lambda: F.linear(x, w))
        res = [f"torch {gb / t_torch:5.0f}GB/s {t_torch * 1e6:6.1f}us"]
        for name in kernels._GEMV_VARIANTS:
            yy = torch.empty(1, N, device="cuda", dtype=torch.bfloat16)
            t = bench(lambda: k.gemv(name, w, x.view(1, K), yy))
            res.append(f"{name[10:]} {gb / t:5.0f}GB/s {t * 1e6:6.1f}us")
        print(f"{K:>6}x{N:<6} pick={k.pick(w):<14} err={err:.4f} (torch err {err_t:.4f}) | " + " | ".join(res))

    # graph capture check
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    kernels.fast_linear(x, w)  # warm / tune
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            out = kernels.fast_linear(x, w)
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    err = (out.float() - F.linear(x.float(), w.float())).abs().max().item()
    print(f"graph replay err {err:.4f} (picked {k.pick(w)})")


if __name__ == "__main__":
    main()
