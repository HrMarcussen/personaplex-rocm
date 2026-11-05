# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Compare batch-1 matmul strategies on ROCm for the shapes Moshi uses."""
import time
import torch
import torch.nn.functional as F

dev = "cuda"
SHAPES = [(4096, 12288), (4096, 4096), (4096, 22528), (11264, 4096), (1024, 3072),
          (1024, 8448), (4224, 1024), (4096, 1024), (1024, 2048), (4096, 32000)]


def bench(fn, n=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def run(label):
    print(f"== {label}")
    for k, n in SHAPES:
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)  # nn.Linear layout (out, in)
        wt = w.t().contiguous()                                  # (in, out)
        x = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
        xv = x[0]
        gb = w.numel() * 2 / 1e9
        r = {
            "linear": bench(lambda: F.linear(x, w)),
            "mv(w,x)": bench(lambda: torch.mv(w, xv)),
            "x@wt": bench(lambda: x @ wt),
            "bmm-split4": None,
        }
        # split along N into 4 chunks (different kernel tiles)
        ws = list(w.chunk(4, 0))
        r["bmm-split4"] = bench(lambda: [F.linear(x, c) for c in ws])
        line = "  ".join(f"{name}={gb / t:5.0f}GB/s" for name, t in r.items())
        print(f"{k:>6}x{n:<6} {line}")


run("default blas: " + str(torch.backends.cuda.preferred_blas_library()))
for lib in ("cublas", "cublaslt"):
    try:
        torch.backends.cuda.preferred_blas_library(lib)
        run(lib)
    except Exception as e:  # noqa: BLE001
        print(lib, "failed:", e)
