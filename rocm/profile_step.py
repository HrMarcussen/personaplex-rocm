# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Profile GPU kernel time of one LM step (random 7B weights)."""
import os
import sys
import time

os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ["NO_CUDA_GRAPH"] = "1"

import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

from moshi.models import loaders  # noqa: E402
from moshi.models.lm import LMGen  # noqa: E402


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    lm = loaders.get_moshi_lm(None, device=dev)
    lm.eval()
    gen = LMGen(lm, device=dev)
    gen.streaming_forever(1)
    codes = torch.zeros(1, 8, 1, dtype=torch.long, device=dev)
    for _ in range(10):
        gen.step(codes)
    torch.cuda.synchronize()

    # Microbench a few GEMV shapes vs. bandwidth roofline
    for (k, n) in [(4096, 4096 * 3), (4096, 11264 * 2), (11264, 4096), (1024, 8448), (4096, 32000)]:
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
        x = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
        for _ in range(5):
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 50
        print(f"gemv bf16 {k}x{n}: {dt * 1e6:7.1f} us  -> {w.numel() * 2 / dt / 1e9:6.0f} GB/s")
        wh = w.half(); xh = x.half()
        for _ in range(5):
            torch.nn.functional.linear(xh, wh)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            torch.nn.functional.linear(xh, wh)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 50
        print(f"gemv fp16 {k}x{n}: {dt * 1e6:7.1f} us  -> {w.numel() * 2 / dt / 1e9:6.0f} GB/s")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            gen.step(codes)
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70)
    print(table)


if __name__ == "__main__":
    main()
