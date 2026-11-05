# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Kernel-level GPU time of one LM step (random 7B weights, eager)."""
import os
os.environ["NO_CUDA_GRAPH"] = "1"
import moshi  # noqa: F401,E402
import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402
from moshi.models import loaders  # noqa: E402
from moshi.models.lm import LMGen  # noqa: E402


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    lm = loaders.get_moshi_lm(None, device=dev)
    gen = LMGen(lm, device=dev)
    gen.streaming_forever(1)
    codes = torch.zeros(1, 8, 1, dtype=torch.long, device=dev)
    for _ in range(30):
        gen.step(codes)
    torch.cuda.synchronize()
    n = 3
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            gen.step(codes)
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA or getattr(e, "self_device_time_total", 0) > 0:
            t = getattr(e, "self_device_time_total", None) or e.self_cuda_time_total
            rows.append((t / n, e.count / n, e.key))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    cnt = sum(r[1] for r in rows)
    print(f"total GPU kernel time per step: {tot / 1000:.2f} ms over {cnt:.0f} kernels")
    for t, c, k in rows[:30]:
        print(f"{t / 1000:7.3f} ms  {c:6.0f}x  {k[:110]}")


if __name__ == "__main__":
    main()
