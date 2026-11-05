# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Time graphed main transformer vs. depformer inside LMGen.step (random weights)."""
import os
import time

os.environ.setdefault("NO_TORCH_COMPILE", "1")
import torch  # noqa: E402
from moshi.models import loaders  # noqa: E402
from moshi.models.lm import LMGen  # noqa: E402

timings: dict[str, list[float]] = {"main": [], "depth": [], "step": []}


def wrap(obj, name):
    orig = obj.__call__

    class Timed:
        def __call__(self, *a):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = orig(*a)
            torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - t)
            return r
    return Timed()


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    lm = loaders.get_moshi_lm(None, device=dev)
    gen = LMGen(lm, device=dev)
    gen.streaming_forever(1)
    st = gen._streaming_state
    st.graphed_main = wrap(st.graphed_main, "main")
    st.graphed_depth = wrap(st.graphed_depth, "depth")
    codes = torch.zeros(1, 8, 1, dtype=torch.long, device=dev)
    for i in range(60):
        torch.cuda.synchronize()
        t = time.perf_counter()
        gen.step(codes)
        torch.cuda.synchronize()
        timings["step"].append(time.perf_counter() - t)
    for k, v in timings.items():
        v = v[20:]
        print(f"{k:>6}: mean {1000 * sum(v) / len(v):6.2f} ms  (n={len(v)})")


if __name__ == "__main__":
    main()
