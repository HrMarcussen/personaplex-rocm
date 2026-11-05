# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Per-frame latency benchmark for the PersonaPlex/Moshi streaming pipeline.

Runs the exact per-frame work the realtime server does (Mimi encode,
LM step incl. depformer, Mimi decode) and reports timings. Realtime
needs < 80 ms per frame (12.5 Hz).

With --random it builds the 7B architecture with random weights, so it
works before the gated HF weights are downloaded.
"""
import argparse
import os
import time

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import torch  # noqa: E402

from moshi.models import loaders  # noqa: E402
from moshi.models.lm import LMGen  # noqa: E402


def build(random_weights: bool, device: torch.device):
    if random_weights:
        orig = loaders.load_model
        loaders.load_model = lambda *a, **k: None
        try:
            mimi = loaders.get_mimi("random.safetensors", device)
        finally:
            loaders.load_model = orig
        lm = loaders.get_moshi_lm(None, device=device)
    else:
        from huggingface_hub import hf_hub_download
        mimi_w = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        lm_w = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        mimi = loaders.get_mimi(mimi_w, device)
        lm = loaders.get_moshi_lm(lm_w, device=device)
    lm.eval()
    return mimi, lm


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"torch {torch.__version__} hip={torch.version.hip} dev={torch.cuda.get_device_name(0)}")
    print(f"NO_CUDA_GRAPH={os.environ.get('NO_CUDA_GRAPH', '')!r} NO_TORCH_COMPILE={os.environ.get('NO_TORCH_COMPILE')!r}")
    t0 = time.time()
    mimi, lm = build(args.random, device)
    print(f"loaded in {time.time() - t0:.1f}s, vram {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    frame = int(mimi.sample_rate / mimi.frame_rate)
    gen = LMGen(lm, device=device, sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate,
                audio_silence_frame_cnt=int(0.5 * mimi.frame_rate))
    mimi.streaming_forever(1)
    gen.streaming_forever(1)

    times = {"enc": [], "lm": [], "dec": [], "total": []}
    for i in range(args.frames + 10):
        chunk = (torch.randn(1, 1, frame, device=device) * 0.05)
        torch.cuda.synchronize()
        a = time.perf_counter()
        codes = mimi.encode(chunk)
        torch.cuda.synchronize()
        b = time.perf_counter()
        tokens = gen.step(codes[:, :, 0:1])
        torch.cuda.synchronize()
        c = time.perf_counter()
        if tokens is not None:
            pcm = mimi.decode(tokens[:, 1:9])
            pcm.cpu()
        torch.cuda.synchronize()
        d = time.perf_counter()
        if i >= 10:  # skip warmup / graph capture
            times["enc"].append(b - a)
            times["lm"].append(c - b)
            times["dec"].append(d - c)
            times["total"].append(d - a)
        elif i in (0, 1, 2):
            print(f"  warmup frame {i}: {1000 * (d - a):.1f} ms")

    def stats(xs):
        xs = sorted(xs)
        return f"mean {1000 * sum(xs) / len(xs):6.1f}  p50 {1000 * xs[len(xs) // 2]:6.1f}  p95 {1000 * xs[int(len(xs) * .95)]:6.1f} ms"

    for k, v in times.items():
        print(f"{k:>6}: {stats(v)}")
    mean_total = sum(times["total"]) / len(times["total"])
    print(f"realtime factor: {0.080 / mean_total:.2f}x  (>1.0 = faster than realtime; frame budget 80 ms)")
    print(f"peak vram {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
