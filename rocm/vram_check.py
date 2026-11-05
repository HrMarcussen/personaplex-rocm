# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Peak VRAM of the full streaming pipeline on the real weights (honours PERSONAPLEX_INT8)."""
import os

import moshi  # noqa: F401
import torch
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from moshi.models.lm import LMGen


@torch.no_grad()
def main():
    repo = loaders.DEFAULT_REPO
    torch.cuda.reset_peak_memory_stats()
    mimi = loaders.get_mimi(hf_hub_download(repo, loaders.MIMI_NAME), "cuda")
    lm = loaders.get_moshi_lm(hf_hub_download(repo, loaders.MOSHI_NAME), device="cuda")
    load_peak = torch.cuda.max_memory_allocated() / 2**30
    after_load = torch.cuda.memory_allocated() / 2**30
    gen = LMGen(lm, device="cuda", sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)
    mimi.streaming_forever(1)
    gen.streaming_forever(1)
    for _ in range(50):
        codes = mimi.encode(torch.zeros(1, 1, 1920, device="cuda"))
        tokens = gen.step(codes[:, :, :1])
        if tokens is not None:
            mimi.decode(tokens[:, 1:9])
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"PERSONAPLEX_INT8={os.environ.get('PERSONAPLEX_INT8', '0')}: "
          f"after load {after_load:.1f} GiB, peak while loading {load_peak:.1f} GiB, "
          f"peak while running {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
          f"reserved by PyTorch {torch.cuda.memory_reserved() / 2**30:.1f} GiB, "
          f"device used {(total - free) / 2**30:.1f}/{total / 2**30:.0f} GiB")


if __name__ == "__main__":
    main()
