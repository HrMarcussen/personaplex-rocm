"""Graphed Mimi streaming encode/decode vs. eager, on real audio, plus timing."""
import os
import time

import moshi  # noqa: F401
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from moshi.models.lm import load_audio

REPO = os.environ.get("MIMI_REPO", "kyutai/moshiko-pytorch-bf16")
WAV = os.path.join(os.path.dirname(__file__), "..", "assets", "test", "input_assistant.wav")


@torch.no_grad()
def run(mimi, pcm, graphs: bool, frames: int, reset_at: int):
    os.environ["NO_CUDA_GRAPH"] = "" if graphs else "1"
    mimi.reset_streaming()
    F = 1920
    codes_all, pcm_all, t_enc, t_dec = [], [], [], []
    for i in range(frames):
        if i == reset_at:
            mimi.reset_streaming()
        x = torch.from_numpy(pcm[None, None, i * F:(i + 1) * F]).cuda()
        torch.cuda.synchronize(); a = time.perf_counter()
        c = mimi.encode(x)
        torch.cuda.synchronize(); b = time.perf_counter()
        y = mimi.decode(c) if c.shape[-1] else None
        torch.cuda.synchronize(); d = time.perf_counter()
        if i > 5 and i != reset_at and i != reset_at + 1:
            t_enc.append(b - a); t_dec.append(d - b)
        codes_all.append(c.cpu())
        if y is not None:
            pcm_all.append(y.cpu())
    return codes_all, torch.cat(pcm_all, -1), np.mean(t_enc) * 1e3, np.mean(t_dec) * 1e3


def main():
    mimi = loaders.get_mimi(hf_hub_download(REPO, loaders.MIMI_NAME), "cuda")
    mimi.streaming_forever(1)
    pcm = load_audio(WAV, mimi.sample_rate)[0].astype(np.float32)
    frames, reset_at = 120, 60
    ce, pe, te, tde = run(mimi, pcm, False, frames, reset_at)
    cg, pg, tg, tdg = run(mimi, pcm, True, frames, reset_at)
    same = sum(int(torch.equal(a, b)) for a, b in zip(ce, cg))
    err = (pe - pg).abs().max().item()
    print(f"codes identical in {same}/{len(ce)} frames; max pcm diff {err:.2e}")
    print(f"eager : encode {te:.2f} ms  decode {tde:.2f} ms")
    print(f"graphs: encode {tg:.2f} ms  decode {tdg:.2f} ms")
    # Note: upstream only ever feeds streaming Mimi single 1920-sample frames
    # (encode_from_sphn forces max_batch=1); multi-frame input after capture
    # was already rejected by upstream's own CUDAGraphed transformer wrapper.
    ok = same == len(ce) and err < 1e-3
    print("OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
