"""Fused HIP attention / RMSNorm vs. the original PyTorch path."""
import os

import moshi  # noqa: F401
import torch

from moshi.modules.rope import RotaryEmbedding
from moshi.modules.transformer import StreamingMultiheadAttention, RMSNorm


def run_mha(mha, xs, fused: bool):
    os.environ["PERSONAPLEX_FUSED_ATTN"] = "1" if fused else "0"
    outs = []
    with torch.no_grad(), mha.streaming(1):
        for x in xs:
            outs.append(mha(x, x, x).float())
    return torch.cat(outs, 1)


def check_mha(dim, heads, context, rope, steps=300, wps=0):
    torch.manual_seed(0)
    mha = StreamingMultiheadAttention(
        dim, heads, causal=True, context=context,
        rope=RotaryEmbedding(10000) if rope else None, weights_per_step=wps,
        device="cuda", dtype=torch.bfloat16)
    for p in mha.parameters():
        torch.nn.init.normal_(p, std=dim ** -0.5)
    n = wps if wps else steps
    xs = [torch.randn(1, 1, dim, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
    ref = run_mha(mha, xs, False)
    out = run_mha(mha, xs, True)
    err = (out - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    ok = rel < 3e-2
    print(f"MHA dim={dim} heads={heads} ctx={context} rope={rope} wps={wps} steps={n}: "
          f"max abs err {err:.4f}, rel {rel:.4f} -> {'OK' if ok else 'FAIL'}")
    return ok


def check_norm(dim):
    norm = RMSNorm(dim, eps=1e-5, dtype=torch.float32, device="cuda").to(torch.bfloat16)
    torch.nn.init.normal_(norm.alpha, 1.0, 0.2)
    x = torch.randn(1, 1, dim, device="cuda", dtype=torch.bfloat16) * 3
    os.environ["PERSONAPLEX_FUSED_NORM"] = "0"
    ref = norm(x).float()
    os.environ["PERSONAPLEX_FUSED_NORM"] = "1"
    out = norm(x).float()
    err = (out - ref).abs().max().item()
    ok = err <= 2 * ref.abs().max().item() * 2 ** -8
    print(f"RMSNorm dim={dim}: max abs err {err:.5f} -> {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [
        check_norm(4096), check_norm(1024),
        check_mha(4096, 32, context=3000, rope=True, steps=200),   # main transformer
        check_mha(4096, 32, context=50, rope=True, steps=300),     # ring wrap-around x6
        check_mha(1024, 16, context=8, rope=False, wps=16),        # depformer
        check_mha(512, 8, context=40, rope=True, steps=120),       # head dim 64 + rope
    ]
    print("ALL OK" if all(results) else "SOME FAILED")
