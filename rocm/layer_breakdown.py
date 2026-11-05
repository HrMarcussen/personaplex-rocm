"""Graph-captured timing of the pieces of one main-transformer layer (random weights).

Kineto/roctracer are not available in the Windows ROCm build, so instead of a
kernel trace we capture each sub-block in a HIP graph and time replays.
"""
import time

import moshi  # noqa: F401
import torch
from moshi.models import loaders


def time_graph(fn, reps=20, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters / reps


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    lm = loaders.get_moshi_lm(None, device=dev)
    lm.streaming_forever(1)
    layer = lm.transformer.layers[0]
    # Fill the KV cache context a bit (like mid-conversation)
    x = torch.randn(1, 1, 4096, device=dev, dtype=torch.bfloat16)
    for _ in range(200):
        layer(x)
    attn = layer.self_attn
    res = {
        "full layer": lambda: layer(x),
        "norm1": lambda: layer.norm1(x),
        "self_attn (proj+rope+kv+sdpa+out)": lambda: attn(x, x, x),
        "gating ffn": lambda: layer.gating(x),
        "norm2": lambda: layer.norm2(x),
    }
    print("per-call time inside a HIP graph:")
    for k, fn in res.items():
        print(f"  {k:36s} {time_graph(fn) * 1e6:8.1f} us")
    from moshi.kernels import fast_linear
    w_qkv = attn.in_proj_weight
    print(f"  {'qkv gemv only':36s} {time_graph(lambda: fast_linear(x, w_qkv)) * 1e6:8.1f} us")
    print(f"  {'out-proj gemv only':36s} {time_graph(lambda: fast_linear(x, attn.out_proj.weight)) * 1e6:8.1f} us")
    q = torch.randn(1, 32, 1, 128, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(1, 32, 3000, 128, device=dev, dtype=torch.bfloat16)
    mask = torch.ones(1, 3000, dtype=torch.bool, device=dev)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    print(f"  {'sdpa only (3000 ctx, bool mask)':36s} {time_graph(lambda: sdpa(q, kv, kv, mask)) * 1e6:8.1f} us")
    print(f"  {'sdpa only (3000 ctx, no mask)':36s} {time_graph(lambda: sdpa(q, kv, kv)) * 1e6:8.1f} us")
    print(f"  {'empty-ish op (x+1)':36s} {time_graph(lambda: x + 1) * 1e6:8.1f} us")


if __name__ == "__main__":
    main()
