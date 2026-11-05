"""int8 weight-only GEMV vs a dequantized reference."""
import moshi  # noqa: F401
import torch
import torch.nn.functional as F

from moshi import kernels


def main():
    k = kernels._get()
    assert k is not None
    for K, N in [(64, 8), (4096, 4096), (4096, 12288), (11264, 4096)]:
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K ** 0.5
        p = torch.nn.Parameter(w.clone(), requires_grad=False)
        kernels.quantize_int8_([p])
        q, scale, _ = kernels._QUANT[id(p)]
        deq = q.float() * scale[:, None]
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x.float(), deq)
        ref_bf16 = F.linear(x.float(), w.float())
        for name in kernels._I8_VARIANTS:
            y = torch.empty(1, N, device="cuda", dtype=torch.bfloat16)
            k.gemv_i8(name, q, scale, x, y)
            torch.cuda.synchronize()
            err = (y.float() - ref).abs().max().item()
            print(f"{K}x{N} {name}: err vs dequant {err:.4f} | y[:4]={y[0, :4].float().tolist()} ref[:4]={ref[0, :4].tolist()}")
        print(f"  quantization error vs bf16 weights: {(ref - ref_bf16).abs().max().item():.4f} "
              f"(output scale {ref_bf16.abs().max().item():.2f})")


if __name__ == "__main__":
    main()
