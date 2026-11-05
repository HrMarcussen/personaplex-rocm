# Technical notes

How the AMD/Windows port of PersonaPlex works, what was changed, and how it was validated.
For installation and usage see the [README](../README.md). This is an independent community
project, not affiliated with or endorsed by NVIDIA or AMD.


## Performance history

How each step changed the time to produce one 80 ms audio frame on an RX 7900 XTX. During
development these were measured with `rocm/bench_step.py --random` (the 7B architecture with random
weights, which has identical compute); the final numbers were then confirmed on the real weights.

| Step | Per frame (dev bench) | vs. real time |
|---|---:|---:|
| NVIDIA's code on ROCm PyTorch, eager | 211 ms | 0.38x |
| + HIP graphs (upstream's CUDA-graph code works on ROCm) | 88 ms | 0.91x |
| + TunableOp GEMM tuning, unused second Mimi codec removed | ~64 ms | ~1.25x |
| + custom GEMV, fused attention/RoPE/KV-cache, fused RMSNorm | 39 ms | 2.0x |
| + whole-step graph capture of the Mimi codec | 32 ms | 2.5x |
| optional: `PERSONAPLEX_INT8=1` | 26 ms | 3.0x |

On the real PersonaPlex weights (`rocm/validate_offline.py`, 40 s test conversation, voice NATF2):
**32.5 ms** mean / 34.9 ms worst frame in bf16, **23.8 ms** with int8, and 57.7 ms with the custom
kernels switched off. A live websocket session (`rocm/ws_client_test.py`) streamed 45 s of speech in
real time with zero drift and a constant 0.215 s audio round trip, of which ~160 ms is inherent to
the model (80 ms frames plus one frame of acoustic delay).

With `PERSONAPLEX_INT8=1` the GPU holds ~13.0 GB instead of ~18.4 GB in total (the big matrices are
quantized while loading; measured with `rocm/vram_check.py`), so 16 GB cards should fit. int8 is
opt-in: its text output stays coherent, but judge the audio on your setup.

## What was changed, and why

| Area | Change |
|---|---|
| `moshi/pyproject.toml` | Drop upstream's `torch < 2.5` cap (AMD's Windows wheels are torch 2.9.1+rocm7.2.1); require torch ≥ 2.3. |
| `moshi/_platform_env.py` | Before torch loads: disable `torch.compile` when Triton is missing (it is on Windows ROCm); enable PyTorch **TunableOp** on ROCm, which benchmarks rocBLAS/hipBLASLt kernels per GEMM shape (LM step 63 → 52 ms). |
| `server.py`, `offline.py` | Removed a second Mimi codec instance whose encode/decode outputs were always discarded (~12 ms per frame of pure overhead). The server binds `--host` and prints a `localhost` URL (browsers only allow mic access on localhost or https). |
| `kernels/gemv.hip` | Batch-1 bf16 GEMV kernels. Every projection in token-by-token decoding is a matrix-vector product; the stock BLAS kernels reach only 150–500 GB/s on several of Moshi's shapes, these reach ~850–885 GB/s (peak 960). Autotuned per shape against `F.linear`, so a shape only switches when it is faster. |
| `kernels/fused.hip` | One kernel for RoPE + ring-KV-cache write, one split-K flash-decoding kernel with Moshi's exact ring-position/context masking (+ a combine kernel), and a fused fp32 RMSNorm. Each replaces 10–40 tiny PyTorch ops. |
| `modules/streaming.py`, `utils/compile.py`, `models/compression.py` | Streaming conv state is updated in place when its shape is unchanged (same values, stable addresses), so a whole Mimi encode/decode step can be captured in one HIP graph (`StepGraph`). Codes are bit-identical to eager. |
| `kernels/gemv.hip` (int8), `models/loaders.py` | Optional int8 weight-only quantization (per-row absmax) of the temporal transformer and text head, done while loading. |
| `server.py` (Windows) | Requests 1 ms timer resolution: the server's audio loops poll with `asyncio.sleep(0.001)`, which really sleeps ~15 ms at Windows' default resolution (measured), adding up to ~30 ms of latency per frame. |

The kernels are **header-free HIP** compiled device-only with
`clang -x hip --cuda-device-only -nogpuinc -nogpulib`, loaded with
`hipModuleLoadData` and launched through `hipModuleLaunchKernel` via ctypes on
PyTorch's current stream. That avoids needing a host C++ compiler on Windows, and HIP
graph capture records them like any other kernel. Everything falls back to stock
PyTorch automatically on other hardware, or with `PERSONAPLEX_KERNELS=0`.

Switches: `PERSONAPLEX_KERNELS=0` (all custom kernels off), `PERSONAPLEX_FAST_GEMV=0`,
`PERSONAPLEX_FUSED_ATTN=0`, `PERSONAPLEX_FUSED_NORM=0`, `NO_CUDA_GRAPH=1`,
`PYTORCH_TUNABLEOP_ENABLED=0`.

## Tests and benchmarks (`rocm/`)

| Script | What it does |
|---|---|
| `test_fast_gemv.py` | GEMV correctness and bandwidth vs `F.linear` for all Moshi shapes, plus graph capture. |
| `test_fused.py` | Fused attention vs the PyTorch path over hundreds of tokens (incl. ring-buffer wrap-around, depformer config, head dim 64/128); RMSNorm. |
| `test_mimi_graph.py` | Graphed vs eager Mimi on real audio (bit-identical codes). |
| `bench_step.py --random` | Per-frame latency of the full pipeline (works without weights). |
| `validate_offline.py` | Real weights: streams a wav through the model with kernels on and off and compares. |
| `ws_client_test.py` | Headless client for the running server (same protocol as the web UI): streams a wav at real-time pace, reports lag/drift and the model's text. |
| `test_int8.py` | int8 GEMV vs a dequantized reference. |
| `vram_check.py` | Peak VRAM of the full pipeline on the real weights (bf16 or int8). |
| `split_lm.py`, `layer_breakdown.py` | Where the time goes (main transformer vs depformer, per-layer pieces). |

## Related work

- [gain774/moshi-rocm](https://github.com/gain774/moshi-rocm) is a HIP build of the C++
  `moshi.cpp` engine (Linux, ROCm 6.x). This port instead keeps NVIDIA's reference PyTorch
  code and web UI and targets native Windows.
