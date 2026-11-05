# PersonaPlex on AMD Radeon, native Windows (ROCm)

This branch runs NVIDIA's [PersonaPlex](https://github.com/NVIDIA/personaplex), a
real-time, full-duplex speech-to-speech model that you can interrupt mid-sentence,
on an **AMD Radeon RX 7900 XTX under Windows 11**. No WSL, no Linux, no CUDA.
It uses AMD's official PyTorch-on-Windows ROCm wheels, plus a handful of custom HIP
kernels that make it comfortably faster than real time.

| RX 7900 XTX, per 80 ms audio frame | time | vs. realtime |
|---|---:|---:|
| Upstream code, ROCm PyTorch, eager | 211 ms | 0.38x |
| + HIP graphs (upstream CUDA-graph code, works on ROCm) | 88 ms | 0.91x |
| + TunableOp GEMM tuning, drop unused 2nd Mimi codec | ~64 ms | ~1.25x |
| + custom GEMV, fused attention/RoPE/KV-cache, fused RMSNorm | 39 ms | 2.0x |
| + whole-step graph capture of the Mimi codec | **32 ms** | **2.5x** |
| optional: `PERSONAPLEX_INT8=1` (int8 weights for the 7B transformer) | 26 ms | 3.0x |

With `PERSONAPLEX_INT8=1` the model needs ~10 GB of VRAM instead of ~18 GB (the big
matrices are quantized while loading), so 16 GB cards such as the RX 7800 XT or
RX 9070 XT should fit. It is opt-in until its quality has been compared on real weights.

## Requirements

- Windows 11, AMD Radeon RX 7000 (RDNA3) or RX 9000 (RDNA4) with ≥ 20 GB VRAM
  (the 7B model uses ~18 GB in bf16). Tested on RX 7900 XTX (gfx1100).
- AMD Adrenalin driver **26.2.2 or newer**.
- Python **3.12** from python.org (AMD's wheels are cp312 only).
- ~40 GB free disk (ROCm SDK + PyTorch ~10 GB, model ~17 GB).
- A Hugging Face account that has accepted the
  [PersonaPlex model license](https://huggingface.co/nvidia/personaplex-7b-v1).

No Visual Studio / MSVC and no CUDA are needed. The custom kernels are compiled by
the clang that ships inside AMD's ROCm SDK wheel.

## Install

```powershell
git clone -b rocm-windows <this repo> personaplex-rocm
cd personaplex-rocm
powershell -ExecutionPolicy Bypass -File rocm\install.ps1
```

Then log in to Hugging Face once (paste a *Read* token from
https://huggingface.co/settings/tokens; the input is hidden):

```powershell
& .\.venv\Scripts\huggingface-cli.exe login
```

## Run

```powershell
powershell -ExecutionPolicy Bypass -File rocm\run-server.ps1
```

Open **http://localhost:8998** in Chrome or Edge and allow the microphone. Use
headphones, otherwise the model hears itself. Pick a voice and a role prompt (see the
[upstream README](README.md#prompting-guide)) and just talk. You can interrupt it at
any time; that is what full duplex means.

The first start downloads the weights and runs a one-time GEMM/GEMV autotune for
your GPU (results are cached in `%LOCALAPPDATA%\personaplex-rocm`). Later starts
take about half a minute.

Offline (wav in, wav out) works as upstream: `python -m moshi.offline ...`.

## What was changed, and why

| Area | Change |
|---|---|
| `moshi/pyproject.toml` | Allow torch ≥ 2.5 (AMD's Windows wheels are torch 2.9.1+rocm7.2.1). |
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
| `split_lm.py`, `layer_breakdown.py` | Where the time goes (main transformer vs depformer, per-layer pieces). |

## Related

- [gain774/moshi-rocm](https://github.com/gain774/moshi-rocm) is a HIP build of the
  C++ `moshi.cpp` engine (Linux, ROCm 6.x). This branch instead keeps NVIDIA's
  reference PyTorch code and web UI and targets native Windows.

## License

Code: MIT (as upstream; Moshi/Audiocraft-derived parts keep their licenses). The model
weights are under the NVIDIA Open Model License; accept it on Hugging Face.
