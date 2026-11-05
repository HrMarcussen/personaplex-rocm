# Notices

This repository is a fork of [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) that adds
support for AMD Radeon GPUs on native Windows. It contains code from several sources, each under its
own license, which is kept in place:

| Component | Copyright / origin | License | Where |
|---|---|---|---|
| AMD ROCm / Windows port (custom HIP kernels, platform setup, scripts, docs, and the modifications listed in [docs/TECHNICAL.md](docs/TECHNICAL.md)) | © 2026 HrMarcussen | MIT, [LICENSE](LICENSE) | `moshi/moshi/kernels/`, `moshi/moshi/_platform_env.py`, `rocm/`, `docs/`, and changes across `moshi/` |
| PersonaPlex code and web client | NVIDIA | MIT, [LICENSE-MIT](LICENSE-MIT), [client/LICENSE](client/LICENSE) | `moshi/`, `client/`, `assets/` |
| Moshi | © Kyutai | MIT, [moshi/LICENSE.moshi](moshi/LICENSE.moshi) | `moshi/` |
| Audiocraft (parts of the Moshi code) | © Meta Platforms, Inc. and affiliates | MIT, [moshi/LICENSE.audiocraft](moshi/LICENSE.audiocraft) | `moshi/` |

## Model weights

No model weights, voice embeddings, or other model files are included in this repository. The
PersonaPlex weights are downloaded from
[huggingface.co/nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) at run
time and are governed by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
which each user must accept on Hugging Face.

## Trademarks

NVIDIA and PersonaPlex are trademarks of NVIDIA Corporation. AMD, Radeon, and ROCm are trademarks of
Advanced Micro Devices, Inc. Windows is a trademark of Microsoft Corporation. They are used here
only to describe compatibility. This is an independent community project, not affiliated with,
endorsed by, or supported by NVIDIA, AMD, or Microsoft.
