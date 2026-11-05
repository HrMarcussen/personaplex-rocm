"""End-to-end numerical validation on real weights.

Streams a wav through Mimi + the LM exactly like the server (frame by frame,
HIP graphs on), once with the custom HIP kernels and once with stock PyTorch
ops, and reports:
  * the model's text stream for both runs (to eyeball coherence),
  * how long the greedy token streams agree,
  * per-frame latency for both runs.

Works with the public Kyutai Moshiko weights (same architecture), so it does
not need access to the gated PersonaPlex repo:
    python rocm/validate_offline.py --hf-repo kyutai/moshiko-pytorch-bf16 --voice-prompt ''
"""
import argparse
import os
import time

import moshi  # noqa: F401  (ROCm env defaults)
import numpy as np
import sentencepiece
import sphn
import torch
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from moshi.models.lm import LMGen, load_audio
from moshi.offline import wrap_with_system_tags


@torch.no_grad()
def run(args, mimi, lm, tok, pcm, kernels_on: bool):
    os.environ["PERSONAPLEX_KERNELS"] = "1" if kernels_on else "0"
    torch.manual_seed(args.seed)
    gen = LMGen(lm, device="cuda", sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate,
                audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                use_sampling=not args.greedy, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)
    frame = int(mimi.sample_rate / mimi.frame_rate)
    mimi.streaming_forever(1)
    gen.streaming_forever(1)
    if args.voice_prompt_path:
        # Same prompt phase as the server: voice prompt, silence, role text, silence.
        if args.voice_prompt_path.endswith(".pt"):
            gen.load_voice_prompt_embeddings(args.voice_prompt_path)
        else:
            gen.load_voice_prompt(args.voice_prompt_path)
        gen.text_prompt_tokens = tok.encode(wrap_with_system_tags(args.text_prompt)) if args.text_prompt else None
    mimi.reset_streaming()
    gen.reset_streaming()
    if args.voice_prompt_path:
        t_p = time.perf_counter()
        gen.step_system_prompts(mimi)
        mimi.reset_streaming()
        print(f"[{'kernels' if kernels_on else 'pytorch'}] prompt phase took {time.perf_counter() - t_p:.2f}s")
    texts, tokens_all, out_pcm, times = [], [], [], []
    n = pcm.shape[-1] // frame
    for i in range(n):
        chunk = torch.from_numpy(pcm[None, None, i * frame:(i + 1) * frame]).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        codes = mimi.encode(chunk)
        tokens = gen.step(codes[:, :, :1])
        if tokens is not None:
            out = mimi.decode(tokens[:, 1:9]).cpu()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if tokens is None:
            continue
        out_pcm.append(out[0, 0].numpy())
        tokens_all.append(tokens[0, :, 0].cpu())
        t = tokens[0, 0, 0].item()
        if t not in (0, 3):
            texts.append(tok.id_to_piece(t).replace("▁", " "))
    mimi.reset_streaming()
    times = np.array(times[10:]) * 1000
    return "".join(texts), torch.stack(tokens_all), np.concatenate(out_pcm), times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    ap.add_argument("--input-wav", default=os.path.join(os.path.dirname(__file__), "..", "assets", "test", "input_assistant.wav"))
    ap.add_argument("--out-prefix", default="validate")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42424242)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--only-kernels", action="store_true")
    ap.add_argument("--voice-prompt", default="NATF2.pt", help="PersonaPlex voice; '' to skip prompts (e.g. Moshiko)")
    ap.add_argument("--text-prompt", default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.")
    args = ap.parse_args()
    args.voice_prompt_path = None
    if args.voice_prompt:
        from moshi.offline import _get_voice_prompt_dir
        args.voice_prompt_path = os.path.join(_get_voice_prompt_dir(None, args.hf_repo), args.voice_prompt)

    mimi = loaders.get_mimi(hf_hub_download(args.hf_repo, loaders.MIMI_NAME), "cuda")
    tok = sentencepiece.SentencePieceProcessor(hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME))
    lm = loaders.get_moshi_lm(hf_hub_download(args.hf_repo, loaders.MOSHI_NAME), device="cuda")
    lm.eval()
    pcm = load_audio(args.input_wav, mimi.sample_rate)  # (C, T)
    pcm = pcm[0, : int(args.seconds * mimi.sample_rate)].astype(np.float32)
    print(f"input {args.input_wav}: {pcm.shape[-1] / mimi.sample_rate:.1f}s")

    modes = [True] if args.only_kernels else [True, False]
    res = {}
    for on in modes:
        name = "kernels" if on else "pytorch"
        text, toks, out, times = run(args, mimi, lm, tok, pcm, on)
        res[name] = toks
        sphn.write_wav(f"{args.out_prefix}_{name}.wav", out, mimi.sample_rate)
        print(f"\n=== {name}: frame mean {times.mean():.1f} ms, p95 {np.percentile(times, 95):.1f} ms, "
              f"max {times.max():.1f} ms  (budget 80 ms) -> {args.out_prefix}_{name}.wav")
        print(f"text: {text.strip()}")
    if len(res) == 2:
        a, b = res["kernels"], res["pytorch"]
        same = (a == b).all(dim=1)
        first_diff = int((~same).nonzero()[0]) if (~same).any() else len(same)
        print(f"\ntoken frames identical: {int(same.sum())}/{len(same)}; first divergence at frame {first_diff}"
              f" ({first_diff / 12.5:.1f}s)")


if __name__ == "__main__":
    main()
