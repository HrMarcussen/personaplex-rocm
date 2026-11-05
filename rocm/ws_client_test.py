"""Headless end-to-end test of the realtime server, speaking the web client's protocol.

Streams a wav file to ws://HOST/api/chat as Opus at real-time pace (like a
microphone), records what the server sends back, and reports:
  * prompt-processing time until the handshake,
  * whether the server keeps up (lag between audio sent and audio received),
  * the model's text stream,
and writes the received audio to a wav file.

    python rocm/ws_client_test.py --input-wav assets/test/input_assistant.wav
"""
import argparse
import asyncio
import os
import time
import urllib.parse

import aiohttp
import numpy as np
import sphn

from moshi.models.lm import load_audio

SR = 24000
FRAME = 1920  # 80 ms


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8998")
    ap.add_argument("--input-wav", default=os.path.join(os.path.dirname(__file__), "..", "assets", "test", "input_assistant.wav"))
    ap.add_argument("--voice-prompt", default="NATF2.pt")
    ap.add_argument("--text-prompt", default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--tail", type=float, default=5, help="seconds of silence sent after the input")
    ap.add_argument("--out-wav", default="ws_test_out.wav")
    args = ap.parse_args()

    pcm = load_audio(args.input_wav, SR)[0].astype(np.float32)[: int(args.seconds * SR)]
    pcm = np.concatenate([pcm, np.zeros(int(args.tail * SR), np.float32)])
    q = urllib.parse.urlencode({"text_prompt": args.text_prompt, "voice_prompt": args.voice_prompt,
                                "text_seed": 42, "audio_seed": 42})
    ws_url = args.url.replace("http", "ws", 1) + "/api/chat?" + q

    writer = sphn.OpusStreamWriter(SR)
    reader = sphn.OpusStreamReader(SR)
    received: list[np.ndarray] = []
    recv_times: list[tuple[float, int]] = []  # (time, total samples received)
    text: list[str] = []
    n_recv = 0

    async with aiohttp.ClientSession() as session:
        t_connect = time.perf_counter()
        async with session.ws_connect(ws_url, max_msg_size=0) as ws:
            msg = await ws.receive()
            assert msg.type == aiohttp.WSMsgType.BINARY and msg.data[:1] == b"\x00", msg
            t_hs = time.perf_counter()
            print(f"handshake after {t_hs - t_connect:.2f}s (voice + text prompt processing)")

            async def recv_loop():
                nonlocal n_recv
                async for m in ws:
                    if m.type != aiohttp.WSMsgType.BINARY:
                        break
                    kind, payload = m.data[0], m.data[1:]
                    if kind == 1:
                        reader.append_bytes(payload)
                        out = reader.read_pcm()
                        if out.shape[-1]:
                            received.append(out)
                            n_recv += out.shape[-1]
                            recv_times.append((time.perf_counter(), n_recv))
                    elif kind == 2:
                        text.append(payload.decode("utf-8", errors="replace"))

            recv_task = asyncio.create_task(recv_loop())
            t0 = time.perf_counter()
            for i in range(0, len(pcm) - FRAME + 1, FRAME):
                writer.append_pcm(pcm[i:i + FRAME])
                data = writer.read_bytes()
                if data:
                    await ws.send_bytes(b"\x01" + data)
                # real-time pacing, like a microphone
                target = t0 + (i + FRAME) / SR
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            sent_s = len(pcm) / SR
            await asyncio.sleep(1.0)
            await ws.close()
            recv_task.cancel()

    out = np.concatenate(received) if received else np.zeros(0, np.float32)
    sphn.write_wav(args.out_wav, out, SR)
    # Lag: for each receive event, wall time since start minus audio duration received.
    lags = np.array([(t - t0) - n / SR for t, n in recv_times]) if recv_times else np.array([np.nan])
    print(f"sent {sent_s:.1f}s of audio, received {len(out) / SR:.1f}s -> {args.out_wav}")
    print(f"lag (wall clock - audio received): start {lags[:10].mean():.3f}s, "
          f"end {lags[-10:].mean():.3f}s, max {lags.max():.3f}s")
    drift = lags[-10:].mean() - lags[10:20].mean() if len(lags) > 30 else float("nan")
    print(f"drift over the session: {drift * 1000:+.0f} ms "
          f"({'keeps up with real time' if abs(drift) < 0.2 else 'FALLING BEHIND'})")
    print("model text:", "".join(text).strip())
    report_turn_taking(pcm, out, recv_times, t0)


def _frame_rms(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    return np.sqrt((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1) + 1e-12)


def report_turn_taking(inp: np.ndarray, out: np.ndarray, recv_times, t0: float) -> None:
    """Response latency in the spirit of NVIDIA's chart (FullDuplexBench style).

    The server emits exactly one output frame per input frame, so input and
    output share a frame timeline. For every point where the user stops
    talking (>= 0.5 s of silence after speech), find the first frame where the
    model's output becomes voiced. That is the model's own reaction time; the
    pipeline delay measured on this PC (audio sent -> audio back) adds to it.
    """
    rin, rout = _frame_rms(inp), _frame_rms(out)
    if len(rout) < 20:
        print("turn-taking: not enough output to analyse")
        return
    thr_in = max(0.01, 0.1 * np.percentile(rin, 95))
    thr_out = max(0.01, 0.1 * np.percentile(rout, 95))
    vin, vout = rin > thr_in, rout > thr_out
    gap = int(0.5 * 12.5)
    lat = []
    i = gap
    while i < min(len(vin), len(vout)) - gap:
        # user was talking in the previous frames and is silent for `gap` frames now
        if vin[i - 1] and not vin[i:i + gap].any():
            j = i
            while j < len(vout) and not vout[j]:
                j += 1
            if j < len(vout) and j - i < 12.5 * 3:  # answered within 3 s
                if not vout[max(0, i - 3):i].any():  # model was not already talking
                    lat.append((j - i) * 0.08)
            i += gap
        else:
            i += 1
    # Pipeline delay: when output frame k arrived vs when input frame k was sent (t0 + k*80ms).
    delays = [t - (t0 + n / SR) for t, n in recv_times]
    pipe = float(np.median(delays)) if delays else float("nan")
    if lat:
        lat = np.array(lat)
        print(f"turn-taking: {len(lat)} user turn ends; model starts speaking after "
              f"median {np.median(lat):.2f}s (model timeline, 80 ms resolution)")
        print(f"  + pipeline delay on this PC (median, audio sent -> audio back): {pipe * 1000:.0f} ms")
        print(f"  => perceived response latency ~ {np.median(lat) + pipe:.2f}s "
              f"(NVIDIA's chart: 0.17s smooth turn-taking)")
    else:
        print(f"turn-taking: no clean user-turn ends found; pipeline delay {pipe * 1000:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
