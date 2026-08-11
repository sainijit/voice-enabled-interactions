#!/usr/bin/env python3
"""Demo-day microphone noise-floor test.

Measures the background noise level of the kiosk microphone and reports
whether the venue is workable, using exactly the same energy maths
(``BaseAudioSession._rms`` over ``DEFAULT_BLOCK_DURATION_SECONDS`` frames)
that the live VAD uses. Run it in the venue before doors open.

Usage:
    # list capture devices
    python mic_floor_test.py --list

    # measure the default device for 5 seconds (STAY SILENT while it runs)
    python mic_floor_test.py

    # measure a specific device for 10 seconds
    python mic_floor_test.py --device 0 --seconds 10

    # measure a WAV file recorded elsewhere (e.g. via `pw-record`)
    python mic_floor_test.py --wav /tmp/venue.wav

The kiosk calibrates itself at runtime, so this tool is a *verification* aid,
not a tuning step — it tells you whether the venue is quiet enough for the
microphone to work at all, which is a question no amount of code can fix.
"""
from __future__ import annotations

import argparse
import sys
import wave

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from kiosk_core import config  # noqa: E402

FRAME_SECONDS = config.DEFAULT_BLOCK_DURATION_SECONDS
SAMPLE_RATE = 16000


def _dbfs(rms: float) -> float:
    return 20.0 * np.log10(max(rms, 1e-9) / 32767.0)


def _frame_rms(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    n = max(1, int(sample_rate * FRAME_SECONDS))
    usable = (len(samples) // n) * n
    if usable == 0:
        return np.array([])
    blocks = samples[:usable].astype(np.float32).reshape(-1, n)
    return np.sqrt(np.mean(blocks * blocks, axis=1))


def _import_sounddevice():
    """Import sounddevice, or explain the WAV fallback if PortAudio is absent.

    PortAudio is only installed inside the kiosk-core container, so the live
    capture path fails on a bare host. Under demo-day pressure a raw
    ImportError traceback is useless — point at the fallback instead.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415 — optional, device path only
    except OSError as exc:
        raise SystemExit(
            f"Cannot open the audio backend: {exc}\n\n"
            "PortAudio is only present inside the kiosk-core container. Either:\n"
            "  1. run this inside it:\n"
            "       docker exec -it kiosk-core python mic_floor_test.py\n"
            "  2. or record on the host and analyse the file:\n"
            "       pw-record --rate 16000 --channels 1 --format s16 /tmp/venue.wav\n"
            "       python mic_floor_test.py --wav /tmp/venue.wav"
        ) from exc
    return sd


def _capture(device: int | None, seconds: float) -> tuple[np.ndarray, int]:
    sd = _import_sounddevice()

    print(f"Recording {seconds:.0f}s — STAY SILENT (measuring background noise)…")
    frames = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()
    return frames[:, 0], SAMPLE_RATE


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM, got {handle.getsampwidth() * 8}-bit")
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        if handle.getnchannels() > 1:
            data = data.reshape(-1, handle.getnchannels())[:, 0]
        return data, handle.getframerate()


def _report(rms: np.ndarray) -> int:
    floor = float(np.percentile(rms, config.DEFAULT_VAD_FLOOR_PERCENTILE))
    gate = floor * (10.0 ** (config.DEFAULT_VAD_MARGIN_DB / 20.0))
    clamped = min(max(gate, config.DEFAULT_VAD_THRESHOLD_MIN), config.DEFAULT_VAD_THRESHOLD_MAX)

    print("\n── Measured background noise ─────────────────────────────")
    print(f"  frames analysed      : {len(rms)}  ({FRAME_SECONDS * 1000:.0f} ms each)")
    print(f"  noise floor (p{config.DEFAULT_VAD_FLOOR_PERCENTILE:.0f})    : {floor:8.0f} RMS   ({_dbfs(floor):+6.1f} dBFS)")
    print(f"  median               : {np.median(rms):8.0f} RMS   ({_dbfs(float(np.median(rms))):+6.1f} dBFS)")
    print(f"  p95                  : {np.percentile(rms, 95):8.0f} RMS")
    print(f"  max frame            : {rms.max():8.0f} RMS")

    print("\n── Resulting speech gate ─────────────────────────────────")
    print(f"  gate = floor + {config.DEFAULT_VAD_MARGIN_DB:.0f} dB : {gate:8.0f} RMS")
    print(f"  after clamping       : {clamped:8.0f} RMS   "
          f"[min {config.DEFAULT_VAD_THRESHOLD_MIN}, max {config.DEFAULT_VAD_THRESHOLD_MAX}]")
    print(f"  legacy fixed gate    : {config.DEFAULT_SILENCE_THRESHOLD:8d} RMS   "
          f"({'BROKEN — below the floor' if config.DEFAULT_SILENCE_THRESHOLD < floor else 'above floor'})")

    print("\n── Verdict ───────────────────────────────────────────────")
    if clamped < gate:
        print("  ⚠  VENUE TOO LOUD for this microphone.")
        print("     The gate saturates at its ceiling, so the kiosk falls back to")
        print("     permissive detection: it will still transcribe, but silence")
        print("     detection is unreliable and Whisper receives noisy audio.")
        print("     → Use push-to-talk, and move the mic closer to the customer.")
        return 2
    if floor > 2500:
        print("  ⚠  HIGH noise floor. Adaptive VAD will cope, but speech must be")
        print(f"     louder than {clamped:.0f} RMS to register — keep the mic close.")
        return 1
    print("  ✓  Noise floor is workable. Adaptive VAD will gate at "
          f"{clamped:.0f} RMS.")
    print("     Speak a test order and confirm the transcript looks correct.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure kiosk microphone noise floor.")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--seconds", type=float, default=5.0, help="capture duration")
    parser.add_argument("--wav", type=str, default=None, help="analyse a WAV file instead")
    parser.add_argument("--list", action="store_true", help="list input devices and exit")
    args = parser.parse_args()

    if args.list:
        print(_import_sounddevice().query_devices())
        return 0

    samples, rate = _read_wav(args.wav) if args.wav else _capture(args.device, args.seconds)
    rms = _frame_rms(samples, rate)
    if rms.size == 0:
        raise SystemExit("Recording too short to analyse.")
    return _report(rms)


if __name__ == "__main__":
    raise SystemExit(main())
