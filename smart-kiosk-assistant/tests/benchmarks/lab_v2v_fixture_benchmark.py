#!/usr/bin/env python3
"""V2V fixture benchmark for the kiosk-voice-lab machine -- the VEI-side
counterpart of tests/benchmarks/v2v_fixture_benchmark.py, built so the two
numbers can be quoted side by side in the same report.

WHERE TO RUN THIS
  This does NOT run against the VEI stack (kiosk-core/audio-analyzer/etc).
  It drives the LAB's own pipeline.orchestrator.AdaptivePipeline directly
  (the same class experiments/e7_endpoint.py etc. benchmark), on the lab's
  Intel Core Ultra box, using the lab's own NPU/GPU models. Copy this file
  into the kiosk-voice-lab-main checkout's experiments/ directory (next to
  e7_endpoint.py) and run it there:

    cp lab_v2v_fixture_benchmark.py ~/kvl-qsr/experiments/
    cd ~/kvl-qsr
    ~/kvl-venv/bin/python experiments/lab_v2v_fixture_benchmark.py \\
        --runs 10 --label lab-order-simple

  Needs the same deps build_fixtures.py needs (kokoro_onnx, numpy,
  soundfile, librosa) plus whatever pipeline/orchestrator.py itself needs
  (openvino, the QSR knowledge base under kb/qsr_kb.json, etc) -- all of
  which `make setup` / `make demo-qsr` already provide on the box.

WHY THIS SCRIPT EXISTS
  The lab's own experiments (e4_baseline.py, e7_endpoint.py, ...) only ever
  replay the 80-question golden_set.json retail fixtures. There is no
  QSR-ordering, single-line fixture in that set that matches VEI's own
  order-simple benchmark line, so a true side-by-side number never existed.
  This script synthesizes THE SAME scripted line VEI's
  tests/benchmarks/v2v_scripted_conversation_benchmark.py uses for
  --script order-simple ("Hi, I would like to order one Classic Chicken
  Burger, please.") through the lab's own Kokoro TTS, using the lab's own
  build_fixtures.py helpers for room-tone padding and end_of_speech_sample
  detection (so the ground-truth EOS marker is computed exactly the way
  every other lab fixture's is), then runs it through
  AdaptivePipeline.run_turn() -- the lab's own, most-optimized pipeline --
  the requested number of times.

WHAT "v2v_ms" MEANS HERE (READ BEFORE COMPARING NUMBERS)
  The lab's run_turn() measures v2v_ms as (last real speech sample in the
  fixture) -> (first audio sample written), using the fixture's OWN
  end_of_speech_sample ground truth -- i.e. it is computed the same way as
  VEI's --explicit-end-mark mode (tests/benchmarks/v2v_scripted_conversation_
  benchmark.py), NOT VEI's default customer-facing number, which additionally
  waits out a real silence-timeout/endpoint-completeness-shortcut before
  committing the turn. Compare this script's v2v_ms against VEI's
  --explicit-end-mark run (results/v2v-explicit-end-mark-v2.json,
  ~1200 ms median at last measurement) or against
  voice_to_voice_post_endpoint_ms (~330-570 ms median), NOT against VEI's
  raw voice_to_voice_ms (which bakes in the deliberate silence wait and will
  always look worse in a same-line comparison).

OUTPUT
  Writes <out-dir>/<label>.json in the same shape as VEI's
  tests/benchmarks/v2v_fixture_benchmark.py summary (mean/median/p90/p95/
  min/max per stage, plus every raw per-run row) so the two JSON files can be
  diffed/tabulated together for the final report.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import qsr_config  # noqa: E402
from pipeline.orchestrator import AdaptivePipeline  # noqa: E402
from experiments.build_fixtures import build_clip, speech_bounds, TARGET_SR  # noqa: E402

try:
    import soundfile as sf
    from kokoro_onnx import Kokoro
except ImportError as e:  # pragma: no cover - lab-machine-only deps
    print(
        f"Missing lab-machine dependency ({e}). This script must run on the "
        "kiosk-voice-lab box, inside its own venv (~/kvl-venv or "
        "~/kvl-tts-venv) -- see this file's module docstring.",
        file=sys.stderr,
    )
    raise

# The exact line VEI's tests/benchmarks/v2v_scripted_conversation_benchmark.py
# uses for --script order-simple. Keep these in sync by hand if either side's
# wording changes -- there is no shared import between the two repos.
DEFAULT_LINE = "Hi, I would like to order one Classic Chicken Burger, please."
DEFAULT_VOICE = "af_heart"  # matches PipelineConfig.tts_voice default

# Fields returned by AdaptivePipeline.run_turn() worth summarizing across runs.
_STAGE_FIELDS = [
    "v2v_ms",
    "compute_ttfa_ms",
    "endpoint_delay_ms",
    "asr_visible_ms",
    "rag_ms",
    "llm_ttft_ms",
    "first_clause_ms",
    "tts_first_ms",
    "llm_total_ms",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(pct * (len(s) - 1)))))
    return s[idx]


def _summarize(rows: list[dict], field: str) -> dict[str, float]:
    vals = [r[field] for r in rows if r.get(field) is not None]
    if not vals:
        return {"mean": None, "median": None, "p90": None, "p95": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.mean(vals), 1),
        "median": round(statistics.median(vals), 1),
        "p90": round(_percentile(vals, 0.90), 1),
        "p95": round(_percentile(vals, 0.95), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
        "n": len(vals),
    }


def build_fixture(text: str, voice: str, out_dir: Path) -> tuple[Path, int]:
    """Synthesize `text` through the lab's own Kokoro TTS + room-tone padding
    (build_fixtures.build_clip), the same way every golden_set.json fixture is
    built, and return (wav_path, end_of_speech_sample)."""
    from pipeline.config import PipelineConfig

    cfg = PipelineConfig()
    kokoro = Kokoro(str(Path(cfg.tts_model).expanduser()), str(Path(cfg.tts_voices).expanduser()))
    qid = "v2v-parity-order-simple"
    clean, _pause_trap = build_clip(kokoro, qid, text, voice)
    start_idx, end_idx = speech_bounds(clean, TARGET_SR)
    if end_idx is None:
        raise RuntimeError(f"No speech activity detected in synthesized fixture for: {text!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{qid}.wav"
    sf.write(str(wav_path), clean, TARGET_SR, subtype="PCM_16")
    return wav_path, int(end_idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--line", default=DEFAULT_LINE, help="Scripted line to synthesize and benchmark")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="Kokoro voice id (default: af_heart)")
    ap.add_argument("--runs", type=int, default=10, help="Number of repeated turns (default: 10)")
    ap.add_argument("--label", default="lab-v2v-scripted", help="Basename for the JSON report")
    ap.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory to write the JSON report to (created if missing)",
    )
    ap.add_argument(
        "--fixture-dir",
        default=str(Path(__file__).resolve().parent / "lab_fixtures"),
        help="Directory to write the synthesized WAV fixture to (created if missing)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    fixture_dir = Path(args.fixture_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lab-v2v] synthesizing fixture: {args.line!r} (voice={args.voice})")
    wav_path, eos_sample = build_fixture(args.line, args.voice, fixture_dir)
    print(f"[lab-v2v] wrote {wav_path} (end_of_speech_sample={eos_sample})")

    print("[lab-v2v] loading QSR ordering pipeline (models, KB, index)...")
    cfg = qsr_config(tools=True)
    pipe = AdaptivePipeline(cfg)
    print("[lab-v2v] pipeline ready")

    rows = []
    for i in range(1, args.runs + 1):
        row = pipe.run_turn(str(wav_path), eos_sample)
        row.update({"run": i, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        rows.append(row)
        print(
            f"[lab-v2v] run {i}/{args.runs}: v2v={row['v2v_ms']} ms  "
            f"compute_ttfa={row['compute_ttfa_ms']} ms  spec_hit={row['spec_hit']}  "
            f"false_cut={row['false_cut']}  transcript={row['transcript']!r}"
        )

    summary = {field: _summarize(rows, field) for field in _STAGE_FIELDS}
    spec_hits = sum(1 for r in rows if r.get("spec_hit"))
    false_cuts = sum(1 for r in rows if r.get("false_cut"))

    print("\n" + "=" * 78)
    print(f"LAB V2V FIXTURE BENCHMARK  label={args.label!r}  line={args.line!r}")
    print("=" * 78)
    print(f"{'stage':<22}{'mean':>10}{'median':>10}{'p90':>10}{'p95':>10}{'min':>10}{'max':>10}")
    for field in _STAGE_FIELDS:
        s = summary[field]
        if s["n"] == 0:
            print(f"{field:<22}{'n/a':>10}")
            continue
        print(
            f"{field:<22}{s['mean']:>10}{s['median']:>10}{s['p90']:>10}"
            f"{s['p95']:>10}{s['min']:>10}{s['max']:>10}"
        )
    print(f"\nspec_hit: {spec_hits}/{len(rows)}   false_cut: {false_cuts}/{len(rows)}")
    print(
        "\nNote: v2v_ms here is ground-truth-EOS-anchored (like VEI's "
        "--explicit-end-mark mode), NOT VEI's default silence-timeout-inclusive "
        "voice_to_voice_ms -- see this script's module docstring before quoting "
        "a side-by-side comparison."
    )

    report = {
        "label": args.label,
        "line": args.line,
        "voice": args.voice,
        "eos_sample": eos_sample,
        "runs": len(rows),
        "summary": summary,
        "spec_hit_rate": spec_hits / len(rows) if rows else None,
        "false_cut_rate": false_cuts / len(rows) if rows else None,
        "rows": rows,
    }
    out_path = results_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[lab-v2v] wrote {out_path}")


if __name__ == "__main__":
    main()
