#!/usr/bin/env python3
"""Baseline benchmark harness for FP8/INT8 HIP microkernels.

This script compiles and runs `benchmarks/baseline_kernels_bench.cpp` and records
robust timing stats (mean/median/stddev/p95) using long-run sampling to reduce
interactive-system variance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

RESULT_RE = re.compile(r"^RESULT\s+(.*)$")
INFO_RE = re.compile(r"^INFO\s+(.*)$")


def parse_kv_blob(blob: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in blob.strip().split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        out[k] = v.strip('"')
    return out


def parse_numeric(record: dict[str, str], key: str) -> float:
    value = record.get(key)
    if value is None:
        raise ValueError(f"missing key: {key}")
    return float(value)


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def maybe_build(
    binary: Path,
    source: Path,
    arch: str,
    rebuild: bool,
    cwd: Path,
    extra_flags: list[str],
) -> None:
    if not rebuild and binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise RuntimeError("hipcc not found in PATH")
    binary.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        hipcc,
        "-O3",
        f"--offload-arch={arch}",
        *extra_flags,
        str(source),
        "-o",
        str(binary),
    ]
    print("[build]", " ".join(cmd))
    proc = run_cmd(cmd, cwd)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())


def parse_output(stdout: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    for line in stdout.splitlines():
        m_info = INFO_RE.match(line)
        if m_info:
            info = parse_kv_blob(m_info.group(1))
            continue

        m_result = RESULT_RE.match(line)
        if not m_result:
            continue
        raw = parse_kv_blob(m_result.group(1))
        parsed: dict[str, Any] = {"mode": raw.get("mode", "unknown")}
        numeric_keys = [
            "runs",
            "elapsed_s",
            "mean_ms",
            "stddev_ms",
            "median_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
            "cv_pct",
            "checksum",
        ]
        for key in numeric_keys:
            if key in raw:
                parsed[key] = parse_numeric(raw, key)
        results.append(parsed)

    if not results:
        raise RuntimeError("benchmark produced no RESULT lines")
    return info, results


def print_summary(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'mode':<6} {'runs':>6} {'mean_ms':>12} {'stddev_ms':>12} "
        f"{'cv%':>8} {'median_ms':>12} {'p95_ms':>12} {'elapsed_s':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['mode']:<6} {int(r.get('runs', 0)):>6d} "
            f"{r.get('mean_ms', float('nan')):>12.6f} "
            f"{r.get('stddev_ms', float('nan')):>12.6f} "
            f"{r.get('cv_pct', float('nan')):>8.3f} "
            f"{r.get('median_ms', float('nan')):>12.6f} "
            f"{r.get('p95_ms', float('nan')):>12.6f} "
            f"{r.get('elapsed_s', float('nan')):>10.3f}"
        )


def _cmd_int(cmd: list[str], key: str, default: int) -> int:
    try:
        idx = cmd.index(key)
    except ValueError:
        return default
    if idx + 1 >= len(cmd):
        return default
    return int(cmd[idx + 1])


def aggregate_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        reps_per_run = _cmd_int(trial["command"], "--reps-per-run", 1)
        for r in trial["results"]:
            item = dict(r)
            item["_reps_per_run"] = reps_per_run
            by_mode.setdefault(r["mode"], []).append(item)

    agg: list[dict[str, Any]] = []
    for mode, items in sorted(by_mode.items()):
        mean_vals = [float(x["mean_ms"]) for x in items]
        median_vals = [float(x["median_ms"]) for x in items]
        p95_vals = [float(x["p95_ms"]) for x in items]
        cv_vals = [float(x["cv_pct"]) for x in items]
        elapsed_vals = [float(x["elapsed_s"]) for x in items]
        runs_vals = [int(x["runs"]) for x in items]
        total_kernel_invocations = sum(int(x["runs"]) * int(x["_reps_per_run"]) for x in items)

        agg.append(
            {
                "mode": mode,
                "trials": len(items),
                "mean_of_mean_ms": statistics.mean(mean_vals),
                "stdev_of_mean_ms": statistics.stdev(mean_vals) if len(mean_vals) > 1 else 0.0,
                "mean_of_median_ms": statistics.mean(median_vals),
                "stdev_of_median_ms": (
                    statistics.stdev(median_vals) if len(median_vals) > 1 else 0.0
                ),
                "mean_of_p95_ms": statistics.mean(p95_vals),
                "stdev_of_p95_ms": statistics.stdev(p95_vals) if len(p95_vals) > 1 else 0.0,
                "mean_cv_pct": statistics.mean(cv_vals),
                "mean_runs": statistics.mean(runs_vals),
                "total_runs": sum(runs_vals),
                "total_kernel_invocations": total_kernel_invocations,
                "total_elapsed_s": sum(elapsed_vals),
            }
        )

    return agg


def print_trial_aggregate(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'mode':<6} {'trials':>6} {'mean(mean_ms)':>14} {'stdev(mean_ms)':>15} "
        f"{'mean(median_ms)':>16} {'mean(p95_ms)':>13} {'mean(cv%)':>10} {'elapsed_s':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['mode']:<6} {int(r['trials']):>6d} {r['mean_of_mean_ms']:>14.6f} "
            f"{r['stdev_of_mean_ms']:>15.6f} {r['mean_of_median_ms']:>16.6f} "
            f"{r['mean_of_p95_ms']:>13.6f} {r['mean_cv_pct']:>10.3f} "
            f"{r['total_elapsed_s']:>10.3f}"
        )


def mode_summary_from_aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row["mode"])
        out[mode] = dict(row)
    return out


def mode_summary_from_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "mode_summary" in payload and isinstance(payload["mode_summary"], dict):
        return payload["mode_summary"]
    if "aggregate" in payload and isinstance(payload["aggregate"], list):
        return mode_summary_from_aggregate(payload["aggregate"])
    return {}


def classify_against_reference(
    current: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    metric: str,
    min_uncertainty_pct: float,
    cv_scale: float,
) -> dict[str, Any]:
    metric_key = "mean_of_median_ms" if metric == "median" else "mean_of_mean_ms"
    metric_sd_key = "stdev_of_median_ms" if metric == "median" else "stdev_of_mean_ms"

    per_mode: list[dict[str, Any]] = []
    for mode in sorted(set(current) & set(reference)):
        cur = current[mode]
        ref = reference[mode]

        cur_value = float(cur.get(metric_key, float("nan")))
        ref_value = float(ref.get(metric_key, float("nan")))
        if not math.isfinite(cur_value) or not math.isfinite(ref_value) or ref_value <= 0:
            continue

        cur_n = max(1, int(cur.get("trials", 1)))
        ref_n = max(1, int(ref.get("trials", 1)))
        cur_sd = max(0.0, float(cur.get(metric_sd_key, 0.0)))
        ref_sd = max(0.0, float(ref.get(metric_sd_key, 0.0)))

        # 95% CI-style uncertainty from trial-to-trial spread where available.
        moe_trials = 1.96 * math.sqrt((cur_sd * cur_sd) / cur_n + (ref_sd * ref_sd) / ref_n)

        # Conservative fallback when trial spread is missing or too optimistic.
        cur_cv = max(0.0, float(cur.get("mean_cv_pct", 0.0))) / 100.0
        ref_cv = max(0.0, float(ref.get("mean_cv_pct", 0.0))) / 100.0
        fallback_pct = max(min_uncertainty_pct / 100.0, cv_scale * max(cur_cv, ref_cv))
        moe_fallback = ref_value * fallback_pct

        uncertainty_ms = max(moe_trials, moe_fallback)
        uncertainty_pct = 100.0 * uncertainty_ms / ref_value

        delta_ms = ref_value - cur_value
        delta_pct = 100.0 * delta_ms / ref_value

        if abs(delta_ms) <= uncertainty_ms:
            verdict = "unsure"
        elif delta_ms > uncertainty_ms:
            verdict = "keep"
        else:
            verdict = "drop"

        per_mode.append(
            {
                "mode": mode,
                "metric": metric,
                "reference_ms": ref_value,
                "current_ms": cur_value,
                "delta_ms": delta_ms,
                "delta_pct": delta_pct,
                "uncertainty_ms": uncertainty_ms,
                "uncertainty_pct": uncertainty_pct,
                "verdict": verdict,
                "reference_trials": ref_n,
                "current_trials": cur_n,
            }
        )

    verdicts = {row["verdict"] for row in per_mode}
    if not per_mode:
        overall = "no_overlap"
    elif "drop" in verdicts and "keep" in verdicts:
        overall = "unsure"
    elif "drop" in verdicts:
        overall = "drop"
    elif "unsure" in verdicts:
        overall = "unsure"
    else:
        overall = "keep"

    return {
        "overall": overall,
        "metric": metric,
        "min_uncertainty_pct": min_uncertainty_pct,
        "cv_scale": cv_scale,
        "per_mode": per_mode,
    }


def print_classification(classification: dict[str, Any]) -> None:
    rows = classification.get("per_mode", [])
    if not rows:
        print("Classification: no comparable modes")
        return

    header = (
        f"{'mode':<6} {'ref_ms':>12} {'cur_ms':>12} {'delta%':>10} "
        f"{'uncert%':>10} {'verdict':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:<6} {row['reference_ms']:>12.6f} {row['current_ms']:>12.6f} "
            f"{row['delta_pct']:>10.3f} {row['uncertainty_pct']:>10.3f} {row['verdict']:>10}"
        )
    print(f"overall: {classification['overall']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline FP8/INT8 HIP benchmarks")
    parser.add_argument("--mode", choices=["both", "int8", "fp8"], default="both")
    parser.add_argument("--arch", default="gfx1151")
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--items-per-thread", type=int, default=1)
    parser.add_argument("--force-runtime-inner-loops", action="store_true")
    parser.add_argument("--force-scalar-int8-io", action="store_true")
    parser.add_argument("--force-packed-int8-io", action="store_true")
    parser.add_argument("--force-inloop-scale-bias", action="store_true")
    parser.add_argument("--force-per-iter-requant", action="store_true")
    parser.add_argument("--split-interior-edge", action="store_true")
    parser.add_argument("--lds-stage-input", action="store_true")
    parser.add_argument("--lds-stage-weight", action="store_true")
    parser.add_argument("--lds-padding", action="store_true")
    parser.add_argument("--lds-double-buffer", action="store_true")
    parser.add_argument("--force-unfused-post", action="store_true")
    parser.add_argument("--force-two-pass", action="store_true")
    parser.add_argument("--force-mixed-int8-path", action="store_true")
    parser.add_argument("--fp8-quantized-io", action="store_true")
    parser.add_argument("--inner-int8", type=int, default=64)
    parser.add_argument("--inner-fp8", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=40)
    parser.add_argument("--min-runs", type=int, default=120)
    parser.add_argument("--max-runs", type=int, default=200000)
    parser.add_argument("--reps-per-run", type=int, default=200)
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=120.0,
        help="minimum wall-clock benchmark time per mode",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="independent benchmark trials to run (seed increments each trial)",
    )
    parser.add_argument("--reference-stats", type=Path)
    parser.add_argument("--classify-metric", choices=["median", "mean"], default="median")
    parser.add_argument("--min-uncertainty-pct", type=float, default=3.0)
    parser.add_argument(
        "--cv-scale",
        type=float,
        default=0.5,
        help="fallback uncertainty scale from cv_pct (0.5 means half of max CV)",
    )
    parser.add_argument("--stats-out", type=Path)
    parser.add_argument(
        "--latest-stats-file",
        type=Path,
        default=Path("results/latest-benchmark-stats.json"),
    )
    parser.add_argument("--classification-out", type=Path)
    parser.add_argument(
        "--latest-classification-file",
        type=Path,
        default=Path("results/latest-benchmark-classification.json"),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--hipcc-flag",
        action="append",
        default=[],
        help="extra flag to pass to hipcc during build (repeatable)",
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    source = root / "benchmarks" / "baseline_kernels_bench.cpp"
    binary = root / "build" / "baseline_kernels_bench"

    if not source.exists():
        raise FileNotFoundError(source)

    if not args.no_build:
        maybe_build(binary, source, args.arch, args.force_rebuild, root, args.hipcc_flag)
    elif not binary.exists():
        raise FileNotFoundError(f"binary not found: {binary}")

    if args.trials <= 0:
        raise ValueError("--trials must be >= 1")
    if args.min_uncertainty_pct < 0:
        raise ValueError("--min-uncertainty-pct must be >= 0")
    if args.cv_scale < 0:
        raise ValueError("--cv-scale must be >= 0")

    trial_records: list[dict[str, Any]] = []
    for trial_idx in range(args.trials):
        seed = args.seed + trial_idx
        cmd = [
            str(binary),
            "--mode",
            args.mode,
            "--elements",
            str(args.elements),
            "--threads",
            str(args.threads),
            "--items-per-thread",
            str(args.items_per_thread),
            "--inner-int8",
            str(args.inner_int8),
            "--inner-fp8",
            str(args.inner_fp8),
            "--warmup-runs",
            str(args.warmup_runs),
            "--min-runs",
            str(args.min_runs),
            "--max-runs",
            str(args.max_runs),
            "--reps-per-run",
            str(args.reps_per_run),
            "--target-seconds",
            str(args.target_seconds),
            "--seed",
            str(seed),
        ]
        if args.force_runtime_inner_loops:
            cmd.append("--force-runtime-inner-loops")
        if args.force_scalar_int8_io:
            cmd.append("--force-scalar-int8-io")
        if args.force_packed_int8_io:
            cmd.append("--force-packed-int8-io")
        if args.force_inloop_scale_bias:
            cmd.append("--force-inloop-scale-bias")
        if args.force_per_iter_requant:
            cmd.append("--force-per-iter-requant")
        if args.split_interior_edge:
            cmd.append("--split-interior-edge")
        if args.lds_stage_input:
            cmd.append("--lds-stage-input")
        if args.lds_stage_weight:
            cmd.append("--lds-stage-weight")
        if args.lds_padding:
            cmd.append("--lds-padding")
        if args.lds_double_buffer:
            cmd.append("--lds-double-buffer")
        if args.force_unfused_post:
            cmd.append("--force-unfused-post")
        if args.force_two_pass:
            cmd.append("--force-two-pass")
        if args.force_mixed_int8_path:
            cmd.append("--force-mixed-int8-path")
        if args.fp8_quantized_io:
            cmd.append("--fp8-quantized-io")

        print(f"[run {trial_idx + 1}/{args.trials}]", " ".join(cmd))
        proc = run_cmd(cmd, root)
        if proc.stderr.strip():
            print(proc.stderr.strip())

        info, results = parse_output(proc.stdout)
        print_summary(results)
        trial_records.append(
            {
                "trial": trial_idx + 1,
                "seed": seed,
                "command": cmd,
                "info": info,
                "results": results,
            }
        )

    aggregate = aggregate_trials(trial_records)
    mode_summary = mode_summary_from_aggregate(aggregate)
    if args.trials > 1:
        print("\nAggregate Across Trials")
        print_trial_aggregate(aggregate)

    payload: dict[str, Any] = {
        "schema_version": 2,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "mode": args.mode,
            "arch": args.arch,
            "elements": args.elements,
            "threads": args.threads,
            "inner_int8": args.inner_int8,
            "inner_fp8": args.inner_fp8,
            "warmup_runs": args.warmup_runs,
            "min_runs": args.min_runs,
            "max_runs": args.max_runs,
            "reps_per_run": args.reps_per_run,
            "target_seconds": args.target_seconds,
            "trials": args.trials,
            "seed": args.seed,
            "force_runtime_inner_loops": args.force_runtime_inner_loops,
            "force_scalar_int8_io": args.force_scalar_int8_io,
            "force_packed_int8_io": args.force_packed_int8_io,
            "force_inloop_scale_bias": args.force_inloop_scale_bias,
            "force_per_iter_requant": args.force_per_iter_requant,
            "split_interior_edge": args.split_interior_edge,
            "lds_stage_input": args.lds_stage_input,
            "lds_stage_weight": args.lds_stage_weight,
            "lds_padding": args.lds_padding,
            "lds_double_buffer": args.lds_double_buffer,
            "force_unfused_post": args.force_unfused_post,
            "force_two_pass": args.force_two_pass,
            "force_mixed_int8_path": args.force_mixed_int8_path,
            "fp8_quantized_io": args.fp8_quantized_io,
            "hipcc_flags": args.hipcc_flag,
        },
        "trials": trial_records,
        "aggregate": aggregate,
        "mode_summary": mode_summary,
    }

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.stats_out if args.stats_out else out_dir / f"baseline-benchmark-{stamp}.json"
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")

    latest_stats = args.latest_stats_file
    if not latest_stats.is_absolute():
        latest_stats = root / latest_stats
    latest_stats.parent.mkdir(parents=True, exist_ok=True)
    latest_stats.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {latest_stats}")

    classification: dict[str, Any] | None = None
    if args.reference_stats:
        reference_path = args.reference_stats
        if not reference_path.is_absolute():
            reference_path = root / reference_path
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_summary = mode_summary_from_payload(reference_payload)
        classification = classify_against_reference(
            current=mode_summary,
            reference=reference_summary,
            metric=args.classify_metric,
            min_uncertainty_pct=args.min_uncertainty_pct,
            cv_scale=args.cv_scale,
        )
        classification["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        classification["reference_stats"] = str(reference_path)
        classification["current_stats"] = str(out_path)
        payload["classification"] = classification

        # Refresh stats files so the main payload also contains classification.
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        latest_stats.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        print("\nClassification")
        print_classification(classification)

        class_out = (
            args.classification_out
            if args.classification_out
            else out_dir / f"baseline-benchmark-classification-{stamp}.json"
        )
        if not class_out.is_absolute():
            class_out = root / class_out
        class_out.parent.mkdir(parents=True, exist_ok=True)
        class_out.write_text(json.dumps(classification, indent=2) + "\n", encoding="utf-8")
        print(f"[saved] {class_out}")

        latest_class = args.latest_classification_file
        if not latest_class.is_absolute():
            latest_class = root / latest_class
        latest_class.parent.mkdir(parents=True, exist_ok=True)
        latest_class.write_text(json.dumps(classification, indent=2) + "\n", encoding="utf-8")
        print(f"[saved] {latest_class}")

    high_cv: list[tuple[int, str, float]] = []
    for tr in trial_records:
        for r in tr["results"]:
            cv = float(r.get("cv_pct", 0.0))
            if cv > 3.0:
                high_cv.append((int(tr["trial"]), str(r["mode"]), cv))
    if high_cv:
        print("[note] high variance detected (cv_pct > 3):")
        for trial, mode, cv in high_cv:
            print(f"  - trial={trial} mode={mode}: cv_pct={cv:.3f}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stdout)
        sys.stderr.write(exc.stderr)
        raise
