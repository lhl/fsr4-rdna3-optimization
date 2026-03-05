#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_result(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    cfg = payload.get("config", {})
    mode_summary = payload.get("mode_summary", {})
    classification = payload.get("classification")

    if classification is None:
        sidecar = path.with_name(path.stem + "-classification.json")
        if sidecar.exists():
            classification = load_json(sidecar)

    return {
        "path": str(path),
        "name": path.stem,
        "config": cfg,
        "mode_summary": mode_summary,
        "classification": classification or {},
    }


def is_comparable(cfg: dict[str, Any], mode: str) -> bool:
    if str(cfg.get("mode")) != mode:
        return False
    if int(cfg.get("elements", -1)) != 262144:
        return False
    if int(cfg.get("threads", -1)) <= 0:
        return False
    if int(cfg.get("reps_per_run", -1)) != 200:
        return False
    if int(cfg.get("min_runs", -1)) != 200:
        return False
    if float(cfg.get("target_seconds", -1.0)) != 60.0:
        return False
    return True


def mode_rows(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if not is_comparable(rec["config"], mode):
            continue
        summary = rec["mode_summary"].get(mode)
        if not summary:
            continue
        mean_ms = float(summary.get("mean_of_mean_ms", float("nan")))
        median_ms = float(summary.get("mean_of_median_ms", float("nan")))
        cv_pct = float(summary.get("mean_cv_pct", float("nan")))
        verdict = "n/a"
        delta_pct = float("nan")
        for item in rec["classification"].get("per_mode", []):
            if item.get("mode") == mode:
                verdict = str(item.get("verdict", "n/a"))
                delta_pct = float(item.get("delta_pct", float("nan")))
                break
        rows.append(
            {
                "name": rec["name"],
                "mean_ms": mean_ms,
                "median_ms": median_ms,
                "cv_pct": cv_pct,
                "delta_pct": delta_pct,
                "verdict": verdict,
                "path": rec["path"],
            }
        )
    rows.sort(key=lambda x: x["mean_ms"])
    return rows


def mixed_policy_rows(result_dir: Path, reference_path: Path) -> list[dict[str, Any]]:
    ref = load_json(reference_path)
    ref_int8 = float(ref["mode_summary"]["int8"]["mean_of_mean_ms"])
    ref_fp8 = float(ref["mode_summary"]["fp8"]["mean_of_mean_ms"])
    ref_total = ref_int8 + ref_fp8

    def _val(path: Path, mode: str) -> float:
        payload = load_json(path)
        return float(payload["mode_summary"][mode]["mean_of_mean_ms"])

    rows = [
        {
            "policy": "canonical-default-both",
            "int8_ms": ref_int8,
            "fp8_ms": ref_fp8,
        },
        {
            "policy": "candidate-A (int8 items=4, fp8 items=1)",
            "int8_ms": _val(result_dir / "todo-p1-candA-int8-items4-trials3.json", "int8"),
            "fp8_ms": _val(result_dir / "todo-p1-candA-fp8-items1-trials3.json", "fp8"),
        },
        {
            "policy": "candidate-B (int8 inner=16, fp8 inner=8)",
            "int8_ms": _val(result_dir / "todo-p1-candB-int8-inner16-trials3.json", "int8"),
            "fp8_ms": _val(result_dir / "todo-p1-candB-fp8-inner8-trials3.json", "fp8"),
        },
        {
            "policy": "candidate-C (A+B combined)",
            "int8_ms": _val(result_dir / "todo-p1-candC-int8-items4-inner16-trials3.json", "int8"),
            "fp8_ms": _val(result_dir / "todo-p1-candC-fp8-items1-inner8-trials3.json", "fp8"),
        },
    ]

    for row in rows:
        total = row["int8_ms"] + row["fp8_ms"]
        row["total_ms"] = total
        row["delta_total_pct_vs_ref"] = 100.0 * (ref_total - total) / ref_total
    rows.sort(key=lambda x: x["total_ms"])
    return rows


def table_mode(rows: list[dict[str, Any]], limit: int) -> str:
    hdr = "| rank | config | mean_ms | median_ms | cv% | delta% vs ref | verdict |\n|---:|---|---:|---:|---:|---:|---|\n"
    lines = []
    for i, row in enumerate(rows[:limit], start=1):
        lines.append(
            f"| {i} | `{row['name']}` | {row['mean_ms']:.6f} | {row['median_ms']:.6f} | "
            f"{row['cv_pct']:.3f} | {row['delta_pct']:.3f} | {row['verdict']} |"
        )
    return hdr + "\n".join(lines) + ("\n" if lines else "")


def table_mixed(rows: list[dict[str, Any]]) -> str:
    hdr = "| rank | policy | int8_ms | fp8_ms | total_ms | total delta% vs canonical |\n|---:|---|---:|---:|---:|---:|\n"
    lines = []
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {row['policy']} | {row['int8_ms']:.6f} | {row['fp8_ms']:.6f} | "
            f"{row['total_ms']:.6f} | {row['delta_total_pct_vs_ref']:.3f} |"
        )
    return hdr + "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize gfx1100 todo benchmark results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/gfx1100-final-default-optimized-trials3.json"),
    )
    parser.add_argument("--out-md", type=Path, default=Path("results/todo-summary.md"))
    parser.add_argument("--out-json", type=Path, default=Path("results/todo-summary.json"))
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    results_dir = args.results_dir if args.results_dir.is_absolute() else root / args.results_dir
    reference = args.reference if args.reference.is_absolute() else root / args.reference
    out_md = args.out_md if args.out_md.is_absolute() else root / args.out_md
    out_json = args.out_json if args.out_json.is_absolute() else root / args.out_json

    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("todo-*.json")):
        if path.name.endswith("-classification.json"):
            continue
        records.append(parse_result(path))

    int8 = mode_rows(records, "int8")
    fp8 = mode_rows(records, "fp8")
    mixed = mixed_policy_rows(results_dir, reference)

    md = []
    md.append("# gfx1100 TODO Result Summary")
    md.append("")
    md.append("## INT8 Comparable Ranking")
    md.append("")
    md.append(table_mode(int8, args.top))
    md.append("## FP8 Comparable Ranking")
    md.append("")
    md.append(table_mode(fp8, args.top))
    md.append("## Mixed Policy Ranking")
    md.append("")
    md.append(table_mixed(mixed))
    md.append("")

    payload = {
        "int8": int8,
        "fp8": fp8,
        "mixed_policy": mixed,
        "reference": str(reference),
    }

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_md}")
    print(f"[saved] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
