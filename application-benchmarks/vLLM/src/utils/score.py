#!/usr/bin/env python3
"""
Score a sweep: per-subtask Pareto AUC.

A *subtask* is one (model, workload-shape) pair, where the workload shape is the
(input-len, output-len) of the dataset. For each subtask we take every swept point
(all concurrency levels across all parallelism configs), extract the Pareto frontier
of tok/s/GPU (y) vs tok/s/user (x), and compute its area under the curve (AUC) — the
headline metric. A higher AUC means better throughput across the whole
online<->offline operating range, not just at one point.

Outputs:
  * a table printed to the terminal
  * scores.csv  and  scores.json  written into the results dir

The CSV/JSON schema is one flat row per (model, input_len, output_len), plus per-model
mean and an overall mean. The flat layout is deliberate: results from several
submissions can be concatenated and normalised across parties later (README §6) without
reshaping.

Reuses the metric helpers from plot_pareto.py so the AUC here is identical to the value
drawn on the Pareto plots.

Usage:
    python score.py RESULTS_DIR
"""
import argparse
import csv
import json
import re
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("pandas is required: pip install pandas") from e

from plot_pareto import (
    THROUGHPUT_KEY,
    _compute_auc,
    _json_load,
    _pareto_frontier,
    _prepare_records,
)

_SHAPE_RE = re.compile(r"_i(\d+)_o(\d+)")


def _input_len(record: dict) -> int | None:
    """Recover the dataset input length, preferring an explicit field then the path."""
    for key in ("input-len", "input_len"):
        if key in record and record[key] is not None:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                pass
    path = record.get("dataset-path") or record.get("dataset_path") or ""
    m = _SHAPE_RE.search(str(path))
    return int(m.group(1)) if m else None


def _output_len(record: dict) -> int | None:
    for key in ("custom-output-len", "custom_output_len"):
        if key in record and record[key] is not None:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                pass
    path = record.get("dataset-path") or record.get("dataset_path") or ""
    m = _SHAPE_RE.search(str(path))
    return int(m.group(2)) if m else None


def _model(record: dict) -> str:
    return record.get("model") or record.get("model_id") or "?"


def compute_scores(output_dir: Path) -> list[dict]:
    """Return one score row per (model, input_len, output_len) subtask."""
    raw_data = [
        run_data
        for path in output_dir.rglob("**/summary.json")
        for run_data in _json_load(path)
    ]
    if not raw_data:
        raise ValueError(f"No sweep results found under {output_dir}")

    prepared, skipped = _prepare_records(
        raw_data, user_count_var="max_concurrency", gpu_count_var=None
    )
    if skipped:
        print(f"Skipped {skipped} runs (missing {THROUGHPUT_KEY} or user count).")
    if not prepared:
        raise ValueError(f"No data points with {THROUGHPUT_KEY} + user count available.")

    # Tag every prepared record with its subtask key.
    for rec in prepared:
        rec["_model"] = _model(rec)
        rec["_input_len"] = _input_len(rec)
        rec["_output_len"] = _output_len(rec)

    df = pd.DataFrame.from_records(prepared)

    rows: list[dict] = []
    for (model, in_len, out_len), group in df.groupby(
        ["_model", "_input_len", "_output_len"], dropna=False
    ):
        group = group.dropna(subset=["tokens_per_user", "tokens_per_gpu"])
        if group.empty:
            continue
        frontier = _pareto_frontier(group, "tokens_per_user", "tokens_per_gpu")
        frontier = frontier.sort_values("tokens_per_user")
        auc = _compute_auc(
            frontier["tokens_per_user"].tolist(),
            frontier["tokens_per_gpu"].tolist(),
        )
        rows.append({
            "model": model,
            "input_len": None if pd.isna(in_len) else int(in_len),
            "output_len": None if pd.isna(out_len) else int(out_len),
            "auc": auc,
            "n_points": int(len(group)),
            "n_frontier": int(len(frontier)),
        })

    rows.sort(key=lambda r: (r["model"], r["input_len"] or 0, r["output_len"] or 0))
    return rows


def _shape_label(row: dict) -> str:
    i = row["input_len"]
    o = row["output_len"]
    return f"{i if i is not None else '?'}->{o if o is not None else '?'}"


def print_table(rows: list[dict]) -> None:
    header = f"{'model':40}  {'shape(in->out)':16}  {'AUC':>14}  {'pts':>5}  {'front':>5}"
    print("\n" + header)
    print("-" * len(header))
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        print(f"{row['model']:40.40}  {_shape_label(row):16}  "
              f"{row['auc']:14.2f}  {row['n_points']:5d}  {row['n_frontier']:5d}")
        by_model.setdefault(row["model"], []).append(row)
    print("-" * len(header))
    for model, mrows in by_model.items():
        mean_auc = sum(r["auc"] for r in mrows) / len(mrows)
        print(f"{model:40.40}  {'mean':16}  {mean_auc:14.2f}")
    overall = sum(r["auc"] for r in rows) / len(rows)
    print(f"{'ALL':40}  {'mean':16}  {overall:14.2f}\n")


def write_files(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Write scores.csv and scores.json (with per-model + overall aggregates)."""
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    model_means = {m: sum(r["auc"] for r in rs) / len(rs) for m, rs in by_model.items()}
    overall_mean = sum(r["auc"] for r in rows) / len(rows) if rows else 0.0

    csv_path = output_dir / "scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "input_len", "output_len", "auc", "n_points", "n_frontier"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = output_dir / "scores.json"
    json_path.write_text(json.dumps(
        {"subtasks": rows, "model_mean_auc": model_means, "overall_mean_auc": overall_mean},
        indent=2,
    ))
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("OUTPUT_DIR", help="Directory containing the sweep results (summary.json files).")
    args = parser.parse_args()

    output_dir = Path(args.OUTPUT_DIR)
    if not output_dir.exists():
        parser.error(f"Directory not found: {output_dir}")

    rows = compute_scores(output_dir)
    print_table(rows)
    csv_path, json_path = write_files(rows, output_dir)
    print(f"Wrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
