#!/usr/bin/env python3
"""
Score a run: tok/s/GPU at each fixed operating point.

A *cell* is one (workload shape, operating point) pair, where the shape is the
(input-len, output-len) of the dataset and the point is the `_point` label carried by the
bench config. Nothing here assumes how many points there are, or what they are called: the
point set is read from the data.

For each cell we report the best tok/s/GPU achieved across every parallelism layout that was
run. Taking the max *is* the pooling across layouts: a submission may run several layouts and
each cell is credited with the best result, exactly as the earlier Pareto frontier pooled all
layouts' points into one curve.

Only raw values are written. The final score requires normalising each cell against the best
submission on that same cell and then averaging, which can only be done once every submission
is in — so that step belongs to the organisers, not to this script. The flat one-row-per-cell
schema is designed to be concatenated across submissions for exactly that purpose.

Outputs:
  * a table printed to the terminal
  * scores.csv and scores.json written into the results dir

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
    _first_present,
    _json_load,
    _prepare_records,
)

_SHAPE_RE = re.compile(r"_i(\d+)_o(\d+)")

# The scored quantity. Everything else in a row is context for auditing the number.
SCORED_METRIC = "tokens_per_gpu"


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
    # A permitted FP4 expert checkpoint (README §3) scores as its base model, so its cells stay
    # comparable with base-checkpoint results; the checkpoint itself is reported separately.
    return (record.get("_base_model") or record.get("model") or record.get("model_id")
            or "?")


def _checkpoint(record: dict) -> str:
    return record.get("model") or record.get("model_id") or "?"


def _point(record: dict) -> str:
    """Operating-point label, as tagged by the bench config."""
    value = _first_present(record, ["_point"])
    return str(value) if value is not None else "?"


def _int_or_none(record: dict, keys: list[str]) -> int | None:
    value = _first_present(record, keys)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_scores(output_dir: Path) -> list[dict]:
    """Return one row per (model, shape, point), holding the best layout's tok/s/GPU."""
    raw_data = [
        run_data
        for path in output_dir.rglob("**/summary.json")
        for run_data in _json_load(path)
    ]
    if not raw_data:
        raise ValueError(f"No results found under {output_dir}")

    prepared, skipped = _prepare_records(
        raw_data, user_count_var="max_concurrency", gpu_count_var=None
    )
    if skipped:
        print(f"Skipped {skipped} runs (missing {THROUGHPUT_KEY} or user count).")
    if not prepared:
        raise ValueError(f"No data points with {THROUGHPUT_KEY} + user count available.")

    for rec in prepared:
        rec["_model"] = _model(rec)
        rec["_input_len"] = _input_len(rec)
        rec["_output_len"] = _output_len(rec)
        rec["_point_label"] = _point(rec)

    df = pd.DataFrame.from_records(prepared)

    rows: list[dict] = []
    for (model, in_len, out_len, point), group in df.groupby(
        ["_model", "_input_len", "_output_len", "_point_label"], dropna=False
    ):
        group = group.dropna(subset=[SCORED_METRIC])
        if group.empty:
            continue
        # Pool across layouts: the best result for this cell wins.
        best = group.loc[group[SCORED_METRIC].idxmax()].to_dict()
        rows.append({
            "model": model,
            "input_len": None if pd.isna(in_len) else int(in_len),
            "output_len": None if pd.isna(out_len) else int(out_len),
            "point": point,
            "max_concurrency": _int_or_none(best, ["max_concurrency"]),
            "tokens_per_gpu": float(best[SCORED_METRIC]),
            "tokens_per_user": float(best.get("tokens_per_user") or 0.0),
            "tensor_parallel_size": _int_or_none(best, ["tensor-parallel-size"]),
            "pipeline_parallel_size": _int_or_none(best, ["pipeline-parallel-size"]),
            "max_num_seqs": _int_or_none(best, ["max_num_seqs"]),
            "n_layouts": int(len(group)),
            "checkpoint": _checkpoint(best),
        })

    rows.sort(key=lambda r: (r["model"], r["input_len"] or 0, r["output_len"] or 0,
                             r["max_concurrency"] or 0))
    return rows


def _point_order(rows: list[dict]) -> list[str]:
    """Point labels ordered by the concurrency they ran at, whatever the labels happen to be."""
    by_label: dict[str, int] = {}
    for r in rows:
        c = r["max_concurrency"] or 0
        by_label[r["point"]] = max(by_label.get(r["point"], 0), c)
    return sorted(by_label, key=lambda label: (by_label[label], label))


def _shape_label(row: dict) -> str:
    i, o = row["input_len"], row["output_len"]
    return f"{i if i is not None else '?'}->{o if o is not None else '?'}"


def _point_means(rows: list[dict]) -> dict[str, float]:
    means: dict[str, float] = {}
    for label in _point_order(rows):
        vals = [r["tokens_per_gpu"] for r in rows if r["point"] == label]
        if vals:
            means[label] = sum(vals) / len(vals)
    return means


def print_table(rows: list[dict]) -> None:
    header = (f"{'model':34}  {'shape(in->out)':16}  {'point':6}  {'conc':>6}  "
              f"{'tok/s/GPU':>12}  {'tok/s/user':>11}  {'lay':>4}")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['model']:34.34}  {_shape_label(row):16}  {row['point']:6.6}  "
              f"{row['max_concurrency'] or 0:6d}  {row['tokens_per_gpu']:12.2f}  "
              f"{row['tokens_per_user']:11.2f}  {row['n_layouts']:4d}")
    print("-" * len(header))

    means = _point_means(rows)
    for label, mean in means.items():
        print(f"{'mean over shapes':34}  {'':16}  {label:6.6}  {'':6}  {mean:12.2f}")
    if means:
        overall = sum(r["tokens_per_gpu"] for r in rows) / len(rows)
        print(f"{'mean over all cells':34}  {'':16}  {'':6}  {'':6}  {overall:12.2f}")
    print("\nRaw values. Normalisation across submissions is done by the organisers "
          "(see README §7).\n")


def write_files(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Write scores.csv and scores.json (with per-point + overall aggregates)."""
    fieldnames = ["model", "input_len", "output_len", "point", "max_concurrency",
                  "tokens_per_gpu", "tokens_per_user", "tensor_parallel_size",
                  "pipeline_parallel_size", "max_num_seqs", "n_layouts", "checkpoint"]

    csv_path = output_dir / "scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overall = (sum(r["tokens_per_gpu"] for r in rows) / len(rows)) if rows else 0.0
    json_path = output_dir / "scores.json"
    json_path.write_text(json.dumps(
        {"cells": rows,
         "point_mean_tokens_per_gpu": _point_means(rows),
         "overall_mean_tokens_per_gpu": overall},
        indent=2,
    ))
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("OUTPUT_DIR", help="Directory containing the results (summary.json files).")
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
