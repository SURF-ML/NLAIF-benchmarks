# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import math
import re
from itertools import groupby
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("pandas is required: pip install pandas") from e

# matplotlib is imported lazily inside _plot_fig so that the metric helpers
# (_pareto_frontier, _compute_auc, _prepare_records) can be reused — e.g. by
# score.py — in environments without matplotlib installed.


# ---------------------------------------------------------------------------
# Helpers replacing vllm internal utilities
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def full_groupby(items, key):
    """Return a list of (group_key, group_items) sorted & grouped by key."""
    tagged = [(key(item), item) for item in items]
    tagged.sort(key=lambda x: str(x[0]))
    result = []
    for k, group in groupby(tagged, key=lambda x: x[0]):
        result.append((k, [item for _, item in group]))
    return result


def _json_load(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric helpers (ported from upstream plot_pareto.py)
# ---------------------------------------------------------------------------

def _first_present(run_data: dict, keys: list[str]):
    for key in keys:
        for candidate in (key, key.replace("_", "-"), key.replace("-", "_")):
            if candidate in run_data:
                return run_data[candidate]
    return None


def _get_numeric(run_data: dict, keys: list[str], *, allow_zero: bool = True):
    value = _first_present(run_data, keys)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected numeric for one of {keys}, got {value!r}"
        ) from exc
    if not allow_zero and numeric == 0:
        return None
    return numeric


def _infer_user_count(run_data: dict, user_count_var: str | None):
    candidates = [user_count_var] if user_count_var else []
    candidates.append("request_rate")
    user_count = _get_numeric(run_data, candidates, allow_zero=False)
    if user_count is not None:
        return user_count
    return _get_numeric(run_data, ["max_concurrent_requests"], allow_zero=False)


def _infer_gpu_count(run_data: dict, gpu_count_var: str | None) -> float:
    direct = _get_numeric(run_data, [gpu_count_var] if gpu_count_var else [], allow_zero=False)
    if direct:
        return direct
    tp = _get_numeric(run_data, ["tensor_parallel_size", "tp"]) or 1.0
    pp = _get_numeric(run_data, ["pipeline_parallel_size", "pp"]) or 1.0
    dp = _get_numeric(run_data, ["data_parallel_size", "dp"]) or 1.0
    return tp * pp * dp


# The throughput metric used throughout: total_token_throughput
# = (input tokens + output tokens) / second, summed across all concurrent requests.
THROUGHPUT_KEY = "total_token_throughput"


def _prepare_records(
    all_data: list[dict],
    *,
    user_count_var: str | None,
    gpu_count_var: str | None,
) -> tuple[list[dict], int]:
    prepared = []
    skipped = 0
    for record in all_data:
        throughput = _get_numeric(record, [THROUGHPUT_KEY])
        if throughput is None:
            skipped += 1
            continue
        user_count = _infer_user_count(record, user_count_var)
        if user_count is None:
            skipped += 1
            continue
        gpu_count = _infer_gpu_count(record, gpu_count_var)
        prepared.append({
            **record,
            "tokens_per_user": throughput / user_count,
            "tokens_per_gpu": throughput / gpu_count,
            "user_count_estimate": user_count,
            "gpu_count": gpu_count,
        })
    return prepared, skipped


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def _pareto_frontier(df: "pd.DataFrame", x_col: str, y_col: str, *, epsilon: float = 1e-9):
    sorted_df = df.sort_values([x_col, y_col], ascending=[False, False])
    frontier_indices = []
    best_y = -math.inf
    for idx, row in sorted_df.iterrows():
        if row[y_col] >= best_y - epsilon:
            frontier_indices.append(idx)
            best_y = max(best_y, row[y_col])
    return df.loc[frontier_indices]


# ---------------------------------------------------------------------------
# Staircase path and AUC
# ---------------------------------------------------------------------------

def _staircase_path(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """
    Convert frontier points (sorted by x ascending, y descending) into a
    staircase path that starts at (0, y0) and drops to y=0 after the last point.

    Shape per segment:
      (0, y0) -> (x0, y0)           horizontal to first point
               -> (x0, y1)           vertical drop to next y
               -> (x1, y1)           horizontal to next point
               ...
               -> (xn, yn)           arrive at last point
               -> (xn, 0)            final vertical drop to baseline
    """
    path_x: list[float] = [0.0]
    path_y: list[float] = [ys[0]]
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        path_x.append(xi)
        path_y.append(yi)
        next_y = ys[i + 1] if i + 1 < len(ys) else 0.0
        path_x.append(xi)
        path_y.append(next_y)
    return path_x, path_y


def _compute_auc(xs: list[float], ys: list[float]) -> float:
    """Area under the staircase Pareto curve (sum of width * height rectangles from x=0)."""
    auc = 0.0
    x_prev = 0.0
    for xi, yi in zip(xs, ys):
        auc += (xi - x_prev) * yi
        x_prev = xi
    return auc


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _get_fig_path(fig_dir: Path, fig_group: tuple) -> Path:
    parts = ["PARETO"]
    if fig_group:
        parts.extend(f"{k}={v}" for k, v in fig_group)
    return fig_dir / sanitize_filename("-".join(parts) + ".png")


def _curve_label(curve_group: tuple) -> str:
    if not curve_group:
        return "all"
    return ", ".join(f"{k}={v}" for k, v in curve_group)


def _workload_subtitle(df: "pd.DataFrame") -> str:
    """Build a subtitle string showing the input/output sizes present in the data."""
    parts = []
    for in_key in ("input-len", "input_len"):
        if in_key in df.columns:
            vals = sorted(df[in_key].dropna().unique().tolist())
            parts.append("in=" + "/".join(str(int(v)) for v in vals))
            break
    for out_key in ("custom-output-len", "custom_output_len"):
        if out_key in df.columns:
            vals = sorted(df[out_key].dropna().unique().tolist())
            parts.append("out=" + "/".join(str(int(v)) for v in vals))
            break
    return "  |  ".join(parts)


def _plot_fig(
    fig_dir: Path,
    fig_group: tuple,
    fig_data: list[dict],
    *,
    curve_by: list[str],
    label_by: list[str],
    dry_run: bool,
):
    fig_path = _get_fig_path(fig_dir, fig_group)

    print("[BEGIN FIGURE]")
    print(f"Group: {dict(fig_group)}")
    print(f"Output file: {fig_path}")

    if dry_run:
        print("[END FIGURE]")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError as e:
        raise ImportError("matplotlib is required to plot: pip install matplotlib") from e

    df = pd.DataFrame.from_records(fig_data)
    df = df.dropna(subset=["tokens_per_user", "tokens_per_gpu"])

    if df.empty:
        print("No data points after filtering; skipping.")
        print("[END FIGURE]")
        return

    fig, ax = plt.subplots()

    # Group into curves — one color per unique combination of curve_by values.
    if curve_by:
        def curve_key(row):
            return tuple((k, str(row[k])) for k in curve_by if k in row)
        curve_groups = full_groupby(df.to_dict("records"), key=curve_key)
    else:
        curve_groups = [((), df.to_dict("records"))]

    colors = cm.tab10.colors  # up to 10 distinct colors

    for i, (curve_group, curve_records) in enumerate(curve_groups):
        color = colors[i % len(colors)]
        label = _curve_label(curve_group)

        cdf = pd.DataFrame.from_records(curve_records)
        cdf = cdf.dropna(subset=["tokens_per_user", "tokens_per_gpu"])
        if cdf.empty:
            continue

        # Scatter: all points for this curve (same color, faded)
        ax.scatter(
            cdf["tokens_per_user"],
            cdf["tokens_per_gpu"],
            color=color,
            alpha=0.4,
            s=40,
            label=f"{label} (all)" if curve_by else "All runs",
        )

        # Pareto frontier + staircase for this curve
        frontier = _pareto_frontier(cdf, "tokens_per_user", "tokens_per_gpu")
        frontier = frontier.sort_values("tokens_per_user")

        fxs = frontier["tokens_per_user"].tolist()
        fys = frontier["tokens_per_gpu"].tolist()

        # AUC computed separately per curve
        auc = _compute_auc(fxs, fys)
        auc_label = f"AUC={auc:.1f}"

        step_xs, step_ys = _staircase_path(fxs, fys)

        curve_label_full = (
            f"{label} ({auc_label})" if curve_by else f"Pareto ({auc_label})"
        )
        ax.plot(step_xs, step_ys, color=color, linewidth=1.5, label=curve_label_full)
        ax.fill_between(step_xs, step_ys, alpha=0.08, color=color)
        ax.scatter(fxs, fys, color=color, s=40, zorder=5)

        # Annotate frontier points
        if label_by:
            for _, row in frontier.iterrows():
                parts = [f"{k}={row[k]}" for k in label_by if k in row]
                if parts:
                    ax.text(
                        row["tokens_per_user"],
                        row["tokens_per_gpu"],
                        "\n".join(parts),
                        fontsize=7,
                        color=color,
                        va="bottom",
                    )

        print(f"  Curve '{label}': {len(cdf)} points, {len(frontier)} on frontier, AUC={auc:.2f}")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # Axis labels make explicit that total (input+output) token throughput is used
    ax.set_xlabel(f"Total tok/s/user  [{THROUGHPUT_KEY}]")
    ax.set_ylabel(f"Total tok/s/GPU  [{THROUGHPUT_KEY}]")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Title: fig_group variables (if any) + workload profile (input/output sizes)
    title_parts = []
    if fig_group:
        title_parts.append(", ".join(f"{k}={v}" for k, v in fig_group))
    subtitle = _workload_subtitle(df)
    if subtitle:
        title_parts.append(subtitle)
    if title_parts:
        ax.set_title("\n".join(title_parts), fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print("[END FIGURE]")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_pareto(
    output_dir: Path,
    *,
    fig_by: list[str],
    curve_by: list[str],
    user_count_var: str | None,
    gpu_count_var: str | None,
    label_by: list[str],
    dry_run: bool,
):
    fig_dir = output_dir / "pareto"

    raw_data = [
        run_data
        for path in output_dir.rglob("**/summary.json")
        for run_data in _json_load(path)
    ]

    if not raw_data:
        raise ValueError(f"No sweep results found under {output_dir}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    prepared, skipped = _prepare_records(
        raw_data,
        user_count_var=user_count_var,
        gpu_count_var=gpu_count_var,
    )

    if skipped:
        print(f"Skipped {skipped} runs (missing {THROUGHPUT_KEY} or user count).")

    if not prepared:
        raise ValueError(f"No data points with {THROUGHPUT_KEY} + user count available.")

    def fig_key(item):
        return tuple((k, str(item[k])) for k in fig_by if k in item)

    fig_groups = full_groupby(prepared, key=fig_key)

    for fig_group, fig_data in fig_groups:
        _plot_fig(
            fig_dir,
            fig_group,
            fig_data,
            curve_by=curve_by,
            label_by=label_by,
            dry_run=dry_run,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            f"Plot Pareto frontiers using {THROUGHPUT_KEY} (input+output tok/s) "
            "normalized by user count and GPU count."
        )
    )
    parser.add_argument(
        "OUTPUT_DIR",
        help="Directory containing the sweep results (summary.json files).",
    )
    parser.add_argument(
        "--fig-by",
        type=str,
        default="",
        help="Comma-separated variables to split into separate figures.",
    )
    parser.add_argument(
        "--curve-by",
        type=str,
        default="",
        help="Comma-separated variables to draw separate curves/frontiers for, "
             "each with its own color and AUC "
             "(e.g. 'tensor_parallel_size' or 'input-len,custom-output-len').",
    )
    parser.add_argument(
        "--user-count-var",
        type=str,
        default="max_concurrency",
        help="Result key for concurrent user count. Falls back to "
             "max_concurrent_requests if missing. Default: max_concurrency",
    )
    parser.add_argument(
        "--gpu-count-var",
        type=str,
        default=None,
        help="Result key for GPU count. If omitted, inferred from "
             "tensor_parallel_size * pipeline_parallel_size.",
    )
    parser.add_argument(
        "--label-by",
        type=str,
        default="max_concurrency,gpu_count",
        help="Comma-separated fields to annotate on Pareto frontier points. "
             "Default: max_concurrency,gpu_count",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print figure info without rendering.",
    )

    args = parser.parse_args()

    output_dir = Path(args.OUTPUT_DIR)
    if not output_dir.exists():
        parser.error(f"Directory not found: {output_dir}")

    plot_pareto(
        output_dir,
        fig_by=[s for s in args.fig_by.split(",") if s],
        curve_by=[s for s in args.curve_by.split(",") if s],
        user_count_var=args.user_count_var,
        gpu_count_var=args.gpu_count_var,
        label_by=[s for s in args.label_by.split(",") if s],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
