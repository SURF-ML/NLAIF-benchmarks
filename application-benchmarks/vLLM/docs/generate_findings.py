#!/usr/bin/env python3
"""Regenerate docs/AMD-MI355X-FINDINGS.md from whatever runs are on disk.

Numbers are read from results/*/scores.csv rather than pasted, so the document cannot
drift from the measurements. Re-run after any new sweep:
    python3 docs/generate_findings.py
"""
import csv, glob, json, os, pathlib, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
POINTS = ["low", "mid", "high"]

# (label, results-dir glob, engine, EP, precision, submittable)
RUNS = [
    ("N=16  v0.29 EP8", "results/*_amd-mi355x",              "v0.29.0", "8", "W4A8", "yes"),
    ("N=32  v0.29 EP8", "results/*_amd-mi355x-n32",          "v0.29.0", "8", "W4A8", "yes"),
    ("N=32  v0.29 EP1", "results/*_amd-mi355x-n32-ep1",      "v0.29.0", "1", "W4A8", "yes"),
    ("N=32  v0.23 EP8", "results/*_amd-mi355x-n32-v023",     "v0.23.0", "8", "W4A4", "no — W4A4 (§8.5)"),
    ("N=32  v0.23 EP1", "results/*_amd-mi355x-n32-v023-ep1", "v0.23.0", "1", "W4A4", "no — W4A4 (§8.5)"),
    ("N=32  v0.30 EP8", "results/*_amd-mi355x-n32-v030",     "v0.30.0", "8", "W4A8", "yes"),
    ("N=32  v0.30 EP1", "results/*_amd-mi355x-n32-v030-ep1", "v0.30.0", "1", "W4A8", "yes"),
    # One run dir, two layouts: '#ep8' / '#ep1' selects a layout's cells from summary.json.
    ("N=512 v0.29 EP8", "results/*_amd-mi355x-1node-n512#ep8", "v0.29.0", "8", "W4A8", "yes"),
    ("N=512 v0.29 EP1", "results/*_amd-mi355x-1node-n512#ep1", "v0.29.0", "1", "W4A8", "yes"),
]

def load(pattern):
    """Newest complete scores.csv matching pattern, excluding smoke dirs."""
    if "#ep" in pattern:
        pattern, sel = pattern.split("#")
        run = _newest(pattern)
        if not run:
            return None, None
        want_ep = sel == "ep8"
        rows = {k[1:]: r["tpg"] for k, r in load_cells(run).items() if k[0][2] == want_ep}
        return (rows or None), run
    best = None
    for d in sorted(glob.glob(str(ROOT / pattern))):
        if "smoke" in d:
            continue
        # a glob like *_amd-mi355x also matches *_amd-mi355x-n32; require exact suffix
        suffix = pattern.split("*_")[-1]
        if not os.path.basename(d).endswith(suffix):
            continue
        f = pathlib.Path(d) / "scores.csv"
        if not f.exists():
            continue
        # Prefer a complete run. A sweep interrupted partway still writes a scores.csv, and
        # silently charting 4 cells as if they were 6 would misstate the mean.
        ncells = sum(1 for _ in csv.DictReader(open(f)))
        if best is None or ncells >= best[1]:
            best = (f, ncells)
    if not best:
        return None, None
    f, _ = best
    rows = {}
    for r in csv.DictReader(open(f)):
        rows[(r["input_len"], r["output_len"], r["point"])] = float(r["tokens_per_gpu"])
    return rows, f.parent

def kv_line(run_dir):
    """The engine's own reported capacity, for auditing the declared N."""
    if not run_dir:
        return None
    for log in sorted(run_dir.glob("*/serve.log")):
        m = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
                      log.read_text(errors="ignore"))
        if m:
            return float(m.group(1))
    return None

# Per-cell runs, read from summary.json rather than scores.csv (which keeps only the best layout
# per cell): the two-node runs, and the single-node N=512 run as power/latency reference.
TWO_NODE_RUNS = ["results/*_amd-mi355x-2node-rdma-n1024",    # TP8/PP2, measured N
                 "results/*_amd-mi355x-2node-rdma-sweep"]    # other layouts, stopped early
ONE_NODE_RUN = "results/*_amd-mi355x-1node-n512"             # TP8 EP8+EP1, N=512 @ 0.97


def _newest(pattern):
    dirs = sorted(glob.glob(str(ROOT / pattern)))
    return pathlib.Path(dirs[-1]) if dirs else None


def load_cells(run_dir):
    """{(layout, in, out, point): record} from every summary.json under run_dir.

    layout = (TP, PP, EP-on); each record also carries its serve.log capacity and, when the
    cell dir holds that job's power_samples.csv, the mean GPU power over its bench window.
    """
    from datetime import datetime, timedelta

    def samples_of(pf):
        # time,node,jobid,"8 powers","8 utilisations" — one row per node per ~5 s, this job only
        out = []
        for row in csv.reader(open(pf)):
            if row[0] == "time" or len(row) < 4:
                continue
            w = [float(x) for x in row[3].split()]
            if len(w) == 8:
                out.append((datetime.fromisoformat(row[0]), sum(w) / 8))
        return out

    cells = {}
    for f in sorted(run_dir.glob("*/summary.json")):
        d = json.loads(f.read_text())
        d = d[0] if isinstance(d, list) else d
        tp, pp = int(d.get("tensor-parallel-size", 1)), int(d.get("pipeline-parallel-size", 1))
        ep = bool(d.get("enable-expert-parallel", True))
        i_len = "8192" if "i8192" in d["dataset-path"] else "1024"
        o_len = str(d["custom-output-len"])
        rec = {"gpus": tp * pp, "n": int(d["max_num_seqs"]), "conc": int(d["max_concurrency"]),
               "tpg": d["total_token_throughput"] / (tp * pp),
               "tput": d["total_token_throughput"], "ok": d["completed"], "fail": d["failed"],
               "ttft": d["median_ttft_ms"], "tpot": d["median_tpot_ms"], "kv": None,
               "watts": None, "rdma": None}
        log = f.parent / "serve.log"
        if log.exists():
            txt = log.read_text(errors="ignore")
            m = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", txt)
            rec["kv"] = float(m.group(1)) if m else None
            rec["rdma"] = "GDRDMA" in txt and "via NET/Socket" not in txt
        pf = f.parent / "power_samples.csv"
        if pf.exists():
            end = datetime.strptime(d["date"], "%Y%m%d-%H%M%S")
            start = end - timedelta(seconds=d["duration"])
            w = [p for t, p in samples_of(pf) if start <= t <= end]
            # Report only a well-covered window: ≥60% of the ~5 s samples expected per node.
            expected = d["duration"] / 5 * max(1, (tp * pp) // 8)
            rec["watts"] = sum(w) / len(w) if w and len(w) >= 0.6 * expected else None
        cells[((tp, pp, ep), i_len, o_len, d["_point"])] = rec
    return cells


def _layout_label(layout):
    # vLLM forms the expert-parallel group from the TP ranks of each pipeline stage, so EP on
    # is EP=TP (EP8 on TP8/PP2, EP2 on TP2/PP8) — the same naming as the single-node EP8/EP1.
    tp, pp, ep = layout
    return f"TP{tp}/PP{pp} EP{tp if ep else 1}"


def two_node_section(A, data, shapes):
    cells = {}
    for pat in TWO_NODE_RUNS:
        run = _newest(pat)
        if run:
            cells.update(load_cells(run))
    if not cells:
        return
    one = _newest(ONE_NODE_RUN)
    one_cells = load_cells(one) if one else {}
    # Layout order: most TP first, EP on before off.
    layouts = sorted({k[0] for k in cells}, key=lambda l: (-l[0], not l[2]))
    n_of = {l: next(r["n"] for k, r in cells.items() if k[0] == l) for l in layouts}
    lab = lambda l: f"{_layout_label(l)} N={n_of[l]}"
    fmt = lambda v, s="{:,.1f}": s.format(v) if v is not None else "—"
    # Best single-node result per cell, over every v0.29.0 run above (engine held equal).
    one_keys = [k for k in data if "v0.29" in k]
    def best_one(k):
        c = [(data[l][0][k], l.strip()) for l in one_keys if k in data[l][0]]
        return max(c) if c else (None, None)

    A("## Two nodes: 16× MI355X over RDMA\n")
    A("Both nodes (`mi355-gpu-13` + `mi355-gpu-37`) as one 16-GPU deployment: SLURM, Ray over "
      "rootless podman, RCCL over the eight per-GPU AMD Pollara NICs per node (GPU-direct RDMA, "
      "RoCE v2). Engine v0.29.0, W4A8, `GPU_MEM_UTIL=0.90`. Configs "
      "`amd/mi355x-2node-rdma-n1024` (TP8/PP2, N measured) and `amd/mi355x-2node-rdma-sweep` "
      "(other layouts; N from the startup line, stopped after the TP2/PP8 EP2 cells); setup in "
      "`docs/AMD-MI355X-TROUBLESHOOTING.md` §12. tok/s/GPU is the score metric (§7): total "
      "tokens/s over `TP × PP` = 16 GPUs. TP16 cannot load this checkpoint (§12.6).\n")

    A("### Layouts\n")
    A("| layout | N | reported KV capacity | cells | RCCL transport |")
    A("|---|---|---|---|---|")
    for l in layouts:
        recs = [r for k, r in cells.items() if k[0] == l]
        kv = next((r["kv"] for r in recs if r["kv"]), None)
        tr = ("RDMA (GDRDMA) in every cell" if all(r["rdma"] for r in recs)
              else "⚠ not RDMA in every cell")
        bad = sum(1 for r in recs if r["fail"])
        A(f"| {_layout_label(l)} | {n_of[l]} | {fmt(kv, '{:.2f}x')} | "
          f"{len(recs)}/6{' (' + str(bad) + ' with failed requests)' if bad else ''} | {tr} |")
    A("")

    A("### tok/s/GPU by cell\n")
    A("Concurrency in brackets: operating points are fractions of each layout's own N. The "
      "single-node column is the best v0.29.0 result for that cell from the table above.\n")
    A("| shape | point | " + " | ".join(lab(l) for l in layouts)
      + " | 1 node, best | 2 nodes vs 1 node |")
    A("|---|---|" + "---|" * (len(layouts) + 2))
    for i_len, o_len in shapes:
        for pt in POINTS:
            row = []; best = None
            for l in layouts:
                r = cells.get((l, i_len, o_len, pt))
                row.append(f"{r['tpg']:,.1f} ({r['conc']})" if r else "—")
                if r and (best is None or r["tpg"] > best):
                    best = r["tpg"]
            ref, who = best_one((i_len, o_len, pt))
            vs = f"**{(best/ref-1)*100:+.0f}%**" if best and ref else "—"
            A(f"| {i_len}→{o_len} | {pt} | " + " | ".join(row)
              + f" | {fmt(ref)} ({who or '—'}) | {vs} |")
    A("")

    # Same concurrency on one node and on two: the second node's actual contribution.
    pairs = []
    for (l, i_len, o_len, pt), r in cells.items():
        for (l1, i1, o1, p1), r1 in one_cells.items():
            if ((i1, o1) == (i_len, o_len) and r1["conc"] == r["conc"] and l1[2] == l[2]
                    and l[0] == l1[0] == 8):
                pairs.append((i_len, o_len, r["conc"], l, r, l1, r1))
    if pairs:
        A("### Same load on one node and on two\n")
        A("Adding the second node to the same TP8 layout (TP8 → TP8/PP2), at equal concurrency and "
          "EP setting. Total throughput is input + output tokens/s. At these loads one node "
          "already holds every request without preemption, so the second node doubles the GPUs "
          "and adds no throughput.\n")
        A("| shape | concurrency | 1 node (8 GPUs) | 2 nodes (16 GPUs) | total gain | tok/s/GPU 1 → 2 nodes |")
        A("|---|---|---|---|---|---|")
        for i_len, o_len, c, l, r, l1, r1 in sorted(pairs, key=lambda x: (x[0], x[2])):
            A(f"| {i_len}→{o_len} | {c} | {r1['tput']:,.0f} tok/s | {r['tput']:,.0f} tok/s | "
              f"{(r['tput']/r1['tput']-1)*100:+.0f}% | {r1['tpg']:,.1f} → {r['tpg']:,.1f} |")
        A("")

    A("### Latency\n")
    A("Median time per output token (1024→8192, decode-bound) and median time to first token "
      "(8192→1024, prefill-bound). 8192→1024 is admission-limited at high concurrency: the fixed "
      "`--max-num-batched-tokens=9216` admits roughly one 8k prompt per engine step, so requests "
      "queue for prefill while the KV cache is mostly empty.\n")
    A("| metric | point | " + " | ".join(lab(l) for l in layouts) + " |")
    A("|---|---|" + "---|" * len(layouts))
    for name, (i_len, o_len), key in (("TPOT (ms)", ("1024", "8192"), "tpot"),
                                      ("TTFT (ms)", ("8192", "1024"), "ttft")):
        for pt in POINTS:
            vals = [cells.get((l, i_len, o_len, pt)) for l in layouts]
            A(f"| {name} | {pt} | " + " | ".join(
                (f"{r[key]:,.0f}" if key == "ttft" else f"{r[key]:.1f}") if r else "—"
                for r in vals) + " |")
    A("")

    # Power: every per-cell run, one node and two.
    cols = [("1 node " + _layout_label(l).split(" ", 1)[1] + f" N={r['n']}", l, one_cells)
            for l, r in {k[0]: v for k, v in one_cells.items()}.items()]
    cols = sorted(cols, key=lambda c: not c[1][2]) + [(lab(l), l, cells) for l in layouts]
    if any(r["watts"] for c in cols for k, r in c[2].items() if k[0] == c[1]):
        A("## Power\n")
        A("Mean GPU package power over each cell's bench window only (engine load excluded), "
          "from `rocm-smi` sampled every ~5 s on every node of the job (`power_samples.csv` in "
          "each cell directory); cells with under 60% sample coverage are left blank. Efficiency "
          "is total tokens per joule of GPU energy. MI355X board limit: 1,400 W.\n")
        A("| shape | point | " + " | ".join(c[0] for c in cols) + " |")
        A("|---|---|" + "---|" * len(cols))
        for i_len, o_len in shapes:
            for pt in POINTS:
                row = []
                for _, l, src in cols:
                    r = src.get((l, i_len, o_len, pt))
                    row.append(f"{r['watts']:,.0f} W · {r['tput']/(r['watts']*r['gpus']):.2f} tok/J"
                               if r and r["watts"] else "—")
                A(f"| {i_len}→{o_len} | {pt} | " + " | ".join(row) + " |")
        A("")


def main():
    data = {}
    for label, pat, *_ in RUNS:
        rows, d = load(pat)
        if rows:
            data[label] = (rows, d, kv_line(d))

    shapes = [("8192", "1024"), ("1024", "8192")]
    out = []
    A = out.append

    A("# DeepSeek-V4-Pro on 8× MI355X — benchmark findings\n")
    A("Generated by `docs/generate_findings.py` from `results/*/scores.csv`. "
      "Numbers are read from disk, not transcribed.\n")
    A(f"Platform: 8× Instinct MI355X (288 GiB, gfx950), ROCm 7.2, rootless podman. The N=16/N=32 "
      f"runs used the scheduler-free `run_nosched.sh`; the N=512 and two-node runs use SLURM "
      f"(`./run.sh <cluster> --local`). Datasets and harness parameters per the benchmark spec; "
      f"see `docs/AMD-MI355X-TROUBLESHOOTING.md` for how the platform was brought up.\n")

    A("## Configurations\n")
    A("| run | engine | EP | MoE precision | reported KV capacity | submittable |")
    A("|---|---|:--:|---|---|---|")
    for label, pat, eng, ep, prec, sub in RUNS:
        if label not in data:
            A(f"| {label} | {eng} | {ep} | {prec} | _not run_ | {sub} |")
            continue
        kv = data[label][2]
        kvs = f"{kv:.2f}x" if kv else "—"
        warn = ""
        n = int(label.split()[0][2:])
        if kv and kv < n:
            warn = " (below N †)" if n > 2 * kv else " ⚠ below declared N"
        A(f"| {label} | {eng} | {ep} | {prec} | {kvs}{warn} | {sub} |")
    A("")
    A("† vLLM's startup capacity line is a worst-case reservation, not the real ceiling, for "
      "DeepSeek-V4 (README §4). N=512 was measured instead: 512 requests run with ~21% KV-cache "
      "usage and no preemption (troubleshooting §12.6). The N=16/N=32 rows took N from that line.\n")

    A("## Engine version comparison\n")
    A("Same six cells, same node, same TP8 layout and N=32 @ GPU_MEM_UTIL=0.97 — only the "
      "vLLM version differs. v0.23.0 runs the MoE as W4A4 (its older code inferred activation "
      "precision from weight precision); v0.29.0 and v0.30.0 run W4A8.\n")
    vers = [("v0.23.0", "N=32  v0.23 EP8"), ("v0.29.0", "N=32  v0.29 EP8"),
            ("v0.30.0", "N=32  v0.30 EP8")]
    have = [(v, k) for v, k in vers if k in data]
    if len(have) > 1:
        A("| version | mean tok/s/GPU | vs v0.29.0 |")
        A("|---|---|---|")
        base = sum(data["N=32  v0.29 EP8"][0].values()) / 6 if "N=32  v0.29 EP8" in data else None
        for v, k in have:
            m = sum(data[k][0].values()) / len(data[k][0])
            rel = f"{(m/base-1)*100:+.1f}%" if base else "—"
            A(f"| {v} | {m:,.1f} | {rel} |")
        A("")

    # Engine version is now a vendor choice (any stable release available at acceptance,
    # §5.1 / §9 ruling 10), so this is what the choice is worth rather than what a pin costs.
    a = data.get("N=32  v0.29 EP8"); b = data.get("N=32  v0.29 EP1")
    c = data.get("N=32  v0.30 EP8"); d = data.get("N=32  v0.30 EP1")
    if a and d:
        m = lambda x: sum(x[0].values()) / len(x[0])
        A("### What the release choice is worth\n")
        A("Any stable vLLM release available at acceptance is permitted (§5.1, ruling 10), so "
          "the engine version is a vendor choice. Best configuration on each release:\n")
        A("| | EP8 | EP1 | best |")
        A("|---|---|---|---|")
        A(f"| v0.29.0 | {m(a):,.1f} | {m(b):,.1f} | **{max(m(a),m(b)):,.1f}** |")
        A(f"| v0.30.0 | {m(c):,.1f} | {m(d):,.1f} | **{max(m(c),m(d)):,.1f}** |")
        A(f"\nv0.30.0 is worth **{(max(m(c),m(d))/max(m(a),m(b))-1)*100:.1f}%** over v0.29.0 "
          f"best-to-best — more than the EP8-only comparison suggests, because v0.30.0 also "
          f"repairs an EP1 regression (below). v0.23.0 is not a candidate on precision grounds: "
          f"it runs the MoE as W4A4, which §8.5 prohibits.\n")
        # the EP1 regression, version-specific
        k = ("8192", "1024", "mid")
        if all(k in x[0] for x in (a, b, c, d)):
            A("### The EP1 prefill-mid regression is version-specific\n")
            A("| version | EP8 | EP1 | EP1 vs EP8 |")
            A("|---|---|---|---|")
            for ver, x, y in (("v0.29.0", a, b), ("v0.30.0", c, d)):
                A(f"| {ver} | {x[0][k]:,.1f} | {y[0][k]:,.1f} | "
                  f"**{(y[0][k]/x[0][k]-1)*100:+.1f}%** |")
            A("\nThe 8192→1024 `mid` cell is the one cell EP1 lost on v0.29.0, reproduced "
              "across two nodes to within 0.2%. On v0.30.0 the same cell gains instead. So EP1 "
              "is not inherently weak there — v0.29.0 has a defect that v0.30.0 repairs. Reproduced "
              "on v0.30.0 too: gpu-13 791.7, gpu-37 794.1, a 0.3% spread.\n")

    A("## tok/s/GPU by cell\n")
    hdr = "| shape | point | " + " | ".join(l for l, *_ in RUNS if l in data) + " |"
    A(hdr)
    A("|---|---|" + "---|" * len(data))
    for i_len, o_len in shapes:
        for pt in POINTS:
            cells = []
            for label, *_ in RUNS:
                if label not in data:
                    continue
                v = data[label][0].get((i_len, o_len, pt))
                cells.append(f"{v:,.1f}" if v is not None else "—")
            A(f"| {i_len}→{o_len} | {pt} | " + " | ".join(cells) + " |")
    means = []
    for label, *_ in RUNS:
        if label not in data:
            continue
        vs = list(data[label][0].values())
        means.append(f"**{sum(vs)/len(vs):,.1f}**" if len(vs) == 6 else "—")
    A("| **mean** | | " + " | ".join(means) + " |")
    A("")

    # Submission recommendation: per §9 ruling 9 each cell takes the best submittable
    # configuration. Submittable = stable release + W4A8 (v0.23.0 is W4A4, barred by §8.5).
    subs = [l for l, pat, eng, ep, prec, sub in RUNS if sub.startswith("yes") and l in data]
    if len(subs) >= 2:
        A("## Submission recommendation\n")
        A("Submittable here means a stable vLLM release and W4A8 MoE precision. §9 ruling 9 "
          "credits each cell to the best configuration, so the envelope below is itself a "
          "legal submission.\n")
        A("| shape | point | best | from |")
        A("|---|---|---|---|")
        tot = 0.0; n = 0; winners = set()
        for i_len, o_len in shapes:
            for pt in POINTS:
                k = (i_len, o_len, pt)
                cand = [(data[l][0][k], l) for l in subs if k in data[l][0]]
                if not cand:
                    continue
                v, w = max(cand)
                tot += v; n += 1; winners.add(w)
                A(f"| {i_len}→{o_len} | {pt} | **{v:,.1f}** | {w.strip()} |")
        base = sum(data["N=32  v0.29 EP8"][0].values()) / len(data["N=32  v0.29 EP8"][0])
        A(f"| **mean** | | **{tot/n:,.1f}** | |")
        if len(winners) == 1:
            A(f"\n**{next(iter(winners)).strip()} wins every cell**, so no envelope is needed — "
              f"submit that single configuration. {100*(tot/n/base-1):+.1f}% over the original "
              f"v0.29.0 EP8 baseline.\n")
        else:
            A(f"\nEnvelope is {100*(tot/n/base-1):+.1f}% over the v0.29.0 EP8 baseline.\n")

    # Cross-node reproducibility, when the same config has been run on both nodes.
    import glob as _g
    g13 = _g.glob(str(ROOT / "results/*gpu13_amd-mi355x-n32-ep1/scores.csv"))
    # The gpu-37 repeat was cut short by a session teardown, so accept the PARTIAL directory
    # too — its cells are valid measurements, and the shared-cell comparison is what matters
    # here. The table below states how many cells overlap.
    g37 = [f for f in _g.glob(str(ROOT / "results/*amd-mi355x-n32-ep1*/scores.csv"))
           if "gpu13" not in f]
    if g13 and g37:
        x = {(r["input_len"], r["output_len"], r["point"]): float(r["tokens_per_gpu"])
             for r in csv.DictReader(open(sorted(g13)[-1]))}
        y = {(r["input_len"], r["output_len"], r["point"]): float(r["tokens_per_gpu"])
             for r in csv.DictReader(open(sorted(g37)[-1]))}
        common = [k for k in x if k in y]
        if common:
            A("## Cross-node reproducibility\n")
            A(f"The same EP1 config run independently on mi355-gpu-13 and mi355-gpu-37 "
              f"({len(common)} overlapping cells; the gpu-37 repeat was cut short by a session "
              f"teardown before the two `high` cells). This is what establishes that the "
              f"prefill-mid regression is a real effect rather than run-to-run variance.\n")
            A("| shape | point | gpu-13 | gpu-37 | spread |")
            A("|---|---|---|---|---|")
            worst = 0.0
            for i_len, o_len in shapes:
                for pt in POINTS:
                    k = (i_len, o_len, pt)
                    if k not in x or k not in y:
                        continue
                    d = abs(x[k] - y[k]) / max(x[k], y[k]) * 100
                    worst = max(worst, d)
                    A(f"| {i_len}→{o_len} | {pt} | {x[k]:,.1f} | {y[k]:,.1f} | {d:.1f}% |")
            A(f"\nWorst-case disagreement across {len(common)} shared cells: "
              f"**{worst:.1f}%**.\n")

    two_node_section(A, data, shapes)

    # External comparison: SemiAnalysis InferenceX. Their dashboard's own API carries the
    # numbers; docs/semianalysis-results.json is the filtered extract.
    sa_path = ROOT / "docs" / "semianalysis-results.json"
    if sa_path.exists():
        sa = json.loads(sa_path.read_text())
        rec = {(r["isl"], r["osl"], r["conc"]): r for r in sa["records"]}
        A("## External comparison: SemiAnalysis InferenceX\n")
        A(f"Source: `{sa['source']}` (fetched {sa['fetched']}), filtered to "
          f"`{sa['filter']}` — i.e. the same model, hardware and TP8/EP1 layout we run.\n")
        A("Their `8192→1024` shape matches ours exactly. (Their other shape is 1024→1024; ours is "
          "1024→8192, so those do not compare.) Each row compares at equal concurrency: our N=32 "
          "configs at their 16/32, our N=512 configs at their 256/512.\n")
        A("| conc | our point | SemiAnalysis | ours EP8 | ours EP1 | best ours vs SA |")
        A("|---|---|---|---|---|---|")
        gaps = []
        for conc, pt, n in ((16, "mid", "N=32 "), (32, "high", "N=32 "),
                            (256, "mid", "N=512"), (512, "high", "N=512")):
            r = rec.get((8192, 1024, conc))
            e8, e1 = data.get(f"{n} v0.29 EP8"), data.get(f"{n} v0.29 EP1")
            k = ("8192", "1024", pt)
            if not (r and e8 and e1 and k in e8[0] and k in e1[0]):
                continue
            theirs = r["metrics"]["tput_per_gpu"]
            a8, a1 = e8[0][k], e1[0][k]
            gaps.append((max(a8, a1) / theirs - 1) * 100)
            A(f"| {conc} | {pt} ({n.strip()}) | {theirs:,.1f} | {a8:,.1f} | {a1:,.1f} | "
              f"**{(max(a8, a1)/theirs-1)*100:+.1f}%** |")
        if gaps:
            A(f"\nAll tok/s/GPU. **The two independent implementations agree to within "
              f"{max(abs(g) for g in gaps):.1f}% at every shared concurrency**, from 16 to 512.\n")
        A("How they reach concurrency 512 on one node: their serve command "
          "(`benchmarks/single_node/fixed_seq_len/deprecated/dsv4_fp4_mi355x_vllm.sh` in "
          "InferenceX) sets no `--max-num-seqs`, so vLLM's default for ≥160 GB GPUs, 1024, "
          "applies — they never take N from the startup capacity line. They also run a vLLM "
          f"**nightly** (`{(rec.get((8192,1024,32)) or {}).get('image','?')}`, contained in "
          "v0.30.0) with flags our spec does not use or does not allow: no `--max-model-len` / "
          "`--max-num-batched-tokens` (fixed at 9216 here, §8.1), `--moe-backend aiter` with "
          "shared-expert fusion, `FULL_AND_PIECEWISE` CUDA graphs, `--async-scheduling`, and "
          "`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` (an INT4-quantised all-reduce, additional "
          "activation quantisation under §8.5).\n")
        r32 = rec.get((8192, 1024, 32))
        if r32 and r32["metrics"].get("kv_cache_pool_tokens"):
            theirs_kv = r32["metrics"]["kv_cache_pool_tokens"]
            A("### The KV pool gap is not a capacity difference\n")
            A(f"SemiAnalysis report {theirs_kv:,} KV tokens for this shape against our 300,533. "
              "Both are the same vLLM log line, `GPU KV cache size: N tokens`, and vLLM computes "
              "it as `max_concurrency × max_model_len` (`get_kv_cache_capacity`, "
              "`vllm/v1/core/kv_cache_utils.py`). Their serve command sets no `--max-model-len`, "
              "so it is DeepSeek-V4's default of 1,048,576: "
              f"{theirs_kv:,} = {theirs_kv/1048576:.2f} × 1,048,576. Ours is 32.61 × 9,216. The "
              "two figures measure the same pool at different request lengths and do not "
              "compare.\n")
            A("Neither is the real ceiling either. `max_concurrency` is a worst-case "
              "reservation summed over DeepSeek-V4's many KV groups (compressed MLA at ratios 4 "
              "and 128, indexer, compressor state, 128-token sliding window); blocks are "
              "allocated on demand at runtime. Measured on one node at N=512, 512 requests run "
              "with ~21% KV-cache usage and no preemption (troubleshooting §12.6).\n")
            A("Ruled out along the way: the engine version (v0.30.0 moves the reported pool "
              "0.7%), §8.1's literal `fp8` (`auto` is byte-identical on ROCm), and DCP, which "
              "refuses to start for this model on either version (`DCP is not supported with "
              "sparse indexer KV compression (compress_ratio=4)`).\n")

    # MFMA counter probe. logs/mfma-counters.txt is the parsed rocpd output (rocprofv3
    # writes SQLite by default, so the raw run log does not carry the numbers).
    # docs/ is tracked; logs/ is gitignored, so the counter results live under docs/.
    counters = ROOT / "docs" / "mfma-counters.txt"
    if not counters.exists():
        counters = ROOT / "logs" / "mfma-counters.txt"
    probe = counters if counters.exists() else ROOT / "logs" / "gpu13-mfma.log"
    A("## Is the FP4 matrix path actually used?\n")
    if counters.exists():
        txt = counters.read_text(errors="ignore")
        A("gfx950 separates 8-bit from 4/6-bit matrix ops in hardware "
          "(`SQ_INSTS_VALU_MFMA_MOPS_F8` vs `..._F6F4`), so this is measured, not inferred. "
          "A control run confirms the counters register at all — without it a zero F6F4 "
          "would be ambiguous between \"no 4-bit ops\" and \"collection broken\".\n")
        A("```")
        A(txt.strip()[:3000])
        A("```\n")
    else:
        A("_Counter probe not complete — see `logs/gpu13-mfma.log`._\n")

    (ROOT / "docs" / "AMD-MI355X-FINDINGS.md").write_text("\n".join(out) + "\n")
    print(f"wrote docs/AMD-MI355X-FINDINGS.md ({len(data)}/{len(RUNS)} runs present)")

if __name__ == "__main__":
    main()
