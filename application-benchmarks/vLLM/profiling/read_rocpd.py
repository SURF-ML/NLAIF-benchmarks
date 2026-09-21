#!/usr/bin/env python3
"""Sum MFMA operand-width counters out of a rocprofv3 rocpd (SQLite) result file.

rocprofv3 defaults to rocpd now, and suffixes its table names with a per-run GUID, so
tables are discovered rather than hard-coded.

Usage: read_rocpd.py <results.db> [<results.db> ...]
"""
import sqlite3, sys, collections, pathlib

def totals(path):
    con = sqlite3.connect(path)
    tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    info = next((t for t in tabs if t.startswith("rocpd_info_pmc")), None)
    ev   = next((t for t in tabs if t.startswith("rocpd_pmc_event")), None)
    if not (info and ev):
        return None
    names = {r[0]: r[1] for r in con.execute(f"select id, name from {info}")}
    tot = collections.Counter()
    for pmc_id, val in con.execute(f"select pmc_id, value from {ev}"):
        tot[names.get(pmc_id, f"pmc_{pmc_id}")] += val or 0
    return tot

for p in sys.argv[1:]:
    label = pathlib.Path(p).stem
    print(f"--- {label} ---")
    tot = totals(p)
    if not tot:
        print("  (no pmc tables)"); continue
    for k, v in sorted(tot.items()):
        if "MFMA" in k.upper():
            print(f"  {k:36s} {v:,.0f}")
    f8 = sum(v for k, v in tot.items() if k.endswith("MFMA_MOPS_F8"))
    f4 = sum(v for k, v in tot.items() if k.endswith("MFMA_MOPS_F6F4"))
    bf = sum(v for k, v in tot.items() if k.endswith("MFMA_MOPS_BF16"))
    print(f"  {'':36s} F8={f8:,.0f}  F6F4={f4:,.0f}  BF16={bf:,.0f}")
    if f8 + f4:
        print(f"  -> 4/6-bit share of low-precision MFMA: {100*f4/(f8+f4):.2f}%")
    else:
        print("  -> no low-precision MFMA counted")
