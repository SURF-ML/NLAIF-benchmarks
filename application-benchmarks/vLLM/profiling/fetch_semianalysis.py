#!/usr/bin/env python3
"""Pull SemiAnalysis InferenceX results for the configs comparable to ours.

Their dashboard is a Next.js app; the numbers come from its own public API. The model
parameter is the DISPLAY name, not the URL slug — /inference/deepseek-v4 renders from
?model=DeepSeek-V4-Pro, and passing "deepseek-v4" returns {"error":"Unknown model"}.

    python3 profiling/fetch_semianalysis.py            # refresh docs/semianalysis-results.json

(The GitHub Actions artifacts on SemiAnalysisAI/InferenceX carry the same data but need a
token — artifact downloads are authenticated even for public repos. This API does not.)
"""
import json, pathlib, sys, urllib.request

URL = "https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4-Pro"
OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "semianalysis-results.json"
KEEP = ("tput_per_gpu", "input_tput_per_gpu", "output_tput_per_gpu",
        "kv_cache_pool_tokens", "mean_ttft", "mean_tpot", "mean_e2el")

# The server gzips regardless of Accept-Encoding, so decompress explicitly.
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0",
                                           "Accept-Encoding": "gzip"})
raw = urllib.request.urlopen(req, timeout=90).read()
if raw[:2] == b"\x1f\x8b":
    import gzip
    raw = gzip.decompress(raw)
data = json.loads(raw)

# Match our configuration: single-node MI355X, vLLM, FP4, TP8/EP1, no disagg, no speculation.
m = [r for r in data
     if r.get("hardware") == "mi355x" and r.get("framework") == "vllm"
     and r.get("precision") == "fp4" and not r.get("disagg")
     and r.get("decode_tp") == 8 and r.get("decode_ep") == 1
     and not r.get("is_multinode") and str(r.get("spec_method")) in ("none", "None")]
if not m:
    sys.exit("no matching records — check whether the filter fields changed")

out = {"source": URL,
       "filter": "hardware=mi355x framework=vllm precision=fp4 TP8/EP1 non-disagg no-spec single-node",
       "fetched": __import__("datetime").date.today().isoformat(),
       "records": [{"isl": r["isl"], "osl": r["osl"], "conc": r["conc"],
                    "image": r.get("image"), "date": r.get("date"),
                    "metrics": {k: r["metrics"].get(k) for k in KEEP}}
                   for r in sorted(m, key=lambda r: (r["isl"], r["conc"]))]}
OUT.write_text(json.dumps(out, indent=2))
print(f"wrote {OUT} with {len(out['records'])} records")
for r in out["records"]:
    if r["isl"] == 8192 and r["conc"] in (16, 32):
        print(f"  {r['isl']}->{r['osl']} conc{r['conc']:<4} "
              f"{r['metrics']['tput_per_gpu']:8.1f} tok/s/GPU  "
              f"KV pool {r['metrics']['kv_cache_pool_tokens']:,}")
