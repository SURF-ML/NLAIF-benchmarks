#!/usr/bin/env python3
"""Pull SemiAnalysis InferenceX results for the configs comparable to ours.

Their artifacts are on a public repo but GitHub requires auth to download artifacts, so
this needs a token with public_repo scope:

    GITHUB_TOKEN=ghp_... python3 profiling/fetch_semianalysis.py [RUN_ID]

Writes docs/semianalysis-results.json and prints the cells that line up with ours.
Their run 28593351944 is DeepSeek-V4 FP4 on MI355X at TP8/EP1 — same model, hardware and
parallel layout we use, so the shape+concurrency pairs below are directly comparable
modulo the caveats printed at the end.
"""
import io, json, os, re, sys, urllib.request, zipfile, pathlib

REPO = "SemiAnalysisAI/InferenceX"
RUN = sys.argv[1] if len(sys.argv) > 1 else "28593351944"
TOK = os.environ.get("GITHUB_TOKEN")
if not TOK:
    sys.exit("set GITHUB_TOKEN (public_repo scope is enough)")

def api(url):
    r = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOK}",
                                             "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(r, timeout=60)

arts = json.load(api(f"https://api.github.com/repos/{REPO}/actions/runs/{RUN}/artifacts?per_page=100"))
out = {}
for a in arts["artifacts"]:
    m = re.match(r"bmk_dsv4_(\w+?)_fp4_vllm_(tp\d+-ep\d+)[^_]*_.*?_conc(\d+)_", a["name"])
    if not m:
        continue
    shape, layout, conc = m.group(1), m.group(2), int(m.group(3))
    try:
        z = zipfile.ZipFile(io.BytesIO(api(a["archive_download_url"]).read()))
    except Exception as e:
        print(f"  skip {a['name']}: {e}"); continue
    for n in z.namelist():
        body = z.read(n).decode("utf-8", "replace")
        try:
            out[f"{shape}/{layout}/conc{conc}"] = json.loads(body)
        except json.JSONDecodeError:
            out[f"{shape}/{layout}/conc{conc}"] = {"_raw": body[:2000], "_file": n}

dst = pathlib.Path("docs/semianalysis-results.json")
dst.write_text(json.dumps(out, indent=2, sort_keys=True))
print(f"wrote {dst} with {len(out)} configs\n")

# Cells where their grid and ours coincide exactly.
print("Directly comparable to our runs (same model/hardware/TP8-EP1/shape/concurrency):")
for key, ours in [("8k1k/tp8-ep1/conc32",  "8192->1024 high  ours EP1 = 1127.0 tok/s/GPU"),
                  ("8k1k/tp8-ep1/conc16",  "8192->1024 mid   ours EP1 =  632.6 tok/s/GPU")]:
    print(f"  {key:26s} {ours}")
    v = out.get(key)
    print(f"      theirs: {json.dumps(v)[:400] if v else 'NOT FOUND'}")
print("""
Caveats before treating any delta as meaningful:
  * vLLM version — they pin a nightly, we pin v0.29.0. Worth 5.5x between v0.23.0 and
    v0.29.0, so version is not a rounding error here.
  * --max-model-len — ours is fixed at 9216 by the spec; theirs is not stated. This sets
    KV footprint per sequence and therefore the concurrency their server can admit.
  * --kv-cache-dtype fp8, prefix caching off, --ignore-eos — fixed for us by §8.1; unknown
    for them.
  * Normalisation — our tok/s/GPU divides total throughput by TP*PP = 8. Check whether
    their figure is total or per-GPU before comparing.
""")
