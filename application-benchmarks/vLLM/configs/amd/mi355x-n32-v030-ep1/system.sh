# v0.30.0 with expert parallelism OFF (TP8/EP1), N=32 @ 0.97 — the v0.30.0 counterpart of
# ../mi355x-n32-ep1, so the EP8-vs-EP1 comparison can be made on both engine versions.
# NOT SUBMITTABLE: §9 ruling 10 pins the version.
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-n32-v030" && pwd)/system.sh"
