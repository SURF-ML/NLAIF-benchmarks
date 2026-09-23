# v0.29.0, N=32 @ GPU_MEM_UTIL=0.97, but expert parallelism OFF (TP8 / EP1).
#
# Single-variable change from ../mi355x-n32: same image, same checkpoint, same layout, same N —
# only --enable-expert-parallel is dropped.
#
# THIS ONE IS SUBMITTABLE. §8.4 lists --enable-expert-parallel as Tunable ("typically required
# for this MoE model", but tunable), and v0.29.0 is a stable release. So if EP1 scores
# higher it is a legitimate submission, unlike the v0.23.0 and W4A4 probes.
#
# Motivation: SemiAnalysis's InferenceX recipe for this model on this hardware
# (dsv4-fp4-mi355x-vllm) runs TP8/EP1, not TP8/EP8. Worth knowing whether that layout choice is
# worth points on our operating-point grid.
#
# Watch the KV capacity. On v0.23.0, dropping EP cost ~32 GiB of KV cache (169.66 -> 137.84 GiB,
# 36.17x -> 29.38x) because the experts stop being distributed across ranks and more HBM goes to
# weights. If the same happens here, 32.61x could fall below 32 and the high operating point
# would preempt — check before trusting the high cell:
#     grep "Maximum concurrency" results/<run>/*/serve.log
# If it lands under 32.0, the honest move is to re-declare N for this layout rather than report a
# preempting high cell.
#
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-n32" && pwd)/system.sh"
