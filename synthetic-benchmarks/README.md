# Synthetic benchmarks

> **_NOTE:_** The synthetic benchmarks do **NOT** require commitments on performance metrics. They are just required to
> be run by the vendor based on the specifications **during acceptance**. These are for us (SURF/EuroHPC) to understand
> the system design in terms of practical performance numbers which should represent the best system capability for the
> modality being measured.

## List of synthetic benchmarks

<!-- prettier-ignore-start -->

| Test                 | Modality                                                           | Rules/Specifications                   |
|----------------------|--------------------------------------------------------------------|----------------------------------------|
| HPL-MxP              | GPU compute compared to peak performance                           | Specified in the main document         |
| STREAM               | CPU DRAM memory bandwidth                                          | Check README in the relevant directory |
| BabelStream          | GPU VRAM memory bandwidth                                          | Check README in the relevant directory |
| OSU Micro-Benchmarks | Network fabric latencies and bandwidth (both intra and inter node) | Check README in the relevant directory |
| GEMM                 | GPU MMA capabilities under different workloads                     | Check README in the relevant directory |

<!-- prettier-ignore-end -->
