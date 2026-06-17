# NLAIF Pre-training benchmarks

Pre-training benchmark suite for the NLAIF procurement. This suite contains three pre-training benchmarks. Two from MLPerf's training benchmark suite (Deepseek V3 and GPT-OSS-20B) and one custom computer vision suite (DINOv3). 

---

## Benchmarks

| Benchmark | Model | Target hardware | Key dataset | Quality target |
|-----------|-------|----------------|-------------|----------------|
| **DINOv3 7B** | 7  ViT (teacher‑student) | 32 GPUs (e.g., 4 × 8‑GPU nodes) | ImageNet‑1k (upsampled to 512 × 512) | 36 % top‑5 k‑NN accuracy on ImageNet‑val  |
| **GPT‑OSS‑20B** | 20B MoE LLM | 8 GPUs (single or 2‑node) | C4 EN 3.0.1  | Validation loss ≤ 3.34 (log‑perplexity)  |
| **DeepseekV3 671B** | 671B MoE LLM | Full partition | C4 EN 3.0.1  | Validation loss ≤ 3.34 (log‑perplexity) | 

---

## Repository layout

```
/training-benchmarks
│
├─ DINOv3-7B/                    # DINOv3 benchmark
│   ├─ dinov3_src/               # Meta DINOv3 repo (submodule)
│   ├─ configs/train/            # Training configs
│   ├─ eval/                     # Evaluation outputs
│   ├─ logs/                     # Job logs
│   ├─ stored_checkpoints/       # Saved checkpoints
│   ├─ imagenet1k_prep.py        # Dataset preparation script
│   ├─ record_time_and_run.sh    # Timed run wrapper
│   ├─ run_benchmark.job         # SLURM job script
│   ├─ run_eval.job              # SLURM eval job script
│   ├─ extract_time.sh           # Timing extraction helper
│   ├─ requirements.txt
│   ├─ RULES_AND_SCORING.md
│   └─ README.md
│
├─ MLPerf/                       # MLPerf benchmarks
│   ├─ GPT-OSS-20B/             # GPT‑OSS‑20B benchmark
│   │   └─ primus/              # Primus framework
│   │       ├─ src/             # Source code
│   │       ├─ conf/            # Configuration files
│   │       ├─ patches/         # Framework patches
│   │       ├─ config_*.sh      # System‑specific configs
│   │       ├─ launch_apptainer.sh
│   │       ├─ run_and_time.sh
│   │       ├─ requirements.txt
│   │       ├─ RULES_AND_SCORING.md
│   │       └─ README.md
│   │
│   └─ DeepseekV3/              # DeepseekV3 671B benchmark
│       ├─ nemo/                # Reference implementation
│       ├─ RULES_AND_SCORING.md
│       └─ README.md
│
└─ README.md                     # this file
```

## Getting started
Navigate to the target repository folder, consult the `README.md`, `RULES_AND_SCORING.md` as well as the requirements document. Depending on the rules, tune the benchmark and system to the used system and follow the instructions obtain a result.