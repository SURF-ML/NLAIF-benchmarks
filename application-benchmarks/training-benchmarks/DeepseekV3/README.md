# MLPerf DeepseekV3 671B MoE Pre-training benchmark
## Introduction

 Clone of https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo. Adaption to Snellius was not possible as the system is too small for a run of this scale. The reference implementation is therefore left as is, and free to adapt.


This pre-training benchmark targets whole accelerated partition, small input data, large Mixture-of-Experts model (671B) pre-training, using inter-node communiciation intensive parallelization strategies (expert parallel). 

This benchmark follows all the CLOSED division rulesets from the [MLPerf training rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc), as well as some additional rules of our own. In short, this means that the framework and exact implementation of the training script is flexible, as long as the result is mathematically equivalent to the reference implementation beyond the allowed tunable hyperparameters (such as batch size and learning rate). 


On top of these rules, we restrict the precision format to be 8-bit precision (e.g., FP8/MXFP8). The number of GPUs used for the run is up to the benchmarker, but (unlike the other benchmarks) the resulting score will **not** be weak-scaled to the entire system. Hence, the only reason for using fewer GPUs than available would be if it yields higher performance (perhaps due to a more convenient parallelization strategy on a subset of the GPUs). For more details on these rules, as well as the scoring mechanism we refer to [`RULES_AND_SCORING.md`](RULES_AND_SCORING.md).
