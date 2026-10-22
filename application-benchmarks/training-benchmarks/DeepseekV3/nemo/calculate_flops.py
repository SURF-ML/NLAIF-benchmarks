from megatron.training.training import num_floating_point_operations
from typing_extensions import TypedDict, Unpack
from megatron.bridge import AutoBridge
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.recipes.deepseek.deepseek_v3 import (
    DeepSeekV3CommonKwargs, set_deepseek_v3_pipeline_model_parallel_layout, apply_flex_dispatcher_backend
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
import torch
from typing import List, Optional, Union
from megatron.bridge.training.comm_overlap import CommOverlapConfig
import os
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing

from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
import math

def _deepseek_v3_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 16,
    pipeline_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 64,
    sequence_parallel: bool = True,
    use_megatron_fsdp: bool = False,
    check_for_nan_in_grad: bool = True,
    # Recompute configuration
    recompute_granularity: Optional[str] = "selective",
    recompute_modules: Optional[List[str]] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    # MTP support
    mtp_num_layers: Optional[int] = 1,
    mtp_loss_scaling_factor: Optional[float] = 0.1,
    # Training hyperparameters
    train_iters: int = 1_000_000,
    global_batch_size: int = 4096,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 2000,
    save_interval: int = 2000,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    moe_flex_dispatcher_backend: str = None,
    apply_rope_fusion: bool = False,
    layout: Optional[Union[str, List[List[str]]]] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for DeepSeek-V3 models using a given HuggingFace path.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )
    
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    model_cfg.expert_tensor_parallel_size = 1
    # MTP configuration (allow None to disable by setting to 0)
    model_cfg.mtp_num_layers = 0 if mtp_num_layers is None else mtp_num_layers
    model_cfg.mtp_loss_scaling_factor = mtp_loss_scaling_factor
    model_cfg.init_method_std = 0.006
    model_cfg.rotary_base = 10000.0
    model_cfg.rotary_scaling_factor = 40
    model_cfg.rotary_base = float(model_cfg.rotary_base)
    model_cfg.rotary_scaling_factor = int(model_cfg.rotary_scaling_factor)

    model_cfg.recompute_granularity = recompute_granularity
    model_cfg.recompute_modules = recompute_modules
    model_cfg.recompute_method = recompute_method
    model_cfg.recompute_num_layers = recompute_num_layers
    set_deepseek_v3_pipeline_model_parallel_layout(model_cfg, layout)

    # Pipeline split for asymmetric stages are specified with map_pp_vp_to_layout below
    model_cfg.account_for_embedding_in_pipeline_split = False
    model_cfg.account_for_loss_in_pipeline_split = False
    model_cfg.num_layers_in_first_pipeline_stage = None
    model_cfg.num_layers_in_last_pipeline_stage = None

    # Performance optimization knobs
    model_cfg.moe_permute_fusion = True
    # apply_flex_dispatcher_backend(model_cfg, moe_flex_dispatcher_backend)

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
    )
    opt_config.use_precision_aware_optimizer = True
    opt_config.main_params_dtype = torch.float32
    opt_config.main_grads_dtype = torch.bfloat16
    opt_config.exp_avg_dtype = torch.bfloat16
    opt_config.exp_avg_sq_dtype = torch.bfloat16

    if precision_config is None:
        precision_config = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=5,
            manual_gc_eval=5,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=check_for_nan_in_grad,
            grad_reduce_in_fp32=False,  # V3 recipe sets this to False
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            data_sharding=True,
            dataloader_type="single",
            num_workers=8,
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer" if use_null_tokenizer else "HuggingFaceTokenizer",
            tokenizer_model=hf_path if not use_null_tokenizer else None,
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE if use_null_tokenizer else None,
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=False,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )
    if apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # mla rope fusion is experimental

    # Ensure comm_overlap exists with old default tp_comm_overlap=False when not provided
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    return cfg

def deepseek_v3_pretrain_config(**user_kwargs: Unpack[DeepSeekV3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for DeepSeek-V3.

    See `_deepseek_v3_common` for the full list of parameters.
    """
    recommended_kwargs: DeepSeekV3CommonKwargs = {
        "hf_path": "deepseek-ai/DeepSeek-V3",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 16,
        "expert_model_parallel_size": 64,
        "pipeline_dtype": torch.bfloat16,
        # Old recipe-compatible defaults passed via wrapper
        "recompute_granularity": "selective",
        "precision_config": MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    }
    combined_kwargs: DeepSeekV3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _deepseek_v3_common(**combined_kwargs)


thing = deepseek_v3_pretrain_config()
# print(thing)
model_cfg = thing.model
model_cfg.recompute_modules = "mlp,moe_act".split(',')
model_cfg.cuda_graph_implementation = "none"
    # MoE params
model_cfg.moe_token_dispatcher_type = "alltoall"
model_cfg.moe_grouped_gemm = True
model_cfg.moe_permute_fusion = False
model_cfg.moe_router_fusion = False
model_cfg.moe_router_force_load_balancing = False
model_cfg.cuda_graph_implementation = "none"
model_cfg.group_query_attention = False
model_cfg.swiglu = True
model_cfg.padded_vocab_size = 128000
model_cfg.vocab_size = 128000
model_cfg.num_experts = model_cfg.num_moe_experts
gbs = 15360
flop_per_training_batch = num_floating_point_operations(model_cfg, gbs)
flop_per_eval_batch = (num_floating_point_operations(model_cfg, 1024)/ 3)
# One eval for every training iter
flop_per_batch = flop_per_training_batch + flop_per_eval_batch
print()
print(f"FLOP per training batch: {flop_per_training_batch / 1e15:.2f} PFLOP")
print(f"FLOP per eval batch: {flop_per_eval_batch / 1e15:.2f} PFLOP")
print(f"FLOP per batch: {flop_per_batch / 1e15:.2f} PFLOP")
print(f"Total FLOP per sample: {(flop_per_batch / gbs) / 1e12:.2f} TFLOP/sample")

n_samples = {15360: [721920, 721920, 721920, 737280, 691200,  737280],
16384: [770048, 786432, 770048, 753664, 753664, 770048],
18432: [847872, 866304, 866304, 829440, 866304, 866304]}

gbs = 15360
# Discard worst and best case
sorted_samples = sorted(n_samples[gbs])[1:-1]
#
avg_samples = sum(sorted_samples) / len(sorted_samples)
n_batches = math.ceil(avg_samples / gbs)
print(f"Average samples for full training run: {avg_samples:.2f}")
print(f"Average number of batches for full training run (GBS={gbs}): {n_batches:.2f}")
print(f"Total FLOP for {n_batches} batches: {flop_per_batch * n_batches / 1e18:.2f} EFLOP")

print(f"Assuming 1.5 hours training time, with 256 GPUs")
total_time = 1.5 * 3600
num_gpus = 256

print(f"Required PFLOP/s to achieve in {total_time/3600:.2f} hours: {flop_per_batch * n_batches / total_time / 1e15:.2f} PFLOP/s")


print(f"Required TFLOP/s per GPU ({num_gpus} GPUs): {flop_per_batch * n_batches / total_time / num_gpus / 1e12:.2f} TFLOP/s per GPU")


flop_per_device = {"GB300": 1290, "GB200": 1090, "B300": 814, "B200": 732, "H100": 0.18 * 989 * 2}
num_virtual_gpus=768
scaling = 0.8
for device, device_flop in flop_per_device.items():
    virtual_time = flop_per_batch * n_batches / (device_flop * 1e12 * num_virtual_gpus * scaling)
    print(f"Estimated optimized training time on {device} with {num_virtual_gpus} virtual GPUs: {virtual_time/3600:.2f} hours")