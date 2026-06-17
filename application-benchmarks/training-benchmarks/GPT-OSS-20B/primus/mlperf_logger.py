###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLPERF Logging support for Primus Megatron backend."""

import os
import time
from typing import Dict, Any, Optional


class MLPerfLogger:
    """Wrapper around mllog library with rank-aware logging."""
    
    def __init__(self):
        from mlperf_logging import mllog
        self.mllogger = mllog.get_mllogger()
        self._configured = False
        self._rank = None
    
    def configure(self, filepath: str, args) -> Dict[str, Any]:
        from mlperf_logging import mllog
        mllog.config(filename=filepath, default_stack_offset=3)
        self._configured = True
        self._rank = self._get_rank()
        return self.extract_mlperf_configs(args)
    
    def _get_rank(self) -> int:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _is_logging_rank(self) -> bool:
        """Return True on the single rank that should emit MLPerf log events.

        With PP=1 this is global rank 0.  With PP>1 the loss is only available
        on the *last* pipeline stage, so we must log from the rank that is
        simultaneously: TP rank 0, DP rank 0, and on the last PP stage.
        Falls back to global rank 0 when Megatron parallel state is not yet
        initialised (e.g. during __init__).
        """
        try:
            from megatron.core import parallel_state as mpu
            if not mpu.model_parallel_is_initialized():
                return self._rank == 0
            return (
                mpu.is_pipeline_last_stage()
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_data_parallel_rank() == 0
            )
        except Exception:
            return self._rank == 0

    def log_event_all_ranks(self, key: str, value: Any, metadata: Optional[Dict] = None):
        self.mllogger.event(key=key, value=value, metadata=metadata or {})
    
    def log_event(self, key: str, value: Any, metadata: Optional[Dict] = None, stack_offset: int = 2):
        if self._is_logging_rank():
            self.mllogger.event(key=key, value=value, metadata=metadata or {}, stack_offset=stack_offset)
    
    def log_start(self, key: str, metadata: Optional[Dict] = None, stack_offset: int = 2):
        if self._is_logging_rank():
            self.mllogger.start(key=key, metadata=metadata or {}, stack_offset=stack_offset)
    
    def log_end(self, key: str, metadata: Optional[Dict] = None, stack_offset: int = 2):
        if self._is_logging_rank():
            self.mllogger.end(key=key, metadata=metadata or {}, stack_offset=stack_offset)
    
    def extract_mlperf_configs(self, args) -> Dict[str, Any]:
        """Extract MLPERF config parameters from Megatron args."""
        from mlperf_logging.mllog import constants
        
        data_parallel_size = getattr(args, 'data_parallel_size', 1)
        if data_parallel_size == 0:
            data_parallel_size = 1
        micro_batches = args.global_batch_size // (args.micro_batch_size * data_parallel_size)
        
        eval_samples = args.eval_iters * args.global_batch_size if args.eval_iters > 0 else 1024
        
        optimizer_name = getattr(args, 'optimizer', 'adam')
        if optimizer_name.lower() == 'adam':
            optimizer_name = 'adamw'
        else:
            optimizer_name = optimizer_name.lower()
        
        configs = {
            constants.GLOBAL_BATCH_SIZE: args.global_batch_size,
            constants.GRADIENT_ACCUMULATION_STEPS: micro_batches,
            'max_sequence_length': args.seq_length,
            'eval_samples': eval_samples,
            constants.SEED: args.seed,
            'init_checkpoint_step': args.iteration,
            constants.OPT_NAME: optimizer_name,
            constants.OPT_BASE_LR: args.lr,
            constants.OPT_ADAMW_BETA_1: getattr(args, 'adam_beta1', 0.9),
            constants.OPT_ADAMW_BETA_2: getattr(args, 'adam_beta2', 0.999),
            constants.OPT_ADAMW_EPSILON: getattr(args, 'adam_eps', 1e-8),
            constants.OPT_ADAMW_WEIGHT_DECAY: args.weight_decay,
            'opt_gradient_clip_norm': getattr(args, 'clip_grad', 1.0),
            'opt_end_learning_rate': args.min_lr,
            'opt_learning_rate_warmup_steps': getattr(args, 'lr_warmup_iters', 0),
            'opt_learning_rate_decay_steps': args.lr_decay_iters if args.lr_decay_iters else args.train_iters,
            'opt_learning_rate_decay_schedule': self._get_lr_schedule_name(args.lr_decay_style),
            'max_steps': args.train_iters,
            constants.SUBMISSION_BENCHMARK: os.getenv('MLLOG_SUBMISSION_BENCHMARK', ''),
            constants.SUBMISSION_DIVISION: os.getenv('MLLOG_SUBMISSION_DIVISION', ''),
            constants.SUBMISSION_STATUS: os.getenv('MLLOG_SUBMISSION_STATUS', ''),
            constants.SUBMISSION_ORG: os.getenv('MLLOG_SUBMISSION_ORG', ''),
            constants.SUBMISSION_PLATFORM: os.getenv('MLLOG_SUBMISSION_PLATFORM', ''),
        }
        return configs
    
    def _get_lr_schedule_name(self, style: str) -> str:
        mapping = {
            'cosine': 'cosine with linear warmup',
            'linear': 'linear',
            'constant': 'constant',
        }
        return mapping.get(style, style)


class ThroughputTimer:
    """Timer for tracking training throughput (samples/sec)."""
    
    def __init__(self, global_batch_size: int):
        self.gbs = global_batch_size
        self.start_time = None
        self.last_log_time = None
        self.last_log_samples = 0
    
    def start(self):
        """Reset timer for new training block."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def get_throughput(self, current_samples: int) -> Optional[float]:
        """Calculate throughput (samples/sec) since last measurement.
        
        Args:
            current_samples: Total consumed samples (cumulative)
            
        Returns:
            Samples per second for the interval since last call, or None if no previous measurement
        """
        if self.last_log_time is None:
            # First call - just initialize tracking
            self.last_log_time = time.time()
            self.last_log_samples = current_samples
            return None
        
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        samples_delta = current_samples - self.last_log_samples
        
        # Update tracking for next call
        self.last_log_time = current_time
        self.last_log_samples = current_samples
        
        # Calculate throughput for this interval
        if elapsed > 0 and samples_delta >= 0:
            return samples_delta / elapsed
        return 0.0

