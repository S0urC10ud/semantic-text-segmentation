from dataclasses import dataclass
from typing import Tuple, List
import jax.numpy as jnp

# Language and class mapping
LANG2ID = {"html": 0, "css": 1, "javascript": 2, "c":3, "cpp": 4, "csv":5, "java":6, "json":7, "python":8, "text":9}
ID2LANG = {v: k for k, v in LANG2ID.items()}
NUM_CLASSES = 10

# Special IDs for padding in labels and inputs
PAD_ID = NUM_CLASSES  # Label PAD is 3 (masked out of loss/metrics)
BYTE_VOCAB_SIZE = 256
PAD_BYTE_ID = 256
NUM_TOKEN_EMBEDDINGS = BYTE_VOCAB_SIZE + 1  # 257 for embeddings table

@dataclass
class DataConfig:
    """Configuration for data loading, augmentation, and batching."""
    # Data source
    data_root: str = "data"
    allow_hf_fallback: bool = True
    num_proc: int = 2
    seed: int = 42

    # Windowing and batching
    window_min_bytes: int = 512
    window_max_bytes: int = 512
    bucket_step: int = 128
    batch_size: int = 16
    mix_prob: float = 0.5
    min_seg_len: int = 64
    pure_prob: float = 0.25
    line_inject_prob: float = 0.5
    # remaining percentage is for mixed prob!

    # Prefetching
    prefetch_batches: int = 4
    num_workers: int = 2
    bucket_hold_steps: int = 10

    # Line injection augmentation
    line_inject_max_injections: int = 12
    line_inject_exp_rate: float = 0.06  # λ for exponential line count
    line_inject_max_lines: int = 100
    line_inject_min_single_len: int = 4
    allow_same_lang_injection: bool = True
    reindent_prob: float = 0.5
    start_with_newline_prob: float = 0.5
    strip_weights: Tuple[float, float, float, float] = (0.1, 0.2, 0.2, 0.5) # none, l, r, both
    host_skip_top_min: int = 5
    host_skip_top_max: int = 20
    donor_skip_top_min: int = 5
    donor_skip_top_max: int = 20

    def buckets(self) -> List[int]:
        """Generate window size buckets from min to max."""
        if self.window_min_bytes == self.window_max_bytes:
            return [self.window_min_bytes]
        return list(range(self.window_min_bytes, self.window_max_bytes + 1, self.bucket_step))

@dataclass
class TrainConfig:
    """Configuration for model training."""
    steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 100
    dtype: jnp.dtype = jnp.bfloat16
    model_dim: int = 128
    channels: Tuple[int, ...] = (128, 256, 384, 512)
    dropout_rate: float = 0.1
    rng_seed: int = 123
    log_every: int = 50
    eval_every: int = 250
    eval_batches: int = 250
    ckpt_path: str = "checkpoints/seg-unet1d.msgpack"
    sweep_id: str = ""
    no_jit: bool = False
    preview_only: bool = False
    preview_start: int = 0
    preview_count: int = 10
