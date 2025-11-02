from dataclasses import dataclass
from typing import Tuple, List, Dict
import jax.numpy as jnp

# Language and class mapping
LANG2ID = {
    # Real languages (must stay in sync with downloader outputs – see downloader/0_main.py)
    "php": 0,
    "csharp": 1,
    "javascript_typescript": 2,
    "go": 3,
    "sql": 4,
    "rust": 5,
    "yaml": 6,
    "ruby": 7,
    "python": 8,
    "java": 9,
    "c_family": 10,
    "json": 11,
    "css": 12,
    "html": 13,
    "text": 14,
    "csv": 15,
    "shell": 16,
    "powershell": 17,
    "visual_basic": 18,
    "dockerfile": 19,
    # Derived encodings
    "encoding_hex": 100,
    "encoding_base64": 101,
    "encoding_base32": 102,
    "encoding_base58": 103,
    "encoding_base85": 104,
}

# These will be updated dynamically based on available data
ID2LANG: Dict[int, str] = {}
NUM_CLASSES = 0

def update_lang_mappings():
    """Update ID2LANG and NUM_CLASSES based on current LANG2ID state"""
    global NUM_CLASSES, PAD_ID
    ID2LANG.clear()
    ID2LANG.update({v: k for k, v in LANG2ID.items()})
    NUM_CLASSES = len(LANG2ID)
    PAD_ID = NUM_CLASSES

# Initialize mappings
update_lang_mappings()

# Special IDs for padding in labels and inputs
PAD_ID = NUM_CLASSES  # Label PAD is masked out of loss/metrics
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
    window_min_bytes: int = 1536
    window_max_bytes: int = 1536
    bucket_step: int = 128
    batch_size: int = 16
    min_seg_len: int = 64
    pure_prob: float = 0.4
    mix_prob: float = 0.25
    line_inject_prob: float = 0.25
    markdown_prob: float = 0.1
    max_mixed_languages: int = 3
    markdown_inline_code_prob: float = 0.25
    # remaining probability mass is used for mixed windows

    # Prefetching
    prefetch_batches: int = 4
    num_workers: int = 2
    bucket_hold_steps: int = 10

    # Line injection augmentation
    line_inject_max_injections: int = 4
    line_inject_exp_rate: float = 1  # λ for exponential line count
    line_inject_max_lines: int = 100
    line_inject_min_single_len: int = 6
    line_inject_min_letters: int = 6
    line_inject_strip_prob: float = 0.5
    allow_same_lang_injection: bool = True
    reindent_prob: float = 0.5
    start_with_newline_prob: float = 0.5
    strip_weights: Tuple[float, float, float, float] = (0.1, 0.2, 0.2, 0.5) # none, l, r, both
    inject_extra_newlines_max: int = 4
    host_skip_top_min: int = 5
    host_skip_top_max: int = 20
    donor_skip_top_min: int = 5
    donor_skip_top_max: int = 20
    both_prob: float = 0 # probability to overlay/inject substrings after the window ways built

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
    accum_steps: int = 1
    dtype: jnp.dtype = jnp.bfloat16
    model_dim: int = 128
    channels: Tuple[int, ...] = (128, 256, 384, 512)
    dropout_rate: float = 0.1
    rng_seed: int = 123
    log_every: int = 50
    eval_every: int = 250
    eval_batches: int = 50
    ckpt_path: str = "checkpoints/seg-unet1d.msgpack"
    sweep_id: str = ""
    no_jit: bool = False
    preview_only: bool = False
    preview_start: int = 0
    preview_count: int = 10
