from dataclasses import dataclass
from typing import Tuple, List, Dict
import jax.numpy as jnp

# Fixed window size (bytes / tokens) used throughout the project.
MODEL_WINDOW_BYTES = 1536

# Canonical language/class ordering used across training, preprocessing, and monitor eval.
# Keep this list in the desired ID order; indices are assigned sequentially.
LANG_ORDER = [
    # Base / code-like languages
    "php",
    "csharp",
    "javascript_typescript",
    "go",
    "sql",
    "rust",
    "yaml",
    "ruby",
    "python",
    "java",
    "c_family",
    "json",
    "css",
    "html",
    "text",
    "csv",
    "shell",
    "powershell",
    "visual_basic",
    "dockerfile",
    "dart",
    "gettext_catalog",
    "kotlin",
    "markdown",
    "restructuredtext",
    "scala",
    "swift",
    "tex",
    "xml",
    "svg",
    # Derived encodings
    "encoding_hex",
    "encoding_base64",
    "encoding_base32",
    "encoding_base58",
    "encoding_base85",
]

# Language and class mapping (populated from LANG_ORDER)
LANG2ID = {lang: idx for idx, lang in enumerate(LANG_ORDER)}

# These will be updated dynamically based on available data
ID2LANG: Dict[int, str] = {}
NUM_CLASSES = 0
# Optional labels that may not have full train/val coverage but should be kept
OPTIONAL_LABELS = ("other",)

# Index reserved for a derived "other" bucket in metrics/confusion, not a model logit.
OTHER_CLASS_INDEX: int | None = None

def update_lang_mappings():
    """Update ID2LANG and NUM_CLASSES based on current LANG2ID state."""
    global NUM_CLASSES, PAD_ID, OTHER_CLASS_INDEX
    ID2LANG.clear()
    ID2LANG.update({v: k for k, v in LANG2ID.items()})
    NUM_CLASSES = len(LANG2ID)
    # Reserve the next id after the trained classes for a derived "other" class
    # used only in evaluation/monitor metrics. The model never has an explicit
    # logit for this bucket.
    OTHER_CLASS_INDEX = NUM_CLASSES
    # Padding id is kept distinct from both real classes and the derived "other".
    PAD_ID = NUM_CLASSES + 1

# Initialize mappings
update_lang_mappings()

# Special IDs for padding in labels and inputs
# Label PAD is masked out of loss/metrics
PAD_ID = NUM_CLASSES + 1
BYTE_VOCAB_SIZE = 256
PAD_BYTE_ID = 256
NUM_TOKEN_EMBEDDINGS = BYTE_VOCAB_SIZE + 1  # 257 for embeddings table

# Byte/token IDs that should be ignored for supervision + metrics (space, tab, newline, carriage return)
IGNORED_TRAINING_TOKEN_IDS: Tuple[int, ...] = tuple(
    sorted({ord(" "), ord("\t"), ord("\n"), ord("\r")})
)

@dataclass
class DataConfig:
    """Configuration for data loading, augmentation, and batching."""
    # Data source
    data_root: str = "data"
    allow_hf_fallback: bool = False
    num_proc: int = 2
    seed: int = 42

    # Windowing and batching
    window_min_bytes: int = MODEL_WINDOW_BYTES
    window_max_bytes: int = MODEL_WINDOW_BYTES
    bucket_step: int = 128
    batch_size: int = 16
    min_seg_len: int = 64
    pure_prob: float = 0.65
    mix_prob: float = 0.15
    line_inject_prob: float = 0.15
    markdown_prob: float = 0.1
    max_mixed_languages: int = 3
    markdown_inline_code_prob: float = 0.20
    language_pair_mode_prob: float = 0.0
    # remaining probability mass is used for mixed windows

    # Prefetching
    prefetch_batches: int = 4
    num_workers: int = 2
    bucket_hold_steps: int = 10

    # Line injection augmentation
    line_inject_max_injections: int = 4
    line_inject_exp_rate: float = 1  # λ for exponential line count
    line_inject_max_lines: int = 6
    line_inject_min_single_len: int = 7
    line_inject_min_letters: int = 7
    line_inject_strip_prob: float = 0.5
    allow_same_lang_injection: bool = True
    reindent_prob: float = 0.5
    start_with_newline_prob: float = 0.5
    strip_weights: Tuple[float, float, float, float] = (0.1, 0.2, 0.2, 0.5) # none, l, r, both
    inject_extra_newlines_max: int = 4
    host_skip_top_min: int = 0
    host_skip_top_max: int = 0
    donor_skip_top_min: int = 5
    donor_skip_top_max: int = 20
    both_prob: float = 0 # probability to overlay/inject substrings after the window ways built

    # Encoded/hex content augmentation
    hex_spacing_aug_prob: float = 0.1

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
    monitor_eval_every: int = 0
    monitor_eval_limit: int = 4096
    monitor_eval_root: str = "../downloader/monitor_preprocessed"
    monitor_other_threshold: float = 0.0
    MODEL_WINDOW_BYTES = 1536
    # Fine-tuning options (used by train/main.py)
    fine_tune: bool = False
    fine_tune_run_id: str = ""
    fine_tune_train_root: str = ""
    fine_tune_val_root: str = ""
    fine_tune_step: int | None = None
