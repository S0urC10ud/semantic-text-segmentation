import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Any, Optional
from train.utils.model import MambaBlock1D
from flax import linen as nn
import numpy as np
import time

class StatefulMambaEngine:
    def __init__(self, apply_fn, params: dict, num_layers: int, chunk_size: int):
        self.apply_fn = apply_fn
        self.params = params
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self._cache_text = ""
        self._cache_fwd_states = [] # list of dicts: one dict per layer per chunk boundary
        self._cache_bwd_states = []
        self._cache_f_outs = [] # layer -> list of chunk outputs

    # We will implement the layer-by-layer chunked eval!
