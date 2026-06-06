import argparse
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import tensorflow as tf
from jax.experimental import jax2tf
from flax import serialization

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.utils.model import build_model
from train.utils.config import TrainConfig
import viewers.core as core

def export_to_onnx(ckpt_path: Path, out_path: Path, max_input_bytes: int):
    print(f"Loading checkpoint {ckpt_path}...")
    restored = serialization.msgpack_restore(ckpt_path.read_bytes())
    if isinstance(restored, dict) and "params" in restored:
        params = restored["params"]
    else:
        params = restored

    hparams = core._load_checkpoint_hparams(ckpt_path)
    label_names = hparams.get("label_names")
    if label_names:
        core._apply_label_mapping(label_names)
    inferred = core._infer_checkpoint_architecture(ckpt_path)
    
    cfg = TrainConfig(
        arch=inferred.get("arch", "mamba"),
        model_dim=int(inferred.get("model_dim", 256)),
        mamba_layers=int(inferred.get("mamba_layers", 6)),
        mamba_d_state=int(inferred.get("mamba_d_state", 16)),
        mamba_expand=int(inferred.get("mamba_expand", 1)),
        mamba_dt_rank=int(inferred.get("mamba_dt_rank", 16)),
        mamba_conv=int(inferred.get("mamba_conv", 4)),
        mamba_bidirectional=bool(inferred.get("mamba_bidirectional", True)),
        dtype=jnp.float32, # Export to fp32 for maximum compatibility
    )
    
    import train.utils.config as config_mod
    num_classes = len(config_mod.LANG_ORDER)

    # Monkey patch jax.nn.softplus to avoid log1p (which tf2onnx doesn't support well)
    original_softplus = jax.nn.softplus
    def _onnx_safe_softplus(x):
        return jnp.log(1.0 + jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)
    jax.nn.softplus = _onnx_safe_softplus

    # No sequential scan patch, we rely on the native associative scan implementation
    # which is O(log N) tree depth and should trace perfectly with fixed sequence length!

    # Rebuild model with use_remat=False to avoid remat boundary issues during export
    model = build_model(cfg, num_classes)

        
    def predict_fn(tokens):
        # returns logits
        return model.apply({"params": params}, tokens, train=False, return_auxiliary=False)

    print("Tracing with jax2tf...")
    # Trace for dynamic batch size but fixed sequence length to optimize ONNX graph

    # We will export with FIXED sequence length because dynamic sequence length + associative_scan
    # Constants for tracing
    SEQ_LEN = 512
    # fails to trace, and sequential scan is too slow (18 seconds per inference).
    # Input shape: (1, 512)
    tf_predict = jax2tf.convert(
        predict_fn,
        polymorphic_shapes=None,
        enable_xla=False # Avoid XLA ops in the SavedModel, use standard TF ops
    )
    
    # Wrap in tf.Module
    class ExportModule(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, SEQ_LEN], dtype=tf.int32, name="tokens")])
        def __call__(self, tokens):
            return {"logits": tf_predict(tokens)}

    module = ExportModule()
    tmp_saved_model = out_path.parent / "tmp_savedmodel"
    print(f"Saving temporary TF SavedModel to {tmp_saved_model}...")
    tf.saved_model.save(module, str(tmp_saved_model))

    print("Converting to ONNX using tf2onnx...")
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", str(tmp_saved_model),
        "--output", str(out_path),
        "--opset", "17",
        "--large_model"
    ]
    subprocess.run(cmd, check=True)
    print(f"ONNX model saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-bytes", type=int, default=6144)
    args = parser.parse_args()
    
    export_to_onnx(Path(args.ckpt), Path(args.out), args.max_bytes)
