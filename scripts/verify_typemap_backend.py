"""Compare the pure-numpy backend against the JAX/Flax float32 reference."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "train"):
    sys.path.insert(0, str(p))

import jax, jax.numpy as jnp                      # noqa: E402
from flax import serialization                    # noqa: E402
import viewers.core as core                        # noqa: E402
from viewers.core import UNet1D, Mamba1D, _sanitize_model_bytes  # noqa: E402
from typemap._numpy_backend import unet_forward, mamba_forward, flatten_params, Weights  # noqa: E402

PAD = 256


def load_flat(name):
    tree = serialization.msgpack_restore((REPO / "checkpoints" / f"{name}.msgpack").read_bytes())
    if isinstance(tree, dict) and "params" in tree and "Embed_0" not in tree:
        tree = tree["params"]
    return tree


def sample_tokens(n=384):
    from datasets import load_from_disk
    out = []
    for task in ["realistic", "sequence_pair", "markdown_mix"]:
        ds = load_from_disk(str(REPO / "evaluation" / "test" / task))
        for i in range(3):
            b = _sanitize_model_bytes(np.frombuffer(ds[i]["content"].encode("utf-8", "ignore"), dtype=np.uint8))
            arr = np.full((n,), PAD, dtype=np.int32)
            L = min(n, b.shape[0])
            arr[:L] = b[:L].astype(np.int32)
            out.append(arr)
    return np.stack(out)  # (B, n)


def main():
    toks = sample_tokens(384)
    params_u = load_flat("unet_al")
    params_m = load_flat("mamba_al")
    flat_u = Weights(flatten_params(params_u))
    flat_m = Weights(flatten_params(params_m))

    # JAX reference (float32)
    um = UNet1D(num_classes=35, emb_dim=256, channels=(32, 64, 64, 128, 128, 128, 128, 256),
                dtype=jnp.float32, num_token_embeddings=130)
    mm = Mamba1D(num_classes=35, d_model=256, n_layers=6, d_state=16, expand=1, dt_rank=16,
                 d_conv=4, bidirectional=True, dtype=jnp.float32, num_token_embeddings=130)
    jx = jnp.asarray(toks)
    # load params through the same flexible loader the Predictor uses, so the
    # checkpoint's CheckpointMambaBlock1D_* keys map onto the model template.
    tmpl_u = um.init({"params": jax.random.PRNGKey(0)}, jx[:1], train=False)["params"]
    tmpl_m = mm.init({"params": jax.random.PRNGKey(0)}, jx[:1], train=False)["params"]
    pu = core._load_params_from_any(str(REPO / "checkpoints" / "unet_al.msgpack"), tmpl_u)
    pm = core._load_params_from_any(str(REPO / "checkpoints" / "mamba_al.msgpack"), tmpl_m)
    ju = np.asarray(jax.jit(lambda t: um.apply({"params": pu}, t, train=False))(jx))
    jm = np.asarray(jax.jit(lambda t: mm.apply({"params": pm}, t, train=False))(jx))

    for name, jref, fn, flat in [("U-Net", ju, unet_forward, flat_u), ("Mamba", jm, mamba_forward, flat_m)]:
        max_abs = 0.0
        arg_mismatch = 0
        total = 0
        for bi in range(toks.shape[0]):
            npl = fn(flat, toks[bi])
            d = float(np.max(np.abs(npl - jref[bi])))
            max_abs = max(max_abs, d)
            am = int(np.count_nonzero(np.argmax(npl, -1) != np.argmax(jref[bi], -1)))
            arg_mismatch += am
            total += npl.shape[0]
        print(f"{name}: max|Δlogit|={max_abs:.3e} | argmax mismatches={arg_mismatch}/{total}")


if __name__ == "__main__":
    main()
