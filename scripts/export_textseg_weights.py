#!/usr/bin/env python3
"""Export slimmed checkpoints + label metadata into the textseg package data dir.

Produces, under packages/textseg/textseg/data/:
  unet_al.npz, mamba_al.npz   - flat float32 weights (keys: "Module__sub__param")
  manifest.json               - label set, class indices, post-processing defaults
"""
import json
import sys
from pathlib import Path

import numpy as np
from flax import serialization

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
import utils.config as cfg  # noqa: E402

DATA = REPO / "packages" / "textseg" / "textseg" / "data"
DATA.mkdir(parents=True, exist_ok=True)


def flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        else:
            out[key] = np.asarray(v, dtype=np.float32)
    return out


def export(name):
    tree = serialization.msgpack_restore((REPO / "checkpoints" / f"{name}.msgpack").read_bytes())
    if isinstance(tree, dict) and "params" in tree and "Embed_0" not in tree:
        tree = tree["params"]
    flat = flatten(tree)
    safe = {k.replace("/", "__"): v for k, v in flat.items()}
    out = DATA / f"{name}.npz"
    np.savez_compressed(out, **safe)
    rows = flat["Embed_0/embedding"].shape[0]
    total = sum(int(v.size) for v in flat.values())
    print(f"  {name}: {len(flat)} arrays, {total:,} params, embed_rows={rows} -> {out} ({out.stat().st_size/1e6:.2f} MB)")


def main():
    for n in ("unet_al", "mamba_al"):
        export(n)
    labels = list(cfg.LANG_ORDER)
    manifest = {
        "labels": labels,
        "num_classes": len(labels),
        "other_label": "other",
        "other_index": len(labels),
        "unet": {"file": "unet_al.npz", "channels": [32, 64, 64, 128, 128, 128, 128, 256], "window_bytes": 1536, "window_stride": 768},
        "mamba": {"file": "mamba_al.npz", "n_layers": 6, "d_state": 16, "dt_rank": 16, "d_conv": 4},
        "defaults": {"other_threshold": 0.30, "min_run_chars": 3, "boundary_snap_max_shift": 2},
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  manifest: {len(labels)} labels -> {DATA/'manifest.json'}")


if __name__ == "__main__":
    main()
