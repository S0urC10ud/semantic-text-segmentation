"""Export U-Net checkpoint weights to .bin + meta.json for browser deployment."""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from flax import serialization

def main():
    ckpt_path = Path("checkpoints/unet_al.msgpack")
    out_dir = Path("viewers/content_type_segmentor_static/assets")

    raw = serialization.msgpack_restore(ckpt_path.read_bytes())
    params = raw["params"] if "params" in raw else raw

    # Flatten the nested param dict
    def flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten(v, key + "/"))
            else:
                out[key] = np.asarray(v, dtype=np.float32)
        return out

    flat = flatten(params)

    # Print all weight shapes for inspection
    print("=== U-Net Weight Shapes ===")
    total_params = 0
    for name in sorted(flat.keys()):
        w = flat[name]
        print(f"  {name}: {w.shape}  ({w.size} params)")
        total_params += w.size
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # Write binary weights
    bin_path = out_dir / "unet_al_weights.bin"
    meta = {}
    offset = 0
    with open(bin_path, "wb") as f:
        for name in sorted(flat.keys()):
            w = flat[name].flatten()
            meta[name] = {
                "offset": offset,
                "length": len(w),
                "shape": list(flat[name].shape),
            }
            f.write(w.tobytes())
            offset += len(w)

    meta_path = out_dir / "unet_al_weights_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {bin_path} ({bin_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Wrote {meta_path}")

if __name__ == "__main__":
    main()
