import argparse
import json
from pathlib import Path
import numpy as np

def convert_npz_to_bin(npz_path: Path, bin_path: Path, meta_path: Path):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    
    metadata = {}
    offset = 0
    
    with open(bin_path, "wb") as f:
        for key in data.files:
            arr = data[key].astype(np.float32)
            arr_bytes = arr.tobytes()
            length = len(arr_bytes) // 4
            
            metadata[key] = {
                "offset": offset,
                "length": length,
                "shape": list(arr.shape)
            }
            
            f.write(arr_bytes)
            offset += length
            
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved binary to {bin_path} ({offset * 4 / 1024 / 1024:.2f} MB)")
    print(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--bin", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    args = parser.parse_args()
    
    convert_npz_to_bin(Path(args.npz), Path(args.bin), Path(args.meta))
