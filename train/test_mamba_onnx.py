import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

def test_onnx(model_path: Path, max_bytes: int = 6144):
    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    # Get input names and shapes
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    
    print(f"Model Inputs: {input_names}")
    print(f"Model Outputs: {output_names}")
    
    # Create dummy input of size 1536 (typical window)
    seq_len = 1536
    dummy_input = np.random.randint(0, 256, size=(1, seq_len), dtype=np.int32)
    
    print(f"Running inference with sequence length {seq_len}...")
    
    # Warmup
    for _ in range(3):
        session.run(output_names, {input_names[0]: dummy_input})
        
    # Measure
    start_time = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        outputs = session.run(output_names, {input_names[0]: dummy_input})
    end_time = time.perf_counter()
    
    avg_ms = (end_time - start_time) * 1000 / iterations
    print(f"Average Inference Time (seq_len={seq_len}): {avg_ms:.2f} ms")
    
    logits = outputs[0]
    print(f"Output shape: {logits.shape}")
    print("Export and inference successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    test_onnx(Path(args.model))
