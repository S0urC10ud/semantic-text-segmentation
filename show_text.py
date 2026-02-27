import json

with open('active_learning/benchmark_data/curated_oracle_segments_v1.json') as f:
    data = json.load(f)

samples = data.get("samples", data.get("data", []))

for s in samples:
    if s['snippet_id'] == 'curated-0041':
        text = s['text']
        print(f"TEXT ({len(text)} chars):\n{repr(text)}\n")
        
        # We know truth boundaries from previous step:
        # [0, 20], [20, 60], [60, 68], [68, 112], [112, 120]
        # Predicted boundaries:
        # [0, 19], [19, 59], [59, 65], [65, 114], ...
        
        # Let's see the characters around 20, 60, 68, etc.
        print("--- TRUTH BOUNDARY 20: 'text' -> 'encoding_base32'")
        print(f"TRUTH left: {repr(text[15:20])}")
        print(f"TRUTH right:{repr(text[20:25])}")
        print(f"PRED  left (19): {repr(text[14:19])}")
        print(f"PRED  right(19): {repr(text[19:24])}")
        
        print("\n--- TRUTH BOUNDARY 60: 'encoding_base32' -> 'text'")
        print(f"TRUTH left: {repr(text[55:60])}")
        print(f"TRUTH right:{repr(text[60:65])}")
        print(f"PRED  left (59): {repr(text[54:59])}")
        print(f"PRED  right(59): {repr(text[59:64])}")
        
        print("\n--- TRUTH BOUNDARY 68: 'text' -> 'encoding_base58'")
        print(f"TRUTH: {repr(text[65:71])}")
        print(f"PRED : {repr(text[62:68])}") # pred was 65
