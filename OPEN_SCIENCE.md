# Open Science Artifact Map

This document maps the paper's experiments to the artifacts in this anonymous
repository. All paths below are relative to the repository root. The repository
is released under the Apache-2.0 license; installation and basic inference
instructions are in [`README.md`](README.md).

## Artifact inventory

| Paper component | Relevant artifacts | What is available |
|---|---|---|
| Model architectures and training | [`train/main.py`](train/main.py), [`train/utils/model.py`](train/utils/model.py), [`train/utils/config.py`](train/utils/config.py), [`train/utils/window_generator.py`](train/utils/window_generator.py) | Training driver, U-Net and Mamba definitions, label configuration, and window generation. |
| Released models and inference | [`checkpoints/unet_al.msgpack`](checkpoints/unet_al.msgpack), [`checkpoints/mamba_al.msgpack`](checkpoints/mamba_al.msgpack), [`checkpoints/unet_sec.msgpack`](checkpoints/unet_sec.msgpack), [`checkpoints/mamba_sec.msgpack`](checkpoints/mamba_sec.msgpack), [`inference/`](inference), [`viewers/interactive_viewer.py`](viewers/interactive_viewer.py) | The two general-domain checkpoints, the U-Net-Sec and Mamba-Sec checkpoints used for the security evaluation, inference backends, and an interactive inspection tool. |
| Data construction and dense annotation | [`downloader/main.py`](downloader/main.py), [`downloader/dense_prompt.template`](downloader/dense_prompt.template), [`downloader/active_learning_prompt.template`](downloader/active_learning_prompt.template), [`active_learning/`](active_learning) | Dataset-building, reconstruction checks for dense annotations, and the active-learning pipeline. The dense prompt is used for files already expected to contain mixed content. |
| Base task construction and scoring | [`evaluation/test/manifest.json`](evaluation/test/manifest.json), [`evaluation/test/`](evaluation/test), [`evaluation/evaluation.py`](evaluation/evaluation.py), [`evaluation/obtain_eval_dataset.py`](evaluation/obtain_eval_dataset.py) | A bundled LLM-labelled task suite, its manifest, the common metric implementation, and the benchmark builder. |
| MalwareBazaar acquisition | [`downloader/malwarebazaar_carriers.py`](downloader/malwarebazaar_carriers.py) | Static API retrieval, text-like eligibility checks, carrier-family normalization, and deterministic 150-to-15 stratified sampling across the 11 retained carrier families. |
| Magika baselines | [`magika_label_map.py`](magika_label_map.py), [`magika_windowed.py`](magika_windowed.py), [`downloader/999_evaluate_magika.py`](downloader/999_evaluate_magika.py) | Label mapping and whole-file/sliding-window baseline implementations. |
| Gemini baselines (base and MalwareBazaar) | [`evaluation/llm_benchmark/dense_prompt_v2.template`](evaluation/llm_benchmark/dense_prompt_v2.template), [`evaluation/llm_benchmark/dense_prompt_security.template`](evaluation/llm_benchmark/dense_prompt_security.template), [`evaluation/llm_benchmark/label_test_set.py`](evaluation/llm_benchmark/label_test_set.py), [`evaluation/score_llm_predictions.py`](evaluation/score_llm_predictions.py) | Exact base and MalwareBazaar prompt templates, API runner, and scoring through the same metric pipeline as the local models. The base-evaluation prompt uses the same dense labelling rules but supports unfiltered single-type files through a token-saving single-segment shortcut. |
| Throughput | [`evaluation/benchmark_parallel_inference.py`](evaluation/benchmark_parallel_inference.py) | Batched character/byte throughput benchmark used for the local models. |
| MalwareBazaar segmentation and routing | [`evaluation/malwarebazaar_finetune_eval.py`](evaluation/malwarebazaar_finetune_eval.py), [`evaluation/malwarebazaar_full_eval.py`](evaluation/malwarebazaar_full_eval.py), [`evaluation/malwarebazaar_routing_eval.py`](evaluation/malwarebazaar_routing_eval.py) | Security-label mapping, crop-level and full-file segmentation evaluation, and the rule-based expansion, route merging, validation, and random-location control used for Table 6. MalwareBazaar sample contents and human annotations are not included. |
| Automated checks | [`tests/`](tests), [`packages/typemap/tests/`](packages/typemap/tests) | Tests for training losses, inference, evaluation, active learning, and deployment backends. |
| Software environment | [`pyproject.toml`](pyproject.toml) | Python version and research-code dependencies. Run `uv sync` from the repository root to create the environment. |

## Security-checkpoint integrity

| Paper model | Released checkpoint | SHA-256 |
|---|---|---|
| U-Net-Sec | [`checkpoints/unet_sec.msgpack`](checkpoints/unet_sec.msgpack) | `f6c26da54ea78ff0ae50a3e27fde608741ffd97478803d958c3906df388e17c7` |
| Mamba-Sec | [`checkpoints/mamba_sec.msgpack`](checkpoints/mamba_sec.msgpack) | `fa95e23a934679a17131bc72637fed2ed70731be5dd3d10d14f2181ddd5716d5` |
