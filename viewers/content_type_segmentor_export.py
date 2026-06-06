#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from flax import serialization


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
STATIC_ROOT = REPO_ROOT / "viewers" / "content_type_segmentor_static"
DEFAULT_OUTPUT_DIR = STATIC_ROOT / "assets"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "sweeps" / "sfullfiles4.msgpack"
DEFAULT_DEMO_TEXT_PATH = DEFAULT_OUTPUT_DIR / "demo_input.txt"
DEFAULT_POSTPROCESS_MIN_RUN_CHARS = 3
DEFAULT_POSTPROCESS_BOUNDARY_SNAP_MAX_SHIFT = 2

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import viewers.core as core  # noqa: E402


def _load_runtime_module():
    runtime_path = STATIC_ROOT / "py" / "runtime.py"
    spec = importlib.util.spec_from_file_location("content_type_segmentor_runtime", runtime_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load runtime module from {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_checkpoint_params(ckpt_path: Path) -> Mapping[str, Any]:
    restored = serialization.msgpack_restore(ckpt_path.read_bytes())
    if isinstance(restored, Mapping) and "params" in restored:
        params = restored["params"]
    else:
        params = restored
    if not isinstance(params, Mapping):
        raise TypeError(f"Unsupported checkpoint payload type: {type(params)}")
    return dict(params)


def _normalize_block_key(params: Mapping[str, Any], layer_idx: int) -> str:
    candidates = (f"MambaBlock1D_{layer_idx}", f"CheckpointMambaBlock1D_{layer_idx}")
    for candidate in candidates:
        if candidate in params:
            return candidate
    raise KeyError(f"Unable to locate Mamba block {layer_idx} in checkpoint.")


def _resolve_label_payload(ckpt_path: Path) -> tuple[list[str], list[str], list[str]]:
    auto_hparams = core._load_checkpoint_hparams(ckpt_path)
    label_names = auto_hparams.get("label_names")
    if label_names:
        core._apply_label_mapping(label_names)
    canonical, display = core._resolve_langs_and_display(None)
    colors = core._resolve_colors(canonical, None)
    return list(canonical), list(display), list(colors)


def _export_weight_arrays(params: Mapping[str, Any], model_meta: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    arrays["embed/embedding"] = np.asarray(params["Embed_0"]["embedding"], dtype=np.float32)
    arrays["final/ln_scale"] = np.asarray(params["LayerNorm_0"]["scale"], dtype=np.float32)
    arrays["final/ln_bias"] = np.asarray(params["LayerNorm_0"]["bias"], dtype=np.float32)
    arrays["final/dense_kernel"] = np.asarray(params["Dense_0"]["kernel"], dtype=np.float32)
    arrays["final/dense_bias"] = np.asarray(params["Dense_0"]["bias"], dtype=np.float32)

    n_layers = int(model_meta["n_layers"])
    for idx in range(n_layers):
        block = params[_normalize_block_key(params, idx)]
        prefix = f"blocks/{idx}/"
        arrays[prefix + "ln_scale"] = np.asarray(block["LayerNorm_0"]["scale"], dtype=np.float32)
        arrays[prefix + "ln_bias"] = np.asarray(block["LayerNorm_0"]["bias"], dtype=np.float32)
        arrays[prefix + "in_proj_kernel"] = np.asarray(block["Dense_0"]["kernel"], dtype=np.float32)
        arrays[prefix + "in_proj_bias"] = np.asarray(block["Dense_0"]["bias"], dtype=np.float32)
        arrays[prefix + "conv_kernel"] = np.asarray(block["Conv_0"]["kernel"][:, 0, :], dtype=np.float32)
        arrays[prefix + "conv_bias"] = np.asarray(block["Conv_0"]["bias"], dtype=np.float32)
        arrays[prefix + "x_proj_kernel"] = np.asarray(block["Dense_1"]["kernel"], dtype=np.float32)
        arrays[prefix + "x_proj_bias"] = np.asarray(block["Dense_1"]["bias"], dtype=np.float32)
        arrays[prefix + "dt_proj_kernel"] = np.asarray(block["Dense_2"]["kernel"], dtype=np.float32)
        arrays[prefix + "dt_proj_bias"] = np.asarray(block["Dense_2"]["bias"], dtype=np.float32)
        arrays[prefix + "out_proj_kernel"] = np.asarray(block["Dense_3"]["kernel"], dtype=np.float32)
        arrays[prefix + "out_proj_bias"] = np.asarray(block["Dense_3"]["bias"], dtype=np.float32)
        arrays[prefix + "a_log"] = np.asarray(block["A_log"], dtype=np.float32)
        arrays[prefix + "d"] = np.asarray(block["D"], dtype=np.float32)
    return arrays


def _build_manifest(
    *,
    checkpoint: Path,
    label_order: list[str],
    display_labels: list[str],
    colors: list[str],
    model_meta: Mapping[str, Any],
    max_input_bytes: int,
    other_threshold: float,
) -> Dict[str, Any]:
    label_colors = {label: colors[idx] for idx, label in enumerate(label_order)}
    return {
        "app_name": "Content Type Segmentor",
        "model_id": checkpoint.stem,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "num_classes": len(label_order),
        "label_order": label_order,
        "display_labels": display_labels,
        "label_colors": label_colors,
        "window_bytes": int(core.DEFAULT_CHUNK_SIZE),
        "window_stride_bytes": int(core.DEFAULT_CHUNK_SIZE // 2),
        "other_threshold": float(other_threshold),
        "max_input_bytes": int(max_input_bytes),
        "postprocess_min_run_chars": int(DEFAULT_POSTPROCESS_MIN_RUN_CHARS),
        "postprocess_boundary_snap_max_shift": int(DEFAULT_POSTPROCESS_BOUNDARY_SNAP_MAX_SHIFT),
        "default_text_path": "assets/demo_input.txt",
        "weights_path": "assets/sfullfiles4_weights.npz",
        "pyodide_js_url": "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js",
        "pyodide_index_url": "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/",
        "model": {
            "arch": "mamba",
            "d_model": int(model_meta["d_model"]),
            "n_layers": int(model_meta["n_layers"]),
            "d_state": int(model_meta["d_state"]),
            "expand": int(model_meta["expand"]),
            "dt_rank": int(model_meta["dt_rank"]),
            "d_conv": int(model_meta["d_conv"]),
            "bidirectional": bool(model_meta["bidirectional"]),
            "dtype": str(model_meta["dtype"]),
        },
    }


def _expand_segments(text_len: int, segments: list[dict]) -> list[int]:
    labels = [0] * text_len
    for segment in segments:
        label_id = int(segment["label_id"])
        for idx in range(int(segment["start"]), int(segment["end"])):
            labels[idx] = label_id
    return labels


def _validate_assets(
    *,
    manifest_path: Path,
    weights_path: Path,
    checkpoint: Path,
    sample_text: str,
) -> None:
    runtime_mod = _load_runtime_module()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = runtime_mod.NumpyMambaSegmentor.from_files(manifest_path, weights_path)

    predictor = core.Predictor(
        ckpt_path=str(checkpoint),
        num_classes=len(manifest["label_order"]),
        model_dim=int(manifest["model"]["d_model"]),
        channels=tuple(core.DEFAULT_CHANNELS),
        arch="mamba",
        mamba_layers=int(manifest["model"]["n_layers"]),
        mamba_d_state=int(manifest["model"]["d_state"]),
        mamba_expand=int(manifest["model"]["expand"]),
        mamba_dt_rank=int(manifest["model"]["dt_rank"]),
        mamba_conv=int(manifest["model"]["d_conv"]),
        mamba_bidirectional=bool(manifest["model"]["bidirectional"]),
        dtype_str=str(manifest["model"]["dtype"]),
        chunk=int(manifest["window_bytes"]),
        other_threshold=float(manifest["other_threshold"]),
        device="cpu",
        inference_backend="legacy",
    )

    normalized = runtime_mod._normalize_input_text(sample_text)
    raw_bytes = np.frombuffer(normalized.encode("utf-8", "ignore"), dtype=np.uint8)
    sanitized = runtime_mod._sanitize_model_bytes(raw_bytes).astype(np.int32, copy=False)

    browser_probs, spans = model.segment_byte_probs(sanitized.astype(np.uint8, copy=False))
    native_logits = predictor._apply_legacy(core.jnp.array(sanitized[None, :], dtype=core.jnp.int32))
    native_probs = np.asarray(core.jax.nn.softmax(native_logits, axis=-1), dtype=np.float32)[0]

    if spans != [(0, int(sanitized.shape[0]))] or not np.allclose(
        browser_probs,
        native_probs,
        atol=1e-5,
        rtol=1e-4,
    ):
        diff = float(np.max(np.abs(browser_probs - native_probs))) if browser_probs.size and native_probs.size else 0.0
        raise AssertionError(
            "Browser runtime validation failed against the native full-sequence Mamba path: "
            f"spans={spans!r} expected=[(0, {int(sanitized.shape[0])})] "
            f"max_abs_diff={diff:.6g}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export browser assets for Content Type Segmentor.")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--demo-text", type=str, default=str(DEFAULT_DEMO_TEXT_PATH))
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=6144,
        help="Public demo input cap in sanitized bytes. Set to 0 to disable the guardrail.",
    )
    parser.add_argument("--other-threshold", type=float, default=0.3)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = Path(args.ckpt).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_text_path = Path(args.demo_text).resolve()
    if not demo_text_path.exists():
        raise FileNotFoundError(f"Demo input text not found: {demo_text_path}")

    inferred = core._infer_checkpoint_architecture(checkpoint)
    if str(inferred.get("arch", "")).lower().strip() != "mamba":
        raise RuntimeError(f"Expected a Mamba checkpoint, got: {inferred}")

    model_meta = {
        "d_model": int(inferred.get("model_dim", 256)),
        "n_layers": int(inferred.get("mamba_layers", 6)),
        "d_state": int(inferred.get("mamba_d_state", 16)),
        "expand": int(inferred.get("mamba_expand", 1)),
        "dt_rank": int(inferred.get("mamba_dt_rank", 16)),
        "d_conv": int(inferred.get("mamba_conv", 4)),
        "bidirectional": bool(inferred.get("mamba_bidirectional", True)),
        "dtype": str(inferred.get("dtype", "float32")),
    }

    label_order, display_labels, colors = _resolve_label_payload(checkpoint)
    params = _load_checkpoint_params(checkpoint)
    arrays = _export_weight_arrays(params, model_meta)
    weights_path = output_dir / "sfullfiles4_weights.npz"
    np.savez_compressed(weights_path, **arrays)

    manifest = _build_manifest(
        checkpoint=checkpoint,
        label_order=label_order,
        display_labels=display_labels,
        colors=colors,
        model_meta=model_meta,
        max_input_bytes=int(args.max_input_bytes),
        other_threshold=float(args.other_threshold),
    )
    manifest_path = output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if not args.skip_validate:
        sample_text = demo_text_path.read_text(encoding="utf-8")
        _validate_assets(
            manifest_path=manifest_path,
            weights_path=weights_path,
            checkpoint=checkpoint,
            sample_text=sample_text,
        )

    print(f"Manifest: {manifest_path}")
    print(f"Weights: {weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
