#!/usr/bin/env python3
"""
Shared model-loading, inference, and label/color utilities for the viewer apps.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import bisect
import dataclasses
import importlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
try:
    import orbax.checkpoint as ocp
except Exception:  # pragma: no cover — orbax version mismatch on some envs
    ocp = None
from flax import linen as nn
from inference.backend import (
    FastInferenceEngine,
    FastInferenceFailure,
    available_backends,
    build_window_spans as backend_build_window_spans,
    format_auto_fallback_message,
    resolve_backend,
)
from inference.mamba_cuda import has_cuda_mamba_kernel, selective_scan_inference

REPO_ROOT = Path(__file__).resolve().parents[1]
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[@-~]")


def _load_train_module(module_name: str):
    """
    Load a training module used by the viewers.

    Prefers the new `utils.<module_name>` layout (under the repo root),
    but falls back to the legacy `train/<module_name>.py` location if needed.
    """
    # Ensure REPO_ROOT and train/ are importable when running from viewers/
    repo_str = str(REPO_ROOT)
    if repo_str not in os.sys.path:
        os.sys.path.insert(0, repo_str)
    train_root = REPO_ROOT / "train"
    train_str = str(train_root)
    if train_root.exists() and train_str not in os.sys.path:
        os.sys.path.insert(0, train_str)

    # New layout: utils.<module_name> package at repo root
    try:
        return importlib.import_module(f"utils.{module_name}")
    except Exception:
        pass

    # Legacy layout: a plain script at train/<module_name>.py
    module_path = train_root / f"{module_name}.py"
    if not module_path.exists():
        raise ImportError(
            f"Expected to find utils.{module_name} or train/{module_name}.py next to segment_viewer, "
            f"but {module_path} does not exist."
        )
    spec = importlib.util.spec_from_file_location(
        f"segment_viewer.train_{module_name}", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for train/{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    TRAIN_CONFIG = _load_train_module("config")
except ImportError:
    TRAIN_CONFIG = None

DEFAULT_CHANNELS: Tuple[int, ...] = (96, 128, 192, 256)
MODEL_WINDOW_BYTES: int = int(getattr(TRAIN_CONFIG, "MODEL_WINDOW_BYTES", 1536))
DEFAULT_CHUNK_SIZE: int = MODEL_WINDOW_BYTES


def _apply_label_mapping(label_names: Sequence[str]) -> None:
    if not label_names or TRAIN_CONFIG is None:
        return
    LANG2ID = getattr(TRAIN_CONFIG, "LANG2ID", None)
    update_fn = getattr(TRAIN_CONFIG, "update_lang_mappings", None)
    if not isinstance(LANG2ID, dict):
        return
    LANG2ID.clear()
    for idx, name in enumerate(label_names):
        LANG2ID[name] = idx
    if callable(update_fn):
        update_fn()


def _configured_languages() -> List[str]:
    if TRAIN_CONFIG is None:
        raise RuntimeError(
            "Training config could not be loaded; unable to resolve label ordering automatically."
        )
    mapping = getattr(TRAIN_CONFIG, "LANG2ID", None)
    if not isinstance(mapping, dict):
        raise RuntimeError("Training config does not define LANG2ID mapping")
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def _resolve_langs_and_display(
    lang_arg: Optional[str],
) -> Tuple[List[str], List[str]]:
    configured_langs = _configured_languages()
    canonical_map = {lang.lower(): lang for lang in configured_langs}

    if not lang_arg:
        return configured_langs, configured_langs[:]

    requested: List[str] = []
    display_map: Dict[str, str] = {}
    for raw in lang_arg.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
        else:
            key = item
            value = ""
        key_lower = key.lower()
        if key_lower not in canonical_map:
            raise ValueError(
                f"Unknown language '{key}'. Available: {sorted(canonical_map.values())}"
            )
        canonical = canonical_map[key_lower]
        requested.append(canonical)
        if value:
            display_map[canonical] = value

    deduped = list(dict.fromkeys(requested))
    if not deduped:
        raise ValueError("No languages resolved from --lang.")

    selection_set = set(deduped)
    ordered = [lang for lang in configured_langs if lang in selection_set]
    if not ordered:
        raise ValueError("No overlap between --lang selection and training labels.")

    if ordered != deduped:
        print(
            f"Subset derived from --lang reordered to match training: {ordered}",
            flush=True,
        )

    display_names = [display_map.get(name, name) for name in ordered]
    return ordered, display_names


def _extract_run_id_from_checkpoint(path: Path) -> Optional[str]:
    name = path.name.lower()
    match = re.search(r"([a-z0-9]{8})", name)
    if not match:
        return None
    return match.group(1)


def _sanitize_label_list(raw: Sequence[Any]) -> List[str]:
    cleaned: List[str] = []
    if not raw:
        return cleaned
    allowed = None
    if TRAIN_CONFIG is not None:
        mapping = getattr(TRAIN_CONFIG, "LANG2ID", None)
        if isinstance(mapping, dict):
            allowed = set(mapping.keys())
    seen = set()
    dropped: List[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        candidate = ANSI_ESCAPE_RE.sub("", entry).strip()
        if not candidate:
            continue
        if allowed is not None and candidate not in allowed:
            dropped.append(candidate)
            continue
        if candidate in seen:
            continue
        cleaned.append(candidate)
        seen.add(candidate)
    if dropped:
        sample = ", ".join(sorted(set(dropped))[:5])
        print(
            f"⚠️  Ignoring {len(dropped)} unknown labels from checkpoint metadata: {sample}",
            flush=True,
        )
    return cleaned

def _parse_label_names_from_output(log_text: str) -> List[str]:
    labels: List[str] = []
    seen = set()
    collecting = False
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[Validation] Per-class metrics"):
            collecting = False
            continue
        if line.startswith("label ") and "support" in line:
            collecting = True
            continue
        if not collecting:
            continue
        if line.startswith("-"):
            continue
        if not line or line.startswith("Step "):
            collecting = False
            continue
        if line.startswith("ALL"):
            continue
        if line.startswith("WARNING") or line.startswith("INFO"):
            collecting = False
            continue
        parts = line.split()
        if not parts:
            continue
        label = parts[0]
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _load_checkpoint_hparams(ckpt_path: Path) -> Dict[str, Any]:
    run_id = _extract_run_id_from_checkpoint(ckpt_path)
    if not run_id:
        return {}
    wandb_root = REPO_ROOT / "train" / "wandb"
    if not wandb_root.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    pattern = f"run-*-{run_id}"
    for run_dir in wandb_root.glob(pattern):
        config_path = run_dir / "files" / "config.yaml"
        config_data: Optional[Dict[str, Any]]
        if config_path.exists():
            try:
                config_data = yaml.safe_load(config_path.read_text())
            except Exception:
                config_data = None
        else:
            config_data = None
        if not isinstance(config_data, dict):
            config_data = {}
        result: Dict[str, Any] = {}
        channels_val = config_data.get("channels", {}).get("value")
        if isinstance(channels_val, (list, tuple)):
            try:
                result["channels"] = [int(x) for x in channels_val]
            except (TypeError, ValueError):
                pass
        arch_val = config_data.get("arch", {}).get("value")
        if isinstance(arch_val, str) and arch_val.strip():
            result["arch"] = arch_val.strip()
        model_dim_val = config_data.get("model_dim", {}).get("value")
        if isinstance(model_dim_val, (int, float)):
            result["model_dim"] = int(model_dim_val)
        dtype_val = config_data.get("dtype", {}).get("value")
        if isinstance(dtype_val, str):
            result["dtype"] = dtype_val.rsplit(".", 1)[-1]
        # Mamba hyperparameters (present only for arch == mamba)
        for key in (
            "mamba_layers",
            "mamba_d_state",
            "mamba_expand",
            "mamba_dt_rank",
            "mamba_conv",
            "mamba_bidirectional",
        ):
            val = config_data.get(key, {}).get("value")
            if isinstance(val, bool):
                result[key] = bool(val)
            elif isinstance(val, (int, float)):
                result[key] = int(val)

        summary_path = run_dir / "files" / "wandb-summary.json"
        if summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text())
            except Exception:
                summary_data = None
            if isinstance(summary_data, dict):
                label_names: Dict[int, str] = {}
                prefix = "val/per_class/"
                for key in summary_data.keys():
                    if not key.startswith(prefix):
                        continue
                    remainder = key[len(prefix):]
                    head = remainder.split("/", 1)[0]
                    if "_" not in head:
                        continue
                    idx_str, label = head.split("_", 1)
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    label_names[idx] = label
                if label_names:
                    ordered = [label_names[i] for i in sorted(label_names)]
                    ordered = _sanitize_label_list(ordered)
                    if not ordered:
                        ordered = _configured_languages()
                    result["label_names"] = ordered
                    result["_label_source"] = "wandb"
        # Fallback: parse label order from output.log
        output_log = run_dir / "files" / "output.log"
        if output_log.exists():
            parsed_labels = _parse_label_names_from_output(output_log.read_text())
            if parsed_labels:
                ordered = _sanitize_label_list(parsed_labels)
                if ordered:
                    result.setdefault("label_names", ordered)
                result.setdefault("_label_source", "output_log")
        if result:
            return result
    return {}

def _infer_checkpoint_architecture(ckpt_path: Path) -> Dict[str, Any]:
    """Best-effort inference of model hyperparameters from a Flax msgpack checkpoint."""
    result: Dict[str, Any] = {}
    p = ckpt_path.resolve()
    if not p.is_file():
        return result
    try:
        params = serialization.msgpack_restore(p.read_bytes())
    except Exception:
        return result
    if isinstance(params, Mapping) and "params" in params:
        params = params["params"]
    params = _normalize_checkpoint_param_tree(params)

    # Architecture (unet1d vs mamba): detect from top-level param keys.
    try:
        top_keys = set(params.keys())
    except Exception:
        top_keys = set()
    is_mamba = any(str(k).startswith("MambaBlock1D_") for k in top_keys)
    if is_mamba:
        result["arch"] = "mamba"
    else:
        result["arch"] = "unet1d"

    # Embedding dimension -> model_dim
    try:
        embedding = params["Embed_0"]["embedding"]
        result["model_dim"] = int(embedding.shape[1])
        result["dtype"] = getattr(embedding.dtype, "name", str(embedding.dtype))
    except Exception:
        pass

    if is_mamba:
        # Infer Mamba hyperparameters from the first block.
        block0 = params.get("MambaBlock1D_0") if isinstance(params, dict) else None
        if isinstance(block0, dict):
            try:
                A_log = block0.get("A_log")
                if A_log is not None:
                    result["mamba_d_state"] = int(A_log.shape[-1])
            except Exception:
                pass
            try:
                in_proj = block0.get("Dense_0", {})
                kernel = in_proj.get("kernel")
                if kernel is not None and result.get("model_dim"):
                    d_model = int(result["model_dim"])
                    d_inner = int(kernel.shape[-1]) // 2
                    if d_model > 0:
                        result["mamba_expand"] = int(max(1, d_inner // d_model))
            except Exception:
                pass
            try:
                x_dbl = block0.get("Dense_1", {})
                kernel = x_dbl.get("kernel")
                if kernel is not None and result.get("mamba_d_state"):
                    d_state = int(result["mamba_d_state"])
                    result["mamba_dt_rank"] = int(kernel.shape[-1]) - 2 * d_state
            except Exception:
                pass
            try:
                conv = block0.get("Conv_0", {})
                kernel = conv.get("kernel")
                if kernel is not None:
                    result["mamba_conv"] = int(kernel.shape[0])
            except Exception:
                pass
        # Count blocks -> mamba_layers
        try:
            indices = []
            for k in top_keys:
                m = re.match(r"^MambaBlock1D_(\d+)$", str(k))
                if m:
                    indices.append(int(m.group(1)))
            if indices:
                result["mamba_layers"] = int(max(indices) + 1)
        except Exception:
            pass
        # Final projection -> num_classes
        try:
            result["num_classes"] = int(params["Dense_0"]["kernel"].shape[-1])
        except Exception:
            pass
        return result

    # Down path channels: walk ConvBlock1D_{0,2,4,...} until we hit the up path
    channels: List[int] = []
    current_in = result.get("model_dim")
    block_idx = 0
    while True:
        name = f"ConvBlock1D_{block_idx}"
        block = params.get(name)
        if block is None:
            break
        conv = block.get("Conv_0")
        if conv is None:
            break
        kernel = conv.get("kernel")
        if kernel is None:
            break
        in_ch = int(kernel.shape[-2])
        out_ch = int(kernel.shape[-1])
        if block_idx == 0 and current_in is None:
            current_in = in_ch
            result["model_dim"] = in_ch
        if channels and current_in is not None and in_ch != current_in:
            break
        channels.append(out_ch)
        current_in = out_ch
        block_idx += 2  # skip the paired block belonging to the same stage
    if channels:
        result["channels"] = channels

    # Output layer -> num_classes
    try:
        result["num_classes"] = int(params["Conv_0"]["kernel"].shape[-1])
    except Exception:
        pass

    return result


def _normalize_checkpoint_param_tree(candidate, template=None):
    """
    Canonicalize checkpoint module names that differ only because training used
    `nn.remat(...)`, which prefixes saved Mamba block names with `Checkpoint`.
    """
    if not isinstance(candidate, Mapping):
        return candidate
    template_map = template if isinstance(template, Mapping) else None
    changed = False
    normalized = {}
    for key, value in candidate.items():
        new_key = key
        if isinstance(key, str) and key.startswith("CheckpointMambaBlock1D_"):
            stripped = key[len("Checkpoint"):]
            if template_map is None or stripped in template_map:
                new_key = stripped
        child_template = template_map.get(new_key) if template_map is not None else None
        normalized[new_key] = _normalize_checkpoint_param_tree(value, child_template)
        changed = changed or new_key != key
    return normalized if changed else candidate

def _resolve_hparam(cli_value, wandb_value, inferred_value, default_value):
    """Pick a hyperparameter value while recording its source."""
    if cli_value not in (None, "", []):
        return cli_value, "cli"
    if wandb_value is not None:
        return wandb_value, "wandb"
    if inferred_value is not None:
        return inferred_value, "checkpoint"
    return default_value, "default"


def auto_color(k: int, n: int) -> str:
    import colorsys

    h = (k / max(n, 1)) % 1.0
    s, l = 0.65, 0.55
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


DEFAULT_COLOR_BY_LABEL = {
    "html": "#f2994a",
    "css": "#3498db",
    "javascript_typescript": "#f1c40f",
    "php": "#9b59b6",
    "python": "#2ecc71",
    "json": "#1abc9c",
    "sql": "#e74c3c",
    "java": "#8e44ad",
    "go": "#16a085",
    "c_family": "#2ecc71",
    "csharp": "#1abc9c",
    "csv": "#e74c3c",
    "ruby": "#8e44ad",
    "rust": "#16a085",
    "text": "#95a5a6",
    "yaml": "#d35400",
    "powershell": "#8e44ad",
    "shell": "#636e72",
}


def _default_color_for_label(name: str, index: int, total: int) -> str:
    return DEFAULT_COLOR_BY_LABEL.get(name.lower(), auto_color(index, total))


def _resolve_colors(base_names: List[str], colors_arg: Optional[str], *, allow_unknown_named: bool = False) -> List[str]:
    total = len(base_names)
    if total == 0:
        return []

    if not colors_arg:
        return [
            _default_color_for_label(name, idx, total)
            for idx, name in enumerate(base_names)
        ]

    entries = [item.strip() for item in colors_arg.split(",") if item.strip()]
    if not entries:
        return [
            _default_color_for_label(name, idx, total)
            for idx, name in enumerate(base_names)
        ]

    named: Dict[str, str] = {}
    positional: List[str] = []
    for entry in entries:
        if "=" in entry:
            key, value = entry.split("=", 1)
            key = key.strip()
            value = value.strip()
            match = next((name for name in base_names if name.lower() == key.lower()), None)
            if match is None:
                if allow_unknown_named:
                    continue
                raise ValueError(f"Color override references unknown class '{key}'.")
            if value:
                named[match] = value
        else:
            positional.append(entry)

    if named:
        if positional:
            raise ValueError("Mixing named and positional colors in --colors is not supported.")
        return [
            named.get(name, _default_color_for_label(name, idx, total))
            for idx, name in enumerate(base_names)
        ]

    # Pure positional overrides
    colors = positional[:total]
    while len(colors) < total:
        idx = len(colors)
        colors.append(_default_color_for_label(base_names[idx], idx, total))
    return colors[:total]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = (hex_color or "").strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        return f"rgba(136, 136, 136, {max(0.0, min(alpha, 1.0)):.2f})"
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return f"rgba(136, 136, 136, {max(0.0, min(alpha, 1.0)):.2f})"
    return f"rgba({r}, {g}, {b}, {max(0.0, min(alpha, 1.0)):.2f})"

def _hex_to_rgba_confidence(hex_color: str, alpha: float, confidence: float) -> str:
    """
    Convert a hex color to rgba while scaling saturation by `confidence` in [0, 1].

    Hue stays constant; saturation -> saturation * confidence.
    Useful for visualising model uncertainty without changing class identity.
    """
    import colorsys

    confidence_f = float(confidence) if confidence is not None else 0.0
    confidence_f = max(0.0, min(confidence_f, 1.0))

    hex_color = (hex_color or "").strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        return f"rgba(136, 136, 136, {max(0.0, min(alpha, 1.0)):.2f})"
    try:
        r_i = int(hex_color[0:2], 16)
        g_i = int(hex_color[2:4], 16)
        b_i = int(hex_color[4:6], 16)
    except ValueError:
        return f"rgba(136, 136, 136, {max(0.0, min(alpha, 1.0)):.2f})"

    r = r_i / 255.0
    g = g_i / 255.0
    b = b_i / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, s * confidence_f))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (
        f"rgba({int(round(r2 * 255))}, {int(round(g2 * 255))}, {int(round(b2 * 255))}, "
        f"{max(0.0, min(alpha, 1.0)):.2f})"
    )

def _looks_like_orbax_step_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    names = {x.name for x in p.iterdir()}
    return (
        "manifest.ocdbt" in names
        or "_CHECKPOINT_METADATA" in names
        or "ocdbt.process_0" in names
        or "_METADATA" in names
    )

def _find_latest_orbax_step_dir(root: Path):
    root = Path(root)
    if _looks_like_orbax_step_dir(root):
        return root
    if not root.exists() or not root.is_dir():
        return None
    step_dirs = []
    for d in root.iterdir():
        if d.is_dir() and _looks_like_orbax_step_dir(d):
            m = re.search(r"-(\d+)$", d.name)
            step = int(m.group(1)) if m else -1
            step_dirs.append((step, d.name, d))
    if not step_dirs:
        return None
    step_dirs.sort(key=lambda t: (t[0], ".msgpack" in t[1]))
    return step_dirs[-1][2]

def _extract_params_tree(obj):
    if hasattr(obj, "params"):
        try:
            return getattr(obj, "params")
        except Exception:
            pass
    try:
        from flax.core.frozen_dict import FrozenDict
        if isinstance(obj, FrozenDict):
            return _extract_params_tree(obj.unfreeze())
    except Exception:
        pass
    if isinstance(obj, Mapping):
        if "params" in obj:
            return obj["params"]
        for key in ("target", "state", "train_state", "flax_state"):
            if key in obj:
                try:
                    return _extract_params_tree(obj[key])
                except Exception:
                    pass
        for v in obj.values():
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    if isinstance(obj, (list, tuple)):
        for v in obj:
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            try:
                return _extract_params_tree(getattr(obj, f.name))
            except Exception:
                continue
    elif hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    raise KeyError("Could not find 'params' subtree in restored checkpoint object.")

def _load_params_from_any(ckpt_path: str, params_template_for_msgpack):
    p = Path(ckpt_path).resolve()

    # --- Case 1: a raw Flax .msgpack ---
    if p.is_file():
        with open(p, "rb") as f:
            raw = f.read()
        # 1a) Template-based restore: raw params at top level
        try:
            return serialization.from_bytes(params_template_for_msgpack, raw)
        except Exception:
            pass
        # 1b) Template-based restore: params wrapped in {"params": ...}
        try:
            dct = serialization.from_bytes({"params": params_template_for_msgpack}, raw)
            return dct["params"]
        except Exception:
            pass
        # 1c) Template-free restore: load dict, then extract params subtree
        try:
            restored = serialization.msgpack_restore(raw)
            # The checkpoint might be raw params, or wrapped in {"params": ...},
            # or be a full TrainState with nested params.
            if "params" in restored:
                candidate = restored["params"]
            else:
                candidate = restored
            candidate = _normalize_checkpoint_param_tree(candidate, params_template_for_msgpack)
            return serialization.from_state_dict(params_template_for_msgpack, candidate)
        except Exception:
            pass
        # 1d) Last resort: use msgpack_restore without template matching.
        #     This skips shape/key validation but works when the checkpoint
        #     was saved with different hyperparameters than the current config.
        try:
            restored = serialization.msgpack_restore(raw)
            if "params" in restored:
                candidate = restored["params"]
            else:
                candidate = restored
            candidate = _normalize_checkpoint_param_tree(candidate, params_template_for_msgpack)
            # Convert numpy arrays to jax arrays recursively
            return jax.tree.map(lambda x: jnp.asarray(x) if hasattr(x, 'shape') else x, candidate)
        except Exception as e:
            # Print diagnostic info to help debug
            try:
                diag = serialization.msgpack_restore(raw)
                print(f"[checkpoint debug] Top-level keys: {sorted(diag.keys())[:10]}", flush=True)
            except Exception:
                pass
            raise ValueError(
                f"Failed to load msgpack checkpoint '{ckpt_path}'. "
                f"Tried template-based and template-free approaches. Last error: {e}"
            ) from e

    # --- Case 2: an Orbax directory (step dir or its parent) ---
    step_dir = _find_latest_orbax_step_dir(p)
    if step_dir is None:
        raise FileNotFoundError(f"Checkpoint path not found/unsupported: {ckpt_path}")
    if ocp is None:
        raise RuntimeError(
            f"Checkpoint '{ckpt_path}' appears to be an Orbax directory, "
            "but orbax-checkpoint failed to import (likely JAX version mismatch). "
            "Try: pip install --upgrade orbax-checkpoint jax jaxlib"
        )
    step_dir_abs = step_dir.resolve().as_posix()
    ckptr = ocp.StandardCheckpointer()

    last_err = None

    # 2a) Untyped restore (often returns a TrainState)
    try:
        restored = ckptr.restore(step_dir_abs)
        return _extract_params_tree(restored)
    except Exception as e:
        last_err = e
        print(f"[orbax] untyped restore failed: {e}", flush=True)

    # 2b) Typed restore: provide only the params subtree as the target
    try:
        tmpl = {"params": params_template_for_msgpack}
        restored = ckptr.restore(step_dir_abs, target=tmpl, strict=False)
        return _extract_params_tree(restored)
    except Exception as e:
        last_err = e
        print(f"[orbax] target={{'params': ...}} restore failed: {e}", flush=True)

    # 2c) Typed restore: provide a dummy TrainState structure as the target
    try:
        import optax
        from flax.training import train_state as ts
        # Any tx works; we only need structure. Identity keeps it light.
        tx = optax.identity()
        dummy_state = ts.TrainState.create(
            apply_fn=lambda *a, **k: None,
            params=params_template_for_msgpack,
            tx=tx,
        )
        restored = ckptr.restore(step_dir_abs, target=dummy_state, strict=False)
        return restored.params
    except Exception as e:
        last_err = e
        print(f"[orbax] target=TrainState restore failed: {e}", flush=True)

    # If we get here, everything failed.
    raise RuntimeError(
        f"Orbax restore failed for '{step_dir_abs}'. "
        f"Tried untyped, target={{'params': ...}}, and target=TrainState. "
        f"Last error: {last_err}"
    )
# ---------------------------
# Input tokenization constants (match training)
# ---------------------------
BYTE_VOCAB_SIZE = 256
PAD_BYTE_ID = 256
NUM_TOKEN_EMBEDDINGS = BYTE_VOCAB_SIZE + 1  # 257

_VISIBLE_ASCII_BYTES = tuple(range(0x20, 0x7F))
_WHITESPACE_BYTES = (0x09, 0x0A, 0x0D)
_CURRENCY_BYTE_ID = np.int32(0xA4)
_ALLOWED_MODEL_BYTE_VALUES = np.array(
    sorted(set(_VISIBLE_ASCII_BYTES) | set(_WHITESPACE_BYTES) | {int(_CURRENCY_BYTE_ID)}),
    dtype=np.int32,
)
_ALLOWED_MODEL_TOKEN_VALUES = np.array(
    sorted(set(_ALLOWED_MODEL_BYTE_VALUES.tolist()) | {int(PAD_BYTE_ID)}),
    dtype=np.int32,
)

_PLACEHOLDER_CHAR = "\u00A4"
_ALLOWED_TEXT_CHARS = {chr(b) for b in _VISIBLE_ASCII_BYTES}
_ALLOWED_TEXT_CHARS.update({" ", "\n", "\t", _PLACEHOLDER_CHAR})
_VISUAL_WHITESPACE_CHARS = (" ", "\t", "\n")
_VISUAL_WHITESPACE_SET = frozenset(_VISUAL_WHITESPACE_CHARS)
_INLINE_WHITESPACE_SET = frozenset((" ", "\t"))
_BOUNDARY_SNAP_DELIMITER_CHARS = frozenset(
    ("<", ">", "/", "\\", '"', "'", "`", "(", ")", "[", "]", "{", "}", ",", ";", ":", "=")
)
_BOUNDARY_SNAP_ADJACENT_CHARS = _BOUNDARY_SNAP_DELIMITER_CHARS | frozenset((" ", "\t"))
_BOUNDARY_SNAP_PROB_MARGIN = 0.1
_BOUNDARY_SNAP_DELIMITER_PROB_MARGIN = 0.15
_BOUNDARY_SNAP_WHITESPACE_PROB_MARGIN = 0.05
_BOUNDARY_SNAP_MIN_IMPROVEMENT = 0.75
_BOUNDARY_WRAP_OPEN_TO_CLOSE = {
    '"': '"',
    "'": "'",
    "`": "`",
    "(": ")",
    "[": "]",
    "{": "}",
}
_BOUNDARY_WRAP_QUOTE_CHARS = frozenset(('"', "'", "`"))
_BOUNDARY_WRAP_PROB_MARGIN = 0.20
_BOUNDARY_WRAP_PAIR_BONUS = 1.35
_BOUNDARY_WRAP_QUOTE_BONUS = 0.35
_BOUNDARY_WRAP_SHELL_DELIMITER_EJECT_BONUS = 0.80
_BOUNDARY_WRAP_SHELL_DELIMITER_SWALLOW_PENALTY = 0.45
_BOUNDARY_WRAP_SHELL_QUOTE_EJECT_BONUS = 0.35
_BOUNDARY_WRAP_MIN_IMPROVEMENT = 0.70
_LOCAL_HOST_POSTPROCESS_RULE_NAMES: Tuple[Tuple[str, str, str], ...] = (
    ("json", "javascript_typescript", "json"),
)
_LOCAL_HOST_SINGLE_SIDE_MIN_CHARS = 8
_POSTPROCESS_STAGE_LABELS = {
    "markdown_structure_fill": "markdown structure fill",
    "boundary_snap": "boundary snap",
    "paired_delimiter_fill": "paired delimiter fill",
    "local_host_fill": "local host fill",
    "min_run": "min-run normalization",
    "newline_snap": "newline snap",
}


def _normalize_input_text(text: Optional[str]) -> str:
    if not text:
        return ""
    out_chars: List[str] = []
    for ch in text:
        if ch == "\r":
            ch = "\n"
        if ch in _ALLOWED_TEXT_CHARS:
            out_chars.append(ch)
        else:
            out_chars.append(_PLACEHOLDER_CHAR)
    return "".join(out_chars)


def _sanitize_model_bytes(arr: np.ndarray) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.uint8)
    if arr_np.size == 0:
        return arr_np
    invalid = ~np.isin(arr_np.astype(np.int32), _ALLOWED_MODEL_BYTE_VALUES)
    if np.any(invalid):
        arr_np = arr_np.copy()
        arr_np[invalid] = np.uint8(_CURRENCY_BYTE_ID)
    return arr_np


def _sanitize_model_tokens(arr: np.ndarray) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.int32)
    if arr_np.size == 0:
        return arr_np
    invalid = ~np.isin(arr_np, _ALLOWED_MODEL_TOKEN_VALUES)
    if np.any(invalid):
        arr_np[invalid] = _CURRENCY_BYTE_ID
    return arr_np


def _relabel_whitespace_from_neighbors(
    text: str,
    labels: List[int],
    char_probs: List[Dict[str, float]],
) -> tuple[List[int], List[Dict[str, float]]]:
    n = len(text)
    if n == 0 or not labels or len(labels) != n:
        return labels, char_probs

    # Fast path: no visual whitespace present.
    if not any(ch in _VISUAL_WHITESPACE_SET for ch in text):
        return labels, char_probs

    new_labels = list(labels)
    new_probs = list(char_probs)

    # Build per-line segments so we can prefer neighbors from the same line.
    line_starts: List[int] = [0]
    for idx, ch in enumerate(text):
        if ch == "\n" and idx + 1 < n:
            line_starts.append(idx + 1)
    # Ensure monotonic order and uniqueness.
    line_starts = sorted(set(line_starts))
    line_segments: List[tuple[int, int]] = []
    for i, start in enumerate(line_starts):
        end = line_starts[i + 1] if i + 1 < len(line_starts) else n
        if start < end:
            line_segments.append((start, end))

    left_same_line = [-1] * n
    right_same_line = [-1] * n
    for start, end in line_segments:
        last_non_ws = -1
        for i in range(start, end):
            if text[i] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = i
            left_same_line[i] = last_non_ws
        last_non_ws = -1
        for i in range(end - 1, start - 1, -1):
            if text[i] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = i
            right_same_line[i] = last_non_ws

    left_any = [-1] * n
    right_any = [-1] * n
    last_non_ws = -1
    for i in range(n):
        if text[i] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = i
        left_any[i] = last_non_ws
    last_non_ws = -1
    for i in range(n - 1, -1, -1):
        if text[i] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = i
        right_any[i] = last_non_ws

    for i, ch in enumerate(text):
        if ch not in _VISUAL_WHITESPACE_SET:
            continue
        src = -1
        ls = left_same_line[i]
        rs = right_same_line[i]
        if ls != -1 or rs != -1:
            if ls == -1:
                src = rs
            elif rs == -1:
                src = ls
            else:
                dist_l = i - ls
                dist_r = rs - i
                src = ls if dist_l <= dist_r else rs
        else:
            la = left_any[i]
            ra = right_any[i]
            if la != -1 or ra != -1:
                if la == -1:
                    src = ra
                elif ra == -1:
                    src = la
                else:
                    dist_l = i - la
                    dist_r = ra - i
                    src = la if dist_l <= dist_r else ra
        if src == -1:
            continue
        new_labels[i] = labels[src]
        if 0 <= src < len(char_probs):
            new_probs[i] = dict(char_probs[src])
    return new_labels, new_probs


def _build_label_runs(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    if not labels:
        return []
    runs: List[Tuple[int, int, int]] = []
    current = int(labels[0])
    start = 0
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label != current:
            runs.append((start, idx, current))
            start = idx
            current = label
    runs.append((start, len(labels), current))
    return runs


def _is_identifier_like_char(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch in ("_", "$", "-"))


def _label_prob_from_mapping(prob_row: Mapping[str, float] | None, label: int) -> float:
    if not isinstance(prob_row, Mapping):
        return 0.0
    try:
        value = prob_row.get(str(int(label)), 0.0)
    except Exception:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _dominant_label_from_prob_rows(
    char_probs: Sequence[Mapping[str, float]],
    start: int,
    end: int,
    *,
    exclude_labels: Sequence[int] = (),
) -> tuple[Optional[int], float, float]:
    if start >= end:
        return None, 0.0, 0.0
    excluded = {int(label) for label in exclude_labels}
    totals: Dict[int, float] = {}
    argmax_counts: Dict[int, int] = {}
    count = 0
    for pos in range(max(0, int(start)), min(int(end), len(char_probs))):
        row = char_probs[pos]
        if not isinstance(row, Mapping):
            continue
        best_label: Optional[int] = None
        best_prob = float("-inf")
        for key, value in row.items():
            try:
                label = int(key)
                prob = float(value)
            except Exception:
                continue
            if label in excluded:
                continue
            totals[label] = totals.get(label, 0.0) + prob
            if prob > best_prob:
                best_prob = prob
                best_label = label
        if best_label is not None:
            argmax_counts[best_label] = argmax_counts.get(best_label, 0) + 1
            count += 1
    if count <= 0 or not totals:
        return None, 0.0, 0.0
    best = max(
        totals.keys(),
        key=lambda label: (totals[label], argmax_counts.get(label, 0), -int(label)),
    )
    mean_support = float(totals[best]) / float(count)
    argmax_fraction = float(argmax_counts.get(best, 0)) / float(count)
    return int(best), mean_support, argmax_fraction


def _matching_wrap_delimiter(left: str, right: str) -> bool:
    return bool(left) and _BOUNDARY_WRAP_OPEN_TO_CLOSE.get(left) == right


def _wrapped_pair_bonus(left: str, right: str) -> float:
    if not _matching_wrap_delimiter(left, right):
        return 0.0
    bonus = _BOUNDARY_WRAP_PAIR_BONUS
    if left in _BOUNDARY_WRAP_QUOTE_CHARS:
        bonus += _BOUNDARY_WRAP_QUOTE_BONUS
    return bonus


def _is_codeish_wrapped_content(text: str, *, wrapper_char: str) -> bool:
    if not text:
        return False
    has_identifier = any(_is_identifier_like_char(ch) for ch in text)
    has_nonwrapper_delimiter = any(
        ch in _BOUNDARY_SNAP_DELIMITER_CHARS and ch != wrapper_char
        for ch in text
    )
    return has_identifier and has_nonwrapper_delimiter


def _boundary_local_score(text: str, boundary: int) -> float:
    if boundary < 0 or boundary > len(text):
        return float("-inf")
    left = text[boundary - 1] if boundary > 0 else ""
    right = text[boundary] if boundary < len(text) else ""
    score = 0.0
    if left in _BOUNDARY_SNAP_DELIMITER_CHARS:
        score += 1.25
    elif left in _INLINE_WHITESPACE_SET:
        score += 0.20
    if right in _BOUNDARY_SNAP_DELIMITER_CHARS:
        score += 1.25
    elif right in _INLINE_WHITESPACE_SET:
        score += 0.20
    if left in _BOUNDARY_SNAP_DELIMITER_CHARS and right in _BOUNDARY_SNAP_DELIMITER_CHARS:
        score += 0.35
    if left in _INLINE_WHITESPACE_SET and right in _INLINE_WHITESPACE_SET:
        score -= 0.25
    if left in ("<", "(", "[", "{") and _is_identifier_like_char(right):
        score += 0.35
    if right in (">", ")", "]", "}") and _is_identifier_like_char(left):
        score += 0.35
    if _is_identifier_like_char(left) and _is_identifier_like_char(right):
        score -= 1.5
    return score


def _configured_local_host_postprocess_rules() -> Tuple[Tuple[int, int, str], ...]:
    if TRAIN_CONFIG is None:
        return ()
    mapping = getattr(TRAIN_CONFIG, "LANG2ID", None)
    if not isinstance(mapping, dict):
        return ()
    resolved: List[Tuple[int, int, str]] = []
    for inner_name, host_name, rule_kind in _LOCAL_HOST_POSTPROCESS_RULE_NAMES:
        inner_idx = mapping.get(inner_name)
        host_idx = mapping.get(host_name)
        if inner_idx is None or host_idx is None:
            continue
        inner_id = int(inner_idx)
        host_id = int(host_idx)
        if inner_id == host_id:
            continue
        resolved.append((inner_id, host_id, str(rule_kind)))
    return tuple(resolved)


def _lookup_train_label_id(name: str) -> Optional[int]:
    if TRAIN_CONFIG is None:
        return None
    mapping = getattr(TRAIN_CONFIG, "LANG2ID", None)
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(str(name))
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _new_lock_mask(length: int) -> List[bool]:
    return [False] * max(0, int(length))


def _mark_locked_range(mask: List[bool], start: int, end: int) -> None:
    lo = max(0, int(start))
    hi = min(len(mask), int(end))
    for pos in range(lo, hi):
        mask[pos] = True


def _merge_lock_masks(base: List[bool], update: Sequence[bool]) -> List[bool]:
    if len(base) != len(update):
        return base
    for idx, flag in enumerate(update):
        if bool(flag):
            base[idx] = True
    return base


def _find_next_backtick_run(text: str, start: int, end: int, *, min_len: int) -> tuple[int, int]:
    pos = max(0, int(start))
    line_end = min(len(text), int(end))
    while pos < line_end:
        if text[pos] != "`":
            pos += 1
            continue
        run_end = pos
        while run_end < line_end and text[run_end] == "`":
            run_end += 1
        if (run_end - pos) >= int(min_len):
            return pos, run_end
        pos = run_end
    return -1, -1


def _infer_uniform_body_label(
    labels: Sequence[int],
    char_probs: Sequence[Mapping[str, float]],
    start: int,
    end: int,
    *,
    markdown_label: int,
) -> int:
    if start >= end:
        return int(markdown_label)
    counts: Dict[int, int] = {}
    supports: Dict[int, float] = {}
    for pos in range(max(0, int(start)), min(int(end), len(labels))):
        label = int(labels[pos])
        counts[label] = counts.get(label, 0) + 1
        supports[label] = supports.get(label, 0.0) + _label_prob_from_mapping(
            char_probs[pos] if pos < len(char_probs) else None,
            label,
        )
    if not counts:
        return int(markdown_label)
    best_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == best_count]
    if len(candidates) == 1:
        return int(candidates[0])
    best_support = max(supports.get(label, 0.0) for label in candidates)
    support_candidates = [label for label in candidates if supports.get(label, 0.0) >= (best_support - 1e-9)]
    if int(markdown_label) in support_candidates:
        return int(markdown_label)
    return int(min(support_candidates))


def _is_jsonish_content(text: str) -> bool:
    trimmed = text.strip()
    if not trimmed:
        return False
    if len(trimmed) >= 2 and (
        (trimmed[0] == "{" and trimmed[-1] == "}")
        or (trimmed[0] == "[" and trimmed[-1] == "]")
        or (trimmed[0] == '"' and trimmed[-1] == '"')
    ):
        return True
    lowered = trimmed.lower()
    if lowered in {"true", "false", "null"}:
        return True
    if re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?", trimmed):
        return True
    if ":" in trimmed and any(ch in trimmed for ch in ('"', "{", "[")):
        return True
    if "," in trimmed and any(ch in trimmed for ch in ('"', "{", "}", "[", "]")):
        return True
    return False


def _local_host_rule_matches_text(rule_kind: str, text: str) -> bool:
    if rule_kind == "json":
        return _is_jsonish_content(text)
    return bool(text)


def _apply_local_host_postprocess_rules(
    text: str,
    labels: List[int],
    *,
    local_host_rules: Sequence[Tuple[int, int, str]],
    min_run_chars: int,
) -> List[int]:
    if not labels or not text or not local_host_rules:
        return labels
    out = labels[:]
    single_side_min_chars = max(int(min_run_chars), int(_LOCAL_HOST_SINGLE_SIDE_MIN_CHARS))
    max_passes = max(1, len(out))
    for _ in range(max_passes):
        runs = _build_label_runs(out)
        changed = False
        for run_idx, (start, end, label) in enumerate(runs):
            for inner_label, host_label, rule_kind in local_host_rules:
                if int(label) != int(inner_label):
                    continue
                left_host_len = 0
                right_host_len = 0
                if run_idx > 0 and int(runs[run_idx - 1][2]) == int(host_label):
                    left_host_len = int(runs[run_idx - 1][1] - runs[run_idx - 1][0])
                if run_idx + 1 < len(runs) and int(runs[run_idx + 1][2]) == int(host_label):
                    right_host_len = int(runs[run_idx + 1][1] - runs[run_idx + 1][0])
                if left_host_len <= 0 and right_host_len <= 0:
                    continue
                if not _local_host_rule_matches_text(str(rule_kind), text[start:end]):
                    continue
                if left_host_len > 0 and right_host_len > 0:
                    should_relabel = True
                else:
                    should_relabel = (left_host_len + right_host_len) >= single_side_min_chars
                if not should_relabel:
                    continue
                for pos in range(start, end):
                    out[pos] = int(host_label)
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    return out


def _score_boundary_candidate(
    text: str,
    labels: Sequence[int],
    char_probs: Sequence[Mapping[str, float]],
    left_run: Tuple[int, int, int],
    right_run: Tuple[int, int, int],
    boundary: int,
) -> Optional[float]:
    current = int(left_run[1])
    left_start, _, left_label = left_run
    _, right_end, right_label = right_run
    if boundary < left_start or boundary > right_end:
        return None
    left_adjacent = text[boundary - 1] if boundary > 0 else ""
    right_adjacent = text[boundary] if boundary < len(text) else ""
    if left_adjacent == "\n" or right_adjacent == "\n":
        return None
    if boundary != current and (
        left_adjacent not in _BOUNDARY_SNAP_ADJACENT_CHARS
        and right_adjacent not in _BOUNDARY_SNAP_ADJACENT_CHARS
    ):
        return None

    score = _boundary_local_score(text, boundary)
    moved_positions: range
    src_label: int
    dest_label: int
    if boundary < current:
        moved_positions = range(boundary, current)
        src_label = int(left_label)
        dest_label = int(right_label)
    else:
        moved_positions = range(current, boundary)
        src_label = int(right_label)
        dest_label = int(left_label)

    for pos in moved_positions:
        ch = text[pos]
        if ch == "\n":
            return None
        src_prob = _label_prob_from_mapping(char_probs[pos] if pos < len(char_probs) else None, src_label)
        dest_prob = _label_prob_from_mapping(char_probs[pos] if pos < len(char_probs) else None, dest_label)
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            if dest_prob < (src_prob - _BOUNDARY_SNAP_DELIMITER_PROB_MARGIN):
                return None
        elif ch in _INLINE_WHITESPACE_SET:
            if dest_prob < (src_prob - _BOUNDARY_SNAP_WHITESPACE_PROB_MARGIN):
                return None
        elif ch not in _BOUNDARY_SNAP_ADJACENT_CHARS and dest_prob < (src_prob - _BOUNDARY_SNAP_PROB_MARGIN):
            return None
        score += 0.5 * (dest_prob - src_prob)
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            score += 0.15
        elif ch in _INLINE_WHITESPACE_SET:
            score += 0.02
    return score


def _score_wrapped_run_candidate(
    text: str,
    char_probs: Sequence[Mapping[str, float]],
    left_run: Tuple[int, int, int],
    middle_run: Tuple[int, int, int],
    right_run: Tuple[int, int, int],
    start: int,
    end: int,
) -> Optional[float]:
    left_start, _, left_label = left_run
    current_start, current_end, middle_label = middle_run
    _, right_end, right_label = right_run
    if int(left_label) != int(right_label) or int(middle_label) == int(left_label):
        return None
    if start < int(left_start) or end > int(right_end) or start >= end:
        return None
    if start <= 0 or end >= len(text):
        return None
    left_delim = text[start - 1]
    right_delim = text[end]
    score = _boundary_local_score(text, start) + _boundary_local_score(text, end)
    score += _wrapped_pair_bonus(left_delim, right_delim)

    host_label = int(left_label)
    inner_label = int(middle_label)
    changed = False
    union_start = min(int(current_start), int(start))
    union_end = max(int(current_end), int(end))
    shift_penalty = 0
    for pos in range(union_start, union_end):
        current_assign = inner_label if int(current_start) <= pos < int(current_end) else host_label
        candidate_assign = inner_label if int(start) <= pos < int(end) else host_label
        if candidate_assign == current_assign:
            continue
        ch = text[pos]
        if ch == "\n":
            return None
        changed = True
        current_prob = _label_prob_from_mapping(
            char_probs[pos] if pos < len(char_probs) else None,
            current_assign,
        )
        candidate_prob = _label_prob_from_mapping(
            char_probs[pos] if pos < len(char_probs) else None,
            candidate_assign,
        )
        if ch not in _BOUNDARY_SNAP_ADJACENT_CHARS and candidate_prob < (current_prob - _BOUNDARY_WRAP_PROB_MARGIN):
            return None
        score += 0.8 * (candidate_prob - current_prob)
        if ch in _BOUNDARY_SNAP_ADJACENT_CHARS:
            score += 0.10
        moving_out = candidate_assign == host_label and current_assign == inner_label
        moving_in = candidate_assign == inner_label and current_assign == host_label
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            if moving_out:
                score += _BOUNDARY_WRAP_SHELL_DELIMITER_EJECT_BONUS
            elif moving_in:
                score -= _BOUNDARY_WRAP_SHELL_DELIMITER_SWALLOW_PENALTY
        if ch in _BOUNDARY_WRAP_QUOTE_CHARS:
            if moving_out:
                score += _BOUNDARY_WRAP_SHELL_QUOTE_EJECT_BONUS
            elif moving_in:
                score -= 0.15
        shift_penalty += 1
    if changed:
        score += 0.60 * (
            _mean_label_support(char_probs, start, end, inner_label)
            - _mean_label_support(char_probs, start, end, host_label)
        )
        score -= 0.05 * max(0, shift_penalty - 2)
    return score


def _apply_boundary_shift(
    labels: List[int],
    current: int,
    boundary: int,
    left_label: int,
    right_label: int,
) -> List[int]:
    out = labels[:]
    if boundary < current:
        for pos in range(boundary, current):
            out[pos] = int(right_label)
    elif boundary > current:
        for pos in range(current, boundary):
            out[pos] = int(left_label)
    return out


def _snap_boundaries_to_delimiters(
    text: str,
    labels: List[int],
    char_probs: Sequence[Mapping[str, float]],
    *,
    max_shift: int = 2,
    min_run_chars: int = 1,
) -> List[int]:
    if max_shift <= 0 or len(labels) <= 1:
        return labels
    out = labels[:]
    max_passes = max(1, len(out) * 2)
    for _ in range(max_passes):
        runs = _build_label_runs(out)
        if len(runs) <= 1:
            break
        changed = False
        for idx in range(len(runs) - 1):
            left_run = runs[idx]
            right_run = runs[idx + 1]
            current = int(left_run[1])
            # Skip snapping when the current boundary already sits adjacent to
            # a delimiter or to whitespace. The raw model has then already
            # found a natural seam and a short snap onto a different delimiter
            # would relabel a syntactic character (for example the closing
            # quote of an HTML attribute) without evidence that the model was
            # wrong. This mirrors the gate used by the evaluation harness.
            left_curr = text[current - 1] if current > 0 else ""
            right_curr = text[current] if current < len(text) else ""
            if (
                left_curr in _BOUNDARY_SNAP_ADJACENT_CHARS
                or right_curr in _BOUNDARY_SNAP_ADJACENT_CHARS
            ):
                continue
            window_start = int(runs[idx - 1][0]) if idx > 0 else int(left_run[0])
            window_end = int(runs[idx + 2][1]) if (idx + 2) < len(runs) else int(right_run[1])
            current_short_count, current_min_len = _count_local_submin_interior_runs(
                out,
                min_run_chars=int(min_run_chars),
                window_start=window_start,
                window_end=window_end,
            )
            current_score = _score_boundary_candidate(text, out, char_probs, left_run, right_run, current)
            if current_score is None:
                current_score = _boundary_local_score(text, current)
            best_boundary = current
            best_score = current_score
            best_short_count = current_short_count
            best_min_len = current_min_len
            for shift in range(-int(max_shift), int(max_shift) + 1):
                if shift == 0:
                    continue
                candidate = current + shift
                score = _score_boundary_candidate(text, out, char_probs, left_run, right_run, candidate)
                if score is None:
                    continue
                candidate_labels = _apply_boundary_shift(out, current, candidate, left_run[2], right_run[2])
                candidate_short_count, candidate_min_len = _count_local_submin_interior_runs(
                    candidate_labels,
                    min_run_chars=int(min_run_chars),
                    window_start=window_start,
                    window_end=window_end,
                )
                better_structure = (
                    candidate_short_count < best_short_count
                    or (
                        candidate_short_count == best_short_count
                        and candidate_min_len > best_min_len
                    )
                )
                same_structure = (
                    candidate_short_count == best_short_count
                    and candidate_min_len == best_min_len
                )
                if better_structure or (same_structure and score > (best_score + 1e-6)):
                    best_boundary = candidate
                    best_score = score
                    best_short_count = candidate_short_count
                    best_min_len = candidate_min_len
            if best_boundary != current and best_score >= (current_score + _BOUNDARY_SNAP_MIN_IMPROVEMENT):
                out = _apply_boundary_shift(out, current, best_boundary, left_run[2], right_run[2])
                changed = True
                break
        if not changed:
            break
    return out


def _apply_wrapped_run_shift(
    labels: List[int],
    current_start: int,
    current_end: int,
    start: int,
    end: int,
    *,
    host_label: int,
    inner_label: int,
) -> List[int]:
    out = labels[:]
    union_start = min(int(current_start), int(start))
    union_end = max(int(current_end), int(end))
    for pos in range(union_start, union_end):
        out[pos] = int(inner_label) if int(start) <= pos < int(end) else int(host_label)
    return out


def _fill_markdown_structure_regions(
    text: str,
    labels: List[int],
    char_probs: Sequence[Mapping[str, float]],
    *,
    markdown_label: Optional[int],
) -> tuple[List[int], List[bool]]:
    out = labels[:]
    locked = _new_lock_mask(len(labels))
    if markdown_label is None or not text or not labels:
        return out, locked
    n = len(text)
    line_start = 0
    while line_start < n:
        line_end = text.find("\n", line_start)
        if line_end == -1:
            line_end = n
        pos = line_start
        while pos < line_end:
            if text[pos] != "`":
                pos += 1
                continue
            run_end = pos
            while run_end < line_end and text[run_end] == "`":
                run_end += 1
            run_len = run_end - pos
            if run_len < 3:
                pos = run_end
                continue
            next_start, next_end = _find_next_backtick_run(text, run_end, line_end, min_len=run_len)
            if next_start != -1 and next_start > run_end:
                for mark_pos in range(pos, run_end):
                    out[mark_pos] = int(markdown_label)
                body_label = _infer_uniform_body_label(
                    out,
                    char_probs,
                    run_end,
                    next_start,
                    markdown_label=int(markdown_label),
                )
                for body_pos in range(run_end, next_start):
                    out[body_pos] = int(body_label)
                for mark_pos in range(next_start, next_end):
                    out[mark_pos] = int(markdown_label)
                _mark_locked_range(locked, pos, next_end)
                pos = next_end
                continue
            token_end = run_end
            while token_end < line_end and text[token_end] not in (" ", "\t"):
                token_end += 1
            for mark_pos in range(pos, token_end):
                out[mark_pos] = int(markdown_label)
            _mark_locked_range(locked, pos, token_end)
            pos = token_end
        line_start = line_end + 1
    return out, locked


def _refine_wrapped_runs(
    text: str,
    labels: List[int],
    char_probs: Sequence[Mapping[str, float]],
    *,
    max_shift: int = 2,
) -> List[int]:
    if max_shift <= 0 or len(labels) <= 2:
        return labels
    out = labels[:]
    max_passes = max(1, len(out))
    for _ in range(max_passes):
        runs = _build_label_runs(out)
        if len(runs) <= 2:
            break
        changed = False
        for idx in range(1, len(runs) - 1):
            left_run = runs[idx - 1]
            middle_run = runs[idx]
            right_run = runs[idx + 1]
            if int(left_run[2]) != int(right_run[2]) or int(middle_run[2]) == int(left_run[2]):
                continue
            current_start = int(middle_run[0])
            current_end = int(middle_run[1])
            current_score = _score_wrapped_run_candidate(
                text,
                char_probs,
                left_run,
                middle_run,
                right_run,
                current_start,
                current_end,
            )
            if current_score is None:
                current_score = _boundary_local_score(text, current_start) + _boundary_local_score(text, current_end)
            best_start = current_start
            best_end = current_end
            best_score = current_score
            for left_shift in range(-int(max_shift), int(max_shift) + 1):
                cand_start = current_start + left_shift
                if cand_start < int(left_run[0]) or cand_start >= current_end:
                    continue
                for right_shift in range(-int(max_shift), int(max_shift) + 1):
                    cand_end = current_end + right_shift
                    if cand_end <= cand_start or cand_end > int(right_run[1]):
                        continue
                    if cand_start == current_start and cand_end == current_end:
                        continue
                    if not _matching_wrap_delimiter(
                        text[cand_start - 1] if cand_start > 0 else "",
                        text[cand_end] if cand_end < len(text) else "",
                    ):
                        continue
                    score = _score_wrapped_run_candidate(
                        text,
                        char_probs,
                        left_run,
                        middle_run,
                        right_run,
                        cand_start,
                        cand_end,
                    )
                    if score is None:
                        continue
                    if score > (best_score + 1e-6):
                        best_start = cand_start
                        best_end = cand_end
                        best_score = score
            if (
                (best_start != current_start or best_end != current_end)
                and best_score >= (current_score + _BOUNDARY_WRAP_MIN_IMPROVEMENT)
            ):
                out = _apply_wrapped_run_shift(
                    out,
                    current_start,
                    current_end,
                    best_start,
                    best_end,
                    host_label=int(left_run[2]),
                    inner_label=int(middle_run[2]),
                )
                changed = True
                break
        if not changed:
            break
    return out


def _mean_label_support(
    char_probs: Sequence[Mapping[str, float]],
    start: int,
    end: int,
    label: int,
) -> float:
    if start >= end:
        return 0.0
    total = 0.0
    count = 0
    for pos in range(start, end):
        total += _label_prob_from_mapping(char_probs[pos] if pos < len(char_probs) else None, label)
        count += 1
    return total / max(count, 1)


def _count_local_submin_interior_runs(
    labels: Sequence[int],
    *,
    min_run_chars: int,
    window_start: int,
    window_end: int,
) -> tuple[int, int]:
    runs = _build_label_runs(labels)
    count = 0
    min_len: Optional[int] = None
    for idx, (start, end, _label) in enumerate(runs):
        if end <= int(window_start) or start >= int(window_end):
            continue
        run_len = int(end - start)
        min_len = run_len if min_len is None else min(min_len, run_len)
        if 0 < idx < (len(runs) - 1) and run_len < int(min_run_chars):
            count += 1
    return count, (int(min_len) if min_len is not None else 0)


def _has_locked_positions(locked_mask: Sequence[bool], start: int, end: int) -> bool:
    lo = max(0, int(start))
    hi = min(len(locked_mask), int(end))
    return any(bool(locked_mask[pos]) for pos in range(lo, hi))


def _normalize_short_runs(
    labels: List[int],
    char_probs: Sequence[Mapping[str, float]],
    *,
    min_run_chars: int,
    locked_mask: Optional[Sequence[bool]] = None,
) -> List[int]:
    if min_run_chars <= 1 or len(labels) <= 2:
        return labels
    out = labels[:]
    locks = list(locked_mask) if locked_mask is not None and len(locked_mask) == len(out) else [False] * len(out)
    max_passes = max(1, len(out))
    for _ in range(max_passes):
        runs = _build_label_runs(out)
        changed = False
        for idx in range(1, len(runs) - 1):
            start, end, _label = runs[idx]
            if (end - start) >= int(min_run_chars):
                continue
            if _has_locked_positions(locks, start, end):
                continue
            left_run = runs[idx - 1]
            right_run = runs[idx + 1]
            left_label = int(left_run[2])
            right_label = int(right_run[2])
            left_len = int(left_run[1] - left_run[0])
            right_len = int(right_run[1] - right_run[0])
            candidates: List[tuple[tuple[float, ...], int]] = []
            seen_targets: set[int] = set()
            for direction, target, neighbor_len in (
                ("left", left_label, left_len),
                ("right", right_label, right_len),
            ):
                if int(target) in seen_targets:
                    continue
                seen_targets.add(int(target))
                candidate = out[:]
                for pos in range(start, end):
                    candidate[pos] = int(target)
                submin_count, min_run_len = _count_local_submin_interior_runs(
                    candidate,
                    min_run_chars=min_run_chars,
                    window_start=int(left_run[0]),
                    window_end=int(right_run[1]),
                )
                support = _mean_label_support(char_probs, start, end, int(target))
                sandwich = 1.0 if left_label == right_label == int(target) else 0.0
                direction_tiebreak = 1.0 if direction == "left" else 0.0
                key = (
                    -float(submin_count),
                    float(min_run_len),
                    sandwich,
                    float(neighbor_len),
                    float(support),
                    direction_tiebreak,
                )
                candidates.append((key, int(target)))
            if not candidates:
                continue
            target = max(candidates, key=lambda item: item[0])[1]
            for pos in range(start, end):
                out[pos] = int(target)
            changed = True
            break
        if not changed:
            break
    return out


def _record_postprocess_stage(
    before: Sequence[int],
    after: Sequence[int],
    *,
    stage: str,
    stage_hits: List[set[str]],
) -> None:
    for idx, (left, right) in enumerate(zip(before, after)):
        if int(left) != int(right):
            stage_hits[idx].add(stage)


def _build_postprocess_trace(
    original: Sequence[int],
    final: Sequence[int],
    stage_hits: Sequence[set[str]],
) -> Dict[str, Any]:
    stage_order = tuple(_POSTPROCESS_STAGE_LABELS.keys())
    changed_mask: List[bool] = []
    stage_lists: List[List[str]] = []
    descriptions: List[str] = []
    for idx in range(len(final)):
        changed = int(original[idx]) != int(final[idx])
        changed_mask.append(changed)
        ordered_hits = [name for name in stage_order if name in stage_hits[idx]]
        stage_lists.append(ordered_hits)
        descriptions.append(", ".join(_POSTPROCESS_STAGE_LABELS[name] for name in ordered_hits))
    return {
        "original_labels": [int(value) for value in original],
        "final_labels": [int(value) for value in final],
        "changed_mask": changed_mask,
        "stages": stage_lists,
        "descriptions": descriptions,
    }


def _snap_newlines_leading_trailing(
    text: str,
    labels: List[int],
) -> List[int]:
    out = labels[:]
    n = len(text)
    if n <= 3:
        return out
        
    line_start = 0
    while line_start < n:
        line_end = text.find("\n", line_start)
        if line_end == -1:
            line_end = n
            
        line_len = line_end - line_start
        if line_len > 3:
            # Front shifting
            # Only apply if it borders an actual newline (not start of file)
            if line_start > 0:
                out_1 = out[line_start + 1]
                out_2 = out[line_start + 2]
                
                if out_1 != out_2 and out_2 == out[line_start + 3]:
                    out[line_start] = out_2
                    out[line_start + 1] = out_2
                elif out[line_start] != out_1 and out_1 == out_2:
                    out[line_start] = out_1
                    
            # Back shifting
            # Only apply if it borders an actual newline (not end of file)
            if line_end < n:
                out_back_2 = out[line_end - 2]
                out_back_3 = out[line_end - 3]
                
                if out_back_2 != out_back_3 and out_back_3 == out[line_end - 4]:
                    out[line_end - 1] = out_back_3
                    out[line_end - 2] = out_back_3
                elif out[line_end - 1] != out_back_2 and out_back_2 == out_back_3:
                    out[line_end - 1] = out_back_2
                
        line_start = line_end + 1
        
    return out


def _postprocess_char_labels_with_trace(
    text: str,
    labels: List[int],
    char_probs: Sequence[Mapping[str, float]],
    *,
    min_run_chars: int,
    boundary_snap_max_shift: int = 2,
    local_host_rules: Sequence[Tuple[int, int, str]] = (),
    markdown_label: Optional[int] = None,
    html_label: Optional[int] = None,
    config: Optional[Dict[str, bool]] = None,
) -> tuple[List[int], Dict[str, Any]]:
    if not labels:
        empty_trace = {
            "original_labels": [],
            "final_labels": [],
            "changed_mask": [],
            "stages": [],
            "descriptions": [],
        }
        return labels, empty_trace
    markdown_label = _lookup_train_label_id("markdown") if markdown_label is None else int(markdown_label)
    html_label = _lookup_train_label_id("html") if html_label is None else int(html_label)
    original = [int(value) for value in labels]
    current = list(original)
    stage_hits: List[set[str]] = [set() for _ in labels]
    
    if config is None:
        config = {k: True for k in _POSTPROCESS_STAGE_LABELS.keys()}
        
    markdown_locked = _new_lock_mask(len(labels))
    if config.get("markdown_structure_fill", True):
        current, markdown_locked = _fill_markdown_structure_regions(
            text,
            current,
            char_probs,
            markdown_label=markdown_label,
        )
        _record_postprocess_stage(original, current, stage="markdown_structure_fill", stage_hits=stage_hits)
    
    if config.get("boundary_snap", True):
        snapped = _snap_boundaries_to_delimiters(
            text,
            current,
            char_probs,
            max_shift=boundary_snap_max_shift,
            min_run_chars=min_run_chars,
        )
        _record_postprocess_stage(current, snapped, stage="boundary_snap", stage_hits=stage_hits)
        current = snapped

    if config.get("paired_delimiter_fill", True):
        wrapped = _refine_wrapped_runs(
            text,
            current,
            char_probs,
            max_shift=boundary_snap_max_shift,
        )
        _record_postprocess_stage(current, wrapped, stage="paired_delimiter_fill", stage_hits=stage_hits)
        current = wrapped

    if config.get("local_host_fill", True):
        local_host_filled = _apply_local_host_postprocess_rules(
            text,
            current,
            local_host_rules=local_host_rules,
            min_run_chars=min_run_chars,
        )
        _record_postprocess_stage(current, local_host_filled, stage="local_host_fill", stage_hits=stage_hits)
        current = local_host_filled

    if config.get("min_run", True):
        locked_mask = _new_lock_mask(len(labels))
        _merge_lock_masks(locked_mask, markdown_locked)
        normalized = _normalize_short_runs(
            current,
            char_probs,
            min_run_chars=min_run_chars,
            locked_mask=locked_mask,
        )
        _record_postprocess_stage(current, normalized, stage="min_run", stage_hits=stage_hits)
        current = normalized

    if config.get("boundary_snap", True):
        snapped_relit = _snap_boundaries_to_delimiters(
            text,
            current,
            char_probs,
            max_shift=boundary_snap_max_shift,
            min_run_chars=min_run_chars,
        )
        _record_postprocess_stage(current, snapped_relit, stage="boundary_snap", stage_hits=stage_hits)
        current = snapped_relit

    if config.get("paired_delimiter_fill", True):
        wrapped_relit = _refine_wrapped_runs(
            text,
            current,
            char_probs,
            max_shift=boundary_snap_max_shift,
        )
        _record_postprocess_stage(current, wrapped_relit, stage="paired_delimiter_fill", stage_hits=stage_hits)
        current = wrapped_relit

    if config.get("markdown_structure_fill", True):
        markdown_relit, _markdown_relock = _fill_markdown_structure_regions(
            text,
            current,
            char_probs,
            markdown_label=markdown_label,
        )
        _record_postprocess_stage(current, markdown_relit, stage="markdown_structure_fill", stage_hits=stage_hits)
        current = markdown_relit

    if config.get("local_host_fill", True):
        final_with_local_host = _apply_local_host_postprocess_rules(
            text,
            current,
            local_host_rules=local_host_rules,
            min_run_chars=min_run_chars,
        )
        _record_postprocess_stage(current, final_with_local_host, stage="local_host_fill", stage_hits=stage_hits)
        current = final_with_local_host
    
    if config.get("newline_snap", True):
        final_snapped = _snap_newlines_leading_trailing(text, current)
        _record_postprocess_stage(current, final_snapped, stage="newline_snap", stage_hits=stage_hits)
        current = final_snapped
    
    return current, _build_postprocess_trace(original, current, stage_hits)


def _postprocess_char_labels(
    text: str,
    labels: List[int],
    char_probs: Sequence[Mapping[str, float]],
    *,
    min_run_chars: int,
    boundary_snap_max_shift: int = 2,
    local_host_rules: Sequence[Tuple[int, int, str]] = (),
    markdown_label: Optional[int] = None,
    html_label: Optional[int] = None,
    config: Optional[Dict[str, bool]] = None,
) -> List[int]:
    final_labels, _trace = _postprocess_char_labels_with_trace(
        text,
        labels,
        char_probs,
        min_run_chars=min_run_chars,
        boundary_snap_max_shift=boundary_snap_max_shift,
        local_host_rules=local_host_rules,
        markdown_label=markdown_label,
        html_label=html_label,
        config=config,
    )
    return final_labels

# ---------------------------
# Model (must mirror training EXACTLY)
# ---------------------------

class ConvBlock1D(nn.Module):
    features: int
    kernel_size: int = 3
    groups: int = 8
    dropout_rate: float = 0.0  # retained for parity; inactive at eval
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train: bool):
        h = nn.Conv(self.features, (self.kernel_size,), padding="SAME",
                    dtype=self.dtype, param_dtype=self.dtype)(x)  # use_bias=True, like training
        h = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(h)
        h = nn.gelu(h)
        if self.dropout_rate > 0.:
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        return h

def upsample_nn_1d(x, factor: int):
    return jnp.repeat(x, repeats=factor, axis=1)

class UNet1D(nn.Module):
    num_classes: int
    emb_dim: int = 128
    channels: Tuple[int, ...] = (128, 256, 384, 512)
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = False):
        # tokens are int32 with possible PAD_BYTE_ID=256
        tok_i32 = tokens.astype(jnp.int32)
        h = nn.Embed(num_embeddings=NUM_TOKEN_EMBEDDINGS, features=self.emb_dim,
                     embedding_init=nn.initializers.normal(stddev=0.02),
                     dtype=self.dtype, param_dtype=self.dtype)(tok_i32)

        skips = []
        # Down-sampling path: two ConvBlocks per level then max-pool (except last)
        for i, ch in enumerate(self.channels):
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            skips.append(h)
            if i < len(self.channels) - 1:
                h = nn.max_pool(h, (2,), strides=(2,), padding="SAME")

        # Up-sampling path: NN upsample, pad/crop to match, concat skip, two ConvBlocks
        for i, ch in enumerate(reversed(self.channels[:-1])):
            h = upsample_nn_1d(h, factor=2)
            skip = skips[-(i + 2)]
            if h.shape[1] != skip.shape[1]:
                if h.shape[1] < skip.shape[1]:
                    h = jnp.pad(h, ((0, 0), (0, skip.shape[1] - h.shape[1]), (0, 0)))
                else:
                    h = h[:, :skip.shape[1], :]
            h = jnp.concatenate([h, skip], axis=-1)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)

        logits_bf16 = nn.Conv(self.num_classes, (1,), padding="SAME",
                              dtype=self.dtype, param_dtype=self.dtype)(h)
        return logits_bf16.astype(jnp.float32)

# ---------------------------
# Model: Mamba (minimal Flax port; must match training)
# ---------------------------

class MambaBlock1D(nn.Module):
    d_model: int
    d_state: int = 8
    expand: int = 1
    dt_rank: int = 16
    d_conv: int = 4
    dropout_rate: float = 0.0
    bidirectional: bool = True
    dtype: jnp.dtype = jnp.bfloat16
    inference_kernel: str = "default"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        h = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype)(x)
        d_inner = int(self.d_model) * int(self.expand)

        xz = nn.Dense(
            2 * d_inner,
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=True,
        )(h)
        u, gate = jnp.split(xz, 2, axis=-1)

        if self.d_conv and int(self.d_conv) > 1:
            u = nn.Conv(
                features=d_inner,
                kernel_size=(int(self.d_conv),),
                padding="SAME",
                feature_group_count=d_inner,
                dtype=self.dtype,
                param_dtype=self.dtype,
                use_bias=True,
            )(u)
        u = jax.nn.silu(u)

        dt_rank = int(self.dt_rank) if int(self.dt_rank) > 0 else max(4, d_inner // 16)
        x_dbl = nn.Dense(
            dt_rank + 2 * int(self.d_state),
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=True,
        )(u)
        dt_raw, B, C = jnp.split(
            x_dbl, [dt_rank, dt_rank + int(self.d_state)], axis=-1
        )
        dt = nn.Dense(
            d_inner,
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=True,
        )(dt_raw)
        dt = jax.nn.softplus(dt).astype(jnp.float32) + 1e-4

        A_log = self.param(
            "A_log",
            nn.initializers.normal(stddev=0.02),
            (d_inner, int(self.d_state)),
            self.dtype,
        )
        A = -jnp.exp(A_log.astype(jnp.float32))
        D = self.param("D", nn.initializers.ones, (d_inner,), self.dtype).astype(jnp.float32)

        y = selective_scan_inference(
            u,
            dt,
            B,
            C,
            A,
            D,
            backend=self.inference_kernel if not train else "default",
        )
        if self.bidirectional:
            y_rev = selective_scan_inference(
                u[:, ::-1, :],
                dt[:, ::-1, :],
                B[:, ::-1, :],
                C[:, ::-1, :],
                A,
                D,
                backend=self.inference_kernel if not train else "default",
            )
            y = y + y_rev[:, ::-1, :]

        y = y.astype(self.dtype)
        y = y * jax.nn.silu(gate)
        y = nn.Dense(
            int(self.d_model),
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=True,
        )(y)
        if self.dropout_rate and float(self.dropout_rate) > 0.0:
            y = nn.Dropout(rate=float(self.dropout_rate), deterministic=not train)(y)
        return x + y


class Mamba1D(nn.Module):
    num_classes: int
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 8
    expand: int = 1
    dt_rank: int = 16
    d_conv: int = 4
    bidirectional: bool = True
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16
    inference_kernel: str = "default"

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = False):
        tok_i32 = tokens.astype(jnp.int32)
        h = nn.Embed(
            num_embeddings=NUM_TOKEN_EMBEDDINGS,
            features=int(self.d_model),
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(tok_i32)

        if self.dropout_rate and float(self.dropout_rate) > 0.0:
            h = nn.Dropout(rate=float(self.dropout_rate), deterministic=not train)(h)

        for _ in range(int(self.n_layers)):
            h = MambaBlock1D(
                d_model=int(self.d_model),
                d_state=int(self.d_state),
                expand=int(self.expand),
                dt_rank=int(self.dt_rank),
                d_conv=int(self.d_conv),
                dropout_rate=float(self.dropout_rate),
                bidirectional=bool(self.bidirectional),
                dtype=self.dtype,
                inference_kernel=str(self.inference_kernel),
            )(h, train=train)

        h = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype)(h)
        logits_bf16 = nn.Dense(
            int(self.num_classes),
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=True,
        )(h)
        return logits_bf16.astype(jnp.float32)

# ---------------------------
# Predictor
# ---------------------------

def make_slug(name: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

class Predictor:
    def __init__(
        self,
        ckpt_path: str,
        num_classes: int,
        model_dim: int,
        channels: Tuple[int, ...],
        arch: str = "unet1d",
        mamba_layers: int = 6,
        mamba_d_state: int = 8,
        mamba_expand: int = 1,
        mamba_dt_rank: int = 16,
        mamba_conv: int = 4,
        mamba_bidirectional: bool = True,
        dtype_str: str = "bfloat16",
        chunk: int = DEFAULT_CHUNK_SIZE,
        other_threshold: Optional[float] = 0.2,
        inference_batch_size: int = 12,
        device: Optional[str] = None,
        inference_backend: str = "auto",
    ):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        backend = resolve_backend(device)
        available = available_backends()
        if backend is not None and backend not in available:
            raise RuntimeError(
                f"Requested backend '{backend}' not available. Available: {sorted(available)}"
            )
        self.num_classes = int(num_classes)
        self.chunk = max(64, int(chunk or DEFAULT_CHUNK_SIZE))
        if self.chunk <= 0:
            raise ValueError("Chunk size must be positive.")
        self.dtype = getattr(jnp, dtype_str)
        self.arch = str(arch).lower().strip()
        self.backend = backend
        execution_backend = str(backend or jax.default_backend()).lower().strip()
        self.inference_backend = str(inference_backend).lower().strip()
        if self.inference_backend not in {"auto", "fast", "legacy"}:
            raise ValueError(
                f"Unknown inference backend '{inference_backend}'. "
                "Expected one of: auto, fast, legacy."
            )
        cuda_kernel_available = False
        self.fast_model = None
        if self.arch in {"mamba", "mamba1d", "bimamba", "ssm"}:
            self.model = Mamba1D(
                num_classes=self.num_classes,
                d_model=int(model_dim),
                n_layers=int(mamba_layers),
                d_state=int(mamba_d_state),
                expand=int(mamba_expand),
                dt_rank=int(mamba_dt_rank),
                d_conv=int(mamba_conv),
                bidirectional=bool(mamba_bidirectional),
                dtype=self.dtype,
            )
        else:
            self.model = UNet1D(
                num_classes=self.num_classes,
                emb_dim=int(model_dim),
                channels=tuple(channels),
                dtype=self.dtype,
            )
        # Init with dummy to create param structure (int32 tokens to allow PAD_BYTE_ID=256)
        dummy_tokens = jnp.full((1, self.chunk), PAD_BYTE_ID, dtype=jnp.int32)
        variables = self.model.init({"params": jax.random.PRNGKey(0)}, dummy_tokens, train=False)
        params_template_for_msgpack = variables["params"]

        # Load params from either a .msgpack file or an Orbax directory/root (untyped restore).
        self.params = _load_params_from_any(ckpt_path, params_template_for_msgpack)

        # Precompile apply fn; JIT caches per-seq-length (shape-polymorphic)
        if (
            self.arch in {"mamba", "mamba1d", "bimamba", "ssm"}
            and execution_backend == "gpu"
            and self.inference_backend in {"auto", "fast"}
            and has_cuda_mamba_kernel()
        ):
            self.fast_model = Mamba1D(
                num_classes=self.num_classes,
                d_model=int(self.model.d_model),
                n_layers=int(self.model.n_layers),
                d_state=int(self.model.d_state),
                expand=int(self.model.expand),
                dt_rank=int(self.model.dt_rank),
                d_conv=int(self.model.d_conv),
                bidirectional=bool(self.model.bidirectional),
                dtype=self.model.dtype,
                inference_kernel="cuda_fast",
            )
            cuda_kernel_available = True

        def _jit_apply(module):
            apply_fn = lambda tok: module.apply({"params": self.params}, tok, train=False)
            if backend:
                return jax.jit(apply_fn, backend=backend)
            return jax.jit(apply_fn)

        self._apply_legacy = _jit_apply(self.model)
        self._apply_fast = _jit_apply(self.fast_model) if self.fast_model is not None else self._apply_legacy
        self._apply = self._apply_legacy
        self._last_window_spans: List[Tuple[int, int]] = []
        self._last_postprocess_traces: List[Dict[str, Any]] = []
        self.last_postprocess_trace: Optional[Dict[str, Any]] = None
        # Optional virtual "other" bucket driven by a confidence threshold
        self.other_threshold: Optional[float] = (
            float(other_threshold) if other_threshold is not None and other_threshold > 0.0 else None
        )
        self.inference_batch_size = max(1, int(inference_batch_size))
        self._fast_engine: Optional[FastInferenceEngine] = None
        if self.inference_backend in {"auto", "fast"}:
            self._fast_engine = FastInferenceEngine(
                apply_tokens=self._apply_fast,
                num_classes=self.num_classes,
                pad_token_id=int(PAD_BYTE_ID),
                chunk_size=self.chunk,
                batch_size=self.inference_batch_size,
                sanitize_bytes=_sanitize_model_bytes,
                sanitize_tokens=_sanitize_model_tokens,
                arch=self.arch,
                inference_backend=self.inference_backend,
                actual_backend=(backend or str(jax.default_backend())),
                log_fn=lambda message: print(message, flush=True),
                model_dim=int(model_dim),
                channels=tuple(channels),
                mamba_layers=int(mamba_layers),
                mamba_d_state=int(mamba_d_state),
                mamba_expand=int(mamba_expand),
                mamba_bidirectional=bool(mamba_bidirectional),
                cuda_kernel_available=cuda_kernel_available,
            )

    @staticmethod
    def threshold_predictions(
        probs: np.ndarray,
        *,
        other_threshold: Optional[float],
        other_id: int,
    ) -> np.ndarray:
        """
        Apply open-set thresholding on probability vectors.

        For each position:
          pred = argmax(probs)
          conf = max(probs)
          if conf < threshold: pred = other_id
        """
        arr = np.asarray(probs, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"threshold_predictions expects shape [N, K], got {arr.shape}"
            )
        pred = np.argmax(arr, axis=-1).astype(np.int32)
        thr = (
            float(other_threshold)
            if other_threshold is not None
            else 0.0
        )
        if thr > 0.0 and pred.size > 0:
            conf = np.max(arr, axis=-1)
            pred[conf < thr] = int(other_id)
        return pred

    @staticmethod
    def _window_weights(length: int) -> np.ndarray:
        """Return center-weighted coefficients for a window of given length."""
        if length <= 1:
            return np.ones((length,), dtype=np.float32)
        positions = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
        sigma = 0.5
        weights = np.exp(-0.5 * (positions / sigma) ** 2)
        return weights.astype(np.float32)

    def _resolve_chunk_size(self, chunk: int = None) -> int:
        chunk_size = self.chunk
        if chunk is not None:
            requested = int(chunk)
            if requested != self.chunk:
                raise ValueError(
                    f"Predictor initialized with chunk={self.chunk} but received chunk={requested}."
                )
        return chunk_size

    @staticmethod
    def _build_window_spans(length: int, chunk_size: int) -> List[Tuple[int, int]]:
        return backend_build_window_spans(length, chunk_size)

    def _segment_bytes_batch_legacy(
        self,
        byte_arrays: Sequence[np.ndarray],
        chunk: int = None,
        return_probs: bool = True,
        return_max_probs: bool = False,
    ) -> tuple[List[np.ndarray], List[Optional[np.ndarray]], List[Optional[np.ndarray]], List[List[Tuple[int, int]]]]:
        chunk_size = self._resolve_chunk_size(chunk)
        arrays = [_sanitize_model_bytes(np.asarray(arr, dtype=np.uint8)) for arr in byte_arrays]
        final_probs_list: List[Optional[np.ndarray]] = []
        final_max_probs_list: List[Optional[np.ndarray]] = []
        byte_labels: List[np.ndarray] = []

        # Internal helper to process a set of windows and accumulate into target arrays
        def _process_windows(
            win_refs: Sequence[Tuple[int, int, int, np.ndarray]],
            target_p_acc: np.ndarray,
            target_w_acc: np.ndarray,
        ) -> None:
            batch_size = int(self.inference_batch_size)
            for i in range(0, len(win_refs), batch_size):
                batch = win_refs[i : i + batch_size]
                actual = len(batch)
                tokens = np.full((batch_size, chunk_size), PAD_BYTE_ID, dtype=np.int32)
                for j, (_, _, _, window_bytes) in enumerate(batch):
                    length = min(int(window_bytes.shape[0]), chunk_size)
                    if length > 0:
                        tokens[j, :length] = window_bytes[:length].astype(np.int32)

                tokens = _sanitize_model_tokens(tokens)
                logits = self._apply_legacy(jnp.array(tokens, dtype=jnp.int32))
                probs_batch = np.asarray(jax.nn.softmax(logits, axis=-1), dtype=np.float32)
                probs_batch = probs_batch[:actual, :chunk_size]

                for j, (_, start, end, _) in enumerate(batch):
                    plen = int(end) - int(start)
                    if plen <= 0:
                        continue
                    weights = self._window_weights(plen)
                    window_probs = probs_batch[j, :plen]
                    target_p_acc[start:end] += window_probs * weights[:, None]
                    target_w_acc[start:end] += weights

        for arr in arrays:
            L = int(arr.shape[0])
            # If the text is huge and we don't need full probs returned, process it in super-chunks
            # to cap the size of the internal probability accumulation matrix.
            # 512KB * 35 classes * 4 bytes = ~70MB.
            super_chunk_size = 512 * 1024
            if L > super_chunk_size and not return_probs:
                labels_out = np.zeros((L,), dtype=np.uint8)
                max_probs_out = np.zeros((L,), dtype=np.float32) if return_max_probs else None

                # Overlap of 2x chunk_size ensures window weighting is stable at boundaries
                overlap = 2 * chunk_size
                step = max(1, super_chunk_size - overlap)
                for start in range(0, L, step):
                    end = min(L, start + super_chunk_size)
                    sub_arr = arr[start:end]
                    sub_L = len(sub_arr)

                    sub_spans = self._build_window_spans(sub_L, chunk_size)
                    sub_win_refs = [(0, s, e, sub_arr[s:e]) for s, e in sub_spans]

                    sub_probs = np.zeros((sub_L, self.num_classes), dtype=np.float32)
                    sub_weights = np.zeros((sub_L,), dtype=np.float32)

                    _process_windows(sub_win_refs, sub_probs, sub_weights)

                    nonzero = sub_weights > 0
                    if np.any(nonzero):
                        sub_probs[nonzero] /= sub_weights[nonzero, None]
                    zero_mask = ~nonzero
                    if np.any(zero_mask):
                        sub_probs[zero_mask] = 1.0 / self.num_classes

                    sub_labels = np.argmax(sub_probs, axis=-1).astype(np.uint8)

                    # Copy results back, prioritizing the middle parts of super-chunks
                    copy_start_in_sub = 0 if start == 0 else (overlap // 2)
                    copy_start_in_full = start + copy_start_in_sub
                    copy_end_in_full = L if end == L else (start + super_chunk_size - (overlap // 2))

                    copy_len = copy_end_in_full - copy_start_in_full
                    if copy_len > 0:
                        labels_out[copy_start_in_full:copy_end_in_full] = sub_labels[copy_start_in_sub : copy_start_in_sub + copy_len]
                        if return_max_probs:
                            max_probs_out[copy_start_in_full:copy_end_in_full] = np.max(sub_probs[copy_start_in_sub : copy_start_in_sub + copy_len], axis=-1)

                    if end == L:
                        break

                byte_labels.append(labels_out)
                final_probs_list.append(None)
                final_max_probs_list.append(max_probs_out)
            else:
                # Standard single-pass accumulation for smaller arrays or when full probs are requested
                p_acc = np.zeros((L, self.num_classes), dtype=np.float32)
                w_acc = np.zeros((L,), dtype=np.float32)

                spans = self._build_window_spans(L, chunk_size)
                win_refs = [(0, s, e, arr[s:e]) for s, e in spans]
                _process_windows(win_refs, p_acc, w_acc)

                nonzero = w_acc > 0
                if np.any(nonzero):
                    p_acc[nonzero] /= w_acc[nonzero, None]
                    p_acc[~nonzero] = 1.0 / self.num_classes
                else:
                    p_acc[:] = 1.0 / self.num_classes

                byte_labels.append(np.argmax(p_acc, axis=-1).astype(np.uint8))
                final_max_probs_list.append(np.max(p_acc, axis=-1) if return_max_probs else None)
                final_probs_list.append(p_acc if return_probs else None)

        spans_by_text = [self._build_window_spans(int(arr.shape[0]), chunk_size) for arr in arrays]
        return byte_labels, final_probs_list, final_max_probs_list, spans_by_text

    def _segment_bytes_batch(
        self,
        byte_arrays: Sequence[np.ndarray],
        chunk: int = None,
        return_probs: bool = True,
        return_max_probs: bool = False,
    ) -> tuple[List[np.ndarray], List[Optional[np.ndarray]], List[Optional[np.ndarray]], List[List[Tuple[int, int]]]]:
        chunk_size = self._resolve_chunk_size(chunk)
        if chunk_size != self.chunk:
            raise ValueError(
                f"Predictor initialized with chunk={self.chunk} but received chunk={chunk_size}."
            )
        if self.inference_backend == "legacy" or self._fast_engine is None:
            return self._segment_bytes_batch_legacy(
                byte_arrays,
                chunk=chunk,
                return_probs=return_probs,
                return_max_probs=return_max_probs,
            )
        try:
            if not return_probs and not return_max_probs:
                byte_labels, spans_by_text = self._fast_engine.segment_bytes_batch_labels_only(byte_arrays)
                none_probs = [None for _ in byte_labels]
                none_max_probs = [None for _ in byte_labels]
                return byte_labels, none_probs, none_max_probs, spans_by_text
            if not return_probs and return_max_probs:
                byte_labels, max_probs_out, spans_by_text = self._fast_engine.segment_bytes_batch_labels_and_max_probs(byte_arrays)
                none_probs = [None for _ in byte_labels]
                return byte_labels, none_probs, max_probs_out, spans_by_text

            byte_labels, byte_probs, spans_by_text = self._fast_engine.segment_bytes_batch(byte_arrays)
            probs_out: List[Optional[np.ndarray]] = byte_probs if return_probs else [None for _ in byte_labels]
            max_probs_out: List[Optional[np.ndarray]] = (
                [np.max(probs, axis=-1).astype(np.float32) if probs.size > 0 else np.zeros((0,), dtype=np.float32) for probs in byte_probs]
                if return_max_probs
                else [None for _ in byte_labels]
            )
            return byte_labels, probs_out, max_probs_out, spans_by_text
        except FastInferenceFailure as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path=exc.source_path,
                        to_path="legacy",
                        trigger=exc.trigger,
                        reason=exc.reason,
                    ),
                    flush=True,
                )
                return self._segment_bytes_batch_legacy(
                    byte_arrays,
                    chunk=chunk,
                    return_probs=return_probs,
                    return_max_probs=return_max_probs,
                )
            raise
        except Exception as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path="fast",
                        to_path="legacy",
                        trigger="runtime_error",
                        reason=str(exc),
                    ),
                    flush=True,
                )
                return self._segment_bytes_batch_legacy(
                    byte_arrays,
                    chunk=chunk,
                    return_probs=return_probs,
                    return_max_probs=return_max_probs,
                )
            raise

    def _segment_bytes(
        self,
        byte_arr: np.ndarray,
        chunk: int = None,
        return_max_probs: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        byte_labels, byte_probs, max_probs, spans_by_text = self._segment_bytes_batch(
            [byte_arr],
            chunk=chunk,
            return_probs=not return_max_probs,
            return_max_probs=return_max_probs,
        )
        self._last_window_spans = list(spans_by_text[0]) if spans_by_text else []
        return byte_labels[0], byte_probs[0], max_probs[0]

    @staticmethod
    def _build_windows_info(text: str, spans: Sequence[Tuple[int, int]]) -> List[Dict[str, int]]:
        if not spans:
            return []
        byte_offsets = [0]
        for ch in text:
            byte_offsets.append(byte_offsets[-1] + len(ch.encode("utf-8", "ignore")))
        windows_info: List[Dict[str, int]] = []
        for idx, (start_byte, end_byte) in enumerate(spans):
            start_char = bisect.bisect_left(byte_offsets, int(start_byte))
            end_char = bisect.bisect_left(byte_offsets, int(end_byte))
            windows_info.append(
                {
                    "index": idx,
                    "start_byte": int(start_byte),
                    "end_byte": int(end_byte),
                    "start_char": int(start_char),
                    "end_char": int(end_char),
                }
            )
        return windows_info

    def _finalize_segmented_text(
        self,
        *,
        text: str,
        byte_labels: np.ndarray,
        byte_probs: np.ndarray,
        spans: Sequence[Tuple[int, int]],
        min_run_chars: int,
        postprocess_config: Optional[Dict[str, bool]] = None,
    ) -> tuple[
        tuple[List[Tuple[int, int, int]], List[int], List[Dict[str, float]], List[Dict[str, int]]],
        Dict[str, Any],
    ]:
        char_labels, char_probs = self._byte_labels_to_char_labels(text, byte_labels, byte_probs)
        char_labels, char_probs = self._apply_other_threshold(char_labels, char_probs)
        char_labels, postprocess_trace = _postprocess_char_labels_with_trace(
            text,
            char_labels,
            char_probs,
            min_run_chars=int(min_run_chars),
            boundary_snap_max_shift=2,
            local_host_rules=_configured_local_host_postprocess_rules(),
            markdown_label=_lookup_train_label_id("markdown"),
            html_label=_lookup_train_label_id("html"),
            config=postprocess_config,
        )
        segs: List[Tuple[int, int, int]] = []
        if len(char_labels) > 0:
            cur = char_labels[0]
            start = 0
            for i in range(1, len(char_labels)):
                if char_labels[i] != cur:
                    segs.append((start, i, cur))
                    start = i
                    cur = char_labels[i]
            segs.append((start, len(char_labels), cur))
        windows_info = self._build_windows_info(text, spans)
        return (segs, char_labels, char_probs, windows_info), postprocess_trace

    def _predict_logits_legacy(self, token_batch: np.ndarray) -> np.ndarray:
        """
        Run the model on a batch of token windows. Accepts shape (batch, <=chunk)
        (or a single 1-D window) of int32 tokens and returns logits with shape
        (batch, chunk, num_classes). Callers can slice to their original length.
        """
        arr = np.asarray(token_batch, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr[None, :]
        length = int(arr.shape[1])
        if length > self.chunk:
            raise ValueError(
                f"predict_logits received window of length {length}, exceeds chunk {self.chunk}"
            )
        if length < self.chunk:
            padded = np.full((arr.shape[0], self.chunk), PAD_BYTE_ID, dtype=np.int32)
            padded[:, :length] = arr
            arr = padded
        arr = _sanitize_model_tokens(arr)
        logits = self._apply_legacy(jnp.array(arr, dtype=jnp.int32))
        return np.asarray(logits, dtype=np.float32)

    def predict_logits(self, token_batch: np.ndarray) -> np.ndarray:
        if self.inference_backend == "legacy" or self._fast_engine is None:
            return self._predict_logits_legacy(token_batch)
        try:
            return self._fast_engine.predict_logits(token_batch)
        except FastInferenceFailure as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path=exc.source_path,
                        to_path="legacy",
                        trigger=exc.trigger,
                        reason=exc.reason,
                    ),
                    flush=True,
                )
                return self._predict_logits_legacy(token_batch)
            raise
        except Exception as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path="fast",
                        to_path="legacy",
                        trigger="runtime_error",
                        reason=str(exc),
                    ),
                    flush=True,
                )
                return self._predict_logits_legacy(token_batch)
            raise

    def _byte_labels_to_char_labels(self, text: str, byte_labels: np.ndarray, byte_probs: np.ndarray = None) -> tuple[List[int], List[Dict[str, float]]]:
        labels: List[int] = []
        char_probs: List[Dict[str, float]] = []
        bpos = 0
        for ch in text:
            cb = ch.encode("utf-8")
            L = len(cb)
            if L == 0:
                labels.append(0)
                char_probs.append({str(i): 0.0 for i in range(self.num_classes)})
                continue
            seg = byte_labels[bpos:bpos+L]
            if len(seg) == 0:
                lbl = 0
                probs = {str(i): 0.0 for i in range(self.num_classes)}
            else:
                if byte_probs is not None:
                    # Average probabilities across bytes in the character, then
                    # derive prediction directly from those averaged probs.
                    avg_probs = np.mean(byte_probs[bpos:bpos+L], axis=0)
                    lbl = int(np.argmax(avg_probs))
                    probs = {str(i): float(avg_probs[i]) for i in range(self.num_classes)}
                else:
                    vals, counts = np.unique(seg, return_counts=True)
                    lbl = int(vals[np.argmax(counts)])
                    probs = {str(i): 1.0 if i == lbl else 0.0 for i in range(self.num_classes)}
            labels.append(lbl)
            char_probs.append(probs)
            bpos += L
        labels, char_probs = _relabel_whitespace_from_neighbors(text, labels, char_probs)
        return labels, char_probs

    def _apply_other_threshold(
        self,
        labels: List[int],
        char_probs: List[Dict[str, float]],
    ) -> tuple[List[int], List[Dict[str, float]]]:
        """
        Optionally map low-confidence characters into a virtual 'other' bucket.

        If other_threshold is set, any character whose maximum softmax
        probability across the trained classes is below this threshold is
        relabeled to an extra 'other' id at index self.num_classes.
        """
        thr = self.other_threshold
        if thr is None or thr <= 0.0 or not labels or not char_probs:
            return labels, char_probs
        other_id = self.num_classes
        probs_arr = np.zeros((len(char_probs), self.num_classes), dtype=np.float32)
        for i, probs in enumerate(char_probs):
            if not probs:
                continue
            for key, val in probs.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < self.num_classes:
                    probs_arr[i, idx] = float(val)
        out_labels = self.threshold_predictions(
            probs_arr,
            other_threshold=thr,
            other_id=other_id,
        ).tolist()
        return out_labels, char_probs

    def _smooth_min_run(self, labels: List[int], min_run: int) -> List[int]:
        return _normalize_short_runs(labels, (), min_run_chars=min_run)

    def segment_texts(
        self,
        texts: Sequence[str],
        min_run_chars: int = 5,
        chunk: int = None,
        postprocess_config: Optional[Dict[str, bool]] = None,
    ) -> List[tuple[List[Tuple[int, int, int]], List[int], List[Dict[str, float]], List[Dict[str, int]]]]:
        byte_arrays = [
            _sanitize_model_bytes(np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8))
            for text in texts
        ]
        byte_labels_by_text, byte_probs_by_text, max_probs_by_text, spans_by_text = self._segment_bytes_batch(
            byte_arrays,
            chunk=chunk,
        )
        results: List[tuple[List[Tuple[int, int, int]], List[int], List[Dict[str, float]], List[Dict[str, int]]]] = []
        traces: List[Dict[str, Any]] = []
        for text, byte_labels, byte_probs, spans in zip(
            texts,
            byte_labels_by_text,
            byte_probs_by_text,
            spans_by_text,
        ):
            result, trace = self._finalize_segmented_text(
                text=text,
                byte_labels=byte_labels,
                byte_probs=byte_probs,
                spans=spans,
                min_run_chars=min_run_chars,
                postprocess_config=postprocess_config,
            )
            results.append(result)
            traces.append(trace)
        self._last_postprocess_traces = traces
        self.last_postprocess_trace = traces[0] if len(traces) == 1 else None
        return results

    def segment_text(
        self, 
        text: str, 
        min_run_chars: int = 5, 
        chunk: int = None,
        postprocess_config: Optional[Dict[str, bool]] = None,
    ):
        result = self.segment_texts(
            [text], 
            min_run_chars=min_run_chars, 
            chunk=chunk,
            postprocess_config=postprocess_config,
        )[0]
        self._last_window_spans = [
            (int(window["start_byte"]), int(window["end_byte"]))
            for window in result[3]
        ]
        if self._last_postprocess_traces:
            self.last_postprocess_trace = self._last_postprocess_traces[0]
        return result
