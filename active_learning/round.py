from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from datasets import load_from_disk  # type: ignore

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg  # noqa: E402
from utils.data import prepare_dsets_by_lang_with_splits  # noqa: E402
from utils.full_sequence import make_training_full_sequence_with_metadata  # noqa: E402
from utils.window_generator import make_training_window_with_metadata  # noqa: E402
from viewers.core import (  # noqa: E402
    DEFAULT_CHANNELS,
    DEFAULT_CHUNK_SIZE,
    Predictor,
    _infer_checkpoint_architecture,
    _load_checkpoint_hparams,
    _normalize_input_text,
    _resolve_hparam,
)

from .acquisition import CandidateSpan, select_candidate_spans
from .label_store import LabelStore, StoredInferenceSample, StoredRefinement
from .oracle import BoundarySnippet, GeminiBoundaryOracle, StubOracle
from .sample_prefetch import FullSequenceSamplePrefetcher


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def _make_snippet_id(sample_hash: str, start: int, end: int, boundary: int) -> str:
    raw = f"{sample_hash}:{int(start)}:{int(end)}:{int(boundary)}"
    # Keep snippet IDs compact for oracle prompts/logs to reduce token usage.
    return hashlib.blake2s(raw.encode("utf-8", "ignore"), digest_size=5).hexdigest()


def _make_sample_prompt_id(sample_hash: str) -> str:
    raw = f"{sample_hash}:full"
    return hashlib.blake2s(raw.encode("utf-8", "ignore"), digest_size=5).hexdigest()


def _probs_dicts_to_matrix(char_probs: Sequence[Dict[str, float]], num_classes: int) -> np.ndarray:
    arr = np.zeros((len(char_probs), int(num_classes)), dtype=np.float32)
    for i, probs in enumerate(char_probs):
        if not isinstance(probs, dict):
            continue
        for key, val in probs.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_classes:
                arr[i, idx] = float(val)
    row_sums = arr.sum(axis=-1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return arr / row_sums


def _segments_from_label_names(labels: Sequence[str]) -> List[Dict[str, object]]:
    if not labels:
        return []
    out: List[Dict[str, object]] = []
    start = 0
    cur = str(labels[0])
    for i in range(1, len(labels)):
        nxt = str(labels[i])
        if nxt != cur:
            out.append({"start": start, "end": i, "label": cur})
            start = i
            cur = nxt
    out.append({"start": start, "end": len(labels), "label": cur})
    return out


def _parse_langs(arg: Optional[str]) -> List[str]:
    if not arg:
        return [name for name, _ in sorted(cfg.LANG2ID.items(), key=lambda kv: kv[1])]
    parts = [part.strip() for part in arg.split(",") if part.strip()]
    out: List[str] = []
    for name in parts:
        low = name.lower()
        if low == "c" or low == "cpp" or low == "c++":
            low = "c_family"
        if low == "js" or low == "ts":
            low = "javascript_typescript"
        if low in cfg.LANG2ID and low not in out:
            out.append(low)
    return out


def _resolve_channels(cli_value: Optional[str], auto_hparams: Dict[str, object], ckpt: Dict[str, object]) -> List[int]:
    channels_cli = None
    if cli_value:
        channels_cli = [int(x) for x in cli_value.split(",") if x.strip()]
    value, _ = _resolve_hparam(
        channels_cli,
        auto_hparams.get("channels"),
        ckpt.get("channels"),
        list(DEFAULT_CHANNELS),
    )
    return [int(x) for x in value]


def _build_predictor(args: argparse.Namespace) -> Predictor:
    ckpt_path = Path(args.ckpt).resolve()
    auto_hparams = _load_checkpoint_hparams(ckpt_path)
    ckpt_inferred = _infer_checkpoint_architecture(ckpt_path)

    arch_value, _ = _resolve_hparam(
        args.arch,
        auto_hparams.get("arch"),
        ckpt_inferred.get("arch"),
        "unet1d",
    )
    arch = str(arch_value).lower().strip()

    model_dim_value, _ = _resolve_hparam(
        args.model_dim,
        auto_hparams.get("model_dim"),
        ckpt_inferred.get("model_dim"),
        256,
    )
    model_dim = int(model_dim_value)
    channels = tuple(_resolve_channels(args.channels, auto_hparams, ckpt_inferred))

    mamba_layers, _ = _resolve_hparam(
        args.mamba_layers,
        auto_hparams.get("mamba_layers"),
        ckpt_inferred.get("mamba_layers"),
        6,
    )
    mamba_d_state, _ = _resolve_hparam(
        args.mamba_d_state,
        auto_hparams.get("mamba_d_state"),
        ckpt_inferred.get("mamba_d_state"),
        8,
    )
    mamba_expand, _ = _resolve_hparam(
        args.mamba_expand,
        auto_hparams.get("mamba_expand"),
        ckpt_inferred.get("mamba_expand"),
        1,
    )
    mamba_dt_rank, _ = _resolve_hparam(
        args.mamba_dt_rank,
        auto_hparams.get("mamba_dt_rank"),
        ckpt_inferred.get("mamba_dt_rank"),
        16,
    )
    mamba_conv, _ = _resolve_hparam(
        args.mamba_conv,
        auto_hparams.get("mamba_conv"),
        ckpt_inferred.get("mamba_conv"),
        4,
    )
    mamba_bidirectional, _ = _resolve_hparam(
        args.mamba_bidirectional,
        auto_hparams.get("mamba_bidirectional"),
        ckpt_inferred.get("mamba_bidirectional"),
        True,
    )

    return Predictor(
        ckpt_path=str(ckpt_path),
        num_classes=cfg.NUM_CLASSES,
        model_dim=model_dim,
        channels=channels,
        arch=arch,
        mamba_layers=int(mamba_layers),
        mamba_d_state=int(mamba_d_state),
        mamba_expand=int(mamba_expand),
        mamba_dt_rank=int(mamba_dt_rank),
        mamba_conv=int(mamba_conv),
        mamba_bidirectional=bool(mamba_bidirectional),
        dtype_str=str(args.dtype),
        chunk=int(args.chunk),
        other_threshold=float(args.other_threshold) if args.other_threshold > 0 else None,
        inference_batch_size=max(1, int(getattr(args, "predict_batch_size", 12))),
    )


def _window_tokens_to_normalized_text(tokens: np.ndarray) -> str:
    arr = np.asarray(tokens, dtype=np.int32)
    if arr.size == 0:
        return ""
    valid = arr[(arr >= 0) & (arr < 256)]
    if valid.size == 0:
        return ""
    text = valid.astype(np.uint8, copy=False).tobytes().decode("utf-8", "ignore")
    return _normalize_input_text(text)


def _collect_window_source_langs(metadata: Optional[Dict[str, object]]) -> List[str]:
    if not isinstance(metadata, dict):
        return []
    out: List[str] = []
    for entry in metadata.get("samples") or []:
        if not isinstance(entry, dict):
            continue
        lang = str(entry.get("language", "")).strip()
        if lang and lang not in out:
            out.append(lang)
    host = metadata.get("host")
    if isinstance(host, dict):
        lang = str(host.get("language", "")).strip()
        if lang and lang not in out:
            out.append(lang)
    mode = str(metadata.get("mode", "")).strip()
    if mode == "markdown" and "markdown" not in out:
        out.insert(0, "markdown")
    return out


def _resolve_window_source_lang(metadata: Optional[Dict[str, object]]) -> str:
    if not isinstance(metadata, dict):
        return "unknown"
    mode = str(metadata.get("mode", "")).strip()
    if mode == "markdown":
        return "markdown"
    host = metadata.get("host")
    if isinstance(host, dict):
        host_lang = str(host.get("language", "")).strip()
        if host_lang:
            return host_lang

    best_lang = ""
    best_bytes = -1
    unique_langs: List[str] = []
    for entry in metadata.get("samples") or []:
        if not isinstance(entry, dict):
            continue
        lang = str(entry.get("language", "")).strip()
        if not lang:
            continue
        if lang not in unique_langs:
            unique_langs.append(lang)
        bytes_used = entry.get("final_bytes", entry.get("bytes", 0))
        try:
            score = int(bytes_used or 0)
        except (TypeError, ValueError):
            score = 0
        if score > best_bytes:
            best_bytes = score
            best_lang = lang
    if best_lang:
        return best_lang
    if len(unique_langs) == 1:
        return unique_langs[0]
    return "unknown"


def _summarize_window_metadata(
    metadata: Optional[Dict[str, object]],
    *,
    window_seed: int,
) -> Dict[str, object]:
    if not isinstance(metadata, dict):
        return {
            "sampling_mode": "training_window",
            "window_seed": int(window_seed),
            "augmentation_mode": "",
            "augmentation_requested_mode": "",
            "source_langs": [],
        }
    host = metadata.get("host")
    host_summary = None
    if isinstance(host, dict):
        host_summary = {
            "language": str(host.get("language", "")).strip(),
            "source": str(host.get("source", "")).strip(),
            "final_bytes": int(host.get("final_bytes", 0) or 0),
        }
    return {
        "sampling_mode": "training_window",
        "window_seed": int(window_seed),
        "augmentation_mode": str(metadata.get("mode", "")).strip(),
        "augmentation_requested_mode": str(metadata.get("requested_mode", "")).strip(),
        "source_langs": _collect_window_source_langs(metadata),
        "host": host_summary,
        "line_injection_count": len(metadata.get("line_injections") or []),
        "markdown_block_count": len(metadata.get("markdown_blocks") or []),
        "source_count": len(metadata.get("samples") or []),
    }


def _collect_full_sequence_source_langs(metadata: Optional[Dict[str, object]]) -> List[str]:
    if not isinstance(metadata, dict):
        return []
    out: List[str] = []
    for component in metadata.get("components") or []:
        langs = _collect_window_source_langs(component if isinstance(component, dict) else None)
        for lang in langs:
            if lang and lang not in out:
                out.append(lang)
    return out


def _resolve_full_sequence_source_lang(metadata: Optional[Dict[str, object]]) -> str:
    if not isinstance(metadata, dict):
        return "unknown"
    counts: Dict[str, int] = {}
    order: List[str] = []
    for component in metadata.get("components") or []:
        lang = _resolve_window_source_lang(component if isinstance(component, dict) else None)
        if not lang or lang == "unknown":
            continue
        counts[lang] = counts.get(lang, 0) + 1
        if lang not in order:
            order.append(lang)
    if not counts:
        return "unknown"
    return sorted(order, key=lambda item: (-counts.get(item, 0), order.index(item)))[0]


def _summarize_full_sequence_metadata(
    metadata: Optional[Dict[str, object]],
    *,
    window_seed: int,
) -> Dict[str, object]:
    if not isinstance(metadata, dict):
        return {
            "sampling_mode": "training_full_sequence",
            "window_seed": int(window_seed),
            "component_count": 0,
            "source_langs": [],
            "target_bytes": int(cfg.MODEL_WINDOW_BYTES),
        }
    return {
        "sampling_mode": "training_full_sequence",
        "window_seed": int(window_seed),
        "component_count": int(metadata.get("component_count", 0) or 0),
        "component_window_bytes": int(metadata.get("component_window_bytes", cfg.MODEL_WINDOW_BYTES) or cfg.MODEL_WINDOW_BYTES),
        "target_bytes": int(metadata.get("target_bytes", cfg.MODEL_WINDOW_BYTES) or cfg.MODEL_WINDOW_BYTES),
        "source_langs": _collect_full_sequence_source_langs(metadata),
    }


def _maybe_make_progress_bar(*, total: int, desc: str):
    if _tqdm is None or total <= 0:
        return None
    return _tqdm(
        total=int(total),
        desc=str(desc),
        unit="sample",
        dynamic_ncols=True,
        leave=True,
    )


def _update_progress_bar(progress_bar, *, attempts: int, skipped_seen: int, skipped_empty: int) -> None:
    if progress_bar is None:
        return
    progress_bar.set_postfix(
        {
            "attempts": int(attempts),
            "dup": int(skipped_seen),
            "empty": int(skipped_empty),
        },
        refresh=False,
    )


def _iter_raw_split_examples(
    data_root: Path,
    split: str,
    langs: Sequence[str],
    max_samples_per_lang: int,
    *,
    rng: np.random.Generator,
    seen_hashes: Optional[set[str]] = None,
) -> Iterable[tuple[str, int, str, str, Dict[str, object]]]:
    seen = seen_hashes if seen_hashes is not None else set()
    for lang in langs:
        ds_path = data_root / split / lang / "dataset"
        if not ds_path.exists():
            continue
        dataset = load_from_disk(str(ds_path))
        total = int(len(dataset))
        if total <= 0:
            continue
        limit = min(int(max_samples_per_lang), total)
        selected = 0
        skipped_seen = 0
        skipped_empty = 0
        shuffled_indices = [int(i) for i in rng.permutation(total).tolist()]
        for idx in shuffled_indices:
            if selected >= limit:
                break
            row = dataset[int(idx)]
            text = row.get("content") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                skipped_empty += 1
                continue
            normalized = _normalize_input_text(text)
            if not normalized:
                skipped_empty += 1
                continue
            sample_hash = _hash_text(normalized)
            if sample_hash in seen:
                skipped_seen += 1
                continue
            seen.add(sample_hash)
            selected += 1
            yield lang, int(idx), normalized, sample_hash, {"sampling_mode": "raw_dataset"}
        print(
            f"Sampling {lang}: selected={selected}/{limit}, "
            f"skipped_seen_hash={skipped_seen}, skipped_empty={skipped_empty}, dataset_size={total}.",
            flush=True,
        )


def _iter_augmented_split_examples(
    data_root: Path,
    split: str,
    langs: Sequence[str],
    max_samples_per_lang: int,
    *,
    rng: np.random.Generator,
    seen_hashes: Optional[set[str]] = None,
) -> Iterable[tuple[str, int, str, str, Dict[str, object]]]:
    seen = seen_hashes if seen_hashes is not None else set()
    target_per_lang = max(0, int(max_samples_per_lang))
    if target_per_lang <= 0:
        return

    dsets = prepare_dsets_by_lang_with_splits(
        str(data_root),
        include_languages=list(langs),
        verbose=False,
    )
    dsets_by_lang = dsets.get(split) or {}
    if not dsets_by_lang:
        print(
            f"No datasets available for augmented AL sampling on split='{split}'; "
            "falling back to raw dataset rows.",
            flush=True,
        )
        yield from _iter_raw_split_examples(
            data_root,
            split,
            langs,
            max_samples_per_lang,
            rng=rng,
            seen_hashes=seen,
        )
        return

    data_cfg = cfg.DataConfig(data_root=str(data_root))
    buckets = data_cfg.buckets()
    total_target = target_per_lang * max(1, len(dsets_by_lang))
    selected = 0
    skipped_seen = 0
    skipped_empty = 0
    attempts = 0
    max_attempts = max(total_target * 5, total_target + 16)
    progress_bar = _maybe_make_progress_bar(
        total=total_target,
        desc=f"Acquire {split} augmented samples",
    )
    try:
        while selected < total_target and attempts < max_attempts:
            attempts += 1
            if attempts == 1 or attempts % 16 == 0:
                _update_progress_bar(
                    progress_bar,
                    attempts=attempts,
                    skipped_seen=skipped_seen,
                    skipped_empty=skipped_empty,
                )
            window_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            py_state = random.getstate()
            np_state = np.random.get_state()
            try:
                random.seed(window_seed)
                np.random.seed(window_seed)
                target_len = int(random.choice(buckets))
                tokens, _, window_meta = make_training_window_with_metadata(
                    dsets_by_lang,
                    target_len,
                    data_cfg,
                )
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)

            normalized = _window_tokens_to_normalized_text(tokens)
            if not normalized:
                skipped_empty += 1
                continue
            sample_hash = _hash_text(normalized)
            if sample_hash in seen:
                skipped_seen += 1
                continue
            seen.add(sample_hash)
            sample_index = selected
            selected += 1
            if progress_bar is not None:
                progress_bar.update(1)
                _update_progress_bar(
                    progress_bar,
                    attempts=attempts,
                    skipped_seen=skipped_seen,
                    skipped_empty=skipped_empty,
                )
            representative_lang = _resolve_window_source_lang(window_meta)
            sample_meta = _summarize_window_metadata(window_meta, window_seed=window_seed)
            yield representative_lang, sample_index, normalized, sample_hash, sample_meta
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(
        "Augmented AL sampling: "
        f"selected={selected}/{total_target}, "
        f"skipped_seen_hash={skipped_seen}, "
        f"skipped_empty={skipped_empty}, "
        f"attempts={attempts}, "
        f"available_langs={len(dsets_by_lang)}.",
        flush=True,
    )


def _iter_augmented_full_split_examples(
    data_root: Path,
    split: str,
    langs: Sequence[str],
    max_samples_per_lang: int,
    *,
    rng: np.random.Generator,
    full_file_max_bytes: int,
    sample_workers: int = 1,
    sample_prefetch: int = 0,
    sample_seed: Optional[int] = None,
    seen_hashes: Optional[set[str]] = None,
) -> Iterable[tuple[str, int, str, str, Dict[str, object]]]:
    seen = seen_hashes if seen_hashes is not None else set()
    target_per_lang = max(0, int(max_samples_per_lang))
    if target_per_lang <= 0:
        return

    dsets = prepare_dsets_by_lang_with_splits(
        str(data_root),
        include_languages=list(langs),
        verbose=False,
    )
    dsets_by_lang = dsets.get(split) or {}
    if not dsets_by_lang:
        print(
            f"No datasets available for full-file AL sampling on split='{split}'; "
            "falling back to raw dataset rows.",
            flush=True,
        )
        yield from _iter_raw_split_examples(
            data_root,
            split,
            langs,
            max_samples_per_lang,
            rng=rng,
            seen_hashes=seen,
        )
        return

    data_cfg = cfg.DataConfig(data_root=str(data_root))
    total_target = target_per_lang * max(1, len(dsets_by_lang))
    selected = 0
    skipped_seen = 0
    skipped_empty = 0
    attempts = 0
    max_attempts = max(total_target * 5, total_target + 16)
    progress_bar = _maybe_make_progress_bar(
        total=total_target,
        desc=f"Acquire {split} full-file samples",
    )
    try:
        worker_count = max(1, int(sample_workers))
        if worker_count > 1:
            prefetch_size = int(sample_prefetch) if int(sample_prefetch) > 0 else max(16, worker_count * 8)
            prefetcher = FullSequenceSamplePrefetcher(
                data_root=str(data_root),
                split=str(split),
                langs=list(langs),
                target_len=int(full_file_max_bytes),
                num_workers=worker_count,
                prefetch=prefetch_size,
                base_seed=sample_seed,
            )
            try:
                while selected < total_target and attempts < max_attempts:
                    attempts += 1
                    if attempts == 1 or attempts % 16 == 0:
                        _update_progress_bar(
                            progress_bar,
                            attempts=attempts,
                            skipped_seen=skipped_seen,
                            skipped_empty=skipped_empty,
                        )
                    item = prefetcher.get(timeout=30.0)
                    status = str(item.get("status", "error"))
                    if status == "empty":
                        skipped_empty += 1
                        continue
                    if status != "ok":
                        raise RuntimeError(
                            f"Full-file AL sample prefetch failed: {item.get('message', 'unknown_error')}"
                        )
                    normalized = str(item.get("normalized", ""))
                    if not normalized:
                        skipped_empty += 1
                        continue
                    sample_hash = str(item.get("sample_hash", ""))
                    if not sample_hash:
                        skipped_empty += 1
                        continue
                    if sample_hash in seen:
                        skipped_seen += 1
                        continue
                    seen.add(sample_hash)
                    sample_index = selected
                    selected += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                        _update_progress_bar(
                            progress_bar,
                            attempts=attempts,
                            skipped_seen=skipped_seen,
                            skipped_empty=skipped_empty,
                        )
                    representative_lang = str(item.get("lang", "unknown"))
                    sample_meta = dict(item.get("metadata") or {})
                    yield representative_lang, sample_index, normalized, sample_hash, sample_meta
                return
            finally:
                prefetcher.close()
        while selected < total_target and attempts < max_attempts:
            attempts += 1
            if attempts == 1 or attempts % 16 == 0:
                _update_progress_bar(
                    progress_bar,
                    attempts=attempts,
                    skipped_seen=skipped_seen,
                    skipped_empty=skipped_empty,
                )
            window_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            py_state = random.getstate()
            np_state = np.random.get_state()
            try:
                random.seed(window_seed)
                np.random.seed(window_seed)
                tokens, _, full_meta = make_training_full_sequence_with_metadata(
                    dsets_by_lang,
                    data_cfg,
                    target_len=int(full_file_max_bytes),
                )
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)

            normalized = _window_tokens_to_normalized_text(tokens)
            if not normalized:
                skipped_empty += 1
                continue
            sample_hash = _hash_text(normalized)
            if sample_hash in seen:
                skipped_seen += 1
                continue
            seen.add(sample_hash)
            sample_index = selected
            selected += 1
            if progress_bar is not None:
                progress_bar.update(1)
                _update_progress_bar(
                    progress_bar,
                    attempts=attempts,
                    skipped_seen=skipped_seen,
                    skipped_empty=skipped_empty,
                )
            representative_lang = _resolve_full_sequence_source_lang(full_meta)
            sample_meta = _summarize_full_sequence_metadata(full_meta, window_seed=window_seed)
            sample_meta["full_file_mode"] = True
            yield representative_lang, sample_index, normalized, sample_hash, sample_meta
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(
        "Augmented AL full-file sampling: "
        f"selected={selected}/{total_target}, "
        f"skipped_seen_hash={skipped_seen}, "
        f"skipped_empty={skipped_empty}, "
        f"attempts={attempts}, "
        f"available_langs={len(dsets_by_lang)}.",
        flush=True,
    )


def _iter_split_examples(
    data_root: Path,
    split: str,
    langs: Sequence[str],
    max_samples_per_lang: int,
    *,
    rng: np.random.Generator,
    full_files: bool = False,
    full_file_max_bytes: int = 10000,
    sample_workers: int = 1,
    sample_prefetch: int = 0,
    sample_seed: Optional[int] = None,
    seen_hashes: Optional[set[str]] = None,
) -> Iterable[tuple[str, int, str, str, Dict[str, object]]]:
    if str(split) in {"train", "val", "test"}:
        if full_files:
            yield from _iter_augmented_full_split_examples(
                data_root,
                split,
                langs,
                max_samples_per_lang,
                rng=rng,
                full_file_max_bytes=int(full_file_max_bytes),
                sample_workers=int(sample_workers),
                sample_prefetch=int(sample_prefetch),
                sample_seed=sample_seed,
                seen_hashes=seen_hashes,
            )
        else:
            yield from _iter_augmented_split_examples(
                data_root,
                split,
                langs,
                max_samples_per_lang,
                rng=rng,
                seen_hashes=seen_hashes,
            )
        return
    yield from _iter_raw_split_examples(
        data_root,
        split,
        langs,
        max_samples_per_lang,
        rng=rng,
        seen_hashes=seen_hashes,
    )


def _build_snippets_from_prediction(
    *,
    normalized_text: str,
    lang: str,
    sample_index: int,
    sample_hash: str,
    sample_metadata: Optional[Dict[str, object]],
    char_labels: Sequence[int],
    char_probs: Sequence[Dict[str, float]],
    max_candidates_per_sample: int,
    context_chars: int,
    min_score: float,
    full_files: bool = False,
) -> tuple[List[tuple[BoundarySnippet, CandidateSpan]], List[Dict[str, object]]]:
    normalized = normalized_text
    if not char_labels or not char_probs:
        return [], []
    probs = _probs_dicts_to_matrix(char_probs, cfg.NUM_CLASSES)
    pred = np.asarray(char_labels, dtype=np.int32)
    id2lang = cfg.ID2LANG
    pred_label_names = [id2lang.get(int(lbl), "other") for lbl in pred.tolist()]
    sample_pred_segments = _segments_from_label_names(pred_label_names)
    candidates = select_candidate_spans(
        probs,
        labels=pred,
        context_chars=int(context_chars),
        top_k=int(max_candidates_per_sample),
        min_score=float(min_score),
        other_id=getattr(cfg, "OTHER_CLASS_INDEX", None),
    )
    if bool(full_files):
        if not candidates:
            return [], sample_pred_segments
        top_cand = max(candidates, key=lambda item: float(item.score))
        trigger_ranges = [
            {
                "start": int(cand.start),
                "end": int(cand.end),
                "boundary": int(cand.boundary),
                "score": float(cand.score),
                "entropy_mean": float(cand.entropy_mean),
                "flip_rate": float(cand.flip_rate),
                "is_other_boundary": bool(cand.is_other_boundary),
                "left_label_id": int(cand.left_label),
                "right_label_id": int(cand.right_label),
                "left_label": id2lang.get(int(cand.left_label), "other"),
                "right_label": id2lang.get(int(cand.right_label), "other"),
            }
            for cand in candidates
        ]
        snippet = BoundarySnippet(
            snippet_id=_make_sample_prompt_id(sample_hash),
            text=normalized,
            global_start=0,
            global_end=int(len(normalized)),
            boundary=int(top_cand.boundary),
            predicted_labels=pred_label_names,
            metadata={
                "source_lang": lang,
                "sample_index": int(sample_index),
                "sample_hash": sample_hash,
                "boundary_global": int(top_cand.boundary),
                "score": float(top_cand.score),
                "entropy_mean": float(top_cand.entropy_mean),
                "flip_rate": float(top_cand.flip_rate),
                "is_other_boundary": bool(top_cand.is_other_boundary),
                "left_label_id": int(top_cand.left_label),
                "right_label_id": int(top_cand.right_label),
                "left_label": id2lang.get(int(top_cand.left_label), "other"),
                "right_label": id2lang.get(int(top_cand.right_label), "other"),
                "trigger_ranges": trigger_ranges,
                "full_file_mode": True,
                **(dict(sample_metadata) if isinstance(sample_metadata, dict) else {}),
            },
        )
        return [(snippet, top_cand)], sample_pred_segments
    out: List[tuple[BoundarySnippet, CandidateSpan]] = []
    for cand in candidates:
        snippet_labels = [
            id2lang.get(int(lbl), "other")
            for lbl in pred[cand.start : cand.end].tolist()
        ]
        boundary_in_snippet = int(cand.boundary - cand.start)
        snippet_id = _make_snippet_id(
            sample_hash,
            int(cand.start),
            int(cand.end),
            int(cand.boundary),
        )
        snippet = BoundarySnippet(
            snippet_id=snippet_id,
            text=normalized[cand.start : cand.end],
            global_start=int(cand.start),
            global_end=int(cand.end),
            boundary=boundary_in_snippet,
            predicted_labels=snippet_labels,
            metadata={
                "source_lang": lang,
                "sample_index": int(sample_index),
                "sample_hash": sample_hash,
                "boundary_global": int(cand.boundary),
                "score": float(cand.score),
                "entropy_mean": float(cand.entropy_mean),
                "flip_rate": float(cand.flip_rate),
                "is_other_boundary": bool(cand.is_other_boundary),
                "left_label_id": int(cand.left_label),
                "right_label_id": int(cand.right_label),
                "left_label": id2lang.get(int(cand.left_label), "other"),
                "right_label": id2lang.get(int(cand.right_label), "other"),
                **(dict(sample_metadata) if isinstance(sample_metadata, dict) else {}),
            },
        )
        out.append((snippet, cand))
    return out, sample_pred_segments


def _build_snippets_for_sample(
    *,
    normalized_text: str,
    lang: str,
    sample_index: int,
    predictor: Predictor,
    max_candidates_per_sample: int,
    context_chars: int,
    min_score: float,
    full_files: bool = False,
    sample_hash: Optional[str] = None,
    sample_metadata: Optional[Dict[str, object]] = None,
) -> tuple[List[tuple[BoundarySnippet, CandidateSpan]], List[Dict[str, object]]]:
    normalized = normalized_text
    _, char_labels, char_probs, _ = predictor.segment_text(normalized, min_run_chars=1)
    resolved_hash = str(sample_hash) if sample_hash else _hash_text(normalized)
    return _build_snippets_from_prediction(
        normalized_text=normalized,
        lang=lang,
        sample_index=sample_index,
        sample_hash=resolved_hash,
        sample_metadata=sample_metadata,
        char_labels=char_labels,
        char_probs=char_probs,
        max_candidates_per_sample=max_candidates_per_sample,
        context_chars=context_chars,
        min_score=min_score,
        full_files=bool(full_files),
    )


def _build_snippets_for_samples_batch(
    *,
    samples: Sequence[tuple],
    predictor: Predictor,
    max_candidates_per_sample: int,
    context_chars: int,
    min_score: float,
    full_files: bool = False,
) -> List[tuple[List[tuple[BoundarySnippet, CandidateSpan]], List[Dict[str, object]]]]:
    if not samples:
        return []
    texts = [str(sample[2]) for sample in samples]
    segmented = predictor.segment_texts(texts, min_run_chars=1)
    if len(segmented) != len(samples):
        raise RuntimeError(
            "Predictor returned a mismatched number of batched segmentation results: "
            f"expected {len(samples)}, got {len(segmented)}."
        )
    out: List[tuple[List[tuple[BoundarySnippet, CandidateSpan]], List[Dict[str, object]]]] = []
    for sample, result in zip(samples, segmented):
        lang = str(sample[0])
        sample_index = int(sample[1])
        normalized = str(sample[2])
        sample_hash = str(sample[3])
        sample_metadata = dict(sample[4]) if len(sample) > 4 and isinstance(sample[4], dict) else None
        _, char_labels, char_probs, _ = result
        out.append(
            _build_snippets_from_prediction(
                normalized_text=normalized,
                lang=lang,
                sample_index=int(sample_index),
                sample_hash=str(sample_hash),
                sample_metadata=sample_metadata,
                char_labels=char_labels,
                char_probs=char_probs,
                max_candidates_per_sample=max_candidates_per_sample,
                context_chars=context_chars,
                min_score=min_score,
                full_files=bool(full_files),
            )
        )
    return out


def _limit_snippets_for_oracle_requests(
    snippets: Sequence[BoundarySnippet],
    *,
    score_by_id: Dict[str, float],
    oracle_batch_size: int,
    max_oracle_requests: Optional[int],
) -> List[BoundarySnippet]:
    if max_oracle_requests is None:
        return list(snippets)
    request_cap = max(0, int(max_oracle_requests))
    batch_size = max(1, int(oracle_batch_size))
    snippet_cap = request_cap * batch_size
    if snippet_cap <= 0:
        return []
    if len(snippets) <= snippet_cap:
        return list(snippets)
    ordered = sorted(
        snippets,
        key=lambda snippet: (float(score_by_id.get(snippet.snippet_id, 0.0)), snippet.snippet_id),
        reverse=True,
    )
    return ordered[:snippet_cap]


def _configure_oracle_parallel_requests(
    oracle: object,
    *,
    max_oracle_requests: Optional[int],
) -> int:
    parallel_requests = 1
    if max_oracle_requests is not None:
        parallel_requests = max(1, int(max_oracle_requests))
    if hasattr(oracle, "max_parallel_requests"):
        setattr(oracle, "max_parallel_requests", int(parallel_requests))
    return int(parallel_requests)


def _looks_like_oracle_unavailable_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "503" in message
        or "status': 'unavailable'" in message
        or '"status": "unavailable"' in message
        or "status': 'service_unavailable'" in message
        or '"status": "service_unavailable"' in message
        or "currently experiencing high demand" in message
        or "unavailable" in message and "oracle failed completely" in message
    )


def _print_oracle_unavailable_banner(*, error: str, queried_count: int, round_id: str) -> None:
    banner = "=" * 100
    print(banner, flush=True)
    print("!!! GEMINI / ORACLE API UNAVAILABLE - SKIPPING NEW LABELS FOR THIS ROUND !!!", flush=True)
    print(
        "!!! NO NEW LABELS WERE STORED THIS ROUND BECAUSE THE ORACLE API IS CURRENTLY DOWN / OVERLOADED !!!",
        flush=True,
    )
    print(
        f"!!! round_id={round_id} queried_snippets={int(queried_count)} error={error}",
        flush=True,
    )
    print(banner, flush=True)


def run_one_round(
    *,
    ckpt_path: str,
    data_root: str,
    split: str,
    langs: Sequence[str],
    store_path: str,
    oracle,
    max_samples_per_lang: int = 16,
    max_candidates_per_sample: int = 3,
    context_chars: int = 250,
    min_score: float = 0.2,
    predictor_kwargs: Optional[Dict[str, object]] = None,
    sample_seed: Optional[int] = None,
    skip_seen_hashes: bool = True,
    max_oracle_requests: Optional[int] = 3,
    full_files: bool = False,
    full_file_max_bytes: int = 10000,
    sample_workers: int = 1,
    sample_prefetch: int = 0,
) -> Dict[str, object]:
    context_chars = min(250, max(8, int(context_chars)))
    full_files = bool(full_files)
    full_file_max_bytes = max(1, int(full_file_max_bytes))
    round_id = datetime.now(timezone.utc).strftime("al-%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    predictor_args = argparse.Namespace(
        ckpt=ckpt_path,
        arch=None,
        model_dim=None,
        channels=None,
        mamba_layers=None,
        mamba_d_state=None,
        mamba_expand=None,
        mamba_dt_rank=None,
        mamba_conv=None,
        mamba_bidirectional=None,
        dtype="bfloat16",
        chunk=DEFAULT_CHUNK_SIZE,
        other_threshold=0.2,
        predict_batch_size=12,
    )
    if predictor_kwargs:
        for key, value in predictor_kwargs.items():
            setattr(predictor_args, key, value)
    if full_files:
        predictor_args.chunk = int(full_file_max_bytes)
    predictor = _build_predictor(predictor_args)
    store = LabelStore(store_path)
    rng = np.random.default_rng(sample_seed)
    scoring_text_batch_size = max(1, int(getattr(predictor, "inference_batch_size", 12)))
    seen_hashes: set[str] = set()
    if skip_seen_hashes:
        seen_hashes = store.existing_sample_hashes()
        print(
            f"Loaded {len(seen_hashes)} previously-seen sample hashes from store; repeats will be skipped.",
            flush=True,
        )

    snippets: List[BoundarySnippet] = []
    score_by_id: Dict[str, float] = {}
    pred_segments_by_id: Dict[str, List[Dict[str, object]]] = {}
    source_by_id: Dict[str, Dict[str, object]] = {}
    inference_rows: List[StoredInferenceSample] = []
    total_samples_seen = 0
    total_samples_with_candidates = 0
    total_candidate_triggers = 0

    def _process_scored_batch(samples_batch: Sequence[tuple]) -> None:
        nonlocal total_samples_with_candidates, total_candidate_triggers
        if not samples_batch:
            return
        batch_results = _build_snippets_for_samples_batch(
            samples=samples_batch,
            predictor=predictor,
            max_candidates_per_sample=max_candidates_per_sample,
            context_chars=context_chars,
            min_score=min_score,
            full_files=bool(full_files),
        )
        for sample, (sample_snippets, sample_pred_segments) in zip(samples_batch, batch_results):
            lang = str(sample[0])
            sample_idx = int(sample[1])
            normalized = str(sample[2])
            sample_hash = str(sample[3])
            sample_metadata = dict(sample[4]) if len(sample) > 4 and isinstance(sample[4], dict) else {}
            if sample_snippets:
                total_samples_with_candidates += 1
            trigger_ranges: List[Dict[str, object]] = []
            sample_candidate_count = 0
            for snippet, cand in sample_snippets:
                snippets.append(snippet)
                score_by_id[snippet.snippet_id] = float(cand.score)
                pred_segments_by_id[snippet.snippet_id] = _segments_from_label_names(
                    snippet.predicted_labels
                )
                source_by_id[snippet.snippet_id] = dict(snippet.metadata)
                if bool(full_files):
                    trigger_ranges = list(snippet.metadata.get("trigger_ranges") or [])
                    sample_candidate_count = max(sample_candidate_count, len(trigger_ranges))
                else:
                    trigger_ranges.append(
                        {
                            "start": int(cand.start),
                            "end": int(cand.end),
                            "boundary": int(cand.boundary),
                            "score": float(cand.score),
                            "entropy_mean": float(cand.entropy_mean),
                            "flip_rate": float(cand.flip_rate),
                            "is_other_boundary": bool(cand.is_other_boundary),
                            "left_label_id": int(cand.left_label),
                            "right_label_id": int(cand.right_label),
                            "left_label": cfg.ID2LANG.get(int(cand.left_label), "other"),
                            "right_label": cfg.ID2LANG.get(int(cand.right_label), "other"),
                        }
                    )
                    sample_candidate_count += 1
            total_candidate_triggers += int(sample_candidate_count)

            inference_rows.append(
                StoredInferenceSample(
                    round_id=round_id,
                    source_split=str(split),
                    source_lang=str(lang),
                    sample_index=int(sample_idx),
                    sample_hash=str(sample_hash),
                    sample_text=normalized,
                    char_count=int(len(normalized)),
                    queried_for_oracle=False,
                    candidate_count=int(sample_candidate_count),
                    trigger_ranges=trigger_ranges,
                    predicted_segments=sample_pred_segments,
                    metadata={
                        "max_candidates_per_sample": int(max_candidates_per_sample),
                        "min_score": float(min_score),
                        "context_chars": int(context_chars),
                        "full_file_mode": bool(full_files),
                        "full_file_max_bytes": int(full_file_max_bytes),
                        **sample_metadata,
                    },
                )
            )

    pending_samples: List[tuple] = []
    for lang, sample_idx, normalized, sample_hash, sample_metadata in _iter_split_examples(
        Path(data_root),
        split,
        langs,
        max_samples_per_lang,
        rng=rng,
        full_files=bool(full_files),
        full_file_max_bytes=int(full_file_max_bytes),
        sample_workers=int(sample_workers),
        sample_prefetch=int(sample_prefetch),
        sample_seed=sample_seed,
        seen_hashes=seen_hashes if skip_seen_hashes else None,
    ):
        total_samples_seen += 1
        pending_samples.append(
            (
                str(lang),
                int(sample_idx),
                normalized,
                str(sample_hash),
                dict(sample_metadata) if isinstance(sample_metadata, dict) else {},
            )
        )
        if len(pending_samples) >= scoring_text_batch_size:
            _process_scored_batch(pending_samples)
            pending_samples = []

    if pending_samples:
        _process_scored_batch(pending_samples)

    inference_inserted = store.add_inference_samples_many(inference_rows)
    print(
        f"Logged {inference_inserted}/{len(inference_rows)} inference samples "
        f"(candidate_samples={total_samples_with_candidates}, "
        f"without_candidates={max(0, inference_inserted - total_samples_with_candidates)}).",
        flush=True,
    )

    if not snippets:
        print(
            "No candidate snippets selected for oracle refinement "
            f"(samples_scanned={total_samples_seen}, split={split}, min_score={min_score}).",
            flush=True,
        )
        return {
            "round_id": round_id,
            "status": "no_candidates",
            "samples": 0,
            "inference_samples": int(inference_inserted),
            "stored": 0,
            "store_path": str(Path(store_path).resolve()),
            "timestamp": _now_utc(),
        }

    oracle_batch_size = max(1, int(getattr(oracle, "batch_size", 1)))
    queried_snippets = _limit_snippets_for_oracle_requests(
        snippets,
        score_by_id=score_by_id,
        oracle_batch_size=oracle_batch_size,
        max_oracle_requests=max_oracle_requests,
    )
    if max_oracle_requests is None:
        print(
            f"Oracle request cap disabled (--unlimited-oracle): querying all {len(queried_snippets)} snippets.",
            flush=True,
        )
    else:
        snippet_cap = max(0, int(max_oracle_requests)) * oracle_batch_size
        print(
            "Oracle request cap active: "
            f"max_requests={max(0, int(max_oracle_requests))}, "
            f"batch_size={oracle_batch_size}, snippet_cap={snippet_cap}, "
            f"selected={len(queried_snippets)}/{len(snippets)}.",
            flush=True,
        )
        if len(queried_snippets) < len(snippets):
            print(
                "Oracle cap selected top-scoring snippets only to reduce request cost.",
                flush=True,
            )

    if not queried_snippets:
        print(
            "Oracle request cap resulted in 0 queried snippets; skipping oracle refinement.",
            flush=True,
        )
        return {
            "round_id": round_id,
            "status": "oracle_capped",
            "samples": 0,
            "candidate_snippets": int(total_candidate_triggers),
            "inference_samples": int(inference_inserted),
            "stored": 0,
            "store_path": str(Path(store_path).resolve()),
            "timestamp": _now_utc(),
        }

    queried_scores = np.asarray(
        [float(score_by_id.get(snippet.snippet_id, 0.0)) for snippet in queried_snippets],
        dtype=np.float64,
    )
    queried_entropy = np.asarray(
        [float(source_by_id.get(snippet.snippet_id, {}).get("entropy_mean", 0.0)) for snippet in queried_snippets],
        dtype=np.float64,
    )
    queried_flip = np.asarray(
        [float(source_by_id.get(snippet.snippet_id, {}).get("flip_rate", 0.0)) for snippet in queried_snippets],
        dtype=np.float64,
    )
    print(
        "Requesting oracle refinement for "
        f"{len(queried_snippets)} snippets from {total_samples_with_candidates}/{total_samples_seen} samples "
        f"(split={split}, min_score={min_score}).",
        flush=True,
    )
    print(
        "Selected spans stats: "
        f"score(avg/min/max)={queried_scores.mean():.3f}/{queried_scores.min():.3f}/{queried_scores.max():.3f}, "
        f"entropy(avg/min/max)={queried_entropy.mean():.3f}/{queried_entropy.min():.3f}/{queried_entropy.max():.3f}, "
        f"flip_rate(avg/min/max)={queried_flip.mean():.3f}/{queried_flip.min():.3f}/{queried_flip.max():.3f}",
        flush=True,
    )
    parallel_requests = _configure_oracle_parallel_requests(
        oracle,
        max_oracle_requests=max_oracle_requests,
    )
    oracle_batch_size = max(1, int(getattr(oracle, "batch_size", 1)))
    estimated_request_batches = int(
        (len(queried_snippets) + oracle_batch_size - 1) // oracle_batch_size
    )
    print(
        "Oracle dispatch config: "
        f"batch_size={oracle_batch_size}, "
        f"parallel_requests={parallel_requests}, "
        f"estimated_batches={estimated_request_batches}.",
        flush=True,
    )

    try:
        refined = oracle.annotate(queried_snippets)
    except RuntimeError as exc:
        if not _looks_like_oracle_unavailable_error(exc):
            raise
        error_text = str(exc).strip()
        _print_oracle_unavailable_banner(
            error=error_text,
            queried_count=len(queried_snippets),
            round_id=round_id,
        )
        return {
            "round_id": round_id,
            "status": "oracle_unavailable",
            "samples": 0,
            "candidate_snippets": int(total_candidate_triggers),
            "inference_samples": int(inference_inserted),
            "stored": 0,
            "oracle_model_outputs": 0,
            "oracle_fallback_snippets": 0,
            "oracle_skipped_snippets": 0,
            "oracle_error": error_text,
            "store_path": str(Path(store_path).resolve()),
            "timestamp": _now_utc(),
        }
    print(
        f"Oracle returned refinements for {len(refined)}/{len(queried_snippets)} snippets.",
        flush=True,
    )
    oracle_name = str(getattr(oracle, "name", "oracle"))
    oracle_sources = getattr(oracle, "last_snippet_sources", None)
    if not isinstance(oracle_sources, dict):
        oracle_sources = {}
    oracle_batches = getattr(oracle, "last_batches", None)
    failed_batches = []
    skipped_parse_failed_total = 0
    skipped_missing_total = 0
    if isinstance(oracle_batches, list) and oracle_batches:
        failed_batches = [b for b in oracle_batches if str(b.get("status", "")).endswith("failed")]
        requested_total = int(sum(int(b.get("requested", 0)) for b in oracle_batches))
        model_total = int(sum(int(b.get("model_output_count", 0)) for b in oracle_batches))
        missing_initial_total = int(sum(int(b.get("initial_missing_snippets", 0)) for b in oracle_batches))
        parse_failed_initial_total = int(
            sum(int(b.get("initial_parse_failed_snippets", 0)) for b in oracle_batches)
        )
        missing_recovered_total = int(sum(int(b.get("recovered_missing_snippets", 0)) for b in oracle_batches))
        parse_failed_recovered_total = int(
            sum(int(b.get("recovered_parse_failed_snippets", 0)) for b in oracle_batches)
        )
        skipped_parse_failed_total = int(
            sum(int(b.get("final_parse_failed_snippets", 0)) for b in oracle_batches)
        )
        skipped_missing_total = int(
            sum(int(b.get("final_missing_snippets", 0)) for b in oracle_batches)
        )
        print(
            "Oracle batch diagnostics: "
            f"batches={len(oracle_batches)}, requested={requested_total}, "
            f"model_outputs={model_total}, failed_batches={len(failed_batches)}, "
            f"missing_initial={missing_initial_total}, missing_recovered={missing_recovered_total}, "
            f"parse_failed_initial={parse_failed_initial_total}, "
            f"parse_failed_recovered={parse_failed_recovered_total}, "
            f"parse_failed_skipped={skipped_parse_failed_total}, "
            f"missing_skipped={skipped_missing_total}",
            flush=True,
        )
        if failed_batches:
            first = failed_batches[0]
            log_path = str(first.get("log_path", "")).strip()
            err = str(first.get("error", "")).strip()
            msg = "Oracle failure detected; fallback predictions were used for affected snippets."
            if log_path:
                msg += f" See {log_path}."
            if err:
                msg += f" First error: {err}"
            print(msg, flush=True)

    if oracle_name != "stub":
        allowed_non_model_states = {"skipped_parse_failed", "skipped_missing"}
        non_model = [
            (snippet.snippet_id, str(oracle_sources.get(snippet.snippet_id, "unknown")))
            for snippet in queried_snippets
            if str(oracle_sources.get(snippet.snippet_id, "unknown")) != "model"
        ]
        unexpected_non_model = [
            (snippet_id, state) for snippet_id, state in non_model if state not in allowed_non_model_states
        ]
        if non_model:
            state_counts = {}
            for _, state in non_model:
                state_counts[state] = state_counts.get(state, 0) + 1
            print(
                "Oracle skipped unresolved snippets after retries: "
                f"count={len(non_model)}, "
                "states="
                + ",".join(f"{key}:{state_counts[key]}" for key in sorted(state_counts))
                + f", first_snippet={non_model[0][0]}",
                flush=True,
            )
        if failed_batches or unexpected_non_model:
            details = []
            if failed_batches:
                first_failed = failed_batches[0]
                details.append(f"failed_batches={len(failed_batches)}")
                err = str(first_failed.get("error", "")).strip()
                if err:
                    details.append(f"first_error={err}")
                log_path = str(first_failed.get("log_path", "")).strip()
                if log_path:
                    details.append(f"log={log_path}")
            if unexpected_non_model:
                state_counts = {}
                for _, state in unexpected_non_model:
                    state_counts[state] = state_counts.get(state, 0) + 1
                details.append(f"non_model_snippets={len(unexpected_non_model)}")
                details.append(
                    "states="
                    + ",".join(f"{key}:{state_counts[key]}" for key in sorted(state_counts))
                )
                details.append(f"first_snippet={unexpected_non_model[0][0]}")
            raise RuntimeError(
                "Oracle refinement failed or returned fallback outputs; aborting round. "
                + " | ".join(details)
            )

    if oracle_sources:
        model_output_snippets = int(
            sum(1 for snippet in queried_snippets if oracle_sources.get(snippet.snippet_id) == "model")
        )
        oracle_fallback_snippets = int(
            sum(
                1
                for snippet in queried_snippets
                if str(oracle_sources.get(snippet.snippet_id, "")).startswith("fallback_")
            )
        )
        oracle_skipped_snippets = int(
            sum(
                1
                for snippet in queried_snippets
                if str(oracle_sources.get(snippet.snippet_id, "")) in {"skipped_parse_failed", "skipped_missing"}
            )
        )
    else:
        model_output_snippets = len(queried_snippets)
        oracle_fallback_snippets = 0
        oracle_skipped_snippets = 0

    rows: List[StoredRefinement] = []
    total_open_set_segments = 0
    for snippet in queried_snippets:
        if oracle_sources and oracle_sources.get(snippet.snippet_id) != "model":
            continue
        segs = refined.get(snippet.snippet_id, [])
        refined_segments = []
        open_set_labels = []
        for seg in segs:
            segment_row = {"start": int(seg.start), "end": int(seg.end), "label": str(seg.label)}
            open_set_label = getattr(seg, "raw_label", None)
            if isinstance(open_set_label, str) and open_set_label:
                segment_row["open_set_label"] = open_set_label
                open_set_labels.append(open_set_label)
                total_open_set_segments += 1
            refined_segments.append(segment_row)

        source_meta = dict(source_by_id.get(snippet.snippet_id, {}))
        if open_set_labels:
            source_meta["oracle_open_set_labels"] = sorted(set(open_set_labels))
            source_meta["oracle_open_set_segments"] = int(len(open_set_labels))
        rows.append(
            StoredRefinement(
                round_id=round_id,
                source_split=str(split),
                source_lang=str(source_meta.get("source_lang", "unknown")),
                sample_index=int(source_meta.get("sample_index", -1)),
                sample_hash=str(source_meta.get("sample_hash", "")),
                boundary_index=int(source_meta.get("boundary_global", -1)),
                snippet_start=int(snippet.global_start),
                snippet_end=int(snippet.global_end),
                snippet_text=snippet.text,
                oracle_name=getattr(oracle, "name", "oracle"),
                oracle_model=str(getattr(oracle, "model", "")),
                oracle_run_id="",
                status="ok",
                acquisition_score=float(score_by_id.get(snippet.snippet_id, 0.0)),
                predicted_segments=pred_segments_by_id.get(snippet.snippet_id, []),
                refined_segments=refined_segments,
                metadata=source_meta,
            )
        )

    if not rows:
        raise RuntimeError("No oracle-backed refinements available to store for this round.")

    if total_open_set_segments > 0:
        print(
            f"Mapped {total_open_set_segments} non-allowed oracle segment labels to 'other' "
            "and preserved their raw labels in metadata/open_set_label.",
            flush=True,
        )
    inserted = store.add_many(rows)
    queried_marked = store.mark_inference_samples_queried(
        round_id=round_id,
        sample_keys={(row.sample_hash, row.sample_index) for row in rows},
    )
    print(
        f"Marked {queried_marked} inference samples as queried after storing refinements.",
        flush=True,
    )
    # ── Aggregate oracle batch-level diagnostics for wandb logging ──
    _oracle_stats: Dict[str, object] = {}
    if isinstance(oracle_batches, list) and oracle_batches:
        _oracle_stats = {
            "llm_requests": int(len(oracle_batches)),
            "llm_retry_requests": int(
                sum(int(b.get("missing_retry_requests", 0)) for b in oracle_batches)
                + sum(int(b.get("parse_failed_retry_requests", 0)) for b in oracle_batches)
            ),
            "llm_rate_limit_retries": int(
                sum(int(b.get("rate_limit_retries", 0)) for b in oracle_batches)
            ),
            "llm_total_requests": int(
                len(oracle_batches)
                + sum(int(b.get("missing_retry_requests", 0)) for b in oracle_batches)
                + sum(int(b.get("parse_failed_retry_requests", 0)) for b in oracle_batches)
            ),
            "llm_prompt_tokens": int(
                sum(int(b.get("prompt_tokens", 0)) for b in oracle_batches)
            ),
            "llm_candidates_tokens": int(
                sum(int(b.get("candidates_tokens", 0)) for b in oracle_batches)
            ),
            "llm_total_tokens": int(
                sum(int(b.get("total_tokens", 0)) for b in oracle_batches)
            ),
            "llm_failed_batches": int(len(failed_batches)),
            "llm_initial_missing": int(
                sum(int(b.get("initial_missing_snippets", 0)) for b in oracle_batches)
            ),
            "llm_initial_parse_failed": int(
                sum(int(b.get("initial_parse_failed_snippets", 0)) for b in oracle_batches)
            ),
            "llm_recovered_missing": int(
                sum(int(b.get("recovered_missing_snippets", 0)) for b in oracle_batches)
            ),
            "llm_recovered_parse_failed": int(
                sum(int(b.get("recovered_parse_failed_snippets", 0)) for b in oracle_batches)
            ),
            "llm_final_parse_failed": int(
                sum(int(b.get("final_parse_failed_snippets", 0)) for b in oracle_batches)
            ),
            "llm_final_missing": int(
                sum(int(b.get("final_missing_snippets", 0)) for b in oracle_batches)
            ),
        }
    return {
        "round_id": round_id,
        "status": "ok",
        "samples": len(queried_snippets),
        "candidate_snippets": int(total_candidate_triggers),
        "inference_samples": int(inference_inserted),
        "stored": inserted,
        "store_path": str(Path(store_path).resolve()),
        "oracle": getattr(oracle, "name", "oracle"),
        "oracle_model": str(getattr(oracle, "model", "")),
        "oracle_model_outputs": int(model_output_snippets),
        "oracle_fallback_snippets": int(oracle_fallback_snippets),
        "oracle_skipped_snippets": int(oracle_skipped_snippets),
        "timestamp": _now_utc(),
        **_oracle_stats,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one active-learning boundary-refinement round.")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--data-root", type=str, default="downloader/arrow_out")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--langs", type=str, default=None, help="Comma-separated language subset.")
    parser.add_argument("--store", type=str, default="active_learning/label_store.sqlite")
    parser.add_argument("--full-files", action="store_true", help="Use full-file/long-sample AL acquisition and oracle querying.")
    parser.add_argument("--full-file-max-bytes", type=int, default=10000)
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=1,
        help="Number of CPU worker processes for long-sample assembly (full-file mode).",
    )
    parser.add_argument(
        "--sample-prefetch",
        type=int,
        default=0,
        help="Maximum prefetched long samples to keep queued ahead of scoring.",
    )
    parser.add_argument("--max-samples-per-lang", type=int, default=16)
    parser.add_argument("--max-candidates-per-sample", type=int, default=3)
    parser.add_argument("--context-chars", type=int, default=250)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--oracle", choices=("stub", "gemini"), default="gemini")
    parser.add_argument("--gemini-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument(
        "--gemini-thinking-level",
        type=str,
        choices=("minimal", "low", "medium", "high"),
        default="medium",
    )
    parser.add_argument("--gemini-batch-size", type=int, default=32)
    parser.add_argument("--gemini-rate-limit-sleep-seconds", type=float, default=65.0)
    parser.add_argument("--gemini-rate-limit-max-retries", type=int, default=8)
    parser.add_argument("--gemini-missing-snippet-retries", type=int, default=2)
    parser.add_argument(
        "--max-oracle-requests",
        type=int,
        default=3,
        help="Maximum number of oracle requests per round (default: 3).",
    )
    parser.add_argument(
        "--unlimited-oracle",
        action="store_true",
        help="Disable oracle request cap and query all candidate snippets (previous behavior).",
    )
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument(
        "--allow-repeat-hashes",
        action="store_true",
        help="Allow samples whose sample_hash already exists in the label store.",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--proxy", type=str, default=None)

    parser.add_argument("--arch", type=str, default=None, choices=("unet1d", "mamba"))
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--mamba-layers", type=int, default=None)
    parser.add_argument("--mamba-d-state", type=int, default=None)
    parser.add_argument("--mamba-expand", type=int, default=None)
    parser.add_argument("--mamba-dt-rank", type=int, default=None)
    parser.add_argument("--mamba-conv", type=int, default=None)
    parser.add_argument("--mamba-bidirectional", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--other-threshold", type=float, default=0.2)
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=12,
        help="Window batch size for local model scoring during AL acquisition.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    langs = _parse_langs(args.langs)
    if not langs:
        raise ValueError("No valid languages selected.")

    if args.oracle == "gemini":
        oracle = GeminiBoundaryOracle(
            model=args.gemini_model,
            thinking_level=args.gemini_thinking_level,
            api_key=args.api_key,
            batch_size=args.gemini_batch_size,
            proxy=args.proxy,
            rate_limit_sleep_seconds=args.gemini_rate_limit_sleep_seconds,
            rate_limit_max_retries=args.gemini_rate_limit_max_retries,
            missing_snippet_retries=args.gemini_missing_snippet_retries,
        )
    else:
        oracle = StubOracle()
        raise NotImplementedError("no reliable oracle set")

    predictor_kwargs = {
        "arch": args.arch,
        "model_dim": args.model_dim,
        "channels": args.channels,
        "mamba_layers": args.mamba_layers,
        "mamba_d_state": args.mamba_d_state,
        "mamba_expand": args.mamba_expand,
        "mamba_dt_rank": args.mamba_dt_rank,
        "mamba_conv": args.mamba_conv,
        "mamba_bidirectional": args.mamba_bidirectional,
        "dtype": args.dtype,
        "chunk": args.chunk,
        "other_threshold": args.other_threshold,
        "predict_batch_size": args.predict_batch_size,
    }
    summary = run_one_round(
        ckpt_path=args.ckpt,
        data_root=args.data_root,
        split=args.split,
        langs=langs,
        store_path=args.store,
        oracle=oracle,
        max_samples_per_lang=args.max_samples_per_lang,
        max_candidates_per_sample=args.max_candidates_per_sample,
        context_chars=min(250, max(8, int(args.context_chars))),
        min_score=float(args.min_score),
        predictor_kwargs=predictor_kwargs,
        sample_seed=args.sample_seed,
        skip_seen_hashes=not bool(args.allow_repeat_hashes),
        max_oracle_requests=None if bool(args.unlimited_oracle) else max(0, int(args.max_oracle_requests)),
        full_files=bool(args.full_files),
        full_file_max_bytes=int(args.full_file_max_bytes),
        sample_workers=max(1, int(args.sample_workers)),
        sample_prefetch=max(0, int(args.sample_prefetch)),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
