from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
repo_str = str(REPO_ROOT)
train_str = str(TRAIN_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)
if train_str not in sys.path:
    sys.path.insert(0, train_str)

import utils.config as cfg  # noqa: E402
from utils.data import prepare_dsets_by_lang_with_splits  # noqa: E402
from utils.full_sequence import make_training_full_sequence_with_metadata  # noqa: E402

_PLACEHOLDER_CHAR = "\u00A4"
_ALLOWED_TEXT_CHARS = {chr(b) for b in range(0x20, 0x7F)}
_ALLOWED_TEXT_CHARS.update({" ", "\n", "\t", _PLACEHOLDER_CHAR})


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def _window_tokens_to_normalized_text(tokens: np.ndarray) -> str:
    arr = np.asarray(tokens, dtype=np.int32)
    if arr.size == 0:
        return ""
    valid = arr[(arr >= 0) & (arr < 256)]
    if valid.size == 0:
        return ""
    text = valid.astype(np.uint8, copy=False).tobytes().decode("utf-8", "ignore")
    out_chars: List[str] = []
    for ch in text:
        if ch == "\r":
            ch = "\n"
        if ch in _ALLOWED_TEXT_CHARS:
            out_chars.append(ch)
        else:
            out_chars.append(_PLACEHOLDER_CHAR)
    return "".join(out_chars)


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


def _resolve_window_source_lang(metadata: Optional[Dict[str, object]]) -> str:
    if not isinstance(metadata, dict):
        return "unknown"
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
) -> Dict[str, object]:
    if not isinstance(metadata, dict):
        return {
            "sampling_mode": "training_full_sequence",
            "component_count": 0,
            "source_langs": [],
            "target_bytes": int(cfg.MODEL_WINDOW_BYTES),
            "full_file_mode": True,
        }
    return {
        "sampling_mode": "training_full_sequence",
        "component_count": int(metadata.get("component_count", 0) or 0),
        "component_window_bytes": int(
            metadata.get("component_window_bytes", cfg.MODEL_WINDOW_BYTES) or cfg.MODEL_WINDOW_BYTES
        ),
        "target_bytes": int(metadata.get("target_bytes", cfg.MODEL_WINDOW_BYTES) or cfg.MODEL_WINDOW_BYTES),
        "source_langs": _collect_full_sequence_source_langs(metadata),
        "full_file_mode": True,
    }


def _full_sequence_worker_entry(
    data_root: str,
    split: str,
    langs: Sequence[str],
    target_len: int,
    q,
    stop_flag,
    wid: int,
    base_seed: int,
) -> None:
    random.seed(int(base_seed) ^ (int(wid) + 1) ^ int(time.time()))
    np.random.seed((int(base_seed) + int(wid) + 1) % (2**32 - 1))
    data_cfg = cfg.DataConfig(data_root=str(data_root))
    dsets = prepare_dsets_by_lang_with_splits(
        str(data_root),
        include_languages=list(langs),
        verbose=False,
    )
    dsets_by_lang = dsets.get(str(split)) or {}
    if not dsets_by_lang:
        return
    while not stop_flag.is_set():
        try:
            tokens, _, full_meta = make_training_full_sequence_with_metadata(
                dsets_by_lang,
                data_cfg,
                target_len=int(target_len),
            )
            normalized = _window_tokens_to_normalized_text(tokens)
            if not normalized:
                payload = {"status": "empty"}
            else:
                payload = {
                    "status": "ok",
                    "lang": _resolve_full_sequence_source_lang(full_meta),
                    "normalized": normalized,
                    "sample_hash": _hash_text(normalized),
                    "metadata": _summarize_full_sequence_metadata(full_meta),
                }
            q.put(payload, timeout=1.0)
        except queue.Full:
            continue
        except Exception as exc:
            try:
                q.put({"status": "error", "message": repr(exc)}, timeout=1.0)
            except queue.Full:
                pass
            return


class FullSequenceSamplePrefetcher:
    def __init__(
        self,
        *,
        data_root: str,
        split: str,
        langs: Sequence[str],
        target_len: int,
        num_workers: int,
        prefetch: int,
        base_seed: Optional[int] = None,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self.q = self._ctx.Queue(maxsize=max(2, int(prefetch)))
        self.stop_flag = self._ctx.Event()
        self.workers: List[mp.Process] = []
        seed = int(base_seed if base_seed is not None else time.time_ns() & 0xFFFFFFFF)
        for wid in range(max(1, int(num_workers))):
            proc = self._ctx.Process(
                target=_full_sequence_worker_entry,
                args=(
                    str(data_root),
                    str(split),
                    list(langs),
                    int(target_len),
                    self.q,
                    self.stop_flag,
                    int(wid),
                    int(seed),
                ),
                daemon=True,
            )
            proc.start()
            self.workers.append(proc)
        time.sleep(1.0)
        self._check_workers_alive("during startup")

    def _check_workers_alive(self, context: str = "") -> None:
        alive = [proc for proc in self.workers if proc.is_alive()]
        if alive:
            return
        dead_codes = [f"worker-{i} exit={proc.exitcode}" for i, proc in enumerate(self.workers)]
        raise RuntimeError(
            f"All FullSequenceSamplePrefetcher workers are dead {context}! "
            f"Statuses: {', '.join(dead_codes)}."
        )

    def get(self, timeout: float = 30.0) -> Dict[str, object]:
        while True:
            self._check_workers_alive("while waiting for AL samples")
            try:
                item = self.q.get(timeout=float(timeout))
            except queue.Empty:
                continue
            if isinstance(item, dict):
                return item
            return {"status": "error", "message": "invalid_prefetch_payload"}

    def close(self) -> None:
        self.stop_flag.set()
        for proc in self.workers:
            proc.join(timeout=2.0)
        for proc in self.workers:
            if proc.is_alive():
                proc.terminate()
        for proc in self.workers:
            proc.join(timeout=1.0)
        try:
            self.q.close()
        except Exception:
            pass
