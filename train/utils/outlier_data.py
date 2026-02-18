from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

import utils.config as cfg
from utils.token_utils import allowed_byte_values

try:
    from datasets import load_from_disk  # type: ignore
except Exception:  # pragma: no cover - handled by runtime fallback
    load_from_disk = None  # type: ignore


class OutlierBatcher:
    """
    Build outlier token batches for OE training.

    Supported sources:
      - "mixed" (default/expected): 50:50 random vs heldout when heldout exists
      - Legacy accepted values ("random", "heldout", "heldout:/path") are
        interpreted compatibly, but mixed is preferred.
    """

    def __init__(
        self,
        *,
        source: str,
        data_root: str,
        heldout_root: str,
        window_bytes: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.source = str(source or "random").strip()
        self.data_root = Path(data_root)
        if heldout_root and str(heldout_root).strip():
            root = Path(str(heldout_root).strip()).expanduser()
        else:
            root = Path(data_root).expanduser().parent / "arrow_out_other"
        self.heldout_root = root.resolve()
        self.window_bytes = max(1, int(window_bytes))
        self.batch_size = max(1, int(batch_size))
        self.rng = np.random.default_rng(int(seed) ^ 0x0E5EED)
        self._alphabet = allowed_byte_values().astype(np.int32, copy=False)

        self.mode = "random"
        self.holdout_mix_prob = 0.5
        self.heldout_langs: List[str] = []
        self.heldout_sets: List[Tuple[str, object]] = []

        src = self.source.lower()
        root_hint = ""
        if src.startswith("heldout:"):
            root_hint = str(self.source.split(":", 1)[1]).strip()
        # Always attempt to load heldout data so we can run mixed sampling by default.
        self._load_heldout(root_hint=root_hint)
        if self.heldout_sets:
            self.mode = "mixed"
        else:
            self.mode = "random"

    def _load_heldout(self, *, root_hint: str = "") -> None:
        if load_from_disk is None:
            return
        root = self.heldout_root
        if root_hint:
            hint_path = Path(root_hint).expanduser()
            if not hint_path.is_absolute():
                hint_path = (Path.cwd() / hint_path).resolve()
            root = hint_path
        if not root.exists():
            return

        dataset_paths: List[Tuple[str, Path]] = []
        primary = root / "train" / "other" / "dataset"
        if primary.exists():
            dataset_paths.append(("other", primary))
        train_root = root / "train"
        if not dataset_paths and train_root.exists():
            for child in sorted(train_root.iterdir()):
                if not child.is_dir():
                    continue
                ds_path = child / "dataset"
                if ds_path.exists():
                    dataset_paths.append((str(child.name).lower(), ds_path))

        loaded: List[Tuple[str, object]] = []
        for lang, ds_path in dataset_paths:
            try:
                ds = load_from_disk(str(ds_path))
            except Exception:
                continue
            try:
                if len(ds) <= 0:
                    continue
            except Exception:
                continue
            loaded.append((lang, ds))
        if loaded:
            self.heldout_langs = [lang for lang, _ in loaded]
            self.heldout_sets = loaded

    def describe(self) -> dict:
        return {
            "source": self.source,
            "mode": self.mode,
            "holdout_mix_prob": float(self.holdout_mix_prob),
            "heldout_root": str(self.heldout_root),
            "heldout_langs": list(self.heldout_langs),
            "window_bytes": int(self.window_bytes),
            "batch_size": int(self.batch_size),
        }

    def _sample_random(self) -> np.ndarray:
        min_len = max(8, self.window_bytes // 8)
        length = int(self.rng.integers(min_len, self.window_bytes + 1))
        idx = self.rng.integers(0, self._alphabet.size, size=length)
        return self._alphabet[idx].astype(np.int32, copy=False)

    def _sample_heldout(self) -> np.ndarray | None:
        if not self.heldout_sets:
            return None
        for _ in range(16):
            ds_idx = int(self.rng.integers(0, len(self.heldout_sets)))
            _, dataset = self.heldout_sets[ds_idx]
            try:
                row_idx = int(self.rng.integers(0, len(dataset)))
                row = dataset[row_idx]
            except Exception:
                continue
            text = row.get("content") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                continue
            arr = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).astype(
                np.int32, copy=False
            )
            if arr.size <= 0:
                continue
            if arr.size <= self.window_bytes:
                return arr
            start = int(self.rng.integers(0, arr.size - self.window_bytes + 1))
            return arr[start : start + self.window_bytes]
        return None

    def get(self) -> np.ndarray:
        xb = np.full(
            (self.batch_size, self.window_bytes),
            int(cfg.PAD_BYTE_ID),
            dtype=np.int32,
        )
        for i in range(self.batch_size):
            sample = None
            use_heldout = bool(self.heldout_sets) and (self.rng.random() < self.holdout_mix_prob)
            if use_heldout:
                sample = self._sample_heldout()
            if sample is None or sample.size <= 0:
                sample = self._sample_random()
            n = min(self.window_bytes, int(sample.size))
            xb[i, :n] = sample[:n]
        return xb
