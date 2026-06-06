#!/usr/bin/env python3
"""Slim released checkpoints to a compact 130-row embedding (lossless).

The full models carry a 257-row byte embedding (bytes 0..255 + PAD=256). Every
input token is sanitized into {9, 10, 13, 32..126, 164 (=0xA4), 256 (=PAD)}
before the embedding lookup, so only those rows are ever read. This script keeps
embedding rows 0..127 as-is, folds the currency placeholder (0xA4 = 164) into
row 128 and the pad (256) into row 129, dropping the 127 dead rows. The model's
in-graph compress table (``utils.token_utils.COMPACT_TOKEN_TABLE``) maps the
original 257-id space into this compact space at inference time, so the slimmed
checkpoint is a drop-in replacement that produces identical predictions.

Usage:
    python train/slim_checkpoints.py            # back up, slim in place, verify
    python train/slim_checkpoints.py --dry-run  # build + verify only, no overwrite
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from flax import serialization

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "train"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LEGACY_VOCAB = 257
COMPACT_VOCAB = 130
PLACEHOLDER_ROW = 0xA4  # 164
PAD_ROW = 256

# (name, arch, predictor kwargs) for the released checkpoints.
TARGETS = [
    dict(
        name="unet_al",
        arch="unet1d",
        model_dim=256,
        channels=(32, 64, 64, 128, 128, 128, 128, 256),
    ),
    dict(
        name="mamba_al",
        arch="mamba",
        model_dim=256,
        mamba_layers=6,
        mamba_d_state=16,
        mamba_expand=1,
        mamba_dt_rank=16,
        mamba_conv=4,
        mamba_bidirectional=True,
    ),
]


def _unwrap(tree):
    if isinstance(tree, dict) and "params" in tree and "Embed_0" not in tree:
        return tree["params"], True
    return tree, False


def slim_tree(tree):
    """Return a copy of the param tree with a 130-row embedding."""
    params, wrapped = _unwrap(tree)
    emb = np.asarray(params["Embed_0"]["embedding"])
    if emb.shape[0] == COMPACT_VOCAB:
        return tree, emb.shape[0]  # already slimmed
    if emb.shape[0] != LEGACY_VOCAB:
        raise ValueError(f"Unexpected embedding rows: {emb.shape[0]} (expected {LEGACY_VOCAB})")
    new = np.empty((COMPACT_VOCAB, emb.shape[1]), dtype=emb.dtype)
    new[:128] = emb[:128]
    new[128] = emb[PLACEHOLDER_ROW]
    new[129] = emb[PAD_ROW]
    params["Embed_0"]["embedding"] = new
    return tree, LEGACY_VOCAB


def _total_params(tree) -> int:
    params, _ = _unwrap(tree)

    def count(d):
        t = 0
        for v in d.values():
            t += count(v) if isinstance(v, dict) else int(np.asarray(v).size)
        return t

    return count(params)


def _load_eval_texts(limit_per_task: int = 5):
    from datasets import load_from_disk

    texts = []
    for task in ["realistic", "near_pure", "sequence_pair", "sequence_triplet", "markdown_mix"]:
        path = REPO_ROOT / "evaluation" / "test" / task
        if not path.is_dir():
            continue
        try:
            ds = load_from_disk(str(path))
        except Exception:
            continue
        for i in range(min(limit_per_task, len(ds))):
            content = ds[i].get("content")
            if isinstance(content, str) and content.strip():
                texts.append((task, content))
    return texts


def _build_predictor(ckpt_path: Path, target: dict):
    import viewers.core as core

    kwargs = dict(
        ckpt_path=str(ckpt_path),
        num_classes=35,
        model_dim=target["model_dim"],
        channels=tuple(target.get("channels", (32, 64, 64, 128, 128, 128, 128, 256))),
        arch=target["arch"],
        dtype_str="bfloat16",
        other_threshold=0.0,  # disable open-set routing -> compare raw argmax
        inference_backend="auto",
    )
    if target["arch"] == "mamba":
        kwargs.update(
            mamba_layers=target["mamba_layers"],
            mamba_d_state=target["mamba_d_state"],
            mamba_expand=target["mamba_expand"],
            mamba_dt_rank=target["mamba_dt_rank"],
            mamba_conv=target["mamba_conv"],
            mamba_bidirectional=target["mamba_bidirectional"],
        )
    return core.Predictor(**kwargs), core


def _predict_labels(predictor, core, text: str):
    normalized = core._normalize_input_text(text)
    _, char_labels, char_probs, _ = predictor.segment_text(normalized, min_run_chars=1)
    labels = np.asarray(char_labels, dtype=np.int32)
    maxp = np.array([max(p.values()) if p else 0.0 for p in char_probs], dtype=np.float32)
    return labels, maxp


def verify_parity(orig_path: Path, slim_path: Path, target: dict, texts) -> bool:
    print(f"  [verify] building ORIGINAL predictor ({orig_path.name}, 257-row) ...", flush=True)
    pred_o, core = _build_predictor(orig_path, target)
    print(f"  [verify] building SLIMMED predictor ({slim_path.name}, 130-row) ...", flush=True)
    pred_s, _ = _build_predictor(slim_path, target)

    n_mismatch = 0
    max_prob_delta = 0.0
    total_chars = 0
    for task, text in texts:
        lo, po = _predict_labels(pred_o, core, text)
        ls, ps = _predict_labels(pred_s, core, text)
        if lo.shape != ls.shape:
            print(f"    [{task}] SHAPE MISMATCH {lo.shape} vs {ls.shape}")
            n_mismatch += 1
            continue
        diff = int(np.count_nonzero(lo != ls))
        pdelta = float(np.max(np.abs(po - ps))) if po.size else 0.0
        max_prob_delta = max(max_prob_delta, pdelta)
        total_chars += lo.size
        if diff:
            n_mismatch += diff
            print(f"    [{task}] {diff}/{lo.size} char labels differ | max|Δprob|={pdelta:.2e}")
    print(f"  [verify] total_chars={total_chars:,} | label mismatches={n_mismatch} | max|Δprob|={max_prob_delta:.2e}")
    return n_mismatch == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build + verify only; do not overwrite")
    ap.add_argument("--ckpt-dir", default=str(REPO_ROOT / "checkpoints"))
    ap.add_argument("--texts-per-task", type=int, default=5)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    backup_dir = ckpt_dir / "orig_257_backup"
    tmp_dir = ckpt_dir / "_slim_tmp"
    backup_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation texts ...", flush=True)
    texts = _load_eval_texts(args.texts_per_task)
    print(f"  {len(texts)} eval samples loaded", flush=True)

    all_ok = True
    for target in TARGETS:
        name = target["name"]
        src = ckpt_dir / f"{name}.msgpack"
        print(f"\n=== {name} ({src}) ===", flush=True)
        if not src.is_file():
            print(f"  SKIP: {src} not found")
            continue

        tree = serialization.msgpack_restore(src.read_bytes())
        params, _ = _unwrap(tree)
        rows = int(np.asarray(params["Embed_0"]["embedding"]).shape[0])
        if rows == COMPACT_VOCAB:
            print("  already slimmed (130 rows) — skipping")
            continue

        before = _total_params(tree)
        slim, _ = slim_tree(tree)
        after = _total_params(slim)
        print(f"  params: {before:,} -> {after:,}  (dropped {before - after:,})")

        # back up original, write slimmed to temp
        backup_path = backup_dir / f"{name}.msgpack"
        if not backup_path.is_file():
            shutil.copy2(src, backup_path)
            print(f"  backed up original -> {backup_path}")
        tmp_path = tmp_dir / f"{name}.msgpack"
        tmp_path.write_bytes(serialization.msgpack_serialize(slim))
        print(f"  wrote slimmed -> {tmp_path}")

        ok = verify_parity(backup_path, tmp_path, target, texts)
        if not ok:
            print(f"  !! PARITY FAILED for {name} — NOT overwriting")
            all_ok = False
            continue
        print(f"  parity OK for {name}")

        if not args.dry_run:
            shutil.copy2(tmp_path, src)
            print(f"  overwrote {src} with slimmed checkpoint")

    print(f"\n{'ALL PARITY OK' if all_ok else 'SOME PARITY FAILURES'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
