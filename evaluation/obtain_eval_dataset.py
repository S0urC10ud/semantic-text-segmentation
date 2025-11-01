#!/usr/bin/env python3
"""Build curated evaluation datasets from the validation split.

This script samples fragments from the downloader-produced validation split and
derives a set of benchmark tasks covering:

  • Pure single-label fragments (baseline accuracy by content type).
  • Needle-in-the-haystack injections with multiple size buckets.
  • Back-to-back multi-label sequences (pairs and triplets).
  • Markdown-flavoured mixtures with optional prose/context blocks.
  • Throughput stress datasets targeting fixed byte budgets.

Each task is written as its own Hugging Face Dataset (Arrow format) under the
requested output directory together with an inventory manifest.

The generated records share a common schema:

  - ``task``: identifier for the evaluation scenario.
  - ``example_id``: deterministic unique id per example.
  - ``content``: UTF-8 text presented to the model.
  - ``segments``: ordered ground-truth segments with ``label``, ``char_start``,
    and ``char_end`` (exclusive) indices.
  - ``source_langs``: distinct labels contributing to the example.
  - ``metadata_json``: JSON-encoded auxiliary metadata (host/donor UIDs, etc.).

The default configuration samples 1,000 host fragments per content type for
each task as requested, but smaller counts can be supplied to keep dataset size
manageable during iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

import datasets as hfds
from datasets import Dataset, Features, Sequence as SeqFeature, Value


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from train import config as cfg  # noqa: E402


@dataclass
class Fragment:
    """Represents a single validation fragment available for sampling."""

    lang: str
    content: str
    uid: Optional[str]
    extra_meta: Optional[dict]


def _log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _load_val_fragments(
    data_root: Path,
    per_lang_target: int,
    *,
    extra_multiplier: int,
    seed: int,
) -> Dict[str, List[Fragment]]:
    """Load and sample validation fragments per language."""

    out: Dict[str, List[Fragment]] = {}
    missing: List[str] = []
    
    langs = sorted(cfg.LANG2ID.keys())
    _log("📂 Loading validation fragments...")
    for lang in tqdm(langs, desc="Loading languages", unit="lang"):
        path = data_root / "val" / lang / "dataset"
        if not path.exists():
            missing.append(lang)
            continue
        try:
            ds = hfds.load_from_disk(str(path))
        except Exception as exc:  # pragma: no cover - informative logging
            _log(f"⚠️  Failed to load validation dataset for '{lang}': {exc}")
            continue

        if len(ds) == 0:
            _log(f"⚠️  Validation dataset for '{lang}' is empty; skipping.")
            continue

        target = min(len(ds), per_lang_target * max(1, extra_multiplier))
        if target <= 0:
            continue

        seed_offset = seed + cfg.LANG2ID.get(lang, 0)
        sampled = ds.shuffle(seed=seed_offset).select(range(target))
        fragments: List[Fragment] = []
        for row in sampled:
            text = row.get("content")
            if not isinstance(text, str) or not text:
                continue
            uid = row.get("uid") if isinstance(row, dict) else None
            extra_meta = {
                k: row[k]
                for k in ("source_repo", "source_repo_path", "source_ext", "license")
                if k in row and isinstance(row[k], str)
            }
            fragments.append(Fragment(lang=lang, content=text, uid=uid, extra_meta=extra_meta or None))

        if not fragments:
            _log(f"⚠️  No usable fragments collected for '{lang}'.")
            continue
        out[lang] = fragments

    if missing:
        _log(
            "⚠️  Missing validation splits for: "
            + ", ".join(sorted(missing))
        )
    return out


def _features_schema() -> Features:
    return Features(
        {
            "task": Value("string"),
            "example_id": Value("string"),
            "content": Value("string"),
            "segments": SeqFeature(
                feature={
                    "label": Value("string"),
                    "char_start": Value("int32"),
                    "char_end": Value("int32"),
                }
            ),
            "source_langs": SeqFeature(feature=Value("string")),
            "metadata_json": Value("string"),
        }
    )


def _hash_example_id(task: str, lang: str, index: int, salt: Optional[str] = None) -> str:
    base = f"{task}|{lang}|{index}"
    if salt:
        base += f"|{salt}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{task}-{lang}-{index:05d}-{digest}"


def _segment(label: str, start: int, end: int) -> dict:
    return {"label": label, "char_start": int(start), "char_end": int(end)}


def _join_parts(parts: Iterable[Tuple[str, str]]) -> Tuple[str, List[dict], List[str]]:
    """Concatenate labelled text parts and return content, segments, labels."""

    segments: List[dict] = []
    langs: List[str] = []
    pieces: List[str] = []
    cursor = 0
    for label, text in parts:
        if not text:
            continue
        pieces.append(text)
        start = cursor
        cursor += len(text)
        segments.append(_segment(label, start, cursor))
        langs.append(label)
    return "".join(pieces), segments, langs


def _make_record(
    task: str,
    lang: str,
    index: int,
    content: str,
    segments: List[dict],
    source_langs: Sequence[str],
    metadata: dict,
) -> dict:
    example_id = _hash_example_id(task, lang, index, salt=str(metadata.get("seed", "")))
    uniq_langs = sorted({lbl for lbl in source_langs if isinstance(lbl, str) and lbl})
    return {
        "task": task,
        "example_id": example_id,
        "content": content,
        "segments": segments,
        "source_langs": uniq_langs,
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def _slice_by_bytes(text: str, min_bytes: int, max_bytes: Optional[int], rng: random.Random) -> Optional[str]:
    if not text:
        return None
    chars = list(text)
    if max_bytes is not None and max_bytes < min_bytes:
        max_bytes = min_bytes

    for _ in range(80):
        start = rng.randrange(0, len(chars))
        end = min(len(chars), start + 1)
        while end <= len(chars):
            snippet = "".join(chars[start:end])
            size = len(snippet.encode("utf-8", "ignore"))
            if size >= min_bytes and (max_bytes is None or size <= max_bytes):
                return snippet
            if max_bytes is not None and size > max_bytes:
                break
            if end == len(chars):
                break
            end += 1

    # Fallback: return a trimmed prefix if possible
    data = text.encode("utf-8", "ignore")
    if len(data) < min_bytes:
        return None
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", "ignore")


def _build_pure_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
) -> Tuple[str, List[dict], str]:
    task = "pure_fragments"
    examples: List[dict] = []
    desc = "Single-label validation fragments (baseline accuracy)."
    for lang, frags in fragments_by_lang.items():
        limit = min(per_label, len(frags))
        for idx in range(limit):
            frag = frags[idx]
            content = frag.content
            segments = [_segment(lang, 0, len(content))]
            meta = {
                "host_lang": lang,
                "host_uid": frag.uid,
                "source_meta": frag.extra_meta,
            }
            record = _make_record(task, lang, idx, content, segments, [lang], meta)
            examples.append(record)
    return task, examples, desc


def _choose_injection(
    fragments_by_lang: Dict[str, List[Fragment]],
    host_lang: str,
    rng: random.Random,
    min_bytes: int,
    max_bytes: Optional[int],
) -> Optional[Tuple[str, Fragment, str]]:
    donor_langs = [lang for lang in fragments_by_lang.keys() if lang != host_lang and fragments_by_lang[lang]]
    if not donor_langs:
        return None

    for _ in range(100):
        d_lang = rng.choice(donor_langs)
        donor = rng.choice(fragments_by_lang[d_lang])
        snippet = _slice_by_bytes(donor.content, min_bytes, max_bytes, rng)
        if snippet:
            return d_lang, donor, snippet
    return None


def _build_injection_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    bucket_name: str,
    min_bytes: int,
    max_bytes: Optional[int],
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = f"needle_{bucket_name}"
    desc = (
        "Host fragments with a foreign-language needle injection sized "
        f"{min_bytes}-{'∞' if max_bytes is None else max_bytes} bytes."
    )
    examples: List[dict] = []
    _log(f"💉 Building injection dataset for size {min_bytes}-{'∞' if max_bytes is None else max_bytes} bytes...")

    for lang, frags in fragments_by_lang.items():
        if not frags:
            continue
        limit = min(per_label, len(frags))
        if limit == 0:
            continue
        for idx in range(limit):
            host = frags[idx]
            inj = _choose_injection(fragments_by_lang, lang, rng, min_bytes, max_bytes)
            if not inj:
                continue
            donor_lang, donor_frag, snippet = inj
            if not snippet:
                continue
            host_text = host.content
            if not host_text:
                continue
            insert_at = rng.randrange(0, len(host_text) + 1)
            left = host_text[:insert_at]
            right = host_text[insert_at:]
            parts = []
            if left:
                parts.append((lang, left))
            parts.append((donor_lang, snippet))
            if right:
                parts.append((lang, right))
            content, segments, langs = _join_parts(parts)
            inj_bytes = len(snippet.encode("utf-8", "ignore"))
            meta = {
                "host_lang": lang,
                "host_uid": host.uid,
                "donor_lang": donor_lang,
                "donor_uid": donor_frag.uid,
                "insertion_char": insert_at,
                "injection_bytes": inj_bytes,
                "size_bucket": bucket_name,
            }
            record = _make_record(task, lang, idx, content, segments, langs, meta)
            examples.append(record)

    return task, examples, desc


def _build_pair_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "sequence_pair"
    desc = "Two-language back-to-back sequences <A><B>."
    examples: List[dict] = []
    _log("🔄 Building sequence pair dataset...")
    labels = [lang for lang, frags in fragments_by_lang.items() if frags]
    for lang in labels:
        partners = [lbl for lbl in labels if lbl != lang]
        if not partners:
            continue
        host_list = fragments_by_lang[lang]
        limit = min(per_label, len(host_list))
        for idx in range(limit):
            host = host_list[idx]
            partner_lang = rng.choice(partners)
            partner_frag = rng.choice(fragments_by_lang[partner_lang])
            parts = [
                (lang, host.content.rstrip() + "\n\n"),
                (partner_lang, partner_frag.content.lstrip()),
            ]
            content, segments, langs = _join_parts(parts)
            meta = {
                "first_lang": lang,
                "second_lang": partner_lang,
                "host_uid": host.uid,
                "partner_uid": partner_frag.uid,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))
    return task, examples, desc


def _build_triplet_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "sequence_triplet"
    desc = "Three-language back-to-back sequences <A><B><C>."
    examples: List[dict] = []
    _log("🔄 Building sequence triplet dataset...")
    labels = [lang for lang, frags in fragments_by_lang.items() if frags]
    for lang in labels:
        others = [lbl for lbl in labels if lbl != lang]
        if len(others) < 2:
            continue
        host_list = fragments_by_lang[lang]
        limit = min(per_label, len(host_list))
        for idx in range(limit):
            host = host_list[idx]
            partner_langs = rng.sample(others, 2)
            mid_frag = rng.choice(fragments_by_lang[partner_langs[0]])
            tail_frag = rng.choice(fragments_by_lang[partner_langs[1]])
            parts = [
                (lang, host.content.rstrip() + "\n\n"),
                (partner_langs[0], mid_frag.content.strip() + "\n\n"),
                (partner_langs[1], tail_frag.content.lstrip()),
            ]
            content, segments, langs = _join_parts(parts)
            meta = {
                "first_lang": lang,
                "second_lang": partner_langs[0],
                "third_lang": partner_langs[1],
                "host_uid": host.uid,
                "mid_uid": mid_frag.uid,
                "tail_uid": tail_frag.uid,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))
    return task, examples, desc


def _maybe_trim_text(text: str, max_chars: int = 512) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n…\n" + text[-tail:]


def _build_markdown_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "markdown_mix"
    desc = "Markdown-like text/code interleavings with optional fences."
    examples: List[dict] = []
    _log("📝 Building markdown mix dataset...")
    text_pool = fragments_by_lang.get("text", [])
    labels = [lang for lang in fragments_by_lang.keys() if lang != "text" and fragments_by_lang[lang]]
    if not text_pool:
        _log("⚠️  No 'text' fragments available; markdown dataset will lack prose segments.")

    def sample_text() -> str:
        if not text_pool:
            return ""
        frag = rng.choice(text_pool)
        return _maybe_trim_text(frag.content.strip(), max_chars=256)

    for lang in labels:
        host_list = fragments_by_lang[lang]
        limit = min(per_label, len(host_list))
        alt_langs = [lbl for lbl in labels if lbl != lang]
        if not alt_langs:
            continue
        for idx in range(limit):
            host = host_list[idx]
            other_lang = rng.choice(alt_langs)
            other = rng.choice(fragments_by_lang[other_lang])

            blocks: List[Tuple[str, str]] = []
            if rng.random() < 0.65:
                blocks.append(("text", sample_text() + "\n\n"))

            wrap_host = rng.random() < 0.5
            if wrap_host:
                blocks.append(("text", "```\n"))
                blocks.append((lang, host.content.strip() + "\n"))
                blocks.append(("text", "```\n"))
            else:
                blocks.append((lang, host.content.strip() + "\n"))

            if rng.random() < 0.7:
                blocks.append(("text", sample_text() + "\n"))

            wrap_other = rng.random() < 0.5
            if wrap_other:
                blocks.append(("text", "```\n"))
                blocks.append((other_lang, other.content.strip() + "\n"))
                blocks.append(("text", "```\n"))
            else:
                blocks.append((other_lang, other.content.strip() + "\n"))

            if rng.random() < 0.55:
                blocks.append(("text", "\n" + sample_text()))

            content, segments, langs = _join_parts(blocks)
            meta = {
                "host_lang": lang,
                "other_lang": other_lang,
                "host_uid": host.uid,
                "other_uid": other.uid,
                "wrapped_host": wrap_host,
                "wrapped_other": wrap_other,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))

    return task, examples, desc


def _assemble_label_block(
    fragments: List[Fragment],
    target_bytes: int,
    rng: random.Random,
) -> Tuple[str, int]:
    parts: List[str] = []
    total = 0
    attempt = 0
    while total < target_bytes and attempt < target_bytes * 4:
        frag = rng.choice(fragments)
        snippet = frag.content.strip()
        if not snippet:
            attempt += 1
            continue
        if parts:
            snippet = "\n" + snippet
        parts.append(snippet)
        total = len("".join(parts).encode("utf-8", "ignore"))
        attempt += 1
    combined = "".join(parts)
    data = combined.encode("utf-8", "ignore")
    if not data:
        return "", 0
    if len(data) > target_bytes:
        data = data[:target_bytes]
        combined = data.decode("utf-8", "ignore")
    return combined, len(data)


def _build_throughput_dataset(
    size_bytes: int,
    num_examples: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = f"throughput_{size_bytes}"
    pretty = {
        1024: "1 KiB",
        10_240: "10 KiB",
        102_400: "100 KiB",
        1_048_576: "1 MiB",
    }.get(size_bytes, f"{size_bytes} bytes")
    desc = f"Synthetic random text blobs for throughput ({pretty})."
    alphabet = string.ascii_letters + string.digits + string.punctuation + "\n "
    examples: List[dict] = []
    sample_count = max(1, num_examples)
    _log(f"🚀 Building throughput dataset for {pretty} with {sample_count} random samples...")
    for idx in range(sample_count):
        chars = [rng.choice(alphabet) for _ in range(size_bytes)]
        content = "".join(chars)
        actual_bytes = len(content.encode("utf-8", "ignore"))
        meta = {
            "target_bytes": size_bytes,
            "actual_bytes": actual_bytes,
            "mode": "synthetic_random",
        }
        segments = [_segment("text", 0, len(content))]
        examples.append(
            _make_record(
                task,
                "synthetic",
                idx,
                content,
                segments,
                ["text"],
                meta,
            )
        )
    return task, examples, desc


def _write_dataset(
    root: Path,
    task: str,
    examples: List[dict],
    features: Features,
    overwrite: bool,
) -> int:
    if not examples:
        _log(f"⚠️  Task '{task}' produced no examples; skipping write.")
        return 0
    out_dir = root / task
    if out_dir.exists():
        if overwrite:
            _log(f"🔄 Removing existing dataset at {out_dir}")
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Dataset already exists at {out_dir}; use --overwrite to replace it.")
    ds = Dataset.from_list(examples, features=features)
    ds_size_mb = sum(sys.getsizeof(ex) for ex in examples) / (1024 * 1024)
    _log(f"💾 Writing {len(ds):,} examples ({ds_size_mb:.1f} MB) to {out_dir}")
    ds.save_to_disk(str(out_dir))
    return len(ds)


def build_all_datasets(args) -> dict:
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fragments_by_lang = _load_val_fragments(
        data_root,
        args.per_label,
        extra_multiplier=args.extra_pool_multiplier,
        seed=args.seed,
    )
    if not fragments_by_lang:
        raise RuntimeError("No validation fragments available – cannot build evaluation datasets.")

    rng = random.Random(args.seed)
    features = _features_schema()
    manifest_tasks = []

    _log("🔧 Building evaluation datasets...")
    builders = []
    _log("📊 Building pure fragments dataset...")
    builders.append(_build_pure_dataset(fragments_by_lang, args.per_label))

    buckets = [
        ("4_15", 4, 15),
        ("16_31", 16, 31),
        ("32_63", 32, 63),
        ("64_plus", 64, None),
    ]
    for name, lo, hi in buckets:
        builders.append(
            _build_injection_dataset(
                fragments_by_lang,
                args.per_label,
                name,
                lo,
                hi,
                rng,
            )
        )

    builders.append(_build_pair_dataset(fragments_by_lang, args.per_label, rng))
    builders.append(_build_triplet_dataset(fragments_by_lang, args.per_label, rng))
    builders.append(_build_markdown_dataset(fragments_by_lang, args.per_label, rng))

    for size in (1024, 10_240, 102_400, 1_048_576):
        builders.append(
            _build_throughput_dataset(size, args.throughput_examples, rng)
        )

    total_examples = 0
    _log("\n� Writing datasets to disk...")
    for task, examples, desc in tqdm(builders, desc="Writing datasets", unit="dataset"):
        count = _write_dataset(output_root, task, examples, features, overwrite=args.overwrite)
        total_examples += count
        manifest_tasks.append({
            "task": task,
            "description": desc,
            "count": count,
            "path": str((output_root / task).relative_to(output_root)),
        })
    _log(f"\n📊 Summary: Generated {len(manifest_tasks)} datasets with {total_examples:,} total examples")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "per_label": args.per_label,
        "throughput_examples": args.throughput_examples,
        "tasks": manifest_tasks,
    }

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Generate evaluation datasets from validation fragments.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "downloader" / "arrow_out"),
        help="Path to downloader outputs (expects <split>/<lang>/dataset directories).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO_ROOT / "evaluation" / "data"),
        help="Directory where evaluation datasets will be written.",
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=100,
        help="Examples per label for most benchmarks (pure, injection, sequences, markdown).",
    )
    parser.add_argument(
        "--throughput-examples",
        type=int,
        default=8,
        help="Number of synthetic samples to create for each throughput size.",
    )
    parser.add_argument(
        "--extra-pool-multiplier",
        type=int,
        default=3,
        help="Multiplier to expand the initial validation sample pool per label.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing datasets in the output directory.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = build_all_datasets(args)
    _log(f"✅ Generated {len(manifest['tasks'])} datasets under {manifest['output_root']}")


if __name__ == "__main__":
    main()
