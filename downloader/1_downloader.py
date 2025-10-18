import argparse
import gzip
import json
import re
import signal
import sys
import time
import gc
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from datasets import load_dataset  # removed interleave_datasets

# --------- Helpers

def safe_filename(name: str) -> str:
    s = (name or "")
    # special-case languages we want pretty folder names for
    if s.lower() == "c++":
        return "cpp"
    if s.lower() in ("c#", "c-sharp"):
        return "csharp"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)

def to_lower_dir(p: Path) -> Path:
    return p.parent / p.name.lower()

def canonical_lang_key(name: str) -> str:
    """
    Canonicalize language names for counters/output-dir purposes.
    Use The Stack's folder names (e.g., 'c++', 'javascript', 'c-sharp').
    """
    s = (name or "").strip().lower()
    if s in ("cpp",):
        return "c++"
    if s in ("js", "node", "nodejs"):
        return "javascript"
    if s in ("ts", "tsx", "typescript"):
        return "typescript"
    if s in ("csharp", "c-sharp", "cs", "c#"):
        return "c-sharp"
    return s

def normalize_lang_dir(name: str) -> str:
    return canonical_lang_key(name)

def get_rss_mb() -> float:
    """
    Best-effort current resident memory usage in MB.
    Tries /proc first (Linux), falls back to resource.ru_maxrss.
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb / 1024.0
    except Exception:
        pass
    try:
        import resource, sys as _sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux this is KB; on macOS it's bytes.
        if _sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        else:
            return rss / 1024.0
    except Exception:
        return 0.0

# --------- License filtering (robust & permissive-family-only)

_ALLOWED_FAMILIES = {"mit", "apache", "bsd", "unlicense", "0bsd", "mit-0"}
# Deliberately exclude "unknown"/"other" from disallowed to avoid false negatives.
_DISALLOWED_KEYWORDS = {
    "gpl", "agpl", "lgpl", "mpl", "epl", "cdla", "cddl", "artistic",
    "cern", "cecill", "affero", "proprietary", "arr", "cc"
}

def _normalize_license_string(s: str) -> str:
    return re.sub(r"[^a-z0-9.+-]+", " ", (s or "").lower())

def _iter_license_fields(example: Dict) -> Iterable[Tuple[str, Union[str, List[str]]]]:
    preferred = [
        "max_stars_repo_license",
        "max_forks_repo_license",
        "max_issues_repo_license",
        "max_stars_repo_licenses",
        "max_forks_repo_licenses",
        "max_issues_repo_licenses",
        "licenses",
        "license",
    ]
    seen = set()
    for k in preferred:
        if k in example and k not in seen and example.get(k) is not None:
            seen.add(k)
            yield k, example[k]
    for k, v in example.items():
        if k in seen:
            continue
        if "license" in k.lower() and v is not None:
            yield k, v

def _extract_license_strings(example: Dict) -> List[str]:
    vals: List[str] = []
    for _, v in _iter_license_fields(example):
        if isinstance(v, list):
            vals.extend([str(x) for x in v if x is not None])
        else:
            vals.append(str(v))
    return vals

def license_is_allowed(example: Dict) -> Tuple[bool, str]:
    lic_strings = _extract_license_strings(example)
    if not lic_strings:
        return False, "no_license_info"
    any_allowed = False
    for raw in lic_strings:
        s = _normalize_license_string(raw)
        if any(k in s for k in _DISALLOWED_KEYWORDS):
            return False, "disallowed_license"
        if any(k in s for k in _ALLOWED_FAMILIES):
            any_allowed = True
    if not any_allowed:
        return False, "no_allowed_license"
    return True, ""

# --------- Core extraction

def stream_language(lang_dir: str, shuffle_buffer: int, use_auth_token_flag: bool):
    load_kwargs = dict(
        path="bigcode/the-stack",
        data_dir=f"data/{lang_dir}",
        split="train",
        streaming=True,
    )
    if use_auth_token_flag:
        try:
            from huggingface_hub import get_token  # type: ignore
            tk = get_token() or True
        except Exception:
            tk = True
        load_kwargs["token"] = tk

    try:
        ds = load_dataset(**load_kwargs)
    except TypeError:
        load_kwargs.pop("token", None)
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        sys.stderr.write(f"[warn] skipping language '{lang_dir}': {e}\n")
        return None

    if shuffle_buffer > 0:
        # Can be memory-heavy; default is 0
        ds = ds.shuffle(seed=42, buffer_size=shuffle_buffer)
    return ds

def ext_filter(allowed: Set[str]):
    allowed_lc = {e.lower() for e in allowed}
    def _ok(example):
        ext = (example.get("ext") or "").lower().lstrip(".")
        return (ext in allowed_lc)
    return _ok

def write_one(
    out_dir: Path,
    example: Dict,
    gzip_output: bool,
    manifest_fp,
    skip_existing: bool,
    seq_id: int,
) -> Tuple[Optional[str], str]:
    """
    Returns (written_path_or_None, status)
    status in: 'written','exists','bad_content','error'
    """
    content = example.get("content", "")
    if not isinstance(content, str):
        return None, "bad_content"

    lang = canonical_lang_key(example.get("lang", "unknown"))
    ext = (example.get("ext", "") or "").lstrip(".").lower() or "txt"

    # File name is a sequential ID to avoid any hashing/dedup
    fname = f"{seq_id:012d}.{safe_filename(ext)}"

    lang_dir = out_dir / safe_filename(lang).lower()
    lang_dir.mkdir(parents=True, exist_ok=True)

    fpath = lang_dir / fname
    target_path = fpath.with_suffix(fpath.suffix + ".gz") if gzip_output else fpath

    if skip_existing and target_path.exists():
        return None, "exists"

    try:
        if gzip_output:
            # Use light compression to avoid CPU spikes; text mode avoids big bytes copies.
            with gzip.open(target_path, "wt", encoding="utf-8", errors="ignore", compresslevel=1) as g:
                g.write(content)
        else:
            with open(target_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(content)
    except MemoryError:
        raise
    except Exception:
        return None, "error"

    rec = {
        "relpath": str(target_path.relative_to(out_dir)),
        "seq_id": seq_id,
        "lang": example.get("lang"),
        "ext": example.get("ext"),
        "size": example.get("size"),
        "hexsha": example.get("hexsha"),
        "max_stars_repo_name": example.get("max_stars_repo_name"),
        "max_stars_repo_path": example.get("max_stars_repo_path"),
        "max_stars_repo_license": example.get("max_stars_repo_license") or example.get("max_stars_repo_licenses"),
        "max_forks_repo_license": example.get("max_forks_repo_license") or example.get("max_forks_repo_licenses"),
        "max_issues_repo_license": example.get("max_issues_repo_license") or example.get("max_issues_repo_licenses"),
        "max_stars_count": example.get("max_stars_count"),
    }
    try:
        manifest_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except MemoryError:
        raise
    except Exception:
        return str(target_path), "error"

    return str(target_path), "written"

def main():
    ap = argparse.ArgumentParser(
        description="Stream-extract files from BigCode The Stack with strict license + extension filtering (fast, visible, resilient)."
    )
    ap.add_argument("--out", type=Path, default=Path("stack_web_sample"), help="Output directory")
    ap.add_argument("--total", type=int, default=1_000_000, help="Total files to write (across all languages).")
    ap.add_argument(
        "--langs",
        type=str,
        default="json,html,javascript,css,csv,text,java,c,c++,python,typescript,php,csharp,go,sql,rust,yaml,ruby",
        help="Comma-separated language dirs in The Stack (e.g., 'c++', 'c-sharp').",
    )
    ap.add_argument("--shuffle-buffer", type=int, default=0, help="Streaming shuffle buffer size (0 = no shuffle).")
    ap.add_argument("--gzip", action="store_true", help="Gzip-compress saved files.")
    ap.add_argument("--use-auth-token", action="store_true", help="Pass your cached Hugging Face auth token to load_dataset.")
    ap.add_argument(
        "--ext-filter",
        type=str,
        default="json,html,htm,css,js,ts,csv,txt,java,c,cc,cpp,cxx,py,php,cs,go,sql,rs,rb,yaml,yml", # purpusefully exclude tsx
        help="Allowed extensions (lowercase, no dots, comma-separated).",
    )
    ap.add_argument("--per-lang-cap", type=int, default=100_000, help="Optional per-language cap.")
    # Visibility & resilience
    ap.add_argument("--progress-every-secs", type=int, default=10, help="Emit a progress line at least this often (seconds).")
    ap.add_argument("--progress-every-writes", type=int, default=1000, help="Emit a progress line every N writes.")
    ap.add_argument("--skip-existing", action="store_true", help="If a target file already exists, skip writing it (useful for resume).")
    ap.add_argument("--max-runtime-secs", type=int, default=0, help="If >0, stop cleanly after this many seconds.")
    ap.add_argument("--mem-soft-limit-mb", type=int, default=0, help="If >0, stop cleanly if RSS exceeds this many MB.")

    args = ap.parse_args()

    args.out = to_lower_dir(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "manifest.jsonl"
    log_path = args.out / "extraction.log"

    allowed_exts = {e.strip().lower() for e in args.ext_filter.split(",") if e.strip()}

    user_langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    lang_dirs: List[str] = [normalize_lang_dir(s) for s in user_langs]
    per_lang_counts = {canonical_lang_key(s): 0 for s in lang_dirs}

    interrupted = {"flag": False}
    def _sigint(_sig, _frm):
        interrupted["flag"] = True
        print("\nInterrupted; finishing current write and flushing state...", file=sys.stderr)
    signal.signal(signal.SIGINT, _sigint)
    try:
        signal.signal(signal.SIGTERM, _sigint)
    except Exception:
        pass

    counts = {
        "seen": 0,
        "written": 0,
        "skipped_ext": 0,
        "skipped_no_license_info": 0,
        "skipped_no_allowed_license": 0,
        "skipped_disallowed_license": 0,
        "skipped_per_lang_cap": 0,  # (rare now; we break instead)
        "exists_skipped": 0,
        "errors": 0,
    }

    t0 = time.time()
    last_report_t = t0
    last_written_at_report = 0

    next_seq_id = 1
    any_loaded = False

    with open(manifest_path, "a", encoding="utf-8") as manifest_fp, open(log_path, "a", encoding="utf-8") as log_fp:
        # Process languages SEQUENTIALLY; break out of a dataset as soon as its cap is hit.
        for lang_dir in lang_dirs:
            if interrupted["flag"] or (args.total and counts["written"] >= args.total):
                break

            ds = stream_language(lang_dir, args.shuffle_buffer, args.use_auth_token)
            if ds is None:
                sys.stderr.write(f"[warn] language '{lang_dir}' not loaded; check dataset folder name.\n")
                continue
            any_loaded = True

            current_lang_key = canonical_lang_key(lang_dir)

            for ex in ds:
                if interrupted["flag"]:
                    break

                now = time.time()
                if args.max_runtime_secs > 0 and (now - t0) >= args.max_runtime_secs:
                    print("[info] max runtime reached; stopping cleanly.", file=sys.stderr)
                    break
                rss_mb = get_rss_mb()
                if args.mem_soft_limit_mb > 0 and rss_mb >= args.mem_soft_limit_mb:
                    print(f"[warn] memory soft limit hit (rss={rss_mb:.1f}MB >= {args.mem_soft_limit_mb}MB); stopping cleanly.", file=sys.stderr)
                    break

                # If we've already hit the cap for this language, stop consuming this dataset and move on.
                if args.per_lang_cap is not None and per_lang_counts.get(current_lang_key, 0) >= args.per_lang_cap:
                    print(f"[info] per-lang-cap reached for '{current_lang_key}' ({per_lang_counts[current_lang_key]}). Moving to next language.", file=sys.stderr)
                    break

                counts["seen"] += 1

                ex_ext = (ex.get("ext") or "").lower().lstrip(".")
                if allowed_exts and ex_ext not in allowed_exts:
                    counts["skipped_ext"] += 1
                else:
                    ok, why = license_is_allowed(ex)
                    if not ok:
                        if      why == "no_license_info":     counts["skipped_no_license_info"] += 1
                        elif    why == "disallowed_license":  counts["skipped_disallowed_license"] += 1
                        else:                                 counts["skipped_no_allowed_license"] += 1
                    else:
                        # Cap check again (in case others modified counts)
                        if args.per_lang_cap is not None and per_lang_counts.get(current_lang_key, 0) >= args.per_lang_cap:
                            counts["skipped_per_lang_cap"] += 1
                            print(f"[info] per-lang-cap reached for '{current_lang_key}' ({per_lang_counts[current_lang_key]}). Moving to next language.", file=sys.stderr)
                            break
                        try:
                            path, status = write_one(
                                args.out, ex, args.gzip, manifest_fp,
                                args.skip_existing, seq_id=next_seq_id
                            )
                        except MemoryError:
                            print("[warn] MemoryError during write; stopping cleanly.", file=sys.stderr)
                            break
                        except Exception as e:
                            log_fp.write(f"ERROR(write): {e}\n")
                            counts["errors"] += 1
                            path, status = None, "error"

                        if status == "written" and path:
                            counts["written"] += 1
                            next_seq_id += 1
                            per_lang_counts[current_lang_key] = per_lang_counts.get(current_lang_key, 0) + 1
                        elif status == "exists":
                            counts["exists_skipped"] += 1
                            next_seq_id += 1
                        elif status == "error":
                            counts["errors"] += 1
                            next_seq_id += 1

                # Early release of large strings from the yielded record
                if isinstance(ex, dict):
                    ex.pop("content", None)
                    ex.clear()
                del ex

                # progress & flush (and occasional GC)
                should_report_time = (now - last_report_t) >= args.progress_every_secs
                should_report_writes = (counts["written"] - last_written_at_report) >= args.progress_every_writes
                if should_report_time or should_report_writes:
                    elapsed = now - t0
                    wps = counts["written"] / elapsed if elapsed > 0 else 0.0
                    sps = counts["seen"] / elapsed if elapsed > 0 else 0.0
                    rss_mb = get_rss_mb()
                    print(
                        f"[progress] wrote={counts['written']} seen={counts['seen']} "
                        f"wps={wps:.2f}/s sps={sps:.2f}/s rss={rss_mb:.1f}MB "
                        f"ext_skip={counts['skipped_ext']} "
                        f"lic(noinfo)={counts['skipped_no_license_info']} "
                        f"lic(noallow)={counts['skipped_no_allowed_license']} "
                        f"lic(bad)={counts['skipped_disallowed_license']} "
                        f"cap_skip={counts['skipped_per_lang_cap']} "
                        f"exists={counts['exists_skipped']} errs={counts['errors']} "
                        f"per_lang={per_lang_counts}",
                        file=sys.stderr
                    )
                    last_report_t = now
                    last_written_at_report = counts["written"]
                    log_fp.flush(); manifest_fp.flush()
                    gc.collect()

                if counts["written"] >= args.total:
                    break

            # close & free everything from this language before moving on
            del ds
            gc.collect()

    if not any_loaded:
        sys.stderr.write("[error] No language streams could be opened. "
                         "Ensure you accepted the dataset ToS and that language names match the dataset folders (e.g., 'c++', not 'cpp').\n")
        sys.exit(2)

    elapsed = time.time() - t0
    wps = counts["written"] / elapsed if elapsed > 0 else 0.0
    sps = counts["seen"] / elapsed if elapsed > 0 else 0.0
    print(
        "==== Summary ====\n"
        f"Output directory: {args.out}\n"
        f"Manifest: {manifest_path}\n"
        f"Written: {counts['written']} files in {elapsed:.1f}s (write rate {wps:.2f}/s, seen rate {sps:.2f}/s)\n"
        f"Seen total: {counts['seen']}\n"
        f"Skipped: ext={counts['skipped_ext']}, lic(noinfo)={counts['skipped_no_license_info']}, "
        f"lic(noallow)={counts['skipped_no_allowed_license']}, lic(bad)={counts['skipped_disallowed_license']}, "
        f"per-lang-cap-skip={counts['skipped_per_lang_cap']}, "
        f"exists={counts['exists_skipped']}, errors={counts['errors']}\n"
        f"Per-language written: {per_lang_counts}\n",
        file=sys.stderr
    )
    print(f"Done. Wrote {counts['written']} files to {args.out} (manifest at {manifest_path}).")

if __name__ == "__main__":
    main()
