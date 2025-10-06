from __future__ import annotations
from pathlib import Path
import argparse
import re
import sys
from typing import Optional, List, Dict, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import math
import shutil
import multiprocessing as mp

# Keep native threadpools from over-subscribing inside workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --- Magika (bulk inference kept ONLY in main process) ---
from magika import Magika

# ============================================================
#               SANITIZATION & REGEX UTILITIES
# ============================================================

def strip_jinja_code(text: str) -> str:
    # remove Jinja2 blocks/comments without touching other whitespace
    return re.sub(r"{%.*?%}|{{.*?}}|{#.*?#}", "", text, flags=re.DOTALL)

def strip_php_code(text: str) -> str:
    # remove PHP blocks without touching other whitespace
    return re.sub(r"<\?php.*?\?>|<\?.*?\?>", "", text, flags=re.DOTALL)

HTML_LIKE_LABELS = {"html", "xhtml", "svg", "xml"}
HTML_LIKE_MIMES_PREFIX = ("text/html", "application/xhtml+xml", "image/svg+xml")

JS_EXTS = {"js"}
CSS_EXTS = {"css"}
HTML_EXTS = {"html", "htm", "xhtml", "xml", "svg"}

def is_html_like(label: Optional[str], mime: Optional[str]) -> bool:
    if label and label.lower() in HTML_LIKE_LABELS:
        return True
    if mime:
        return mime.startswith(HTML_LIKE_MIMES_PREFIX)
    return False

_DIR_ACCEPTS: Dict[str, set] = {
    "text": {"txt"},         # Magika uses "txt" for generic text
    "c__":  {"cpp", "c++"},  # your dataset's C++ bucket
    # everything else: strict equality via fallback
}

def dir_matches_label(dir_label: Optional[str], label: Optional[str], mime: Optional[str]) -> bool:
    """
    True if Magika's label is considered a match for the directory.
    Strict by default; only the aliases above are allowed differences.
    """
    if dir_label is None or not label:
        return False

    dl = dir_label.lower()
    ll = label.lower()

    allowed = _DIR_ACCEPTS.get(dl)
    if allowed is not None:
        return ll in allowed

    # Strict equality for all others (including 'html', 'svg', 'xml', etc.)
    return ll == dl

# ============================================================
#               XML CHAR LEGALITY (HARD-DELETE)
# ============================================================

def _is_xml_legal_char_ord(cp: int) -> bool:
    # XML 1.0 legal set
    return (
        cp == 0x9 or cp == 0xA or cp == 0xD
        or (0x20 <= cp <= 0xD7FF)
        or (0xE000 <= cp <= 0xFFFD)
        or (0x10000 <= cp <= 0x10FFFF)
    )

def contains_illegal_xml_char(s: str) -> bool:
    return any(not _is_xml_legal_char_ord(ord(ch)) for ch in s)

# ============================================================
#           HTML/SVG CLEANING (regex, no normalization)
# ============================================================

# Tokens that match real or encoded angle brackets
_LT = r"(?:<|&lt;|&#0*60;|&#x0*3[cC];)"
_GT = r"(?:>|&gt;|&#0*62;|&#x0*3[eE];)"

# Remove <script>...</script> (real or encoded)
_SCRIPT_BLOCK_RE = re.compile(
    rf"(?is){_LT}\s*script\b(?:(?!{_GT}).)*{_GT}.*?{_LT}\s*/\s*script\b(?:(?!{_GT}).)*{_GT}"
)

# Remove <style>...</style> (real or encoded)
_STYLE_BLOCK_RE = re.compile(
    rf"(?is){_LT}\s*style\b(?:(?!{_GT}).)*{_GT}.*?{_LT}\s*/\s*style\b(?:(?!{_GT}).)*{_GT}"
)

# Remove <link ... rel="stylesheet" ...> (real or encoded)
_LINK_STYLESHEET_RE = re.compile(
    rf"(?is){_LT}\s*link\b(?:(?!{_GT}).)*\brel\s*=\s*(?:'[^']*stylesheet[^']*'|\"[^\"]*stylesheet[^\"]*\"|[^\s>]+)(?:(?!{_GT}).)*{_GT}"
)

# Remove <meta http-equiv="refresh" ...> (real or encoded)
_META_REFRESH_RE = re.compile(
    rf"(?is){_LT}\s*meta\b(?:(?!{_GT}).)*\bhttp-equiv\s*=\s*(?:'refresh'|\"refresh\"|refresh)(?:(?!{_GT}).)*{_GT}"
)

# Generic "tag" wrapper that captures the open token, inside, close token
_TAG_WRAPPER_RE = re.compile(rf"(?is)({_LT})(.*?)(({_GT}))")

# Event-like attributes: on..., :on..., @click, x-on..., hx-on..., th:on...
_EVENT_ATTR_NAME_RE = r"(?:on[\w:-]*|[\w:-]*:on[\w:-]*|@[\w:-]+|x-on[\w:-]*|hx-on[\w:-]*)"
_EVENT_ATTR_RE = re.compile(
    rf"(?is)(?:\s+)(?P<name>{_EVENT_ATTR_NAME_RE})\s*=\s*(?P<val>'[^']*'|\"[^\"]*\"|[^\s'\"`>]+)"
)

# style= attributes (strip entirely)
_STYLE_ATTR_RE = re.compile(
    r"(?is)(?:\s+)style\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s'\"`>]+)"
)

# Dangerous URI attributes (javascript: URLs) to scrub
_URI_ATTR_RE = re.compile(
    r"(?is)(?:\s+)(?P<name>href|src|xlink:href|formaction|action|data|poster)\s*=\s*(?P<val>'[^']*'|\"[^\"]*\"|[^\s'\"`>]+)"
)

# Detect "javascript:" scheme even when obfuscated and optionally wrapped in url(...)
_JAVASCRIPT_SCHEME_RE = re.compile(
    r"""(?is)^\s*(?:url\(\s*)?
        j[^a-z0-9]*a[^a-z0-9]*v[^a-z0-9]*a[^a-z0-9]*s[^a-z0-9]*c[^a-z0-9]*r[^a-z0-9]*i[^a-z0-9]*p[^a-z0-9]*t[^a-z0-9]*\s*
        (?::|&\#x0*3a;|&\#0*58;|&colon;)
    """,
    re.VERBOSE,
)

# Treat these named entities as ignorable whitespace in URL scheme checking
_WHITESPACE_ENTITIES_RE = re.compile(r"(?i)&(?:tab|newline|nbsp|linefeed|cr|lf|space);")

def _strip_attrs_from_tag_inner(inner: str) -> str:
    """
    Remove dangerous attributes from a tag 'inner' (text between <...> or &lt;...&gt;).
    - removes event-like attributes (on*, *:on*, @click/x-on*/hx-on*)
    - removes style= entirely
    - removes URI attributes if value looks like javascript:
    """
    # Remove event-like attributes (repeat until stable in case of multiple)
    prev = None
    while prev != inner:
        prev = inner
        inner = _EVENT_ATTR_RE.sub("", inner)

    # Remove style= attributes
    prev = None
    while prev != inner:
        prev = inner
        inner = _STYLE_ATTR_RE.sub("", inner)

    # Remove dangerous URIs
    def _uri_repl(m: re.Match) -> str:
        raw_val = m.group("val")
        # Strip quotes if present
        if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in "'\"":
            val = raw_val[1:-1]
        else:
            val = raw_val

        # Normalize colon entities and collapse whitespace-like entities
        val_norm = _WHITESPACE_ENTITIES_RE.sub("", val)
        val_norm = re.sub(r"(?i)(?:&colon;|&#0*58;|&#x0*3a;)", ":", val_norm)

        if _JAVASCRIPT_SCHEME_RE.search(val_norm):
            return ""  # drop entire attribute
        return m.group(0)  # keep as-is

    inner = _URI_ATTR_RE.sub(_uri_repl, inner)
    return inner

def clean_html_with_regex(text: str) -> str:
    """
    Remove any JavaScript/CSS without re-serializing HTML.
    - Strips script/style blocks (real or HTML-encoded tags)
    - Strips <link rel=stylesheet>
    - Removes meta refresh
    - Removes event-like attributes, style=, and javascript: URIs inside tags
    No indentation/whitespace normalization beyond deletions.
    """
    # Optionally pre-strip server-template blocks that may contain scripts/styles
    pre = strip_php_code(strip_jinja_code(text))

    # Remove script/style blocks (iterate a few times to be safe with nesting/odd markup)
    cur = pre
    for _ in range(4):
        nxt = _SCRIPT_BLOCK_RE.sub("", cur)
        nxt = _STYLE_BLOCK_RE.sub("", nxt)
        if nxt == cur:
            break
        cur = nxt

    # Remove stylesheet link tags and meta refresh
    cur = _LINK_STYLESHEET_RE.sub("", cur)
    cur = _META_REFRESH_RE.sub("", cur)

    # Scrub attributes inside both real and &lt;encoded&gt; tags
    def _tag_repl(m: re.Match) -> str:
        open_tok, inner, close_tok = m.group(1), m.group(2), m.group(3)
        cleaned_inner = _strip_attrs_from_tag_inner(inner)
        return f"{open_tok}{cleaned_inner}{close_tok}"

    cur = _TAG_WRAPPER_RE.sub(_tag_repl, cur)

    return cur

# ============================================================
#         FRAMEWORK / COMBINED CONTENT DETECTION (optional)
# ============================================================

# JS imports/requires for popular web frameworks (used on JS files)
_FRAMEWORK_MODULE_RE = re.compile(
    r"""(?ix)
    (?: \bfrom\s+["']|
        \brequire\(\s*["']|
        \bimport\s+[^;]*?\s+from\s+["']|
        \bimport\s*["'] )
    (
      react(-dom)?|
      preact|
      vue|@vue/[^"']*|
      svelte|
      solid-js|
      lit(?:-html|-element)?|
      alpine(?:js)?|
      ember|@ember/[^"']*|
      next|nuxt|astro|qwik|
      @angular/[^"']*
    )
    ["']\s*\)?""",
    re.DOTALL,
)

# HTML script src loading those frameworks
_FRAMEWORK_SCRIPT_SRC_RE = re.compile(
    r"""(?is)<script[^>]+src\s*=\s*["'][^"']*
    (react(?:-dom)?|preact|vue|angular|svelte|solid|lit|alpine|ember|next|nuxt|astro|qwik)
    [^"']*["']""",
)

# Vue SFC / combined markers (works for .vue; also catches HTML-ish JS)
_VUE_SFC_RE = re.compile(r"(?is)<template\b.*?</template>.*?<script\b", re.DOTALL)
# Other combined-type hints (Astro/Svelte/MDX files by extension)
_COMBINED_EXTS = {".vue", ".svelte", ".astro", ".mdx"}

def should_delete_framework_or_combined(
    text: str,
    *,
    label: str,
    path: str,
    delete_framework_imports: bool,
    delete_combined: bool,
) -> Optional[str]:
    """
    Returns a reason string if file should be deleted, else None.
    """
    if delete_combined:
        ext = Path(path).suffix.lower()
        if ext in _COMBINED_EXTS:
            return f"combined-ext:{ext}"
        if _VUE_SFC_RE.search(text):
            return "combined:vue-sfc-markers"

    if delete_framework_imports:
        if label in {"javascript", "css"}:
            if label == "javascript" and _FRAMEWORK_MODULE_RE.search(text):
                return "framework-import"
        if label in {"html", "xhtml", "xml", "svg"} or is_html_like(label, None):
            if _FRAMEWORK_SCRIPT_SRC_RE.search(text):
                return "framework-script-src"

    return None

# ============================================================
#                   FILE PROCESSING (WORKERS)
# ============================================================

def write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")

def worker_delete(path: str, *, dry_run: bool, reason: str = "unidentified") -> Dict[str, object]:
    p = Path(path)
    try:
        if not dry_run:
            p.unlink()
        return {"path": path, "action": "deleted", "reason": reason, "removed_chars": 0}
    except Exception as e:
        return {"path": path, "action": "error", "reason": f"delete {reason} failed: {e}", "removed_chars": 0}

def worker_clean(
    path: str,
    *,
    label: str,
    mime: Optional[str],
    dir_label: Optional[str],
    dry_run: bool,
    process_svg: bool,
    keep_wrong_type: bool,
    delete_mismatch: bool,
    delete_framework_imports: bool,
    delete_combined: bool,
) -> Dict[str, object]:
    """
    Runs in a separate process. No Magika here.
    HTML/SVG via robust regex cleaning (no pretty-printing). Others untouched.
    Optional deletion for framework imports / combined content.
    Directory-based classification: delete if Magika label != directory label (depending on flags).
    """
    p = Path(path)
    try:
        data = p.read_bytes()
        try:
            original = data.decode("utf-8")
        except UnicodeDecodeError:
            if not dry_run:
                try:
                    p.unlink()
                except Exception as e:
                    return {"path": path, "action": "error", "reason": f"delete invalid-utf8 failed: {e}", "removed_chars": 0}
            return {"path": path, "action": "deleted", "reason": "invalid-utf8", "removed_chars": 0}

        if contains_illegal_xml_char(original):
            if not dry_run:
                try:
                    p.unlink()
                except Exception as e:
                    return {"path": path, "action": "error", "reason": f"delete illegal-xml-char failed: {e}", "removed_chars": 0}
            return {"path": path, "action": "deleted", "reason": "illegal-xml-char", "removed_chars": 0}

        # Directory ↔ Magika mismatch policy
        mismatch = False
        if dir_label is None:
            # cannot determine class dir; treat as mismatch (unless keep_wrong_type)
            mismatch = True
            dir_display = "unknown-dir"
        else:
            dir_display = dir_label
            if (delete_mismatch or not keep_wrong_type) and (not dir_matches_label(dir_label, label, mime)):
                mismatch = True

        if mismatch:
            reason = f"dir-mismatch:{dir_display}->{label or mime or 'unknown'}"
            return worker_delete(path, dry_run=dry_run, reason=reason)

        # Optional deletion for framework imports / combined content
        reason_fw = should_delete_framework_or_combined(
            original,
            label=label,
            path=path,
            delete_framework_imports=delete_framework_imports,
            delete_combined=delete_combined,
        )
        if reason_fw:
            return worker_delete(path, dry_run=dry_run, reason=reason_fw)

        # Actual processing for HTML-like
        if label in {"html", "xhtml", "xml", "svg"} or is_html_like(label, mime):
            if label == "svg" and not process_svg:
                return {"path": path, "action": "skipped", "reason": None, "removed_chars": 0}
            cleaned = clean_html_with_regex(original)   # keep spacing; only deletions
            if cleaned != original:
                if not dry_run:
                    write_text(p, cleaned)
                removed = max(0, len(original) - len(cleaned))
                return {"path": path, "action": "changed", "reason": "html-regex-cleaned", "removed_chars": removed}
            return {"path": path, "action": "skipped", "reason": "asset-ok", "removed_chars": 0}

        # All other types: no comment stripping or normalization
        return {"path": path, "action": "skipped", "reason": "asset-ok", "removed_chars": 0}

    except Exception as e:
        return {"path": path, "action": "error", "reason": str(e), "removed_chars": 0}

# ============================================================
#                      PROGRESS RENDERING
# ============================================================

def _fmt_int(n: int) -> str:
    return f"{n:,}"

def _fmt_rate(items: float) -> str:
    if items < 1:
        return f"{items:.2f}/s"
    if items < 10:
        return f"{items:.1f}/s"
    return f"{int(items):d}/s"

def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds):
        return "ETA --:--"
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    if h:
        return f"ETA {h:02d}:{m:02d}:{s:02d}"
    return f"ETA {m:02d}:{s:02d}"

def _render_bar(progress: float, width: int = 20) -> str:
    progress = 0 if math.isnan(progress) else max(0.0, min(1.0, progress))
    filled = int(progress * width + 0.5)
    filled = min(filled, width)
    return "▰" * filled + "▱" * (width - filled)

def _update_progress_line(*, processed: int, total: int, changed: int, deleted: int,
                          errors: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(total - processed, 0)
    eta = remaining / rate if rate > 0 else float("inf")

    cols = shutil.get_terminal_size((100, 20)).columns
    bar = _render_bar(processed / total if total else 0.0, width=20)
    parts = [
        bar,
        f"{_fmt_int(processed)}/{_fmt_int(total)}",
        f"|  {_fmt_int(changed)} changed, {_fmt_int(deleted)} deleted, {_fmt_int(errors)} errors",
        f"|  {_fmt_rate(rate)}",
        f"|  {_fmt_eta(eta)}",
    ]
    line = "  ".join(parts)
    if len(line) > cols:
        line = line[: max(10, cols - 1)]
    sys.stderr.write("\r" + line + " " * max(0, cols - len(line) - 1))
    sys.stderr.flush()

# ============================================================
#                           MAIN
# ============================================================

def iter_files(base: Path, splits: List[str]) -> Iterable[str]:
    """
    Yield ALL files under the selected splits.
    Directory layout is expected as: <base>/<split>/<dir_label>/.../file
    where <split> in {train, val, test} and <dir_label> is the target Magika label.
    """
    for split in splits:
        split_root = base / split
        if not split_root.exists():
            print(f"[!] Split directory not found: {split_root}", file=sys.stderr)
            continue
        for root, _dirs, files in os.walk(split_root):
            for fname in files:
                yield os.path.join(root, fname)

def get_dir_label_for_path(base: Path, path: Path) -> Optional[str]:
    """
    Extract the directory label (the first component under the split dir).
    Returns None if the path doesn't look like <base>/<split>/<dir_label>/...
    """
    try:
        rel = path.relative_to(base)
    except Exception:
        return None
    parts = rel.parts
    # Expect at least: split / dir_label / file
    if len(parts) < 3:
        return None
    split = parts[0].lower()
    if split not in {"train", "test", "val"}:
        return None
    dir_label = parts[1].lower()
    return dir_label

def batched(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Classifier/cleaner based on DIRECTORY label (not extensions):\n"
            "- A file must live under <split>/<label>/... and Magika's label must equal that directory name.\n"
            "- HTML/XML/SVG: regex removes JS/CSS/event attrs/meta-refresh (keeps comments & layout).\n"
            "- Others: untouched (no comment stripping).\n"
            "Magika bulk identification in main process; cleaning in ProcessPool (spawn).\n"
            "Optional deletion: framework imports and combined content."
        )
    )
    ap.add_argument("path", nargs="?", default="/home/s0urc10ud/pure-database/ml_trainer/downloader/stack_web_sample",
                    help="Base directory containing split folders (train/test/val)")
    ap.add_argument("--split", choices=["train", "test", "val", "all"], default="all",
                    help="Which data split to process (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report changes/deletions; do not modify or delete files")
    ap.add_argument("--no-svg", action="store_true", help="Detect SVG but do not process it")
    ap.add_argument("--keep-unidentified", action="store_true",
                    help="Do NOT delete files Magika cannot identify")
    ap.add_argument("--keep-wrong-type", action="store_true",
                    help="Do NOT delete files whose DIRECTORY label disagrees with Magika's label")
    ap.add_argument("--delete-mismatch", action="store_true",
                    help="Also delete when the DIRECTORY name doesn't match Magika's suggestion")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 8),
                    help="Number of **process** workers for cleaning (Magika stays in main)")
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="How many files to classify per Magika identify_paths() call")
    ap.add_argument("--progress", choices=["auto", "always", "never"], default="auto",
                    help="Show a live progress line while processing (default: auto)")
    ap.add_argument("--progress-interval", type=float, default=0.2,
                    help="Seconds between progress updates (default: 0.2)")
    ap.add_argument("--delete-framework-imports", action="store_true",
                    help="Delete files that import/load React, Vue, Angular, Svelte, Solid, Preact, Lit, Alpine, Ember, Next/Nuxt/Astro/Qwik")
    ap.add_argument("--delete-combined", action="store_true",
                    help="Delete likely combined content (e.g., Vue SFC markers, .vue/.svelte/.astro/.mdx files)")

    args = ap.parse_args()

    # Use 'spawn' to avoid inheriting any native state in workers
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    base = Path(args.path).resolve()
    if not base.exists() or not base.is_dir():
        print(f"[!] Directory not found: {base}", file=sys.stderr)
        sys.exit(2)

    splits = ["train", "test", "val"] if args.split == "all" else [args.split]

    files = list(iter_files(base, splits))
    total = len(files)
    if total == 0:
        print("No files found.")
        return

    print(f"Discovered {total} file(s). Using {args.workers} worker(s).")

    is_tty = sys.stderr.isatty()
    show_progress = ((args.progress == "always") or (args.progress == "auto" and is_tty))
    last_tick = 0.0
    start_time = time.time()

    changed = 0
    deleted = 0
    removed_chars_total = 0
    errors = 0
    processed = 0

    # One long-lived Magika in the main process
    magika = Magika()

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        try:
            for paths_batch in batched(files, args.batch_size):
                results = magika.identify_paths(paths_batch)

                futures = []
                for res in results:
                    path_str = str(res.path)
                    p = Path(path_str)

                    dir_label = get_dir_label_for_path(base, p)

                    if not res.ok:
                        if args.keep_unidentified:
                            processed += 1
                            continue
                        futures.append(pool.submit(
                            worker_delete,
                            path_str,
                            dry_run=args.dry_run,
                            reason="unidentified",
                        ))
                        continue

                    label = (res.output.label or "").lower()
                    mime = res.output.mime_type

                    futures.append(pool.submit(
                        worker_clean,
                        path_str,
                        label=label,
                        mime=mime,
                        dir_label=dir_label,
                        dry_run=args.dry_run,
                        process_svg=(not args.no_svg),
                        keep_wrong_type=args.keep_wrong_type,
                        delete_mismatch=args.delete_mismatch,
                        delete_framework_imports=args.delete_framework_imports,
                        delete_combined=args.delete_combined,
                    ))

                for fut in as_completed(futures):
                    try:
                        out = fut.result()
                    except Exception as e:
                        errors += 1
                        print(f"\n[ERROR] (worker): {e}", file=sys.stderr)
                        out = None

                    processed += 1
                    if out:
                        path = out.get("path")
                        action = out.get("action")
                        reason = out.get("reason")
                        removed = int(out.get("removed_chars", 0))

                        if action == "deleted":
                            deleted += 1
                            tag = "WOULD DELETE" if args.dry_run else "DELETED"
                            print(f"\n[{tag}] {path}" + (f" (reason: {reason})" if reason else ""))
                        elif action == "changed":
                            changed += 1
                            removed_chars_total += removed
                            tag = "WOULD CHANGE" if args.dry_run else "CHANGED"
                            print(f"\n[{tag}] {path}" + (f" (reason: {reason})" if reason else ""))
                        elif action == "error":
                            errors += 1
                            print(f"\n[ERROR] {path}: {reason}", file=sys.stderr)

                    if show_progress:
                        now = time.time()
                        if now - last_tick >= args.progress_interval or processed == total:
                            _update_progress_line(
                                processed=processed,
                                total=total,
                                changed=changed,
                                deleted=deleted,
                                errors=errors,
                                start_time=start_time
                            )
                            last_tick = now
        finally:
            if show_progress:
                sys.stderr.write("\n")
                sys.stderr.flush()

    print(f"\nScanned: {total} file(s)")
    print(f"Modified: {changed} file(s){' (dry run)' if args.dry_run else ''}")
    print(f"Deleted: {deleted} file(s){' (dry run)' if args.dry_run else ''}")
    if removed_chars_total:
        print(f"Estimated characters removed/replaced (in modified files): {removed_chars_total}")
    if errors:
        print(f"Errors: {errors} file(s)")

if __name__ == "__main__":
    main()
