from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg  # noqa: E402


DEFAULT_TARGET_LEN = 256
DEFAULT_OUTPUT_JSON = REPO_ROOT / "active_learning" / "benchmark_data" / "curated_oracle_segments_v1.json"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "active_learning" / "benchmark_data" / "curated_oracle_segments_v1.jsonl"
DEFAULT_OUTPUT_MARKDOWN = REPO_ROOT / "active_learning" / "benchmark_data" / "curated_oracle_segments_v1.md"

_LABEL_ALIASES = {
    "javascript": "javascript_typescript",
    "typescript": "javascript_typescript",
    "js": "javascript_typescript",
    "ts": "javascript_typescript",
    "c": "c_family",
    "cpp": "c_family",
    "c++": "c_family",
    "vb": "visual_basic",
    "gettext-catalog": "gettext_catalog",
}

ALLOWED_LABELS = set(cfg.LANG2ID.keys()) | {"other"}


@dataclass(frozen=True)
class SegmentPiece:
    label: str
    text: str


@dataclass(frozen=True)
class DraftSample:
    category: str
    mixed_truth: bool
    note: str
    pieces: Tuple[SegmentPiece, ...]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_label(label: str) -> str:
    low = str(label or "").strip().lower().replace("-", "_")
    low = _LABEL_ALIASES.get(low, low)
    if low in ALLOWED_LABELS:
        return low
    return "other"


def _escape_markdown_cell(text: str) -> str:
    out = text.replace("|", "\\|").replace("\n", "\\n")
    if len(out) > 72:
        out = out[:69] + "..."
    return out


def _merge_pieces(pieces: Sequence[SegmentPiece]) -> List[SegmentPiece]:
    merged: List[SegmentPiece] = []
    for piece in pieces:
        label = _canonical_label(piece.label)
        text = str(piece.text)
        if not text:
            continue
        if merged and merged[-1].label == label:
            merged[-1] = SegmentPiece(label=label, text=merged[-1].text + text)
        else:
            merged.append(SegmentPiece(label=label, text=text))
    return merged


def _slice_pieces(
    pieces: Sequence[SegmentPiece],
    *,
    start: int,
    end: int,
) -> List[SegmentPiece]:
    cursor = 0
    out: List[SegmentPiece] = []
    s = max(0, int(start))
    e = max(s, int(end))
    for piece in pieces:
        p_start = cursor
        p_end = cursor + len(piece.text)
        cursor = p_end
        if p_end <= s or p_start >= e:
            continue
        rel_start = max(0, s - p_start)
        rel_end = min(len(piece.text), e - p_start)
        if rel_end <= rel_start:
            continue
        out.append(SegmentPiece(label=piece.label, text=piece.text[rel_start:rel_end]))
    return _merge_pieces(out)


def _pieces_to_segments(pieces: Sequence[SegmentPiece]) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    cursor = 0
    for piece in pieces:
        end = cursor + len(piece.text)
        segments.append(
            {
                "start": int(cursor),
                "end": int(end),
                "label": _canonical_label(piece.label),
            }
        )
        cursor = end
    return segments


def _dense_labels(segments: Sequence[Mapping[str, object]], text_len: int) -> List[str]:
    labels = ["other"] * max(0, int(text_len))
    for segment in segments:
        start = max(0, min(text_len, int(segment["start"])))
        end = max(start, min(text_len, int(segment["end"])))
        label = _canonical_label(str(segment["label"]))
        for idx in range(start, end):
            labels[idx] = label
    return labels


def _segments_from_labels(labels: Sequence[str]) -> List[Dict[str, object]]:
    if not labels:
        return []
    out: List[Dict[str, object]] = []
    start = 0
    cur = _canonical_label(labels[0])
    for idx in range(1, len(labels)):
        nxt = _canonical_label(labels[idx])
        if nxt != cur:
            out.append({"start": int(start), "end": int(idx), "label": cur})
            start = idx
            cur = nxt
    out.append({"start": int(start), "end": int(len(labels)), "label": cur})
    return out


def _boundary_near_center(segments: Sequence[Mapping[str, object]], text_len: int) -> int:
    boundaries = [int(seg["end"]) for seg in segments[:-1]]
    if not boundaries:
        return int(text_len // 2)
    center = int(text_len // 2)
    return min(boundaries, key=lambda value: abs(value - center))


def _predict_from_truth(
    truth_segments: Sequence[Mapping[str, object]],
    text_len: int,
    *,
    mixed: bool,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    truth = _dense_labels(truth_segments, text_len)
    if not mixed:
        return _segments_from_labels(truth)

    pred = list(truth)
    boundaries = [idx for idx in range(1, text_len) if truth[idx] != truth[idx - 1]]
    for boundary in boundaries:
        shift = int(rng.integers(-4, 5))
        if shift == 0:
            continue
        if shift < 0:
            fill = truth[boundary]
            for idx in range(max(0, boundary + shift), boundary):
                pred[idx] = fill
        else:
            fill = truth[boundary - 1]
            for idx in range(boundary, min(text_len, boundary + shift)):
                pred[idx] = fill
    return _segments_from_labels(pred)


def _hex_blob(i: int) -> str:
    raw = f"hex-sample-{i:03d}-payload".encode("utf-8")
    return raw.hex()


def _base64_blob(i: int) -> str:
    raw = f"base64-sample-{i:03d}-payload".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _base32_blob(i: int) -> str:
    raw = f"base32-sample-{i:03d}-payload".encode("utf-8")
    return base64.b32encode(raw).decode("ascii")


def _base58_blob(i: int) -> str:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    h = hashlib.sha256(f"base58-{i}".encode("utf-8")).digest()
    n = int.from_bytes(h, "big")
    out = []
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(alphabet[rem])
    encoded = "".join(reversed(out)) or "1"
    return encoded[:48]


def _base85_blob(i: int) -> str:
    raw = f"base85-sample-{i:03d}-payload".encode("utf-8")
    return base64.a85encode(raw).decode("ascii")


def _pure_variant_text(label: str, i: int) -> str:
    l = _canonical_label(label)
    if l == "python":
        return f"def variant_{i}(x:int)->int:\n    return x + {i}\n"
    if l == "javascript_typescript":
        return f"export const variant{i}=(x:number)=>x+{i};\n"
    if l == "go":
        return f"package main\nfunc variant{i}(x int) int {{ return x + {i} }}\n"
    if l == "sql":
        return f"SELECT id, score FROM runs WHERE score > {i} ORDER BY score DESC;\n"
    if l == "rust":
        return f"pub fn variant_{i}(x:i32)->i32{{x+{i}}}\n"
    if l == "yaml":
        return f"name: variant_{i}\nthreshold: {i}\nenabled: true\n"
    if l == "ruby":
        return f"def variant_{i}(x)\n  x + {i}\nend\n"
    if l == "java":
        return f"class V{i}{{static int run(int x){{return x+{i};}}}}\n"
    if l == "c_family":
        return f"int variant_{i}(int x){{return x+{i};}}\n"
    if l == "json":
        return json.dumps({"name": f"variant_{i}", "threshold": i, "enabled": True}) + "\n"
    if l == "css":
        return f".v{i}{{padding:{i%12+1}px;color:#1a1a1a;background:#f{i%10}{(i+1)%10}{(i+2)%10};}}\n"
    if l == "html":
        return f"<div class=\"v{i}\" data-k=\"{i}\">variant {i}</div>\n"
    if l == "text":
        return f"Plain text variant {i}. Keep this sample as text only.\n"
    if l == "csv":
        return f"id,label,score\n{i},variant_{i},{(i%97)+1}\n"
    if l == "shell":
        return f"#!/usr/bin/env bash\nprintf '%s\\n' \"variant-{i}\" \"done\"\n"
    if l == "powershell":
        return f"$v={i}\nWrite-Output \"variant-$v\"\n"
    if l == "visual_basic":
        return f"Function Variant{i}(x As Integer) As Integer\n  Variant{i} = x + {i}\nEnd Function\n"
    if l == "dockerfile":
        return f"FROM alpine:3.20\nARG BUILD={i}\nRUN echo \"variant {i}\"\n"
    if l == "dart":
        return f"int variant{i}(int x) => x + {i};\n"
    if l == "gettext_catalog":
        return f"msgid \"variant_{i}\"\nmsgstr \"wert_{i}\"\n"
    if l == "kotlin":
        return f"fun variant{i}(x:Int):Int=x+{i}\n"
    if l == "markdown":
        return f"# Variant {i}\nThis is markdown variant `{i}`.\n"
    if l == "restructuredtext":
        return f"Variant {i}\n=========\n\nThis is rst variant {i}.\n"
    if l == "scala":
        return f"object V{i}{{def run(x:Int)=x+{i}}}\n"
    if l == "swift":
        return f"func variant{i}(_ x:Int)->Int{{x+{i}}}\n"
    if l == "tex":
        return f"\\section{{Variant {i}}}\n\\texttt{{value={i}}}\n"
    if l == "xml":
        return f"<variant id=\"{i}\"><value>{i}</value></variant>\n"
    if l == "svg":
        return f"<svg viewBox=\"0 0 20 20\"><text x=\"1\" y=\"10\">v{i}</text></svg>\n"
    if l == "php":
        return f"<?php function variant_{i}($x){{ return $x + {i}; }} ?>\n"
    if l == "csharp":
        return f"static int Variant{i}(int x) {{ return x + {i}; }}\n"
    if l == "encoding_hex":
        return _hex_blob(1000 + i) + "\n"
    if l == "encoding_base64":
        return _base64_blob(1000 + i) + "\n"
    if l == "encoding_base32":
        return _base32_blob(1000 + i) + "\n"
    if l == "encoding_base58":
        return _base58_blob(1000 + i) + "\n"
    if l == "encoding_base85":
        return _base85_blob(1000 + i) + "\n"
    return f"variant {i} for label {l}\n"


def _build_exact_window_from_real_content(
    pieces: Sequence[SegmentPiece],
    *,
    target_len: int,
    draft_idx: int,
) -> List[SegmentPiece]:
    merged = _merge_pieces(pieces)
    if not merged:
        raise ValueError("Cannot build sample from empty pieces.")

    expanded = list(merged)
    total = sum(len(piece.text) for piece in expanded)
    round_idx = 0
    while total < target_len:
        appended = 0
        for piece_idx, piece in enumerate(merged):
            variant_id = int(draft_idx * 1024 + round_idx * len(merged) + piece_idx)
            chunk = _pure_variant_text(piece.label, variant_id)
            if not chunk:
                continue
            expanded.append(SegmentPiece(label=piece.label, text=chunk))
            total += len(chunk)
            appended += len(chunk)
            if total >= target_len:
                break
        round_idx += 1
        if appended <= 0 or round_idx > 128:
            raise ValueError("Unable to extend sample with real content to target length.")

    expanded = _merge_pieces(expanded)
    total = sum(len(piece.text) for piece in expanded)
    if total == target_len:
        return expanded

    start = 0
    end = target_len
    sliced = _slice_pieces(expanded, start=start, end=end)
    final_len = sum(len(piece.text) for piece in sliced)
    if final_len != target_len:
        raise ValueError(f"Final sample length mismatch: {final_len} != {target_len}")
    return sliced


def _pure_drafts() -> List[DraftSample]:
    base: List[Tuple[str, str, str]] = [
        ("python", "def score_batch(values):\n    total = sum(values)\n    return total / max(1, len(values))\n", "pure python"),
        ("javascript_typescript", "export function scoreBatch(values:number[]):number{return values.reduce((a,b)=>a+b,0)/Math.max(1,values.length);}\n", "pure ts"),
        ("html", "<section class=\"card\"><h2>Batch Report</h2><p>Keep bytes stable.</p></section>\n", "pure html"),
        ("css", ".card{display:grid;gap:8px;padding:12px;border:1px solid #444;background:#fafafa;}\n", "pure css"),
        ("sql", "SELECT model_name, AVG(score) AS avg_score FROM run_metrics GROUP BY model_name ORDER BY avg_score DESC;\n", "pure sql"),
        ("yaml", "name: benchmark\nversion: 1\nsettings:\n  sample_length: 256\n  enforce_exact_bytes: true\n", "pure yaml"),
        ("ruby", "def score_batch(values)\n  values.sum.to_f / [values.length, 1].max\nend\n", "pure ruby"),
        ("go", "package main\nfunc scoreBatch(v []float64) float64 { var s float64; for _,x := range v { s+=x }; return s/float64(len(v)) }\n", "pure go"),
        ("rust", "pub fn score_batch(v:&[f64])->f64{let s:f64=v.iter().sum();s/(v.len().max(1) as f64)}\n", "pure rust"),
        ("java", "class Score{static double score(double[] v){double s=0;for(double x:v)s+=x;return s/Math.max(1,v.length);}}\n", "pure java"),
        ("csharp", "static double ScoreBatch(double[] v){double s=0;foreach(var x in v){s+=x;}return s/Math.Max(1,v.Length);} \n", "pure csharp"),
        ("c_family", "double score_batch(double* v,int n){double s=0;for(int i=0;i<n;i++)s+=v[i];return s/(n>0?n:1);} \n", "pure c-family"),
        ("json", "{\"name\":\"bench\",\"sample_length\":256,\"labels\":[\"html\",\"css\",\"javascript_typescript\"]}\n", "pure json"),
        ("shell", "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"running benchmark\" \"done\"\n", "pure shell"),
        ("powershell", "$ErrorActionPreference='Stop'\n$vals=1,2,3\n$avg=($vals|Measure-Object -Average).Average\n", "pure powershell"),
        ("visual_basic", "Function ScoreBatch(vals() As Double) As Double\n  ScoreBatch = 0\nEnd Function\n", "pure visual basic"),
        ("dockerfile", "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nCMD [\"python\",\"main.py\"]\n", "pure dockerfile"),
        ("dart", "double scoreBatch(List<double> v){final s=v.fold(0.0,(a,b)=>a+b);return s/(v.isEmpty?1:v.length);} \n", "pure dart"),
        ("gettext_catalog", "msgid \"run\"\nmsgstr \"execute\"\n\nmsgid \"score\"\nmsgstr \"wert\"\n", "pure gettext"),
        ("kotlin", "fun scoreBatch(v:List<Double>):Double{val s=v.sum();return s/(if(v.isEmpty())1 else v.size)}\n", "pure kotlin"),
        ("markdown", "# Benchmark\nThis snippet should stay markdown only.\n- Keep bytes exact.\n- Score by label.\n", "pure markdown"),
        ("restructuredtext", "Benchmark\n=========\n\nThis is reStructuredText only.\n\n- Keep bytes exact.\n", "pure rst"),
        ("scala", "object Score{def scoreBatch(v:Seq[Double]):Double=v.sum/math.max(1,v.size)}\n", "pure scala"),
        ("swift", "func scoreBatch(_ v:[Double])->Double{let s=v.reduce(0,+);return s/Double(max(1,v.count))}\n", "pure swift"),
        ("tex", "\\section{Benchmark}\\texttt{Keep bytes exact.}\\\\\n\\begin{itemize}\\item score\\end{itemize}\n", "pure tex"),
        ("xml", "<report><title>Benchmark</title><note>Keep bytes exact.</note></report>\n", "pure xml"),
        ("svg", "<svg viewBox=\"0 0 10 10\"><rect width=\"10\" height=\"10\" fill=\"#0af\"/></svg>\n", "pure svg"),
        ("csv", "id,label,score\n1,html,0.97\n2,css,0.91\n3,javascript_typescript,0.93\n", "pure csv"),
        ("text", "This is plain text only. It talks about stable bytes, benchmark labels, and reviewability.\n", "pure text"),
        ("php", "<?php\nfunction score_batch($v){$s=array_sum($v);return $s/max(1,count($v));}\n?>\n", "pure php"),
    ]
    out: List[DraftSample] = []
    for label, text, note in base:
        out.append(
            DraftSample(
                category=f"pure_{label}",
                mixed_truth=False,
                note=note,
                pieces=(SegmentPiece(label=label, text=text),),
            )
        )

    # Extra pure variants for higher-variance languages.
    for i in range(120):
        out.append(
            DraftSample(
                category="pure_python_variant",
                mixed_truth=False,
                note=f"pure python variant {i}",
                pieces=(
                    SegmentPiece(
                        label="python",
                        text=(
                            f"def boundary_{i}(text:str)->int:\n"
                            f"    marker = 'split-{i}'\n"
                            f"    return text.find(marker)\n"
                        ),
                    ),
                ),
            )
        )
    for i in range(120):
        out.append(
            DraftSample(
                category="pure_javascript_variant",
                mixed_truth=False,
                note=f"pure javascript variant {i}",
                pieces=(
                    SegmentPiece(
                        label="javascript_typescript",
                        text=(
                            f"const run{i}=input=>{{const m='split-{i}';return input.indexOf(m);}};\n"
                            f"export default run{i};\n"
                        ),
                    ),
                ),
            )
        )
    for i in range(120):
        out.append(
            DraftSample(
                category="pure_shell_variant",
                mixed_truth=False,
                note=f"pure shell variant {i}",
                pieces=(
                    SegmentPiece(
                        label="shell",
                        text=(
                            "#!/usr/bin/env bash\n"
                            f"set -e\nprintf '%s\\n' \"phase-{i}\" \"done\"\n"
                        ),
                    ),
                ),
            )
        )
    out.append(
        DraftSample(
            category="pure_text_variant",
            mixed_truth=False,
            note="pure text variant final",
            pieces=(
                SegmentPiece(
                    label="text",
                    text=(
                        "Plain text benchmark note: this line intentionally avoids markup, "
                        "code fences, and typed wrappers so it remains a strict single-label sample.\n"
                    ),
                ),
            ),
        )
    )

    # Broad per-label variants to make large curated subsets label-balanced.
    for label in sorted(cfg.LANG2ID.keys()):
        for i in range(48):
            out.append(
                DraftSample(
                    category=f"pure_balanced_{label}",
                    mixed_truth=False,
                    note=f"balanced pure {label} variant {i}",
                    pieces=(SegmentPiece(label=label, text=_pure_variant_text(label, i)),),
                )
            )
    return out


def _mixed_drafts() -> List[DraftSample]:
    out: List[DraftSample] = []
    colors = ["#ff0044", "#0055ff", "#00aa66", "#bb6600", "#6633cc", "#0099aa"]
    actions = ["save", "delete", "archive", "retry", "publish", "deploy"]

    for i in range(108):
        color = colors[i % len(colors)]
        action = actions[i % len(actions)]
        out.append(
            DraftSample(
                category="mixed_html_css_js_inline",
                mixed_truth=True,
                note=f"inline style + onclick {i}",
                pieces=(
                    SegmentPiece("html", f"<button class=\"btn-{i}\" style=\""),
                    SegmentPiece("css", f"color:{color};padding:{i+4}px {i+8}px;border:1px solid #222;"),
                    SegmentPiece("html", "\" onclick=\""),
                    SegmentPiece("javascript_typescript", f"return window.confirm('{action}-{i}?')"),
                    SegmentPiece("html", "\">Run</button>\n"),
                ),
            )
        )

    for i in range(84):
        out.append(
            DraftSample(
                category="mixed_markdown_shell_fence",
                mixed_truth=True,
                note=f"markdown fenced shell {i}",
                pieces=(
                    SegmentPiece("markdown", f"## Deploy {i}\nRun the command:\n```bash\n"),
                    SegmentPiece("shell", f"kubectl apply -f deploy-{i}.yaml\nkubectl rollout status deploy/api-{i}\n"),
                    SegmentPiece("markdown", "```\nIf rollout fails, inspect logs and retry.\n"),
                ),
            )
        )

    for i in range(60):
        out.append(
            DraftSample(
                category="mixed_markdown_frontmatter_html",
                mixed_truth=True,
                note=f"markdown with yaml frontmatter and html {i}",
                pieces=(
                    SegmentPiece(
                        "yaml",
                        (
                            "---\n"
                            f"title: Benchmark {i}\n"
                            f"owner: team-{i%4}\n"
                            "tags: [segment, review]\n"
                            "---\n"
                        ),
                    ),
                    SegmentPiece("markdown", f"\n# Status {i}\nUse the panel below.\n"),
                    SegmentPiece("html", f"<div class=\"note\" data-id=\"{i}\">rendered panel</div>\n"),
                    SegmentPiece("markdown", "\nKeep bytes exact.\n"),
                ),
            )
        )

    for i in range(72):
        out.append(
            DraftSample(
                category="mixed_dockerfile_shell",
                mixed_truth=True,
                note=f"dockerfile run shell {i}",
                pieces=(
                    SegmentPiece("dockerfile", "FROM python:3.11-slim\nWORKDIR /app\nRUN "),
                    SegmentPiece("shell", f"set -eux; pip install -r req-{i}.txt; python -m compileall ."),
                    SegmentPiece("dockerfile", "\nCOPY . .\nCMD [\"python\",\"main.py\"]\n"),
                ),
            )
        )

    for i in range(60):
        color = colors[(i + 2) % len(colors)]
        out.append(
            DraftSample(
                category="mixed_svg_css_attr",
                mixed_truth=True,
                note=f"svg style attr css split {i}",
                pieces=(
                    SegmentPiece("svg", f"<svg viewBox=\"0 0 120 40\"><text style=\""),
                    SegmentPiece("css", f"font-size:{12+i}px;fill:{color};stroke:none;"),
                    SegmentPiece("svg", f"\">Batch {i}</text></svg>\n"),
                ),
            )
        )

    for i in range(60):
        blob = _base64_blob(i)
        out.append(
            DraftSample(
                category="mixed_js_base64",
                mixed_truth=True,
                note=f"javascript + base64 blob {i}",
                pieces=(
                    SegmentPiece("javascript_typescript", f"const payload{i} = \""),
                    SegmentPiece("encoding_base64", blob),
                    SegmentPiece("javascript_typescript", f"\";\nconsole.log(atob(payload{i}));\n"),
                ),
            )
        )

    for i in range(48):
        blob = _hex_blob(i)
        out.append(
            DraftSample(
                category="mixed_js_hex",
                mixed_truth=True,
                note=f"javascript + hex blob {i}",
                pieces=(
                    SegmentPiece("javascript_typescript", f"const payloadHex{i} = \""),
                    SegmentPiece("encoding_hex", blob),
                    SegmentPiece("javascript_typescript", f"\";\nconsole.log(payloadHex{i}.length);\n"),
                ),
            )
        )

    for i in range(48):
        out.append(
            DraftSample(
                category="mixed_python_sql",
                mixed_truth=True,
                note=f"python + sql string {i}",
                pieces=(
                    SegmentPiece("python", "query = \"\"\"\n"),
                    SegmentPiece("sql", f"SELECT user_id, score FROM leaderboard WHERE score > {100+i} ORDER BY score DESC;\n"),
                    SegmentPiece("python", "\"\"\"\nprint(query)\n"),
                ),
            )
        )

    for i in range(48):
        out.append(
            DraftSample(
                category="mixed_php_html_js",
                mixed_truth=True,
                note=f"php + html + js handler {i}",
                pieces=(
                    SegmentPiece("php", "<?php $name = \"runner\"; ?>\n"),
                    SegmentPiece("html", "<a href=\"#\" onclick=\""),
                    SegmentPiece("javascript_typescript", f"return confirm('remove-{i}?')"),
                    SegmentPiece("html", "\">Delete</a>\n"),
                    SegmentPiece("php", "<?php echo $name; ?>\n"),
                ),
            )
        )

    for i in range(48):
        out.append(
            DraftSample(
                category="mixed_rst_shell",
                mixed_truth=True,
                note=f"restructuredtext + shell block {i}",
                pieces=(
                    SegmentPiece("restructuredtext", f"Build Step {i}\n=============\n\nRun this command::\n\n"),
                    SegmentPiece("shell", f"  make stage-{i}\n  make verify-{i}\n"),
                    SegmentPiece("restructuredtext", "\nThen continue with deployment.\n"),
                ),
            )
        )

    for i in range(48):
        out.append(
            DraftSample(
                category="mixed_xml_js_cdata",
                mixed_truth=True,
                note=f"xml with js in cdata {i}",
                pieces=(
                    SegmentPiece("xml", "<root><script><![CDATA["),
                    SegmentPiece("javascript_typescript", f"if(flag{i}){{console.log('x{i}')}}"),
                    SegmentPiece("xml", "]]></script></root>\n"),
                ),
            )
        )

    for i in range(36):
        blob32 = _base32_blob(i)
        blob58 = _base58_blob(i)
        blob85 = _base85_blob(i)
        out.append(
            DraftSample(
                category="mixed_multi_encoding",
                mixed_truth=True,
                note=f"text mixed with three encoding formats {i}",
                pieces=(
                    SegmentPiece("text", f"payloads-{i}: base32="),
                    SegmentPiece("encoding_base32", blob32),
                    SegmentPiece("text", " base58="),
                    SegmentPiece("encoding_base58", blob58),
                    SegmentPiece("text", " base85="),
                    SegmentPiece("encoding_base85", blob85),
                    SegmentPiece("text", "\n"),
                ),
            )
        )

    return out


def _drafts_to_samples(
    drafts: Sequence[DraftSample],
    *,
    target_len: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, object]] = []
    seen_hashes: set[str] = set()

    for idx, draft in enumerate(drafts):
        pieces = _build_exact_window_from_real_content(
            draft.pieces,
            target_len=target_len,
            draft_idx=idx,
        )
        text = "".join(piece.text for piece in pieces)
        if len(text) != target_len:
            raise ValueError(f"Invalid built sample length: {len(text)} != {target_len}")
        text_hash = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        truth_segments = _pieces_to_segments(pieces)
        truth_labels = _dense_labels(truth_segments, target_len)
        uniq_labels = sorted(set(truth_labels))
        mixed = bool(draft.mixed_truth)
        if mixed and len(uniq_labels) < 2:
            continue
        if (not mixed) and len(uniq_labels) != 1:
            continue
        predicted_segments = _predict_from_truth(
            truth_segments,
            target_len,
            mixed=mixed,
            rng=rng,
        )
        boundary = _boundary_near_center(truth_segments, target_len)
        sample_id = f"curated-{len(rows):04d}"
        rows.append(
            {
                "snippet_id": sample_id,
                "task": str(draft.category),
                "example_id": sample_id,
                "text": text,
                "boundary": int(boundary),
                "mixed_truth": mixed,
                "source_langs": uniq_labels,
                "truth_segments": truth_segments,
                "predicted_segments": predicted_segments,
                "metadata": {
                    "note": str(draft.note),
                    "category": str(draft.category),
                    "text_sha256": text_hash,
                },
            }
        )
    return rows


def _select_balanced_subset(
    *,
    pool: Sequence[Mapping[str, object]],
    target: int,
    seed: int,
) -> List[Dict[str, object]]:
    if target <= 0:
        raise ValueError("target must be > 0.")
    if target > len(pool):
        raise RuntimeError(f"target={target} exceeds pool size={len(pool)}")

    rng = np.random.default_rng(int(seed) ^ 0xC0DE)
    source_rows = [dict(row) for row in pool]
    labels = sorted(
        {
            str(label)
            for row in source_rows
            for label in (row.get("source_langs") or [])
            if str(label).strip()
        }
    )
    tasks = sorted({str(row.get("task", "")) for row in source_rows})

    remaining: set[int] = set(range(len(source_rows)))
    selected: List[int] = []
    label_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()

    def _row_labels(idx: int) -> List[str]:
        return [str(x) for x in (source_rows[idx].get("source_langs") or []) if str(x).strip()]

    def _row_task(idx: int) -> str:
        return str(source_rows[idx].get("task", ""))

    def _candidate_score(idx: int) -> float:
        row_labels = sorted(set(_row_labels(idx)))
        if not row_labels:
            return 1e9
        # Prefer rows that improve underrepresented labels and avoid over-used tasks.
        label_score = sum(float(label_counts[label]) for label in row_labels) / float(len(row_labels))
        task_score = float(task_counts[_row_task(idx)])
        size_penalty = 0.15 * float(len(row_labels))
        jitter = float(rng.random()) * 0.01
        return label_score + (0.35 * task_score) + size_penalty + jitter

    def _pick_best(candidates: Sequence[int]) -> int:
        if not candidates:
            raise RuntimeError("No candidates to pick from.")
        best = min(candidates, key=_candidate_score)
        return int(best)

    def _add(idx: int) -> bool:
        if idx not in remaining:
            return False
        remaining.remove(idx)
        selected.append(int(idx))
        row_labels = set(_row_labels(idx))
        for label in row_labels:
            label_counts[label] += 1
        task_counts[_row_task(idx)] += 1
        return True

    # Coverage pass 1: ensure each label appears at least once (if possible).
    for label in labels:
        if len(selected) >= target:
            break
        candidates = [idx for idx in remaining if label in _row_labels(idx)]
        if not candidates:
            continue
        _add(_pick_best(candidates))

    # Coverage pass 2: ensure each task appears at least once (if possible).
    for task in tasks:
        if len(selected) >= target:
            break
        candidates = [idx for idx in remaining if _row_task(idx) == task]
        if not candidates:
            continue
        _add(_pick_best(candidates))

    # Balanced fill: repeatedly prioritize currently underrepresented labels.
    while len(selected) < target and remaining:
        progress = False
        labels_by_need = sorted(labels, key=lambda label: (label_counts[label], float(rng.random())))
        for label in labels_by_need:
            if len(selected) >= target:
                break
            candidates = [idx for idx in remaining if label in _row_labels(idx)]
            if not candidates:
                continue
            _add(_pick_best(candidates))
            progress = True
        if not progress:
            break

    # Final fill, still preferring low-bias rows.
    while len(selected) < target and remaining:
        _add(_pick_best(list(remaining)))

    if len(selected) != target:
        raise RuntimeError(
            f"Balanced selection could not reach target size: selected={len(selected)}, target={target}."
        )

    chosen = [dict(source_rows[idx]) for idx in selected]
    return chosen


def _render_markdown(samples: Sequence[Mapping[str, object]], *, target_len: int, seed: int) -> str:
    lines: List[str] = []
    lines.append("# Curated Oracle Benchmark Set")
    lines.append("")
    lines.append(f"- Created: `{_now_iso()}`")
    lines.append(f"- Seed: `{seed}`")
    lines.append(f"- Sample length: `{target_len}`")
    lines.append(f"- Samples: `{len(samples)}`")
    lines.append("")

    mixed_count = sum(1 for sample in samples if bool(sample.get("mixed_truth", False)))
    pure_count = len(samples) - mixed_count
    lines.append(f"- Mixed: `{mixed_count}`")
    lines.append(f"- Non-mixed: `{pure_count}`")
    lines.append("")

    task_counts = Counter(str(sample.get("task", "unknown")) for sample in samples)
    label_counts = Counter()
    for sample in samples:
        for label in sample.get("source_langs", []):
            label_counts[str(label)] += 1

    lines.append("## Task Distribution")
    lines.append("")
    lines.append("| task | count |")
    lines.append("|---|---:|")
    for task, count in sorted(task_counts.items()):
        lines.append(f"| `{task}` | {count} |")
    lines.append("")

    lines.append("## Label Coverage")
    lines.append("")
    lines.append("| label | sample_count |")
    lines.append("|---|---:|")
    for label, count in sorted(label_counts.items()):
        lines.append(f"| `{label}` | {count} |")
    lines.append("")

    lines.append("## Sample Index")
    lines.append("")
    lines.append("| id | mixed | task | labels | note | text_preview |")
    lines.append("|---|:---:|---|---|---|---|")
    for sample in samples:
        sid = str(sample.get("snippet_id", ""))
        mixed = "yes" if bool(sample.get("mixed_truth", False)) else "no"
        task = str(sample.get("task", ""))
        labels = ",".join(str(label) for label in sample.get("source_langs", []))
        note = str((sample.get("metadata") or {}).get("note", ""))
        preview = _escape_markdown_cell(str(sample.get("text", "")))
        lines.append(f"| `{sid}` | {mixed} | `{task}` | `{labels}` | {note} | `{preview}` |")
    lines.append("")

    return "\n".join(lines)


def _build_dataset(
    *,
    target_len: int,
    seed: int,
    target_samples: int,
) -> Dict[str, object]:
    drafts = _pure_drafts() + _mixed_drafts()
    samples = _drafts_to_samples(drafts, target_len=target_len, seed=seed)
    if len(samples) < 100:
        raise RuntimeError(f"Expected >=100 samples, got {len(samples)}")
    if target_samples <= 0:
        raise ValueError("target_samples must be > 0.")
    if len(samples) < target_samples:
        raise RuntimeError(
            f"Not enough generated samples to satisfy target_samples={target_samples} "
            f"(generated={len(samples)})."
        )
    if len(samples) > target_samples:
        samples = _select_balanced_subset(
            pool=samples,
            target=int(target_samples),
            seed=int(seed),
        )

    # Renumber IDs after optional subsampling.
    renumbered: List[Dict[str, object]] = []
    for idx, row in enumerate(samples):
        item = dict(row)
        sid = f"curated-{idx:04d}"
        item["snippet_id"] = sid
        item["example_id"] = sid
        renumbered.append(item)
    samples = renumbered

    pure_count = sum(1 for sample in samples if not bool(sample.get("mixed_truth", False)))
    mixed_count = len(samples) - pure_count
    if pure_count < 20 or mixed_count < 50:
        raise RuntimeError(
            f"Insufficient diversity after generation (pure={pure_count}, mixed={mixed_count})."
        )

    return {
        "version": "curated-oracle-segments-v1",
        "created_at": _now_iso(),
        "seed": int(seed),
        "sample_length": int(target_len),
        "target_samples": int(target_samples),
        "samples_total": int(len(samples)),
        "mixed_samples": int(mixed_count),
        "non_mixed_samples": int(pure_count),
        "samples": samples,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed, reviewable active-learning oracle benchmark dataset with "
            "100+ diverse 256-char snippets and ground-truth segments."
        )
    )
    parser.add_argument("--target-length", type=int, default=DEFAULT_TARGET_LEN)
    parser.add_argument(
        "--target-samples",
        type=int,
        default=1000,
        help="Number of curated snippets to keep in the final dataset.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-jsonl", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--output-markdown", type=str, default=str(DEFAULT_OUTPUT_MARKDOWN))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    target_len = int(args.target_length)
    if target_len <= 32:
        raise ValueError("--target-length must be > 32.")
    if int(args.target_samples) <= 0:
        raise ValueError("--target-samples must be > 0.")

    payload = _build_dataset(
        target_len=target_len,
        seed=int(args.seed),
        target_samples=int(args.target_samples),
    )
    samples = list(payload.get("samples", []))

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    output_markdown = Path(args.output_markdown).resolve()
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(
        _render_markdown(samples, target_len=target_len, seed=int(args.seed)),
        encoding="utf-8",
    )

    print(
        "Curated benchmark dataset created: "
        f"samples={len(samples)}, mixed={payload['mixed_samples']}, "
        f"non_mixed={payload['non_mixed_samples']}",
        flush=True,
    )
    print(f"JSON: {output_json}", flush=True)
    print(f"JSONL: {output_jsonl}", flush=True)
    print(f"Markdown: {output_markdown}", flush=True)


if __name__ == "__main__":
    main()
