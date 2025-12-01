from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import requests
from google import genai
from google.genai import types
import urllib3
from rich.console import Console

console = Console()

DEFAULT_MODEL = "gemini-2.5-flash"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEGMENTATIONS_DIR = PROJECT_ROOT / "gemini_segmentations"
LOG_OUTPUT_DIR = PROJECT_ROOT / "gemini_output_logs"

# Keep this list aligned with downloader/999_check_gemini.py so downstream
# consumers see consistent type labels.
ALLOWED_TYPE_NAMES = [
    "php",
    "csharp",
    "javascript",
    "typescript",
    "go",
    "sql",
    "rust",
    "yaml",
    "ruby",
    "python",
    "java",
    "c",
    "cpp",
    "json",
    "css",
    "html",
    "csv",
    "shell",
    "powershell",
    "makefile",
    "visual_basic",
    "dockerfile",
    "xml",
    "markdown",
    "svg",
    "gettext-catalog",
    "scala",
    "swift",
    "restructuredtext",
    "kotlin",
    "dart",
    "encoding_hex",
    "encoding_base64",
    "encoding_base32",
    "encoding_base58",
    "encoding_base85",
    "other",
]
_ALLOWED_TYPES = set(ALLOWED_TYPE_NAMES)
_TYPE_SYNONYMS = {
    "js": "javascript",
    "ts": "typescript",
    "bash": "shell",
    "sh": "shell",
    "powershell_core": "powershell",
    "ps": "powershell",
    "vb": "visual_basic",
    "vbnet": "visual_basic",
    "c++": "cpp",
    "c#": "csharp",
}

CLASSIFICATION_EXAMPLES = """
Example (pure python):
<INPUT>
print("hello world")
</INPUT>
<OUTPUT>
{"is_pure": true, "language": "python", "mixed_types": [], "reason": "single python snippet"}
</OUTPUT>

Example (mixed markdown + shell):
<INPUT>
# Quick start
Run `pip install foo` and then `foo --help`.
</INPUT>
<OUTPUT>
{"is_pure": false, "language": "markdown", "mixed_types": ["markdown", "shell"], "reason": "markdown text with inline shell command"}
</OUTPUT>

Example (pure html):
<INPUT>
<div class="cta">Sign up</div>
</INPUT>
<OUTPUT>
{"is_pure": true, "language": "html", "mixed_types": [], "reason": "only HTML markup"}
</OUTPUT>

Example (mixed html + css + javascript):
<INPUT>
<button style="color: red;" onclick="alert('hi')">Click me</button>
</INPUT>
<OUTPUT>
{"is_pure": false, "language": "html", "mixed_types": ["html", "css", "javascript"], "reason": "HTML with inline CSS style and JS onclick"}
</OUTPUT>

Example (mixed markdown + shell):
<INPUT>
## Deploy
Run this:
```bash
kubectl apply -f deploy.yaml
```
</INPUT>
<OUTPUT>
{"is_pure": false, "language": "markdown", "mixed_types": ["markdown", "shell"], "reason": "markdown with fenced bash snippet"}
</OUTPUT>

Example (mixed javascript + base64):
<INPUT>
const payload = "SGVsbG8sIHdvcmxkIQ=="; // base64 string
console.log(atob(payload));
</INPUT>
<OUTPUT>
{"is_pure": false, "language": "javascript", "mixed_types": ["javascript", "encoding_base64"], "reason": "JS code containing literal base64 blob"}
</OUTPUT>

Example (javascript with JSX/custom tags — this is not pure JS!):
<INPUT>
import React from 'react';
function App() {
  return <div className="wrap"><Header /><Main /></div>;
}
</INPUT>
<OUTPUT>
{"is_pure": false, "language": "javascript", "mixed_types": ["javascript", "other_jsx"], "reason": "JS code containing JSX template tags (not plain HTML)"}
</OUTPUT>

Example (Angular template with bindings):
<INPUT>
<div class="card">
  <h1>{{ title }}</h1>
  <button (click)="save()">Save</button>
</div>
</INPUT>
<OUTPUT>
{"is_pure": false, "language": "other_angular_template", "mixed_types": ["other_angular_template", "javascript"], "reason": "Angular template markup with JS expressions/handlers"}
</OUTPUT>

Example (Django template with inline CSS):
<INPUT>
{% block body %}
<div class="card" style="color: red;">Hello {{ user.name|default:"Anonymous" }}</div>
{% endblock %}
</INPUT>
<OUTPUT>
{"is_pure": false, "language": "other_django_template", "mixed_types": ["other_django_template", "css"], "reason": "Django template markup plus inline CSS style attribute"}
</OUTPUT>

Example (dockerfile running shell):
<INPUT>
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl
CMD ["bash", "-c", "echo hi"]
</INPUT>
<OUTPUT>
{"is_pure": false, "language": "dockerfile", "mixed_types": ["dockerfile", "shell"], "reason": "dockerfile with RUN/CMD shell commands"}
</OUTPUT>

Example (Spring properties config):
<INPUT>
spring.application.name=demo
</INPUT>
<OUTPUT>
{"is_pure": true, "language": "other_application_properties", "mixed_types": [], "reason": "configuration file best labeled as other_application_properties"}
</OUTPUT>
""".strip()

SYSTEM_PROMPT = (
    "You are a strict content purity checker. "
    "Decide if an input is entirely one content type (pure) or mixes multiple types. "
    "Follow the allow list and return concise JSON only. "
    "If JS/TS contains JSX/TSX tags (including custom components like <Header /> or <div>), treat that as mixed javascript/typescript + other_jsx (not plain html); likewise, Angular templates are other_angular_template and Django/Jinja templates are other_django_template rather than plain html. "
    "Inline style attributes are css. YAML front matter (--- ... ---) is yaml even inside markdown."
)

PROMPT_TEMPLATE = """Determine whether the content between <INPUT> and </INPUT> is "pure"
(only one substantive content type) or "mixed" (multiple content types).

Rules:
- Choose the single best content type from this allow list (or 'other' if nothing fits):
  {allow_list}
- Treat comments/docstrings as belonging to their language.
- Inline fenced code or embedded language snippets count as additional types; if any appear, mark is_pure=false.
- Pick the real content type even if fences or file hints are mislabeled.
- False negatives are costly: when uncertain (including when you might say "other" or see another unexpected content type), prefer marking is_pure=false and list every plausible type. A false positive (mistakenly saying it is mixed) is much better than a false negative (mistakenly saying it is pure).
- Never answer with plain "other"—always expand it to other_<best_guess> such as other_application_properties when unsure.
- HTML rules: the `<script>` tag itself is html but its contents are javascript; inline handlers/expressions like onClick/onmouseover/onkeydown are already javascript; `javascript:` URLs (e.g., iframe src="javascript:void(0)") are javascript; inline `style="..."` is css.
- Template languages: JSX/TSX/custom component tags belong to other_jsx mixed with javascript/typescript (not plain html); Angular templates are other_angular_template plus any javascript from bindings; Django/Jinja templates are other_django_template. Inline CSS/JS inside these templates still counts as css/javascript.
- Dockerfiles that execute anything (RUN/CMD/ENTRYPOINT/HEALTHCHECK/ONBUILD shells or inline scripts) are mixed: dockerfile + shell (or another detected language) rather than pure dockerfile.
  - IMPORTANT INSTRUCTION: In fact, anything that seems a bit non-pure to you should be classified as non-pure (recall the false-positive thing) - even the slightest JS in HTML or a CSS comment in HTML
- Django or JINJA templates also make files non-pure"
- Comments stay in their host language (e.g., `#` in Dockerfile/python); only surface embedded executable code inside comments as an additional language when it's real code (e.g., JS hidden in a CSS comment).
- Output JSON with: is_pure (boolean), language (string), mixed_types (list; empty if pure),
  reason (short phrase). Respond with JSON only—no markdown fences.

Helpful examples:
{examples}

Your turn:
<INPUT>
{input_text}
</INPUT>
"""


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _proxy_mapping(proxy_url: str | None) -> dict[str, str] | None:
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def _new_run_identifiers() -> tuple[str, str]:
    now = datetime.datetime.utcnow()
    timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    run_id = f"{now.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}"
    return timestamp_iso, run_id


def _serialize_usage_metadata(usage_metadata):
    if usage_metadata is None:
        return None
    return {
        "prompt_token_count": getattr(usage_metadata, "prompt_token_count", None),
        "candidates_token_count": getattr(usage_metadata, "candidates_token_count", None),
        "total_token_count": getattr(usage_metadata, "total_token_count", None),
    }


def _normalize_type(label: str | None) -> str:
    if not label:
        return "other"
    cleaned = label.strip().lower()
    cleaned = _TYPE_SYNONYMS.get(cleaned, cleaned)
    if cleaned in _ALLOWED_TYPES:
        return cleaned
    if cleaned.startswith("other_") or cleaned.startswith("discovered_"):
        return cleaned
    return "other"


def _limit_input_length(content: str, max_length: int) -> tuple[str, bool]:
    if len(content) <= max_length:
        return content, False
    return content[:max_length], True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify whether an input is pure (single content type) before running full segmentation."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--file", type=Path, help="Path to the text file to classify.")
    source_group.add_argument("--text", help="Raw text content to classify.")
    source_group.add_argument("--url", help="URL pointing to content to classify.")
    source_group.add_argument("--stdin", action="store_true", help="Read content to classify from STDIN.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model to call (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="Explicit Google API key. Falls back to GOOGLE_API_KEY env.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=10_000,
        help="Maximum number of characters to send to Gemini (default: 10000).",
    )
    parser.add_argument(
        "--proxy",
        nargs="?",
        const="http://localhost:8080",
        help=(
            "Route Gemini API traffic through an HTTP proxy (e.g. Burp on port 8080). "
            "When specified without a value, defaults to http://localhost:8080."
        ),
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Emit only a single JSON line with the classification result (quiet mode).",
    )
    parser.add_argument(
        "--no-save-snapshot",
        dest="save_snapshot",
        action="store_false",
        help="Skip writing a segments snapshot when the file is pure.",
    )
    args = parser.parse_args()
    if args.max_input_length <= 0:
        parser.error("--max-input-length must be a positive integer.")
    return args


def resolve_api_key(cli_api_key: str | None) -> str:
    if cli_api_key:
        return cli_api_key
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key
    raise SystemExit(
        "[red]Missing API key.[/red] Provide --api-key, export GOOGLE_API_KEY, "
        "or configure google.colab.userdata."
    )


def read_input_text(
    args: argparse.Namespace,
    *,
    proxies: dict[str, str] | None = None,
    verify: bool = True,
) -> tuple[str, str, bool]:
    def _prepare(text: str, source_label: str) -> tuple[str, str, bool]:
        limited, truncated = _limit_input_length(text, args.max_input_length)
        return limited, source_label, truncated

    if args.text:
        return _prepare(args.text, "--text input")
    if args.file:
        try:
            file_text = args.file.read_text()
            return _prepare(file_text, f"contents of {args.file}")
        except OSError as exc:
            raise SystemExit(f"[red]Failed to read file {args.file}: {exc}[/red]")
    if args.url:
        try:
            response = requests.get(args.url, proxies=proxies, verify=verify)
            response.raise_for_status()
            return _prepare(response.text, f"response from {args.url}")
        except requests.RequestException as exc:
            raise SystemExit(f"[red]Failed to fetch URL {args.url}: {exc}[/red]")
    if args.stdin:
        data = sys.stdin.read()
        if not data:
            raise SystemExit("[red]STDIN was selected but no data was provided.[/red]")
        return _prepare(data, "STDIN input")
    raise SystemExit("[red]No input source provided. Use --file/--text/--url/--stdin.[/red]")


def build_prompt(input_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        allow_list=",".join(ALLOWED_TYPE_NAMES),
        examples=CLASSIFICATION_EXAMPLES,
        input_text=input_text,
    )


def classify_content(
    content: str,
    *,
    client: genai.Client,
    model: str,
) -> tuple[str, Any | None]:
    prompt = build_prompt(content)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
    )
    chunks: list[str] = []
    usage_metadata = None
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.usage_metadata is not None:
            usage_metadata = chunk.usage_metadata
        if not chunk.candidates:
            continue
        content_obj = chunk.candidates[0].content
        if not content_obj or not getattr(content_obj, "parts", None):
            continue
        for part in content_obj.parts:
            if getattr(part, "text", None):
                chunks.append(part.text)
    response_text = "".join(chunks).strip()
    return response_text, usage_metadata


def _extract_json_candidate(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}
    return {}


def _emit_result(payload: dict[str, Any], *, json_only: bool) -> None:
    if json_only:
        print(json.dumps(payload))
        return
    console.print(
        (
            "Purity: {pure} • language={lang} • mixed_types={mixed} • "
            "snapshot={snap}"
        ).format(
            pure=payload.get("is_pure"),
            lang=payload.get("language"),
            mixed=",".join(payload.get("mixed_types") or []),
            snap=payload.get("snapshot_path"),
        ),
        style="bold green" if payload.get("is_pure") else "yellow",
    )
    console.print(json.dumps(payload, indent=2, ensure_ascii=False), style="dim")


def save_segments_snapshot(run_id: str, metadata: dict, segments) -> Path:
    _ensure_dir(SEGMENTATIONS_DIR)
    snapshot = {
        "run_id": run_id,
        "metadata": metadata,
        "segments": segments,
    }
    snapshot_path = SEGMENTATIONS_DIR / f"{run_id}.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return snapshot_path


def log_classification(run_id: str, metadata: dict, classification: dict[str, Any]) -> Path:
    _ensure_dir(LOG_OUTPUT_DIR)
    payload = {
        "run_id": run_id,
        "status": "purity_check",
        "metadata": metadata,
        "classification": classification,
    }
    log_path = LOG_OUTPUT_DIR / f"{run_id}_purity.json"
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return log_path


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    proxy_url = args.proxy
    proxy_dict = _proxy_mapping(proxy_url)
    verify_tls = True
    if proxy_dict:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        verify_tls = False
        console.print(
            "Proxy mode enabled: skipping TLS certificate verification (insecure).",
            style="bold yellow",
        )
    content, input_source, input_was_truncated = read_input_text(
        args, proxies=proxy_dict, verify=verify_tls
    )
    timestamp_iso, run_id = _new_run_identifiers()
    input_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    metadata: dict[str, Any] = {
        "timestamp": timestamp_iso,
        "model": args.model,
        "proxy": args.proxy,
        "proxy_skip_tls": not verify_tls,
        "input_source": input_source,
        "input_characters": len(content),
        "input_sha256": input_sha,
        "max_input_length": args.max_input_length,
        "input_was_truncated": input_was_truncated,
        "purity_check": True,
    }

    http_options = None
    if proxy_dict:
        console.print(f"Routing Gemini API calls through proxy {proxy_url}", style="dim")
        sync_httpx_client = httpx.Client(proxy=proxy_url, verify=verify_tls)
        async_httpx_client = httpx.AsyncClient(proxy=proxy_url, verify=verify_tls)
        http_options = types.HttpOptions(
            httpx_client=sync_httpx_client,
            httpx_async_client=async_httpx_client,
        )
    client = genai.Client(api_key=api_key, http_options=http_options)

    response_text, usage_metadata = classify_content(
        content,
        client=client,
        model=args.model,
    )
    metadata["usage_metadata"] = _serialize_usage_metadata(usage_metadata)

    parsed = _extract_json_candidate(response_text)
    is_pure = bool(parsed.get("is_pure"))
    language = _normalize_type(parsed.get("language"))
    mixed_types = [
        _normalize_type(entry)
        for entry in parsed.get("mixed_types", [])
        if isinstance(entry, str)
    ]
    reason = parsed.get("reason") if isinstance(parsed.get("reason"), str) else ""
    classification = {
        "is_pure": is_pure,
        "language": language,
        "mixed_types": mixed_types,
        "reason": reason,
        "raw_response": response_text,
    }

    metadata["classification"] = classification
    snapshot_path: Path | None = None
    if is_pure and args.save_snapshot:
        segments = [{"type": language, "content": content}]
        snapshot_path = save_segments_snapshot(run_id, metadata, segments)
    log_classification(run_id, metadata, classification)

    result_payload = {
        "is_pure": is_pure,
        "language": language,
        "mixed_types": mixed_types,
        "reason": reason,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "run_id": run_id,
        "usage_metadata": metadata.get("usage_metadata"),
        "input_sha256": input_sha,
        "model": metadata.get("model"),
    }
    _emit_result(result_payload, json_only=args.json_output)


if __name__ == "__main__":
    main()
