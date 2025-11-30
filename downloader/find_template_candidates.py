#!/usr/bin/env python3
"""
Heuristically flag segmentation outputs that may contain template/JSX-like markup
misclassified as plain HTML. Writes a candidate list to
gemini_segmentations/template_mislabel_candidates.txt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Common HTML tag names; uppercase variants of these should not be treated as components.
COMMON_HTML_TAGS = {
    "a",
    "abbr",
    "address",
    "article",
    "aside",
    "audio",
    "b",
    "base",
    "bdi",
    "bdo",
    "blockquote",
    "body",
    "br",
    "button",
    "canvas",
    "caption",
    "center",
    "cite",
    "code",
    "col",
    "colgroup",
    "data",
    "datalist",
    "dd",
    "del",
    "details",
    "dfn",
    "dialog",
    "div",
    "dl",
    "dt",
    "em",
    "embed",
    "fieldset",
    "font",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hr",
    "html",
    "i",
    "iframe",
    "img",
    "input",
    "ins",
    "kbd",
    "label",
    "legend",
    "li",
    "link",
    "main",
    "map",
    "mark",
    "meta",
    "meter",
    "nav",
    "noscript",
    "object",
    "ol",
    "optgroup",
    "option",
    "output",
    "p",
    "param",
    "picture",
    "pre",
    "progress",
    "q",
    "rp",
    "rt",
    "ruby",
    "s",
    "samp",
    "script",
    "section",
    "select",
    "small",
    "source",
    "span",
    "strong",
    "style",
    "sub",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "template",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "time",
    "title",
    "tr",
    "track",
    "u",
    "ul",
    "var",
    "video",
}

COMPONENT_TAG_RE = re.compile(r"<\/?([A-Z][A-Za-z0-9_]*)\b")

# Patterns for templating / frameworks
PATTERNS = [
    re.compile(r"\bclassName="),  # JSX className
    re.compile(r"\bon[A-Z][a-z]+\s*=\s*{"),  # JSX event handlers; require braces to avoid HTML onClick=""
    re.compile(r"\{\{.+?\}\}"),  # Angular/Vue/Handlebars style interpolations
    re.compile(r"{%.*?%}"),  # Django/Jinja templating blocks
    re.compile(r"\bng-[a-zA-Z]+="),  # Angular ng- bindings
    re.compile(r"\b\*ng\w+="),  # Angular structural directives
    re.compile(r"\[\w+\]="),  # Angular bindings [routerLink]= etc.
    re.compile(r"\(\w+\)="),  # Angular event bindings (click)= etc.
    re.compile(r"\[\(\w+\)\]="),  # Angular banana-in-a-box [(ngModel)]=
    re.compile(r"\bv-(bind|on):"),  # Vue directives
    re.compile(r"\b@(?:click|change|input|submit)="),  # Vue/Alpine @click=
    re.compile(r"\btemplate\s*="),  # inline template attrs
]


def has_pascal_tag(content: str) -> bool:
    """Detect PascalCase tags that are unlikely to be plain HTML."""
    for match in COMPONENT_TAG_RE.finditer(content):
        if match.group(1).lower() not in COMMON_HTML_TAGS:
            return True
    return False


def scan() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "gemini_segmentations" / "monitor"
    if not root.exists():
        return []
    candidates: list[Path] = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        segs = data.get("segments")
        if not isinstance(segs, list):
            continue
        hit = False
        for seg in segs:
            if seg.get("type") != "html":
                continue
            content = seg.get("content") or ""
            if has_pascal_tag(content) or any(p.search(content) for p in PATTERNS):
                hit = True
                break
        if hit:
            candidates.append(path)
    return sorted(candidates)


def main() -> None:
    candidates = scan()
    out_path = Path("gemini_segmentations/template_mislabel_candidates.txt")
    out_path.write_text("\n".join(str(p) for p in candidates), encoding="utf-8")
    print(f"wrote {len(candidates)} candidates to {out_path}")


if __name__ == "__main__":
    main()
