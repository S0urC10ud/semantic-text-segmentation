"""ANSI 24-bit terminal colouring for segments.

Single source of the label palette, shared by ``Segment.__repr__`` and the
``examples/segcat.py`` renderer so terminal output stays consistent and close to
the interactive viewer. Colour is emitted only when the output is a TTY; honour
``NO_COLOR`` and the ``TYPEMAP_COLOR`` (``auto``/``always``/``never``) override.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

# 24-bit accent colour per label (close to the interactive viewer).
PALETTE = {
    "html": (231, 76, 60), "css": (46, 204, 113), "javascript_typescript": (190, 200, 40),
    "sql": (52, 152, 219), "shell": (155, 89, 182), "powershell": (125, 95, 200),
    "python": (53, 114, 165), "json": (230, 126, 34), "yaml": (241, 196, 15),
    "xml": (211, 84, 0), "svg": (192, 57, 43), "markdown": (127, 140, 141),
    "c_family": (52, 73, 94), "java": (192, 57, 43), "go": (0, 173, 216),
    "rust": (183, 65, 14), "text": (149, 165, 166), "other": (120, 120, 120),
    "encoding_base64": (26, 188, 156), "encoding_hex": (22, 160, 133),
}


def accent(label: str) -> Tuple[int, int, int]:
    """Stable accent RGB for a label (palette entry, else hashed hue)."""
    if label in PALETTE:
        return PALETTE[label]
    h = sum(ord(c) * 131 for c in label)
    return (80 + h % 150, 80 + (h // 7) % 150, 80 + (h // 53) % 150)


def tint(rgb: Tuple[int, int, int], f: float = 0.78) -> Tuple[int, int, int]:
    """Blend ``rgb`` toward white by fraction ``f`` (lighter background)."""
    return tuple(int(c + (255 - c) * f) for c in rgb)


def bg(rgb: Tuple[int, int, int]) -> str:
    return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def fg(rgb: Tuple[int, int, int]) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def color_enabled(stream: Optional["object"] = None) -> bool:
    """Whether to emit ANSI colour. ``TYPEMAP_COLOR`` wins, then ``NO_COLOR``,
    else on only when ``stream`` (default stdout) is a TTY."""
    mode = os.environ.get("TYPEMAP_COLOR", "auto").lower()
    if mode in ("never", "0", "off", "false"):
        return False
    if mode in ("always", "1", "on", "true"):
        return True
    if "NO_COLOR" in os.environ:
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except Exception:
        return False


def colorize(text: str, label: str) -> str:
    """A tinted background chip of ``text`` for ``label`` (dark fg for contrast)."""
    return f"{bg(tint(accent(label)))}{fg((30, 30, 30))}{text}{RESET}"
