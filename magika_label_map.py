from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

OTHER_THESIS_LABEL = "other"

THESIS_ENCODING_LABELS: Tuple[str, ...] = (
    "encoding_hex",
    "encoding_base64",
    "encoding_base32",
    "encoding_base58",
    "encoding_base85",
)

THESIS_MAGIKA_SUPPORTED_LABELS: Tuple[str, ...] = (
    "php",
    "csharp",
    "javascript_typescript",
    "go",
    "sql",
    "rust",
    "yaml",
    "ruby",
    "python",
    "java",
    "c_family",
    "json",
    "css",
    "html",
    "text",
    "csv",
    "shell",
    "powershell",
    "visual_basic",
    "dockerfile",
    "dart",
    "gettext_catalog",
    "kotlin",
    "markdown",
    "restructuredtext",
    "scala",
    "swift",
    "tex",
    "xml",
    "svg",
)


def canonical_label(name: Optional[str]) -> str:
    s = str(name or "").strip().lower().replace(" ", "_")
    if s in {"c++", "cpp", "c", "c_family", "c-family", "cfamily"}:
        return "c_family"
    if s in {"c#", "c-sharp", "csharp", "cs"}:
        return "csharp"
    if s in {"javascript_typescript", "javascript-typescript", "js_ts", "js-ts"}:
        return "javascript_typescript"
    if s in {"shell_batchfile", "shell-batchfile", "shell_batch"}:
        return "shell"
    if s in {"js", "javascript"}:
        return "javascript"
    if s in {"ts", "typescript"}:
        return "typescript"
    if s in {"yml", "yaml"}:
        return "yaml"
    if s in {"vb", "visual-basic", "visualbasic", "vbnet", "vb.net"}:
        return "visual_basic"
    if s in {"ps", "ps1", "powershell"}:
        return "powershell"
    if s in {"batch", "bat", "cmd", "batchfile"}:
        return "batchfile"
    if s in {"docker", "dockerfile"}:
        return "dockerfile"
    if s in {"gettext-catalog", "gettext_catalog", "gettext", "po"}:
        return "gettext_catalog"
    if s in {"latex"}:
        return "tex"
    if s in {"rst", "restructured_text", "restructured-text", "restructuredtext"}:
        return "restructuredtext"
    return s


LABEL_ACCEPTS: Dict[str, Set[str]] = {
    "text": {"txt", "text"},
    "c_family": {"c", "cpp", "c++", "c_family"},
    "csv": {"csv"},
    "csharp": {"c#", "csharp", "c-sharp", "cs"},
    "javascript_typescript": {"javascript", "typescript", "js", "ts"},
    "yaml": {"yaml", "yml"},
    "php": {"php"},
    "go": {"go"},
    "sql": {"sql"},
    "rust": {"rust"},
    "ruby": {"ruby"},
    "python": {"python"},
    "java": {"java"},
    "json": {"json"},
    "css": {"css"},
    "html": {"html", "xhtml"},
    "dart": {"dart"},
    "gettext_catalog": {"gettext-catalog", "gettext_catalog", "gettext", "po"},
    "kotlin": {"kotlin"},
    "markdown": {"markdown", "md"},
    "restructuredtext": {"restructuredtext", "rst"},
    "scala": {"scala"},
    "swift": {"swift"},
    "svg": {"svg"},
    "tex": {"tex", "latex"},
    "xml": {"xml"},
    "shell": {"shell", "bash", "sh", "zsh", "fish", "batchfile", "bat", "cmd", "shell_batchfile"},
    "powershell": {"powershell", "ps1"},
    "visual_basic": {"visual_basic", "visual-basic", "vb", "vba", "vb.net", "visualbasic"},
    "dockerfile": {"dockerfile", "docker"},
}

_MAGIKA_EXPECTED_ALIAS_GROUPS_RAW: Mapping[str, Sequence[str]] = {
    "php": ("php",),
    "csharp": ("cs", "csharp", "c#"),
    "javascript_typescript": ("javascript", "typescript", "javascript_typescript"),
    "go": ("go",),
    "sql": ("sql",),
    "rust": ("rust",),
    "yaml": ("yaml", "yml"),
    "ruby": ("ruby",),
    "python": ("python",),
    "java": ("java",),
    "c_family": ("c", "cpp", "c++", "c_family"),
    "json": ("json",),
    "css": ("css",),
    "html": ("html", "xhtml"),
    "text": ("txt", "randomtxt", "text"),
    "csv": ("csv",),
    "shell": ("shell", "batch", "batchfile", "shell_batchfile"),
    "powershell": ("powershell", "ps1"),
    "visual_basic": ("vba", "vb", "visual_basic", "visual-basic", "visualbasic", "vb.net"),
    "dockerfile": ("dockerfile", "docker"),
    "dart": ("dart",),
    "gettext_catalog": ("po", "gettext", "gettext_catalog", "gettext-catalog"),
    "kotlin": ("kotlin",),
    "markdown": ("markdown",),
    "restructuredtext": ("rst", "restructuredtext"),
    "scala": ("scala",),
    "swift": ("swift",),
    "tex": ("latex", "tex"),
    "xml": ("xml",),
    "svg": ("svg",),
}

MAGIKA_EXPECTED_ALIAS_GROUPS: Dict[str, Tuple[str, ...]] = {
    thesis_label: tuple(sorted({canonical_label(alias) for alias in aliases}))
    for thesis_label, aliases in _MAGIKA_EXPECTED_ALIAS_GROUPS_RAW.items()
}

_SUPPORTED_RAW_MAGIKA_LABELS: Set[str] = {
    alias
    for aliases in MAGIKA_EXPECTED_ALIAS_GROUPS.values()
    for alias in aliases
}

_DIRECT_MAGIKA_TO_THESIS: Dict[str, str] = {
    "php": "php",
    "csharp": "csharp",
    "go": "go",
    "sql": "sql",
    "rust": "rust",
    "yaml": "yaml",
    "ruby": "ruby",
    "python": "python",
    "java": "java",
    "json": "json",
    "css": "css",
    "html": "html",
    "csv": "csv",
    "powershell": "powershell",
    "dockerfile": "dockerfile",
    "dart": "dart",
    "kotlin": "kotlin",
    "markdown": "markdown",
    "restructuredtext": "restructuredtext",
    "scala": "scala",
    "swift": "swift",
    "xml": "xml",
    "svg": "svg",
}


class MagikaMappingError(RuntimeError):
    pass


@dataclass(frozen=True)
class MagikaInventoryValidation:
    package_version: str
    model_name: str
    installed_labels: Tuple[str, ...]
    supported_labels: Tuple[str, ...]
    unsupported_labels: Tuple[str, ...]


def label_matches_target(target: str, magika_label: Optional[str], mime: Optional[str] = None) -> bool:
    del mime
    if not magika_label:
        return False
    tgt = canonical_label(target)
    ml = canonical_label(magika_label)
    accepts = LABEL_ACCEPTS.get(tgt, {tgt})
    return ml in accepts


def magika_label_to_thesis_label(raw_label: Optional[str]) -> str:
    canonical = canonical_label(raw_label)
    if canonical in _DIRECT_MAGIKA_TO_THESIS:
        return _DIRECT_MAGIKA_TO_THESIS[canonical]
    if canonical in {"javascript", "typescript", "javascript_typescript"}:
        return "javascript_typescript"
    if canonical in {"c_family"}:
        return "c_family"
    if canonical in {"shell", "batchfile"}:
        return "shell"
    if canonical in {"visual_basic", "vba"}:
        return "visual_basic"
    if canonical in {"gettext_catalog"}:
        return "gettext_catalog"
    if canonical in {"tex"}:
        return "tex"
    if canonical in {"txt", "randomtxt", "text"}:
        return "text"
    if canonical in {OTHER_THESIS_LABEL, "unknown", "randombytes"}:
        return OTHER_THESIS_LABEL
    if canonical in THESIS_ENCODING_LABELS:
        return OTHER_THESIS_LABEL
    return OTHER_THESIS_LABEL


def encoding_labels_set() -> Set[str]:
    return {canonical_label(label) for label in THESIS_ENCODING_LABELS}


def validate_magika_label_inventory(
    raw_labels: Iterable[str],
    *,
    package_version: str = "unknown",
    model_name: str = "unknown",
) -> MagikaInventoryValidation:
    installed_set = {
        canonical_label(str(label))
        for label in raw_labels
        if str(label).strip()
    }
    if not installed_set:
        raise MagikaMappingError("Magika label inventory is empty.")

    missing: Dict[str, Tuple[str, ...]] = {}
    for thesis_label, aliases in MAGIKA_EXPECTED_ALIAS_GROUPS.items():
        if installed_set.isdisjoint(aliases):
            missing[thesis_label] = aliases
    if missing:
        details = ", ".join(
            f"{label}: expected one of {list(aliases)}"
            for label, aliases in sorted(missing.items())
        )
        raise MagikaMappingError(
            "Installed Magika model is missing required output labels for the thesis mapping contract: "
            f"{details}"
        )

    unexpected_supported = {
        raw_label: mapped_label
        for raw_label in sorted(installed_set)
        for mapped_label in (magika_label_to_thesis_label(raw_label),)
        if mapped_label != OTHER_THESIS_LABEL and raw_label not in _SUPPORTED_RAW_MAGIKA_LABELS
    }
    if unexpected_supported:
        details = ", ".join(
            f"{raw_label}->{mapped_label}"
            for raw_label, mapped_label in sorted(unexpected_supported.items())
        )
        raise MagikaMappingError(
            "Installed Magika model exposes supported labels that are outside the locked alias contract: "
            f"{details}"
        )

    supported_labels = sorted(
        {
            magika_label_to_thesis_label(raw_label)
            for raw_label in installed_set
            if magika_label_to_thesis_label(raw_label) != OTHER_THESIS_LABEL
        }
    )
    unsupported_labels = sorted(
        raw_label
        for raw_label in installed_set
        if magika_label_to_thesis_label(raw_label) == OTHER_THESIS_LABEL
    )
    return MagikaInventoryValidation(
        package_version=str(package_version),
        model_name=str(model_name),
        installed_labels=tuple(sorted(installed_set)),
        supported_labels=tuple(supported_labels),
        unsupported_labels=tuple(unsupported_labels),
    )


def load_and_validate_installed_magika_mapping() -> MagikaInventoryValidation:
    from magika import Magika

    detector = Magika()
    model_config = getattr(detector, "_model_config", None)
    target_labels = getattr(model_config, "target_labels_space", None)
    if target_labels is None:
        raise MagikaMappingError("Installed Magika package does not expose target_labels_space.")
    return validate_magika_label_inventory(
        [str(label) for label in target_labels],
        package_version=str(detector.get_module_version()),
        model_name=str(detector.get_model_name()),
    )
