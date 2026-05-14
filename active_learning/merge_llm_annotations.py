from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "active_learning" / "all_annotations.sqlite"
DEFAULT_GLOB = "*.sqlite"

REFINEMENT_COLUMNS: tuple[str, ...] = (
    "created_at",
    "round_id",
    "source_split",
    "source_lang",
    "sample_index",
    "sample_hash",
    "boundary_index",
    "snippet_start",
    "snippet_end",
    "snippet_text",
    "oracle_name",
    "oracle_model",
    "oracle_run_id",
    "status",
    "acquisition_score",
    "predicted_segments_json",
    "refined_segments_json",
    "metadata_json",
)


@dataclass
class MergeStats:
    source_databases: int = 0
    source_rows: int = 0
    inserted_rows: int = 0
    duplicate_rows: int = 0
    skipped_invalid_rows: int = 0
    skipped_non_llm_rows: int = 0


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def _ensure_output_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS refinements (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          round_id TEXT NOT NULL,
          source_split TEXT NOT NULL,
          source_lang TEXT NOT NULL,
          sample_index INTEGER NOT NULL,
          sample_hash TEXT NOT NULL,
          boundary_index INTEGER NOT NULL,
          snippet_start INTEGER NOT NULL,
          snippet_end INTEGER NOT NULL,
          snippet_text TEXT NOT NULL,
          oracle_name TEXT NOT NULL,
          oracle_model TEXT NOT NULL,
          oracle_run_id TEXT NOT NULL,
          status TEXT NOT NULL,
          acquisition_score REAL NOT NULL,
          predicted_segments_json TEXT NOT NULL,
          refined_segments_json TEXT NOT NULL,
          metadata_json TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_refinements_round
        ON refinements(round_id)
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_refinements_sample
        ON refinements(sample_hash, snippet_start, snippet_end)
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS annotation_sources (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          refinement_id INTEGER NOT NULL,
          source_db TEXT NOT NULL,
          source_row_id INTEGER NOT NULL,
          source_created_at TEXT NOT NULL,
          UNIQUE(source_db, source_row_id),
          FOREIGN KEY(refinement_id) REFERENCES refinements(id) ON DELETE CASCADE
        )
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_annotation_sources_refinement
        ON annotation_sources(refinement_id)
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_annotation_sources_db
        ON annotation_sources(source_db)
        """
    )


def _discover_candidate_databases(
    search_root: Path,
    pattern: str,
    output_path: Path,
) -> list[Path]:
    candidates = []
    for path in sorted(search_root.rglob(pattern)):
        resolved = path.resolve()
        if resolved == output_path:
            continue
        if not resolved.is_file():
            continue
        if _has_refinements_table(resolved):
            candidates.append(resolved)
    return candidates


def _has_refinements_table(path: Path) -> bool:
    try:
        with _connect(path) as con:
            row = con.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name = 'refinements'
                """
            ).fetchone()
            if row is None:
                return False
            columns = {
                str(info["name"])
                for info in con.execute("PRAGMA table_info(refinements)").fetchall()
            }
    except sqlite3.Error:
        return False
    return set(REFINEMENT_COLUMNS).issubset(columns)


def _canonical_json(raw: object, *, kind: type) -> tuple[object, str] | None:
    if raw is None:
        parsed = [] if kind is list else {}
    elif isinstance(raw, (dict, list)):
        parsed = raw
    else:
        try:
            parsed = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(parsed, kind):
        return None
    return parsed, json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _segment_list_is_valid(snippet_text: str, segments: Sequence[object]) -> bool:
    if not snippet_text or not segments:
        return False
    snippet_len = len(snippet_text)
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        try:
            start = int(segment["start"])
            end = int(segment["end"])
        except (KeyError, TypeError, ValueError):
            return False
        label = str(segment.get("label", "")).strip()
        if not label:
            return False
        if start < 0 or end <= start or end > snippet_len:
            return False
    return True


def _normalize_refinement_row(
    row: sqlite3.Row,
    *,
    include_stub: bool,
) -> tuple[tuple[object, ...], tuple[object, ...]] | None:
    status = str(row["status"] or "").strip()
    oracle_name = str(row["oracle_name"] or "").strip()
    if status != "ok":
        return None
    if not include_stub and oracle_name == "stub":
        return None

    snippet_text = str(row["snippet_text"] or "")
    if not snippet_text:
        return None

    predicted = _canonical_json(row["predicted_segments_json"], kind=list)
    refined = _canonical_json(row["refined_segments_json"], kind=list)
    metadata = _canonical_json(row["metadata_json"], kind=dict)
    if predicted is None or refined is None or metadata is None:
        return None

    refined_segments, refined_json = refined
    if not _segment_list_is_valid(snippet_text, refined_segments):
        return None

    normalized_values = (
        str(row["created_at"] or ""),
        str(row["round_id"] or ""),
        str(row["source_split"] or ""),
        str(row["source_lang"] or ""),
        int(row["sample_index"]),
        str(row["sample_hash"] or ""),
        int(row["boundary_index"]),
        int(row["snippet_start"]),
        int(row["snippet_end"]),
        snippet_text,
        oracle_name,
        str(row["oracle_model"] or ""),
        str(row["oracle_run_id"] or ""),
        status,
        float(row["acquisition_score"]),
        predicted[1],
        refined_json,
        metadata[1],
    )
    dedupe_key = normalized_values[1:]
    return normalized_values, dedupe_key


def _iter_source_rows(con: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    query = (
        "SELECT id, "
        + ", ".join(REFINEMENT_COLUMNS)
        + " FROM refinements ORDER BY id ASC"
    )
    yield from con.execute(query)


def _insert_refinement(con: sqlite3.Connection, values: Sequence[object]) -> int:
    placeholders = ", ".join("?" for _ in REFINEMENT_COLUMNS)
    cur = con.execute(
        f"""
        INSERT INTO refinements ({", ".join(REFINEMENT_COLUMNS)})
        VALUES ({placeholders})
        """,
        list(values),
    )
    return int(cur.lastrowid)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine valid LLM-backed refinement annotations from all label-store SQLite "
            "files in the repo into a single SQLite database."
        )
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=REPO_ROOT,
        help="Directory scanned recursively for SQLite files (default: repo root).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=DEFAULT_GLOB,
        help="Filename glob used with recursive search (default: *.sqlite).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination SQLite path (default: active_learning/all_annotations.sqlite).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination database if it already exists.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate rows instead of collapsing exact copies across source databases.",
    )
    parser.add_argument(
        "--include-stub",
        action="store_true",
        help="Include stub-oracle rows as well. By default only non-stub rows are kept.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    search_root = args.search_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass --overwrite to replace it."
            )
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_paths = _discover_candidate_databases(
        search_root=search_root,
        pattern=str(args.glob),
        output_path=output_path,
    )
    if not source_paths:
        raise FileNotFoundError(
            f"No SQLite files with a compatible refinements table were found under {search_root}."
        )

    stats = MergeStats(source_databases=len(source_paths))
    dedupe_to_output_id: dict[tuple[object, ...], int] = {}

    with _connect(output_path) as out_con:
        _ensure_output_schema(out_con)
        for source_path in source_paths:
            print(f"Merging {source_path}", flush=True)
            with _connect(source_path) as source_con:
                for row in _iter_source_rows(source_con):
                    stats.source_rows += 1
                    normalized = _normalize_refinement_row(
                        row,
                        include_stub=bool(args.include_stub),
                    )
                    if normalized is None:
                        oracle_name = str(row["oracle_name"] or "").strip()
                        status = str(row["status"] or "").strip()
                        if status != "ok":
                            stats.skipped_invalid_rows += 1
                        elif not args.include_stub and oracle_name == "stub":
                            stats.skipped_non_llm_rows += 1
                        else:
                            stats.skipped_invalid_rows += 1
                        continue

                    values, dedupe_key = normalized
                    if not args.keep_duplicates and dedupe_key in dedupe_to_output_id:
                        refinement_id = dedupe_to_output_id[dedupe_key]
                        stats.duplicate_rows += 1
                    else:
                        refinement_id = _insert_refinement(out_con, values)
                        stats.inserted_rows += 1
                        if not args.keep_duplicates:
                            dedupe_to_output_id[dedupe_key] = refinement_id

                    out_con.execute(
                        """
                        INSERT OR IGNORE INTO annotation_sources (
                          refinement_id,
                          source_db,
                          source_row_id,
                          source_created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            int(refinement_id),
                            str(source_path),
                            int(row["id"]),
                            str(row["created_at"] or ""),
                        ),
                    )
        out_con.commit()

    print(
        "Merge complete: "
        f"sources={stats.source_databases}, "
        f"source_rows={stats.source_rows}, "
        f"inserted={stats.inserted_rows}, "
        f"duplicates={stats.duplicate_rows}, "
        f"skipped_invalid={stats.skipped_invalid_rows}, "
        f"skipped_non_llm={stats.skipped_non_llm_rows}, "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
