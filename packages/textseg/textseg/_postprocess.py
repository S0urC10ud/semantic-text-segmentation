"""Post-processing of per-character predictions.

This implements the core deployment steps exposed by the interactive viewer:
whitespace relabelling, open-set confidence gating, minimum-run normalisation and
a simple whitespace boundary snap. The viewer additionally applies markdown /
paired-delimiter / local-host heuristics that are not reproduced here.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ._segmentation import Segment

_WHITESPACE = set(" \t\n\r\f\v")


def _runs(labels: List[int]) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    if not labels:
        return runs
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def relabel_whitespace(text: str, labels: List[int]) -> List[int]:
    """Assign whitespace characters the label of their nearest non-whitespace
    neighbour (preferring the preceding host)."""
    n = len(text)
    if n == 0:
        return labels
    is_ws = [c in _WHITESPACE for c in text]
    if not any(is_ws):
        return labels
    left = [-1] * n
    right = [-1] * n
    last = -1
    for i in range(n):
        if not is_ws[i]:
            last = labels[i]
        left[i] = last
    last = -1
    for i in range(n - 1, -1, -1):
        if not is_ws[i]:
            last = labels[i]
        right[i] = last
    out = list(labels)
    for i in range(n):
        if is_ws[i]:
            if left[i] != -1:
                out[i] = left[i]
            elif right[i] != -1:
                out[i] = right[i]
    return out


def confidence_gate(char_probs: np.ndarray, labels: List[int], threshold: float, other_index: int) -> List[int]:
    """Route characters whose maximum class probability is below ``threshold`` to ``other``."""
    if threshold <= 0.0 or char_probs.size == 0:
        return labels
    maxp = char_probs.max(axis=1)
    return [other_index if maxp[i] < threshold else lab for i, lab in enumerate(labels)]


def normalize_short_runs(labels: List[int], char_probs: np.ndarray, min_run_chars: int) -> List[int]:
    """Absorb interior runs shorter than ``min_run_chars`` into the better-supported neighbour."""
    if min_run_chars <= 1 or len(labels) <= 2:
        return labels
    out = list(labels)
    have_probs = char_probs.size > 0
    ncls = char_probs.shape[1] if have_probs else 0
    # Bounded cascade; each pass absorbs every short interior run whose neighbours
    # were not just modified (``guard``), so it is O(passes * n) instead of the
    # O(n^2) restart-after-each-merge the viewer uses.
    for _ in range(16):
        runs = _runs(out)
        if len(runs) <= 2:
            break
        changed = False
        guard = -1
        for idx in range(1, len(runs) - 1):
            s, e, _lab = runs[idx]
            if s <= guard:  # left neighbour was just rewritten this pass
                continue
            if (e - s) >= min_run_chars:
                continue
            left_lab, left_len = runs[idx - 1][2], runs[idx - 1][1] - runs[idx - 1][0]
            right_lab, right_len = runs[idx + 1][2], runs[idx + 1][1] - runs[idx + 1][0]
            best, best_key = None, None
            for cand, nlen in ((left_lab, left_len), (right_lab, right_len)):
                # The virtual "other" class has no column in char_probs; the viewer
                # resolves its support to 0.0 (dict.get fallback), so we match that.
                if have_probs and 0 <= cand < ncls:
                    support = float(char_probs[s:e, cand].mean())
                else:
                    support = 0.0
                sandwich = 1.0 if left_lab == right_lab == cand else 0.0
                key = (sandwich, support, float(nlen))
                if best_key is None or key > best_key:
                    best_key, best = key, cand
            for p in range(s, e):
                out[p] = best
            guard = e
            changed = True
        if not changed:
            break
    return out


# Boundary-snap tuning (ported from the viewer; thesis Step 3).
_SNAP_PROB_MARGIN = 0.10
_SNAP_DELIM_PROB_MARGIN = 0.15
_SNAP_WS_PROB_MARGIN = 0.05
_SNAP_MIN_IMPROVEMENT = 0.75


def _apply_boundary_shift(labels: List[int], current: int, boundary: int,
                          left_label: int, right_label: int) -> List[int]:
    out = list(labels)
    if boundary < current:
        for pos in range(boundary, current):
            out[pos] = int(right_label)
    elif boundary > current:
        for pos in range(current, boundary):
            out[pos] = int(left_label)
    return out


def _score_boundary_candidate(text, labels, char_probs, left_run, right_run, boundary):
    current = left_run[1]
    left_start, _, left_label = left_run
    _, right_end, right_label = right_run
    if boundary < left_start or boundary > right_end:
        return None
    la = text[boundary - 1] if boundary > 0 else ""
    ra = text[boundary] if boundary < len(text) else ""
    if la == "\n" or ra == "\n":
        return None
    if boundary != current and la not in _SNAP_ADJACENT_CHARS and ra not in _SNAP_ADJACENT_CHARS:
        return None
    score = _boundary_local_score(text, boundary)
    if boundary < current:
        moved, src, dest = range(boundary, current), left_label, right_label
    else:
        moved, src, dest = range(current, boundary), right_label, left_label
    for pos in moved:
        ch = text[pos]
        if ch == "\n":
            return None
        sp = _label_prob(char_probs, pos, src)
        dp = _label_prob(char_probs, pos, dest)
        if ch in _SNAP_DELIM_CHARS:
            if dp < (sp - _SNAP_DELIM_PROB_MARGIN):
                return None
        elif ch in _INLINE_WS:
            if dp < (sp - _SNAP_WS_PROB_MARGIN):
                return None
        elif ch not in _SNAP_ADJACENT_CHARS and dp < (sp - _SNAP_PROB_MARGIN):
            return None
        score += 0.5 * (dp - sp)
        if ch in _SNAP_DELIM_CHARS:
            score += 0.15
        elif ch in _INLINE_WS:
            score += 0.02
    return score


def snap_boundaries(text: str, labels: List[int], char_probs: np.ndarray,
                    max_shift: int, min_run_chars: int = 1) -> List[int]:
    """Snap a boundary (<= ``max_shift`` chars) onto a nearby delimiter/whitespace.

    Thesis Step 3: favours boundaries next to delimiter-like symbols
    (``< > " ' \\` ( ) [ ] { } / \\ , ; : =``), and rejects a shift if a moved
    non-delimiter position's destination-label probability is materially lower
    than the model's assigned probability."""
    if max_shift <= 0 or len(labels) <= 1:
        return labels
    out = list(labels)
    # Bounded number of passes; each pass applies every accepted (non-overlapping)
    # snap in a single left-to-right sweep, so total cost is O(passes * n) rather
    # than the O(n^2) restart-after-each-change the viewer uses.
    for _ in range(4):
        runs = _runs(out)
        nruns = len(runs)
        if nruns <= 1:
            break
        changed = False
        guard = -1  # rightmost position already modified this pass (avoid overlap)
        for idx in range(nruns - 1):
            left_run, right_run = runs[idx], runs[idx + 1]
            current = left_run[1]
            if current <= guard:
                continue
            la = text[current - 1] if current > 0 else ""
            ra = text[current] if current < len(text) else ""
            # Already on a natural seam (delimiter/whitespace): leave it.
            if la in _SNAP_ADJACENT_CHARS or ra in _SNAP_ADJACENT_CHARS:
                continue

            # Sub-min interior-run structure over the local window {prev, L, R, next}.
            # Only L and R lengths change with the boundary, so this is O(1) per
            # candidate (vs rebuilding every run, which made snapping O(n^2)).
            l_start, r_end = left_run[0], right_run[1]
            prev_len = (runs[idx - 1][1] - runs[idx - 1][0]) if idx > 0 else None
            next_len = (runs[idx + 2][1] - runs[idx + 2][0]) if (idx + 2) < nruns else None

            def submin(b, _pl=prev_len, _nl=next_len):
                cnt, mn = 0, None
                if _pl is not None:
                    mn = _pl
                    if 0 < (idx - 1) < nruns - 1 and _pl < min_run_chars:
                        cnt += 1
                ll = b - l_start
                if ll > 0:
                    mn = ll if mn is None else min(mn, ll)
                    if 0 < idx < nruns - 1 and ll < min_run_chars:
                        cnt += 1
                rl = r_end - b
                if rl > 0:
                    mn = rl if mn is None else min(mn, rl)
                    if 0 < (idx + 1) < nruns - 1 and rl < min_run_chars:
                        cnt += 1
                if _nl is not None:
                    mn = _nl if mn is None else min(mn, _nl)
                    if 0 < (idx + 2) < nruns - 1 and _nl < min_run_chars:
                        cnt += 1
                return cnt, (mn if mn is not None else 0)

            cur_score = _score_boundary_candidate(text, out, char_probs, left_run, right_run, current)
            if cur_score is None:
                cur_score = _boundary_local_score(text, current)
            cur_short, cur_min = submin(current)
            best_b, best_score, best_short, best_min = current, cur_score, cur_short, cur_min
            for shift in range(-max_shift, max_shift + 1):
                if shift == 0:
                    continue
                cand = current + shift
                score = _score_boundary_candidate(text, out, char_probs, left_run, right_run, cand)
                if score is None:
                    continue
                c_short, c_min = submin(cand)
                better = c_short < best_short or (c_short == best_short and c_min > best_min)
                same = c_short == best_short and c_min == best_min
                if better or (same and score > best_score + 1e-6):
                    best_b, best_score, best_short, best_min = cand, score, c_short, c_min
            if best_b != current and best_score >= cur_score + _SNAP_MIN_IMPROVEMENT:
                lo, hi = (best_b, current) if best_b < current else (current, best_b)
                fill = right_run[2] if best_b < current else left_run[2]
                for p in range(lo, hi):
                    out[p] = fill
                guard = hi
                changed = True
        if not changed:
            break
    return out


# --------------------------------------------------------------------------
# Paired-delimiter fill (ported from the viewer's _refine_wrapped_runs).
# Moves both edges of a run wrapped by the same host (A B A) so the inner run
# sits *inside* a matching delimiter pair, ejecting the delimiters to the host.
# --------------------------------------------------------------------------
_WRAP_OPEN_TO_CLOSE = {'"': '"', "'": "'", "`": "`", "(": ")", "[": "]", "{": "}"}
_WRAP_QUOTE_CHARS = frozenset(('"', "'", "`"))
_SNAP_DELIM_CHARS = frozenset(
    ("<", ">", "/", "\\", '"', "'", "`", "(", ")", "[", "]", "{", "}", ",", ";", ":", "=")
)
_SNAP_ADJACENT_CHARS = _SNAP_DELIM_CHARS | frozenset((" ", "\t"))
_INLINE_WS = frozenset((" ", "\t"))
_WRAP_PROB_MARGIN = 0.20
_WRAP_PAIR_BONUS = 1.35
_WRAP_QUOTE_BONUS = 0.35
_WRAP_DELIM_EJECT_BONUS = 0.80
_WRAP_DELIM_SWALLOW_PENALTY = 0.45
_WRAP_QUOTE_EJECT_BONUS = 0.35
_WRAP_MIN_IMPROVEMENT = 0.70


def _ident_like(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch in ("_", "$", "-"))


def _matching_wrap(left: str, right: str) -> bool:
    return bool(left) and _WRAP_OPEN_TO_CLOSE.get(left) == right


def _wrap_pair_bonus(left: str, right: str) -> float:
    if not _matching_wrap(left, right):
        return 0.0
    bonus = _WRAP_PAIR_BONUS
    if left in _WRAP_QUOTE_CHARS:
        bonus += _WRAP_QUOTE_BONUS
    return bonus


def _label_prob(char_probs: np.ndarray, pos: int, label: int) -> float:
    # The virtual "other" label has no column; treat its prob as 0.0 (matches viewer).
    if char_probs.size and 0 <= pos < char_probs.shape[0] and 0 <= label < char_probs.shape[1]:
        return float(char_probs[pos, label])
    return 0.0


def _mean_support(char_probs: np.ndarray, start: int, end: int, label: int) -> float:
    if start >= end:
        return 0.0
    return sum(_label_prob(char_probs, p, label) for p in range(start, end)) / (end - start)


def _boundary_local_score(text: str, boundary: int) -> float:
    n = len(text)
    if boundary < 0 or boundary > n:
        return float("-inf")
    left = text[boundary - 1] if boundary > 0 else ""
    right = text[boundary] if boundary < n else ""
    score = 0.0
    score += 1.25 if left in _SNAP_DELIM_CHARS else (0.20 if left in _INLINE_WS else 0.0)
    score += 1.25 if right in _SNAP_DELIM_CHARS else (0.20 if right in _INLINE_WS else 0.0)
    if left in _SNAP_DELIM_CHARS and right in _SNAP_DELIM_CHARS:
        score += 0.35
    if left in _INLINE_WS and right in _INLINE_WS:
        score -= 0.25
    if left in ("<", "(", "[", "{") and _ident_like(right):
        score += 0.35
    if right in (">", ")", "]", "}") and _ident_like(left):
        score += 0.35
    if _ident_like(left) and _ident_like(right):
        score -= 1.5
    return score


def _score_wrapped(text, char_probs, left_run, middle_run, right_run, start, end):
    left_start, _, left_label = left_run
    cur_start, cur_end, mid_label = middle_run
    _, right_end, right_label = right_run
    if left_label != right_label or mid_label == left_label:
        return None
    if start < left_start or end > right_end or start >= end:
        return None
    if start <= 0 or end >= len(text):
        return None
    score = _boundary_local_score(text, start) + _boundary_local_score(text, end)
    score += _wrap_pair_bonus(text[start - 1], text[end])
    host, inner = left_label, mid_label
    changed = False
    shift_penalty = 0
    for pos in range(min(cur_start, start), max(cur_end, end)):
        cur_assign = inner if cur_start <= pos < cur_end else host
        cand_assign = inner if start <= pos < end else host
        if cand_assign == cur_assign:
            continue
        ch = text[pos]
        if ch == "\n":
            return None
        changed = True
        cur_p = _label_prob(char_probs, pos, cur_assign)
        cand_p = _label_prob(char_probs, pos, cand_assign)
        if ch not in _SNAP_ADJACENT_CHARS and cand_p < (cur_p - _WRAP_PROB_MARGIN):
            return None
        score += 0.8 * (cand_p - cur_p)
        if ch in _SNAP_ADJACENT_CHARS:
            score += 0.10
        moving_out = cand_assign == host and cur_assign == inner
        moving_in = cand_assign == inner and cur_assign == host
        if ch in _SNAP_DELIM_CHARS:
            if moving_out:
                score += _WRAP_DELIM_EJECT_BONUS
            elif moving_in:
                score -= _WRAP_DELIM_SWALLOW_PENALTY
        if ch in _WRAP_QUOTE_CHARS:
            if moving_out:
                score += _WRAP_QUOTE_EJECT_BONUS
            elif moving_in:
                score -= 0.15
        shift_penalty += 1
    if changed:
        score += 0.60 * (_mean_support(char_probs, start, end, inner)
                         - _mean_support(char_probs, start, end, host))
        score -= 0.05 * max(0, shift_penalty - 2)
    return score


def _apply_wrap_shift(labels, cur_start, cur_end, start, end, host, inner):
    out = list(labels)
    for pos in range(min(cur_start, start), max(cur_end, end)):
        out[pos] = inner if start <= pos < end else host
    return out


def paired_delimiter_fill(text: str, labels: List[int], char_probs: np.ndarray, max_shift: int) -> List[int]:
    """Snap each ``A B A`` wrapped run onto a matching delimiter pair (quotes/brackets)."""
    if max_shift <= 0 or len(labels) <= 2:
        return labels
    out = list(labels)
    # Bounded multi-apply passes (not restart-after-each-change, which is O(n^2)
    # when there are many candidate runs -- e.g. on noisy Mamba output). Within a
    # pass we apply every non-overlapping improvement left-to-right, skipping runs
    # inside a region just rewritten via `guard`, then rebuild runs once.
    for _ in range(8):
        runs = _runs(out)
        if len(runs) <= 2:
            break
        changed = False
        guard = -1
        for idx in range(1, len(runs) - 1):
            left_run, mid_run, right_run = runs[idx - 1], runs[idx], runs[idx + 1]
            if left_run[0] < guard:
                continue  # triple touches a region rewritten earlier this pass (stale)
            if left_run[2] != right_run[2] or mid_run[2] == left_run[2]:
                continue
            cur_start, cur_end = mid_run[0], mid_run[1]
            cur_score = _score_wrapped(text, char_probs, left_run, mid_run, right_run, cur_start, cur_end)
            if cur_score is None:
                cur_score = _boundary_local_score(text, cur_start) + _boundary_local_score(text, cur_end)
            best_start, best_end, best_score = cur_start, cur_end, cur_score
            for ls in range(-max_shift, max_shift + 1):
                cs = cur_start + ls
                if cs < left_run[0] or cs >= cur_end:
                    continue
                for rs in range(-max_shift, max_shift + 1):
                    ce = cur_end + rs
                    if ce <= cs or ce > right_run[1] or (cs == cur_start and ce == cur_end):
                        continue
                    if not _matching_wrap(text[cs - 1] if cs > 0 else "",
                                          text[ce] if ce < len(text) else ""):
                        continue
                    sc = _score_wrapped(text, char_probs, left_run, mid_run, right_run, cs, ce)
                    if sc is not None and sc > best_score + 1e-6:
                        best_start, best_end, best_score = cs, ce, sc
            if (best_start != cur_start or best_end != cur_end) and \
               best_score >= cur_score + _WRAP_MIN_IMPROVEMENT:
                out = _apply_wrap_shift(out, cur_start, cur_end, best_start, best_end,
                                        left_run[2], mid_run[2])
                guard = right_run[1]  # skip the rewritten triple for the rest of this pass
                changed = True
        if not changed:
            break
    return out


def build_segments(text: str, labels: List[int], char_probs: np.ndarray,
                   label_names: List[str], other_index: int, other_label: str) -> Tuple[List[Segment], List[str], List[float]]:
    n = len(text)
    char_conf: List[float] = []
    char_label_names: List[str] = []
    for i in range(n):
        lab = labels[i]
        if char_probs.size:
            known_max = float(char_probs[i].max())
            conf = (1.0 - known_max) if lab == other_index else float(char_probs[i, lab])
        else:
            conf = 0.0
        char_conf.append(conf)
        char_label_names.append(other_label if lab == other_index else label_names[lab])

    segments: List[Segment] = []
    for s, e, lab in _runs(labels):
        name = other_label if lab == other_index else label_names[lab]
        conf = float(np.mean(char_conf[s:e])) if e > s else 0.0
        segments.append(Segment(start=s, end=e, label=name, confidence=conf, text=text[s:e]))
    return segments, char_label_names, char_conf
