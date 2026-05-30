def _apply_local_host_postprocess_rules(
    text: str,
    labels: np.ndarray,
    *,
    local_host_rules: Sequence[Tuple[int, int, str]],
    min_run_chars: int,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if arr.size == 0 or not text or not local_host_rules:
        return arr
    single_side_min_chars = max(int(min_run_chars), int(_LOCAL_HOST_SINGLE_SIDE_MIN_CHARS))
    max_passes = max(1, int(arr.shape[0]))
    for _ in range(max_passes):
        runs = _build_label_runs(arr)
        changed = False
        for run_idx, (start, end, label) in enumerate(runs):
            for inner_label, host_label, rule_kind in local_host_rules:
                if int(label) != int(inner_label):
                    continue
                left_host_len = 0
                right_host_len = 0
                if run_idx > 0 and int(runs[run_idx - 1][2]) == int(host_label):
                    left_host_len = int(runs[run_idx - 1][1] - runs[run_idx - 1][0])
                if run_idx + 1 < len(runs) and int(runs[run_idx + 1][2]) == int(host_label):
                    right_host_len = int(runs[run_idx + 1][1] - runs[run_idx + 1][0])
                if left_host_len <= 0 and right_host_len <= 0:
                    continue
                if not _local_host_rule_matches_text(str(rule_kind), text[int(start):int(end)]):
                    continue
                if left_host_len > 0 and right_host_len > 0:
                    should_relabel = True
                else:
                    should_relabel = (left_host_len + right_host_len) >= single_side_min_chars
                if not should_relabel:
                    continue
                arr[int(start):int(end)] = int(host_label)
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    return arr


def _score_boundary_candidate(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    left_run: Tuple[int, int, int],
    right_run: Tuple[int, int, int],
    boundary: int,
) -> Optional[float]:
    current = int(left_run[1])
    left_start, _, left_label = left_run
    _, right_end, right_label = right_run
    if boundary < left_start or boundary > right_end:
        return None
    left_adjacent = text[boundary - 1] if boundary > 0 else ""
    right_adjacent = text[boundary] if boundary < len(text) else ""
    if left_adjacent == "\n" or right_adjacent == "\n":
        return None
    if boundary != current and (
        left_adjacent not in _BOUNDARY_SNAP_ADJACENT_CHARS
        and right_adjacent not in _BOUNDARY_SNAP_ADJACENT_CHARS
    ):
        return None

    score = _boundary_local_score(text, boundary)
    if boundary < current:
        moved_positions = range(boundary, current)
        src_label = int(left_label)
        dest_label = int(right_label)
    else:
        moved_positions = range(current, boundary)
        src_label = int(right_label)
        dest_label = int(left_label)
    for pos in moved_positions:
        ch = text[pos]
        if ch == "\n":
            return None
        src_prob = float(char_probs[pos, src_label]) if 0 <= src_label < int(char_probs.shape[1]) else 0.0
        dest_prob = float(char_probs[pos, dest_label]) if 0 <= dest_label < int(char_probs.shape[1]) else 0.0
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            if dest_prob < (src_prob - _BOUNDARY_SNAP_DELIMITER_PROB_MARGIN):
                return None
        elif ch in _INLINE_WHITESPACE_SET:
            if dest_prob < (src_prob - _BOUNDARY_SNAP_WHITESPACE_PROB_MARGIN):
                return None
        elif ch not in _BOUNDARY_SNAP_ADJACENT_CHARS and dest_prob < (src_prob - _BOUNDARY_SNAP_PROB_MARGIN):
            return None
        score += 0.5 * (dest_prob - src_prob)
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            score += 0.15
        elif ch in _INLINE_WHITESPACE_SET:
            score += 0.02
    return score


def _score_wrapped_run_candidate(
    text: str,
    char_probs: np.ndarray,
    left_run: Tuple[int, int, int],
    middle_run: Tuple[int, int, int],
    right_run: Tuple[int, int, int],
    start: int,
    end: int,
) -> Optional[float]:
    left_start, _, left_label = left_run
    current_start, current_end, middle_label = middle_run
    _, right_end, right_label = right_run
    if int(left_label) != int(right_label) or int(middle_label) == int(left_label):
        return None
    if start < int(left_start) or end > int(right_end) or start >= end:
        return None
    if start <= 0 or end >= len(text):
        return None
    left_delim = text[start - 1]
    right_delim = text[end]
    score = _boundary_local_score(text, start) + _boundary_local_score(text, end)
    score += _wrapped_pair_bonus(left_delim, right_delim)

    host_label = int(left_label)
    inner_label = int(middle_label)
    changed = False
    union_start = min(int(current_start), int(start))
    union_end = max(int(current_end), int(end))
    shift_penalty = 0
    for pos in range(union_start, union_end):
        current_assign = inner_label if int(current_start) <= pos < int(current_end) else host_label
        candidate_assign = inner_label if int(start) <= pos < int(end) else host_label
        if candidate_assign == current_assign:
            continue
        ch = text[pos]
        if ch == "\n":
            return None
        changed = True
        current_prob = float(char_probs[pos, current_assign]) if 0 <= current_assign < int(char_probs.shape[1]) else 0.0
        candidate_prob = float(char_probs[pos, candidate_assign]) if 0 <= candidate_assign < int(char_probs.shape[1]) else 0.0
        if ch not in _BOUNDARY_SNAP_ADJACENT_CHARS and candidate_prob < (current_prob - _BOUNDARY_WRAP_PROB_MARGIN):
            return None
        score += 0.8 * (candidate_prob - current_prob)
        if ch in _BOUNDARY_SNAP_ADJACENT_CHARS:
            score += 0.10
        moving_out = candidate_assign == host_label and current_assign == inner_label
        moving_in = candidate_assign == inner_label and current_assign == host_label
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            if moving_out:
                score += _BOUNDARY_WRAP_SHELL_DELIMITER_EJECT_BONUS
            elif moving_in:
                score -= _BOUNDARY_WRAP_SHELL_DELIMITER_SWALLOW_PENALTY
        if ch in _BOUNDARY_WRAP_QUOTE_CHARS:
            if moving_out:
                score += _BOUNDARY_WRAP_SHELL_QUOTE_EJECT_BONUS
            elif moving_in:
                score -= 0.15
        shift_penalty += 1
    if changed:
        score += 0.60 * (
            _mean_label_support(char_probs, start, end, inner_label)
            - _mean_label_support(char_probs, start, end, host_label)
        )
        score -= 0.05 * max(0, shift_penalty - 2)
    return score


def _apply_boundary_shift(
    labels: np.ndarray,
    current: int,
    boundary: int,
    left_label: int,
    right_label: int,
) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    if boundary < current:
        out[boundary:current] = int(right_label)
    elif boundary > current:
        out[current:boundary] = int(left_label)
    return out


def _snap_boundaries_to_delimiters(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    max_shift: int = 2,
    min_run_chars: int = 1,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if max_shift <= 0 or arr.size <= 1:
        return arr
    max_passes = max(1, int(arr.shape[0]) * 2)
    for _ in range(max_passes):
        runs = _build_label_runs(arr)
        if len(runs) <= 1:
            break
        changed = False
        for idx in range(len(runs) - 1):
            left_run = runs[idx]
            right_run = runs[idx + 1]
            current = int(left_run[1])
            window_start = int(runs[idx - 1][0]) if idx > 0 else int(left_run[0])
            window_end = int(runs[idx + 2][1]) if (idx + 2) < len(runs) else int(right_run[1])
            current_short_count, current_min_len = _count_local_submin_interior_runs(
                arr,
                min_run_chars=int(min_run_chars),
                window_start=window_start,
                window_end=window_end,
            )
            current_score = _score_boundary_candidate(text, arr, char_probs, left_run, right_run, current)
            if current_score is None:
                current_score = _boundary_local_score(text, current)
            best_boundary = current
            best_score = current_score
            best_short_count = current_short_count
            best_min_len = current_min_len
            for shift in range(-int(max_shift), int(max_shift) + 1):
                if shift == 0:
                    continue
                candidate = current + shift
                score = _score_boundary_candidate(text, arr, char_probs, left_run, right_run, candidate)
                if score is None:
                    continue
                candidate_labels = _apply_boundary_shift(arr, current, candidate, left_run[2], right_run[2])
                candidate_short_count, candidate_min_len = _count_local_submin_interior_runs(
                    candidate_labels,
                    min_run_chars=int(min_run_chars),
                    window_start=window_start,
                    window_end=window_end,
                )
                better_structure = (
                    candidate_short_count < best_short_count
                    or (
                        candidate_short_count == best_short_count
                        and candidate_min_len > best_min_len
                    )
                )
                same_structure = (
                    candidate_short_count == best_short_count
                    and candidate_min_len == best_min_len
                )
                if better_structure or (same_structure and score > (best_score + 1e-6)):
                    best_boundary = candidate
                    best_score = score
                    best_short_count = candidate_short_count
                    best_min_len = candidate_min_len
            if best_boundary != current and best_score >= (current_score + _BOUNDARY_SNAP_MIN_IMPROVEMENT):
                arr = _apply_boundary_shift(arr, current, best_boundary, left_run[2], right_run[2])
                changed = True
                break
        if not changed:
            break
    return arr


def _apply_wrapped_run_shift(
    labels: np.ndarray,
    current_start: int,
    current_end: int,
    start: int,
    end: int,
    *,
    host_label: int,
    inner_label: int,
) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    union_start = min(int(current_start), int(start))
    union_end = max(int(current_end), int(end))
    out[union_start:union_end] = int(host_label)
    out[int(start):int(end)] = int(inner_label)
    return out


def _fill_markdown_structure_regions(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    markdown_label: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(labels, dtype=np.int32).copy()
    locked = _new_lock_mask(arr.shape[0])
    if markdown_label is None or arr.size == 0 or not text:
        return arr, locked
    n = len(text)
    line_start = 0
    while line_start < n:
        line_end = text.find("\n", line_start)
        if line_end == -1:
            line_end = n
        pos = line_start
        while pos < line_end:
            if text[pos] != "`":
                pos += 1
                continue
            run_end = pos
            while run_end < line_end and text[run_end] == "`":
                run_end += 1
            run_len = run_end - pos
            if run_len < 3:
                pos = run_end
                continue
            next_start, next_end = _find_next_backtick_run(text, run_end, line_end, min_len=run_len)
            if next_start != -1 and next_start > run_end:
                arr[pos:run_end] = int(markdown_label)
                body_label = _infer_uniform_body_label(
                    arr,
                    char_probs,
                    run_end,
                    next_start,
                    markdown_label=int(markdown_label),
                )
                arr[run_end:next_start] = int(body_label)
                arr[next_start:next_end] = int(markdown_label)
                _mark_locked_range(locked, pos, next_end)
                pos = next_end
                continue
            token_end = run_end
            while token_end < line_end and text[token_end] not in (" ", "\t"):
                token_end += 1
            arr[pos:token_end] = int(markdown_label)
            _mark_locked_range(locked, pos, token_end)
            pos = token_end
        line_start = line_end + 1
    return arr, locked


def _refine_wrapped_runs(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    max_shift: int = 2,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if max_shift <= 0 or arr.size <= 2:
        return arr
    max_passes = max(1, int(arr.shape[0]))
    for _ in range(max_passes):
        runs = _build_label_runs(arr)
        if len(runs) <= 2:
            break
        changed = False
        for idx in range(1, len(runs) - 1):
            left_run = runs[idx - 1]
            middle_run = runs[idx]
            right_run = runs[idx + 1]
            if int(left_run[2]) != int(right_run[2]) or int(middle_run[2]) == int(left_run[2]):
                continue
            current_start = int(middle_run[0])
            current_end = int(middle_run[1])
            current_score = _score_wrapped_run_candidate(
                text,
                char_probs,
                left_run,
                middle_run,
                right_run,
                current_start,
                current_end,
            )
            if current_score is None:
                current_score = _boundary_local_score(text, current_start) + _boundary_local_score(text, current_end)
            best_start = current_start
            best_end = current_end
            best_score = current_score
            for left_shift in range(-int(max_shift), int(max_shift) + 1):
                cand_start = current_start + left_shift
                if cand_start < int(left_run[0]) or cand_start >= current_end:
                    continue
                for right_shift in range(-int(max_shift), int(max_shift) + 1):
                    cand_end = current_end + right_shift
                    if cand_end <= cand_start or cand_end > int(right_run[1]):
                        continue
                    if cand_start == current_start and cand_end == current_end:
                        continue
                    if not _matching_wrap_delimiter(
                        text[cand_start - 1] if cand_start > 0 else "",
                        text[cand_end] if cand_end < len(text) else "",
                    ):
                        continue
                    score = _score_wrapped_run_candidate(
                        text,
                        char_probs,
                        left_run,
                        middle_run,
                        right_run,
                        cand_start,
                        cand_end,
                    )
                    if score is None:
                        continue
                    if score > (best_score + 1e-6):
                        best_start = cand_start
                        best_end = cand_end
                        best_score = score
            if (
                (best_start != current_start or best_end != current_end)
                and best_score >= (current_score + _BOUNDARY_WRAP_MIN_IMPROVEMENT)
            ):
                arr = _apply_wrapped_run_shift(
                    arr,
                    current_start,
                    current_end,
                    best_start,
                    best_end,
                    host_label=int(left_run[2]),
                    inner_label=int(middle_run[2]),
                )
                changed = True
                break
        if not changed:
            break
    return arr


def _mean_label_support(char_probs: np.ndarray, start: int, end: int, label: int) -> float:
    if start >= end or label < 0 or label >= int(char_probs.shape[1]):
        return 0.0
    window = np.asarray(char_probs[start:end, label], dtype=np.float32)
    if window.size == 0:
        return 0.0
    return float(np.mean(window, dtype=np.float32))


def _count_local_submin_interior_runs(
    labels: np.ndarray,
    *,
    min_run_chars: int,
    window_start: int,
    window_end: int,
) -> Tuple[int, int]:
    runs = _build_label_runs(labels)
    count = 0
    min_len: Optional[int] = None
    for idx, (start, end, _label) in enumerate(runs):
        if end <= int(window_start) or start >= int(window_end):
            continue
        run_len = int(end - start)
        min_len = run_len if min_len is None else min(min_len, run_len)
        if 0 < idx < (len(runs) - 1) and run_len < int(min_run_chars):
            count += 1
    return count, (int(min_len) if min_len is not None else 0)


def _normalize_short_runs(
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    min_run_chars: int,
    locked_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if min_run_chars <= 1 or arr.size <= 2:
        return arr
    if locked_mask is not None and int(locked_mask.shape[0]) == int(arr.shape[0]):
        locks = np.asarray(locked_mask, dtype=bool)
    else:
        locks = _new_lock_mask(arr.shape[0])
    max_passes = max(1, int(arr.shape[0]))
    for _ in range(max_passes):
        runs = _build_label_runs(arr)
        changed = False
        for idx in range(1, len(runs) - 1):
            start, end, _label = runs[idx]
            if (end - start) >= int(min_run_chars):
                continue
            if bool(np.any(locks[start:end])):
                continue
            left_run = runs[idx - 1]
            right_run = runs[idx + 1]
            left_label = int(left_run[2])
            right_label = int(right_run[2])
            left_len = int(left_run[1] - left_run[0])
            right_len = int(right_run[1] - right_run[0])
            candidates: List[Tuple[Tuple[float, ...], int]] = []
            seen_targets: set[int] = set()
            for direction, target, neighbor_len in (
                ("left", left_label, left_len),
                ("right", right_label, right_len),
            ):
                if int(target) in seen_targets:
                    continue
                seen_targets.add(int(target))
                candidate = arr.copy()
                candidate[start:end] = int(target)
                submin_count, min_run_len = _count_local_submin_interior_runs(
                    candidate,
                    min_run_chars=min_run_chars,
                    window_start=int(left_run[0]),
                    window_end=int(right_run[1]),
                )
                support = _mean_label_support(char_probs, start, end, int(target))
                sandwich = 1.0 if left_label == right_label == int(target) else 0.0
                direction_tiebreak = 1.0 if direction == "left" else 0.0
                key = (
                    -float(submin_count),
                    float(min_run_len),
                    sandwich,
                    float(neighbor_len),
                    float(support),
                    direction_tiebreak,
                )
                candidates.append((key, int(target)))
            if not candidates:
                continue
            target = max(candidates, key=lambda item: item[0])[1]
            arr[start:end] = int(target)
            changed = True
            break
        if not changed:
            break
    return arr


def _postprocess_char_labels(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    min_run_chars: int,
    boundary_snap_max_shift: int = 2,
    local_host_rules: Sequence[Tuple[int, int, str]] = (),
    markdown_label: Optional[int] = None,
    html_label: Optional[int] = None,
) -> np.ndarray:
    markdown_filled, markdown_locked = _fill_markdown_structure_regions(
        text,
        labels,
        char_probs,
        markdown_label=markdown_label,
    )
    snapped = _snap_boundaries_to_delimiters(
        text,
        markdown_filled,
        char_probs,
        max_shift=boundary_snap_max_shift,
        min_run_chars=min_run_chars,
    )
    wrapped = _refine_wrapped_runs(
        text,
        snapped,
        char_probs,
        max_shift=boundary_snap_max_shift,
    )
    local_host_filled = _apply_local_host_postprocess_rules(
        text,
        wrapped,
        local_host_rules=local_host_rules,
        min_run_chars=min_run_chars,
    )
    locked_mask = _new_lock_mask(labels.shape[0])
    _merge_lock_masks(locked_mask, markdown_locked)
    normalized = _normalize_short_runs(
        local_host_filled,
        char_probs,
        min_run_chars=min_run_chars,
        locked_mask=locked_mask,
    )
    snapped_relit = _snap_boundaries_to_delimiters(
        text,
        normalized,
        char_probs,
        max_shift=boundary_snap_max_shift,
        min_run_chars=min_run_chars,
    )
    wrapped_relit = _refine_wrapped_runs(
        text,
        snapped_relit,
        char_probs,
        max_shift=boundary_snap_max_shift,
    )
    markdown_relit, _markdown_relock = _fill_markdown_structure_regions(
        text,
        wrapped_relit,
        char_probs,
        markdown_label=markdown_label,
    )
    return _apply_local_host_postprocess_rules(
        text,
        markdown_relit,
        local_host_rules=local_host_rules,
        min_run_chars=min_run_chars,
    )
def _find_next_backtick_run(text: str, start: int, end: int, *, min_len: int) -> Tuple[int, int]:
    pos = max(0, int(start))
    line_end = min(len(text), int(end))
    while pos < line_end:
        if text[pos] != "`":
            pos += 1
            continue
        run_end = pos
        while run_end < line_end and text[run_end] == "`":
            run_end += 1
        if (run_end - pos) >= int(min_len):
            return pos, run_end
        pos = run_end
    return -1, -1


def _infer_uniform_body_label(
    labels: np.ndarray,
    char_probs: np.ndarray,
    start: int,
    end: int,
    *,
    markdown_label: int,
) -> int:
    lo = max(0, int(start))
    hi = min(int(labels.shape[0]), int(end))
    if lo >= hi:
        return int(markdown_label)
    window_labels = np.asarray(labels[lo:hi], dtype=np.int32)
    if window_labels.size == 0:
        return int(markdown_label)
    counts: Dict[int, int] = {}
    supports: Dict[int, float] = {}
    for offset, label in enumerate(window_labels):
        label_int = int(label)
        counts[label_int] = counts.get(label_int, 0) + 1
        if 0 <= label_int < int(char_probs.shape[1]):
            supports[label_int] = supports.get(label_int, 0.0) + float(char_probs[lo + offset, label_int])
        else:
            supports.setdefault(label_int, 0.0)
    best_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == best_count]
    if len(candidates) == 1:
        return int(candidates[0])
    best_support = max(supports.get(label, 0.0) for label in candidates)
    support_candidates = [label for label in candidates if supports.get(label, 0.0) >= (best_support - 1e-9)]
    if int(markdown_label) in support_candidates:
        return int(markdown_label)
    return int(min(support_candidates))

def _new_lock_mask(length: int) -> np.ndarray:
    return np.zeros((max(0, int(length)),), dtype=bool)


def _mark_locked_range(mask: np.ndarray, start: int, end: int) -> None:
    lo = max(0, int(start))
    hi = min(int(mask.shape[0]), int(end))
    if lo < hi:
        mask[lo:hi] = True


def _merge_lock_masks(base: np.ndarray, update: np.ndarray) -> np.ndarray:
    if int(base.shape[0]) != int(update.shape[0]):
        return base
    np.logical_or(base, update, out=base)
    return base

