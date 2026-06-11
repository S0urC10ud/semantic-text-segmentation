const _VISUAL_WHITESPACE_SET = new Set([" ", "\t", "\n"]);
const _INLINE_WHITESPACE_SET = new Set([" ", "\t"]);
const _BOUNDARY_SNAP_DELIMITER_CHARS = new Set(["<", ">", "/", "\\", '"', "'", "`", "(", ")", "[", "]", "{", "}", ",", ";", ":", "="]);
const _BOUNDARY_SNAP_ADJACENT_CHARS = new Set([..._BOUNDARY_SNAP_DELIMITER_CHARS, " ", "\t"]);
const _BOUNDARY_SNAP_PROB_MARGIN = 0.1;
const _BOUNDARY_SNAP_DELIMITER_PROB_MARGIN = 0.15;
const _BOUNDARY_SNAP_WHITESPACE_PROB_MARGIN = 0.05;
const _BOUNDARY_SNAP_MIN_IMPROVEMENT = 0.75;
const _BOUNDARY_WRAP_OPEN_TO_CLOSE = {
    '"': '"',
    "'": "'",
    "`": "`",
    "(": ")",
    "[": "]",
    "{": "}",
};
const _BOUNDARY_WRAP_QUOTE_CHARS = new Set(['"', "'", "`"]);
const _BOUNDARY_WRAP_PROB_MARGIN = 0.20;
const _BOUNDARY_WRAP_PAIR_BONUS = 1.35;
const _BOUNDARY_WRAP_QUOTE_BONUS = 0.35;
const _BOUNDARY_WRAP_SHELL_DELIMITER_EJECT_BONUS = 0.80;
const _BOUNDARY_WRAP_SHELL_DELIMITER_SWALLOW_PENALTY = 0.45;
const _BOUNDARY_WRAP_SHELL_QUOTE_EJECT_BONUS = 0.35;
const _BOUNDARY_WRAP_MIN_IMPROVEMENT = 0.70;

const _LOCAL_HOST_SINGLE_SIDE_MIN_CHARS = 8;
const _NUMERIC_RE = /^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$/;

function _build_label_runs(labels) {
    if (labels.length === 0) return [];
    let runs = [];
    let current = labels[0];
    let start = 0;
    for (let i = 1; i < labels.length; i++) {
        let label = labels[i];
        if (label !== current) {
            runs.push([start, i, current]);
            start = i;
            current = label;
        }
    }
    runs.push([start, labels.length, current]);
    return runs;
}

function _is_identifier_like_char(ch) {
    return Boolean(ch) && (/^[a-zA-Z0-9_\$-]$/.test(ch));
}

function _matching_wrap_delimiter(left, right) {
    return Boolean(left) && (_BOUNDARY_WRAP_OPEN_TO_CLOSE[left] === right);
}

function _wrapped_pair_bonus(left, right) {
    if (!_matching_wrap_delimiter(left, right)) return 0.0;
    let bonus = _BOUNDARY_WRAP_PAIR_BONUS;
    if (_BOUNDARY_WRAP_QUOTE_CHARS.has(left)) {
        bonus += _BOUNDARY_WRAP_QUOTE_BONUS;
    }
    return bonus;
}

function _boundary_local_score(text, boundary) {
    if (boundary < 0 || boundary > text.length) return -Infinity;
    let left = boundary > 0 ? text[boundary - 1] : "";
    let right = boundary < text.length ? text[boundary] : "";
    let score = 0.0;
    if (_BOUNDARY_SNAP_DELIMITER_CHARS.has(left)) score += 1.25;
    else if (_INLINE_WHITESPACE_SET.has(left)) score += 0.20;
    
    if (_BOUNDARY_SNAP_DELIMITER_CHARS.has(right)) score += 1.25;
    else if (_INLINE_WHITESPACE_SET.has(right)) score += 0.20;
    
    if (_BOUNDARY_SNAP_DELIMITER_CHARS.has(left) && _BOUNDARY_SNAP_DELIMITER_CHARS.has(right)) score += 0.35;
    if (_INLINE_WHITESPACE_SET.has(left) && _INLINE_WHITESPACE_SET.has(right)) score -= 0.25;
    
    if (["<", "(", "[", "{"].includes(left) && _is_identifier_like_char(right)) score += 0.35;
    if ([">", ")", "]", "}"].includes(right) && _is_identifier_like_char(left)) score += 0.35;
    if (_is_identifier_like_char(left) && _is_identifier_like_char(right)) score -= 1.5;
    return score;
}

function _is_jsonish_content(text) {
    let trimmed = text.trim();
    if (!trimmed) return false;
    if (trimmed.length >= 2 && (
        (trimmed[0] === '{' && trimmed[trimmed.length - 1] === '}') ||
        (trimmed[0] === '[' && trimmed[trimmed.length - 1] === ']') ||
        (trimmed[0] === '"' && trimmed[trimmed.length - 1] === '"')
    )) {
        return true;
    }
    let lowered = trimmed.toLowerCase();
    if (lowered === 'true' || lowered === 'false' || lowered === 'null') return true;
    if (_NUMERIC_RE.test(trimmed)) return true;
    if (trimmed.includes(":") && (trimmed.includes('"') || trimmed.includes('{') || trimmed.includes('['))) return true;
    if (trimmed.includes(",") && (trimmed.includes('"') || trimmed.includes('{') || trimmed.includes('}') || trimmed.includes('[') || trimmed.includes(']'))) return true;
    return false;
}

function _local_host_rule_matches_text(rule_kind, text) {
    if (rule_kind === "json") return _is_jsonish_content(text);
    return Boolean(text);
}

function _apply_local_host_postprocess_rules(text, labels, local_host_rules, min_run_chars) {
    let arr = new Int32Array(labels);
    if (arr.length === 0 || !text || !local_host_rules || local_host_rules.length === 0) return arr;
    let single_side_min_chars = Math.max(min_run_chars, _LOCAL_HOST_SINGLE_SIDE_MIN_CHARS);
    let max_passes = Math.max(1, arr.length);
    for (let p = 0; p < max_passes; p++) {
        let runs = _build_label_runs(arr);
        let changed = false;
        for (let run_idx = 0; run_idx < runs.length; run_idx++) {
            let start = runs[run_idx][0];
            let end = runs[run_idx][1];
            let label = runs[run_idx][2];
            for (let r of local_host_rules) {
                let inner_label = r[0];
                let host_label = r[1];
                let rule_kind = r[2];
                if (label !== inner_label) continue;
                let left_host_len = 0;
                let right_host_len = 0;
                if (run_idx > 0 && runs[run_idx - 1][2] === host_label) {
                    left_host_len = runs[run_idx - 1][1] - runs[run_idx - 1][0];
                }
                if (run_idx + 1 < runs.length && runs[run_idx + 1][2] === host_label) {
                    right_host_len = runs[run_idx + 1][1] - runs[run_idx + 1][0];
                }
                if (left_host_len <= 0 && right_host_len <= 0) continue;
                if (!_local_host_rule_matches_text(rule_kind, text.substring(start, end))) continue;
                
                let should_relabel = true;
                if (left_host_len > 0 && right_host_len > 0) {
                    should_relabel = true;
                } else {
                    should_relabel = (left_host_len + right_host_len) >= single_side_min_chars;
                }
                if (!should_relabel) continue;
                for (let i = start; i < end; i++) arr[i] = host_label;
                changed = true;
                break;
            }
            if (changed) break;
        }
        if (!changed) break;
    }
    return arr;
}

function _score_boundary_candidate(text, labels, char_probs, left_run, right_run, boundary, numClasses) {
    let current = left_run[1];
    let left_start = left_run[0], left_label = left_run[2];
    let right_end = right_run[1], right_label = right_run[2];
    if (boundary < left_start || boundary > right_end) return null;
    let left_adjacent = boundary > 0 ? text[boundary - 1] : "";
    let right_adjacent = boundary < text.length ? text[boundary] : "";
    if (left_adjacent === "\n" || right_adjacent === "\n") return null;
    if (boundary !== current && !_BOUNDARY_SNAP_ADJACENT_CHARS.has(left_adjacent) && !_BOUNDARY_SNAP_ADJACENT_CHARS.has(right_adjacent)) return null;

    let score = _boundary_local_score(text, boundary);
    let start_pos, end_pos, src_label, dest_label;
    if (boundary < current) {
        start_pos = boundary;
        end_pos = current;
        src_label = left_label;
        dest_label = right_label;
    } else {
        start_pos = current;
        end_pos = boundary;
        src_label = right_label;
        dest_label = left_label;
    }
    for (let pos = start_pos; pos < end_pos; pos++) {
        let ch = text[pos];
        if (ch === "\n") return null;
        let src_prob = (src_label >= 0 && src_label < numClasses) ? char_probs[pos * numClasses + src_label] : 0.0;
        let dest_prob = (dest_label >= 0 && dest_label < numClasses) ? char_probs[pos * numClasses + dest_label] : 0.0;
        if (_BOUNDARY_SNAP_DELIMITER_CHARS.has(ch)) {
            if (dest_prob < (src_prob - _BOUNDARY_SNAP_DELIMITER_PROB_MARGIN)) return null;
        } else if (_INLINE_WHITESPACE_SET.has(ch)) {
            if (dest_prob < (src_prob - _BOUNDARY_SNAP_WHITESPACE_PROB_MARGIN)) return null;
        } else if (!_BOUNDARY_SNAP_ADJACENT_CHARS.has(ch) && dest_prob < (src_prob - _BOUNDARY_SNAP_PROB_MARGIN)) {
            return null;
        }
        score += 0.5 * (dest_prob - src_prob);
        if (_BOUNDARY_SNAP_DELIMITER_CHARS.has(ch)) score += 0.15;
        else if (_INLINE_WHITESPACE_SET.has(ch)) score += 0.02;
    }
    return score;
}

function _mean_label_support(char_probs, start, end, label, numClasses) {
    if (start >= end || label < 0 || label >= numClasses) return 0.0;
    let sum = 0.0;
    for (let i = start; i < end; i++) {
        sum += char_probs[i * numClasses + label];
    }
    return sum / (end - start);
}

function _score_wrapped_run_candidate(text, char_probs, left_run, middle_run, right_run, start, end, numClasses) {
    let left_start = left_run[0], left_label = left_run[2];
    let current_start = middle_run[0], current_end = middle_run[1], middle_label = middle_run[2];
    let right_end = right_run[1], right_label = right_run[2];
    if (left_label !== right_label || middle_label === left_label) return null;
    if (start < left_start || end > right_end || start >= end) return null;
    if (start <= 0 || end >= text.length) return null;
    
    let left_delim = text[start - 1];
    let right_delim = text[end];
    let score = _boundary_local_score(text, start) + _boundary_local_score(text, end);
    score += _wrapped_pair_bonus(left_delim, right_delim);

    let host_label = left_label;
    let inner_label = middle_label;
    let changed = false;
    let union_start = Math.min(current_start, start);
    let union_end = Math.max(current_end, end);
    let shift_penalty = 0;
    
    for (let pos = union_start; pos < union_end; pos++) {
        let current_assign = (pos >= current_start && pos < current_end) ? inner_label : host_label;
        let candidate_assign = (pos >= start && pos < end) ? inner_label : host_label;
        if (candidate_assign === current_assign) continue;
        let ch = text[pos];
        if (ch === "\n") return null;
        changed = true;
        let current_prob = (current_assign >= 0 && current_assign < numClasses) ? char_probs[pos * numClasses + current_assign] : 0.0;
        let candidate_prob = (candidate_assign >= 0 && candidate_assign < numClasses) ? char_probs[pos * numClasses + candidate_assign] : 0.0;
        if (!_BOUNDARY_SNAP_ADJACENT_CHARS.has(ch) && candidate_prob < (current_prob - _BOUNDARY_WRAP_PROB_MARGIN)) return null;
        
        score += 0.8 * (candidate_prob - current_prob);
        if (_BOUNDARY_SNAP_ADJACENT_CHARS.has(ch)) score += 0.10;
        let moving_out = (candidate_assign === host_label && current_assign === inner_label);
        let moving_in = (candidate_assign === inner_label && current_assign === host_label);
        if (_BOUNDARY_SNAP_DELIMITER_CHARS.has(ch)) {
            if (moving_out) score += _BOUNDARY_WRAP_SHELL_DELIMITER_EJECT_BONUS;
            else if (moving_in) score -= _BOUNDARY_WRAP_SHELL_DELIMITER_SWALLOW_PENALTY;
        }
        if (_BOUNDARY_WRAP_QUOTE_CHARS.has(ch)) {
            if (moving_out) score += _BOUNDARY_WRAP_SHELL_QUOTE_EJECT_BONUS;
            else if (moving_in) score -= 0.15;
        }
        shift_penalty += 1;
    }
    if (changed) {
        score += 0.60 * (_mean_label_support(char_probs, start, end, inner_label, numClasses) - _mean_label_support(char_probs, start, end, host_label, numClasses));
        score -= 0.05 * Math.max(0, shift_penalty - 2);
    }
    return score;
}

function _apply_boundary_shift(labels, current, boundary, left_label, right_label) {
    let out = new Int32Array(labels);
    if (boundary < current) {
        for (let i = boundary; i < current; i++) out[i] = right_label;
    } else if (boundary > current) {
        for (let i = current; i < boundary; i++) out[i] = left_label;
    }
    return out;
}

function _count_local_submin_interior_runs(labels, min_run_chars, window_start, window_end) {
    let runs = _build_label_runs(labels);
    let count = 0;
    let min_len = null;
    for (let idx = 0; idx < runs.length; idx++) {
        let start = runs[idx][0], end = runs[idx][1];
        if (end <= window_start || start >= window_end) continue;
        let run_len = end - start;
        min_len = min_len === null ? run_len : Math.min(min_len, run_len);
        if (idx > 0 && idx < runs.length - 1 && run_len < min_run_chars) {
            count += 1;
        }
    }
    return [count, min_len !== null ? min_len : 0];
}

function _snap_boundaries_to_delimiters(text, labels, charProbs, max_shift, min_run_chars, numClasses) {
    let arr = new Int32Array(labels);
    if (max_shift <= 0 || arr.length <= 1) return arr;
    
    const DELIMS = new Set(['<', '>', '"', "'", '`', '[', ']', '(', ')', '{', '}', '/', ',', ';', ':']);
    
    let max_passes = Math.max(1, arr.length * 2);
    for (let p = 0; p < max_passes; p++) {
        let runs = _build_label_runs(arr);
        if (runs.length <= 1) break;
        let changed = false;
        
        for (let idx = 0; idx < runs.length - 1; idx++) {
            let left_run = runs[idx];
            let right_run = runs[idx + 1];
            let boundary = left_run[1]; // index of first char of right_run
            let left_label = left_run[2];
            let right_label = right_run[2];
            
            let best_boundary = boundary;
            let best_score = -1;
            
            // Score a boundary: +1 if adjacent to delimiter
            const getScore = (b) => {
                let s = 0;
                if (b > 0 && DELIMS.has(text[b - 1])) s += 1;
                if (b < text.length && DELIMS.has(text[b])) s += 1;
                return s;
            };
            
            let current_score = getScore(boundary);
            
            for (let shift = -max_shift; shift <= max_shift; shift++) {
                if (shift === 0) continue;
                let cand = boundary + shift;
                // Don't shift past run boundaries
                if (cand <= left_run[0] || cand >= right_run[1]) continue;

                // Never move a boundary onto or across a line break: a line end is
                // already a natural boundary (matches the reference
                // _score_boundary_candidate newline guards).
                if (cand > 0 && text[cand - 1] === '\n') continue;
                if (cand < text.length && text[cand] === '\n') continue;
                let crossesNewline = false;
                for (let i = Math.min(cand, boundary); i < Math.max(cand, boundary); i++) {
                    if (text[i] === '\n') { crossesNewline = true; break; }
                }
                if (crossesNewline) continue;

                let score = getScore(cand);
                if (score <= current_score) continue; // Must strictly improve delimiter score
                
                // Check probability constraint: "rejected if destination label probability is more than 50% lower than assessed by the model."
                let valid = true;
                if (shift < 0) {
                    // Shifting left: chars [cand, boundary) change from left_label to right_label
                    for (let i = cand; i < boundary; i++) {
                        let p_dest = _label_prob(charProbs, i, right_label, numClasses);
                        let p_src = _label_prob(charProbs, i, left_label, numClasses);
                        if (p_dest < p_src * 0.5) { valid = false; break; }
                    }
                } else {
                    // Shifting right: chars [boundary, cand) change from right_label to left_label
                    for (let i = boundary; i < cand; i++) {
                        let p_dest = _label_prob(charProbs, i, left_label, numClasses);
                        let p_src = _label_prob(charProbs, i, right_label, numClasses);
                        if (p_dest < p_src * 0.5) { valid = false; break; }
                    }
                }
                
                if (valid && score > best_score) {
                    best_score = score;
                    best_boundary = cand;
                }
            }
            
            if (best_boundary !== boundary) {
                if (best_boundary < boundary) {
                    for (let i = best_boundary; i < boundary; i++) arr[i] = right_label;
                } else {
                    for (let i = boundary; i < best_boundary; i++) arr[i] = left_label;
                }
                changed = true;
                break; // Restart passes
            }
        }
        if (!changed) break;
    }
    return arr;
}

function _apply_wrapped_run_shift(labels, current_start, current_end, start, end, host_label, inner_label) {
    let out = new Int32Array(labels);
    let union_start = Math.min(current_start, start);
    let union_end = Math.max(current_end, end);
    for (let i = union_start; i < union_end; i++) out[i] = host_label;
    for (let i = start; i < end; i++) out[i] = inner_label;
    return out;
}

function _refine_wrapped_runs(text, labels, char_probs, max_shift, numClasses) {
    let arr = new Int32Array(labels);
    if (max_shift <= 0 || arr.length <= 2) return arr;
    let max_passes = Math.max(1, arr.length);
    for (let p = 0; p < max_passes; p++) {
        let runs = _build_label_runs(arr);
        if (runs.length <= 2) break;
        let changed = false;
        for (let idx = 1; idx < runs.length - 1; idx++) {
            let left_run = runs[idx - 1];
            let middle_run = runs[idx];
            let right_run = runs[idx + 1];
            if (left_run[2] !== right_run[2] || middle_run[2] === left_run[2]) continue;
            let current_start = middle_run[0];
            let current_end = middle_run[1];
            let current_score = _score_wrapped_run_candidate(text, char_probs, left_run, middle_run, right_run, current_start, current_end, numClasses);
            if (current_score === null) {
                current_score = _boundary_local_score(text, current_start) + _boundary_local_score(text, current_end);
            }
            let best_start = current_start;
            let best_end = current_end;
            let best_score = current_score;
            for (let left_shift = -max_shift; left_shift <= max_shift; left_shift++) {
                let cand_start = current_start + left_shift;
                if (cand_start < left_run[0] || cand_start >= current_end) continue;
                for (let right_shift = -max_shift; right_shift <= max_shift; right_shift++) {
                    let cand_end = current_end + right_shift;
                    if (cand_end <= cand_start || cand_end > right_run[1]) continue;
                    if (cand_start === current_start && cand_end === current_end) continue;
                    let cand_left_delim = cand_start > 0 ? text[cand_start - 1] : "";
                    let cand_right_delim = cand_end < text.length ? text[cand_end] : "";
                    if (!_matching_wrap_delimiter(cand_left_delim, cand_right_delim)) continue;
                    let score = _score_wrapped_run_candidate(text, char_probs, left_run, middle_run, right_run, cand_start, cand_end, numClasses);
                    if (score === null) continue;
                    if (score > (best_score + 1e-6)) {
                        best_start = cand_start;
                        best_end = cand_end;
                        best_score = score;
                    }
                }
            }
            if ((best_start !== current_start || best_end !== current_end) && best_score >= (current_score + _BOUNDARY_WRAP_MIN_IMPROVEMENT)) {
                arr = _apply_wrapped_run_shift(arr, current_start, current_end, best_start, best_end, left_run[2], middle_run[2]);
                changed = true;
                break;
            }
        }
        if (!changed) break;
    }
    return arr;
}

function _new_lock_mask(length) {
    return new Uint8Array(Math.max(0, length));
}

function _mark_locked_range(mask, start, end) {
    let lo = Math.max(0, start);
    let hi = Math.min(mask.length, end);
    for (let i = lo; i < hi; i++) mask[i] = 1;
}

function _merge_lock_masks(base, update) {
    if (base.length !== update.length) return base;
    for (let i = 0; i < base.length; i++) {
        if (update[i]) base[i] = 1;
    }
    return base;
}

function _find_next_backtick_run(text, start, end, min_len) {
    let pos = Math.max(0, start);
    let line_end = Math.min(text.length, end);
    while (pos < line_end) {
        if (text[pos] !== "`") {
            pos++;
            continue;
        }
        let run_end = pos;
        while (run_end < line_end && text[run_end] === "`") run_end++;
        if ((run_end - pos) >= min_len) return [pos, run_end];
        pos = run_end;
    }
    return [-1, -1];
}

function _infer_uniform_body_label(labels, char_probs, start, end, markdown_label, numClasses) {
    let lo = Math.max(0, start);
    let hi = Math.min(labels.length, end);
    if (lo >= hi) return markdown_label;
    
    let counts = new Map();
    let supports = new Map();
    
    for (let i = lo; i < hi; i++) {
        let label = labels[i];
        counts.set(label, (counts.get(label) || 0) + 1);
        if (label >= 0 && label < numClasses) {
            supports.set(label, (supports.get(label) || 0.0) + char_probs[i * numClasses + label]);
        } else {
            if (!supports.has(label)) supports.set(label, 0.0);
        }
    }
    if (counts.size === 0) return markdown_label;
    
    let best_count = -1;
    for (let count of counts.values()) {
        if (count > best_count) best_count = count;
    }
    
    let candidates = [];
    for (let [label, count] of counts.entries()) {
        if (count === best_count) candidates.push(label);
    }
    
    if (candidates.length === 1) return candidates[0];
    
    let best_support = -Infinity;
    for (let label of candidates) {
        let sup = supports.get(label) || 0.0;
        if (sup > best_support) best_support = sup;
    }
    
    let support_candidates = [];
    for (let label of candidates) {
        let sup = supports.get(label) || 0.0;
        if (sup >= (best_support - 1e-9)) support_candidates.push(label);
    }
    
    if (support_candidates.includes(markdown_label)) return markdown_label;
    return Math.min(...support_candidates);
}

function _fill_markdown_structure_regions(text, labels, char_probs, markdown_label, numClasses) {
    let arr = new Int32Array(labels);
    let locked = _new_lock_mask(arr.length);
    if (markdown_label === undefined || markdown_label === null || arr.length === 0 || !text) {
        return [arr, locked];
    }
    let n = text.length;
    let line_start = 0;
    while (line_start < n) {
        let line_end = text.indexOf("\n", line_start);
        if (line_end === -1) line_end = n;
        let pos = line_start;
        while (pos < line_end) {
            if (text[pos] !== "`") {
                pos++;
                continue;
            }
            let run_end = pos;
            while (run_end < line_end && text[run_end] === "`") run_end++;
            let run_len = run_end - pos;
            if (run_len < 3) {
                pos = run_end;
                continue;
            }
            let [next_start, next_end] = _find_next_backtick_run(text, run_end, line_end, run_len);
            if (next_start !== -1 && next_start > run_end) {
                for (let i = pos; i < run_end; i++) arr[i] = markdown_label;
                let body_label = _infer_uniform_body_label(arr, char_probs, run_end, next_start, markdown_label, numClasses);
                for (let i = run_end; i < next_start; i++) arr[i] = body_label;
                for (let i = next_start; i < next_end; i++) arr[i] = markdown_label;
                _mark_locked_range(locked, pos, next_end);
                pos = next_end;
                continue;
            }
            let token_end = run_end;
            while (token_end < line_end && text[token_end] !== " " && text[token_end] !== "\t") {
                token_end++;
            }
            for (let i = pos; i < token_end; i++) arr[i] = markdown_label;
            _mark_locked_range(locked, pos, token_end);
            pos = token_end;
        }
        line_start = line_end + 1;
    }
    return [arr, locked];
}

// The virtual "other" label (id === numClasses) has no probability column;
// treat its prob as 0.0 like the Python reference _label_prob, instead of
// reading the next character's row out of bounds.
function _label_prob(charProbs, pos, label, numClasses) {
    if (label >= 0 && label < numClasses) return charProbs[pos * numClasses + label];
    return 0.0;
}

function _mean_prob(charProbs, start, end, targetClass, numClasses) {
    if (start >= end) return 0.0;
    if (targetClass < 0 || targetClass >= numClasses) return 0.0;
    let sum = 0.0;
    for (let i = start; i < end; i++) {
        sum += charProbs[i * numClasses + targetClass];
    }
    return sum / (end - start);
}

function _normalize_short_runs(labels, charProbs, min_run_chars, locked_mask, numClasses, otherId) {
    let arr = new Int32Array(labels);
    if (min_run_chars <= 1 || arr.length <= 2) return arr;
    let max_passes = Math.max(1, arr.length);
    for (let p = 0; p < max_passes; p++) {
        let runs = _build_label_runs(arr);
        let changed = false;
        for (let idx = 1; idx < runs.length - 1; idx++) {
            let start = runs[idx][0], end = runs[idx][1], run_len = end - start;
            if (run_len >= min_run_chars) continue;
            // "other" runs are deliberate abstentions from confidence gating;
            // merging them into a neighbor would assign a label the model gave
            // sub-threshold probability. Leave them alone.
            if (otherId !== undefined && otherId >= 0 && runs[idx][2] === otherId) continue;
            
            let left_run = runs[idx - 1];
            let right_run = runs[idx + 1];
            let left_label = left_run[2], right_label = right_run[2];
            let left_len = left_run[1] - left_run[0];
            let right_len = right_run[1] - right_run[0];
            
            let target = -1;
            
            // 1. If both neighbors share a label, collapse into it
            if (left_label === right_label) {
                target = left_label;
            } 
            // 2. Both neighbors > 10 tokens
            else if (left_len > 10 && right_len > 10) {
                let p_left = _mean_prob(charProbs, Math.max(0, start - 10), Math.min(arr.length, end + 10), left_label, numClasses);
                let p_right = _mean_prob(charProbs, Math.max(0, start - 10), Math.min(arr.length, end + 10), right_label, numClasses);
                target = (p_left >= p_right) ? left_label : right_label;
            } 
            // 3. At least one neighbor <= 10 tokens
            else {
                if (left_len > right_len) target = left_label;
                else if (right_len > left_len) target = right_label;
                else {
                    // Length tie: average softmax over short run + respective side
                    let p_left = _mean_prob(charProbs, left_run[0], end, left_label, numClasses);
                    let p_right = _mean_prob(charProbs, start, right_run[1], right_label, numClasses);
                    target = (p_left >= p_right) ? left_label : right_label;
                }
            }
            
            for (let i = start; i < end; i++) arr[i] = target;
            changed = true;
            break; // restart runs after mutation
        }
        if (!changed) break;
    }
    return arr;
}

/**
 * Step 1: Deterministic whitespace relabeling.
 * Whitespace chars (space, tab, newline) copy the label and prob vector from
 * the nearest non-whitespace character on the same line. If the whole line is
 * whitespace, fall back to the nearest non-whitespace char anywhere. Ties
 * break to the left.
 */
function _relabel_whitespace(text, labels, charProbs, numClasses) {
    const WHITESPACE = new Set([' ', '\t', '\n']);
    const seq = labels.length;
    if (seq === 0) return [new Int32Array(labels), new Float32Array(charProbs)];

    let out = new Int32Array(labels);
    let outProbs = new Float32Array(charProbs);

    // Build line ranges
    let lines = [];
    let start = 0;
    for (let i = 0; i <= seq; i++) {
        if (i === seq || text[i] === '\n') {
            lines.push([start, i]);
            start = i + 1;
        }
    }

    // For each line, find non-whitespace positions
    for (let [lineStart, lineEnd] of lines) {
        let nonWsPositions = [];
        for (let i = lineStart; i < lineEnd; i++) {
            if (!WHITESPACE.has(text[i])) nonWsPositions.push(i);
        }

        for (let i = lineStart; i <= lineEnd && i < seq; i++) {
            if (!WHITESPACE.has(text[i])) continue;

            let nearest = -1;
            if (nonWsPositions.length > 0) {
                // Find nearest on same line, left-favored tie-break
                let bestDist = Infinity;
                for (let p of nonWsPositions) {
                    let d = Math.abs(p - i);
                    if (d < bestDist || (d === bestDist && p < nearest)) {
                        bestDist = d;
                        nearest = p;
                    }
                }
            } else {
                // Whole line is whitespace: cross-line fallback
                let bestDist = Infinity;
                for (let p = 0; p < seq; p++) {
                    if (WHITESPACE.has(text[p])) continue;
                    let d = Math.abs(p - i);
                    if (d < bestDist || (d === bestDist && p < nearest)) {
                        bestDist = d;
                        nearest = p;
                    }
                }
            }

            if (nearest >= 0) {
                out[i] = labels[nearest];
                for (let c = 0; c < numClasses; c++) {
                    outProbs[i * numClasses + c] = charProbs[nearest * numClasses + c];
                }
            }
        }
    }

    return [out, outProbs];
}

/**
 * Step 2: Confidence gating to "other" label.
 * If the max class probability < threshold, assign otherId.
 */
function _threshold_labels(labels, charProbs, numClasses, threshold, otherId) {
    let out = new Int32Array(labels);
    if (threshold <= 0.0 || otherId < 0) return out;
    for (let i = 0; i < labels.length; i++) {
        let maxProb = -1;
        for (let c = 0; c < numClasses; c++) {
            let p = charProbs[i * numClasses + c];
            if (p > maxProb) maxProb = p;
        }
        if (maxProb < threshold) {
            out[i] = otherId;
        }
    }
    return out;
}

/**
 * Same pipeline as postprocessCharLabels, but also returns the label array
 * after every enabled step so callers can attribute label changes to the
 * step that made them: { labels, stages: [{ step, labels }] }.
 */
function postprocessCharLabelsTraced(text, labels, charProbs, options = {}) {
    let min_run_chars = options.min_run_chars !== undefined ? options.min_run_chars : 3;
    let boundary_snap_max_shift = options.boundary_snap_max_shift !== undefined ? options.boundary_snap_max_shift : 2;
    let threshold = options.threshold || 0.3;
    let otherId = options.otherId !== undefined ? options.otherId : -1;

    let numClasses = Math.floor(charProbs.length / labels.length);
    if (numClasses <= 0) return { labels: new Int32Array(labels), stages: [] };

    let pp = options.ppOptions || { whitespace: true, threshold: true, snap: true, shortRuns: true };

    // Start from raw argmax labels
    let currentLabels = new Int32Array(labels);
    let currentProbs = new Float32Array(charProbs);
    let stages = [];

    // Step 1: Deterministic whitespace relabeling
    if (pp.whitespace) {
        let [wsLabels, wsProbs] = _relabel_whitespace(text, currentLabels, currentProbs, numClasses);
        currentLabels = wsLabels;
        currentProbs = wsProbs;
        stages.push({ step: 'whitespace', labels: currentLabels });
    }

    // Step 2: Confidence gating (threshold)
    if (pp.threshold) {
        currentLabels = _threshold_labels(currentLabels, currentProbs, numClasses, threshold, otherId);
        stages.push({ step: 'threshold', labels: currentLabels });
    }

    // Step 3: Local boundary snapping (±2 chars toward delimiters)
    if (pp.snap) {
        currentLabels = _snap_boundaries_to_delimiters(text, currentLabels, currentProbs, boundary_snap_max_shift, min_run_chars, numClasses);
        stages.push({ step: 'snap', labels: currentLabels });
    }

    // Step 4: Minimum-run normalization (remove interior runs < min_run_chars)
    if (pp.shortRuns) {
        let locked_mask = _new_lock_mask(labels.length);
        currentLabels = _normalize_short_runs(currentLabels, currentProbs, min_run_chars, locked_mask, numClasses, otherId);
        stages.push({ step: 'shortRuns', labels: currentLabels });
    }

    return { labels: currentLabels, stages };
}

function postprocessCharLabels(text, labels, charProbs, options = {}) {
    return postprocessCharLabelsTraced(text, labels, charProbs, options).labels;
}