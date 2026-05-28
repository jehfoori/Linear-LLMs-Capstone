from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - margin, center + margin


def _group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def accuracy_summary(rows: list[dict[str, Any]], group_keys: list[str] | None = None) -> list[dict[str, Any]]:
    group_keys = group_keys or _default_group_keys(rows)
    out: list[dict[str, Any]] = []
    for group, group_rows in sorted(_group_rows(rows, group_keys).items()):
        n = len(group_rows)
        correct = sum(bool(row.get("correct")) for row in group_rows)
        lo, hi = wilson_ci(correct, n)
        summary = {key: value for key, value in zip(group_keys, group)}
        summary.update(
            {
                "correct": correct,
                "n": n,
                "accuracy": correct / n if n else float("nan"),
                "accuracy_pct": 100 * correct / n if n else float("nan"),
                "ci_low_pct": 100 * lo,
                "ci_high_pct": 100 * hi,
                "avg_elapsed_sec": _mean(row.get("elapsed_sec") for row in group_rows),
                "avg_peak_mem_gib": _mean(row.get("peak_mem_gib") for row in group_rows),
            }
        )
        out.append(summary)
    return out


def failure_summary(rows: list[dict[str, Any]], group_keys: list[str] | None = None) -> list[dict[str, Any]]:
    group_keys = group_keys or [*_default_group_keys(rows), "failure_type"]
    grouped = _group_rows(rows, group_keys)
    totals = defaultdict(int)
    for group, group_rows in grouped.items():
        parent = group[:-1]
        totals[parent] += len(group_rows)

    out: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        count = len(group_rows)
        total = totals[group[:-1]]
        row = {key: value for key, value in zip(group_keys, group)}
        row.update({"count": count, "fraction": count / total if total else float("nan"), "percent": 100 * count / total if total else float("nan")})
        out.append(row)
    return out


def position_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    binned: list[dict[str, Any]] = []
    for row in rows:
        value = float(row.get("needle_position_fraction", 0.0))
        if value <= 0.33:
            position_bin = "early"
        elif value <= 0.66:
            position_bin = "middle"
        else:
            position_bin = "late"
        with_bin = dict(row)
        with_bin["position_bin"] = position_bin
        binned.append(with_bin)
    keys = _default_group_keys(rows)
    return accuracy_summary(binned, [*keys, "position_bin"])


def _mean(values) -> float:
    clean = [float(value) for value in values if value not in (None, "")]
    return sum(clean) / len(clean) if clean else float("nan")


def _default_group_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = ["model_label", "target_length"]
    distractor_values = {row.get("num_distractors") for row in rows if row.get("num_distractors") not in (None, "")}
    if len(distractor_values) > 1:
        keys.append("num_distractors")
    return keys
