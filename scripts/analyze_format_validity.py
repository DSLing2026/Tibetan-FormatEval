#!/usr/bin/env python3
"""Analyze format-validity diagnostics from scored per-item files.

Expected layout:

results/
  model_a/
    mcqa_4way.json
    mcqa_4way.per_item.jsonl
    mcqa_10way.json
    mcqa_10way.per_item.jsonl
    open_short_answer.json
    open_short_answer.per_item.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any


FORMAT_FILES = {
    "mcqa4": "mcqa_4way",
    "mcqa10": "mcqa_10way",
    "open": "open_short_answer",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rank(values: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(values.items(), key=lambda x: x[1])
    ranks: dict[str, float] = {}
    idx = 0
    while idx < len(sorted_items):
        j = idx
        while j + 1 < len(sorted_items) and sorted_items[j + 1][1] == sorted_items[idx][1]:
            j += 1
        avg_rank = (idx + j) / 2 + 1
        for k in range(idx, j + 1):
            ranks[sorted_items[k][0]] = avg_rank
        idx = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return math.nan
    return cov / math.sqrt(vx * vy)


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return math.nan
    ra = rank({k: a[k] for k in common})
    rb = rank({k: b[k] for k in common})
    return pearson([ra[k] for k in common], [rb[k] for k in common])


def load_results(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    summaries: dict[str, dict[str, Any]] = {}
    per_items: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        summaries[model] = {}
        per_items[model] = {}
        for fmt, stem in FORMAT_FILES.items():
            summary_path = model_dir / f"{stem}.json"
            per_item_path = model_dir / f"{stem}.per_item.jsonl"
            if summary_path.exists() and per_item_path.exists():
                summaries[model][fmt] = read_json(summary_path)
                per_items[model][fmt] = read_jsonl(per_item_path)
    return summaries, per_items


def paired_bootstrap_diff_ci(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    seed: int = 42,
    n_boot: int = 2000,
) -> tuple[float, float]:
    by_a = {row["id"]: row["correct"] for row in rows_a}
    by_b = {row["id"]: row["correct"] for row in rows_b}
    ids = sorted(set(by_a) & set(by_b))
    if not ids:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    diffs = []
    n = len(ids)
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        diff = sum(by_a[i] - by_b[i] for i in sample) / n
        diffs.append(diff)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def pairwise_distinguishable_ratio(per_items: dict[str, dict[str, list[dict[str, Any]]]], fmt: str) -> dict[str, Any]:
    models = sorted(model for model, fmts in per_items.items() if fmt in fmts)
    pairs = list(itertools.combinations(models, 2))
    distinguishable = 0
    details = []
    for a, b in pairs:
        low, high = paired_bootstrap_diff_ci(per_items[a][fmt], per_items[b][fmt])
        is_distinct = int(not (low <= 0 <= high))
        distinguishable += is_distinct
        details.append({"model_a": a, "model_b": b, "diff_ci95": [low, high], "distinguishable": is_distinct})
    return {
        "format": fmt,
        "num_pairs": len(pairs),
        "distinguishable_pairs": distinguishable,
        "pairwise_distinguishable_ratio": distinguishable / len(pairs) if pairs else math.nan,
        "pairs": details,
    }


def all_wrong_rate(per_items: dict[str, dict[str, list[dict[str, Any]]]], fmt: str) -> float:
    item_to_scores: dict[str, list[int]] = {}
    for fmts in per_items.values():
        for row in fmts.get(fmt, []):
            item_to_scores.setdefault(row["id"], []).append(row["correct"])
    if not item_to_scores:
        return math.nan
    return sum(1 for vals in item_to_scores.values() if vals and sum(vals) == 0) / len(item_to_scores)


def score_table(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model, fmts in sorted(summaries.items()):
        row: dict[str, Any] = {"model": model}
        for fmt in ["mcqa4", "mcqa10", "open"]:
            if fmt not in fmts:
                continue
            row[f"{fmt}_accuracy"] = fmts[fmt].get("accuracy")
            row[f"{fmt}_ci95"] = fmts[fmt].get("ci95")
            if "chance_normalized_accuracy" in fmts[fmt]:
                row[f"{fmt}_chance_normalized_accuracy"] = fmts[fmt]["chance_normalized_accuracy"]
            if "not_attempted_rate" in fmts[fmt]:
                row[f"{fmt}_not_attempted_rate"] = fmts[fmt]["not_attempted_rate"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries, per_items = load_results(args.result_root)
    table = score_table(summaries)
    scores_by_format = {
        fmt: {
            row["model"]: row[f"{fmt}_accuracy"]
            for row in table
            if f"{fmt}_accuracy" in row
        }
        for fmt in ["mcqa4", "mcqa10", "open"]
    }
    rank_correlations = {
        "mcqa4_vs_mcqa10": spearman(scores_by_format["mcqa4"], scores_by_format["mcqa10"]),
        "mcqa4_vs_open": spearman(scores_by_format["mcqa4"], scores_by_format["open"]),
        "mcqa10_vs_open": spearman(scores_by_format["mcqa10"], scores_by_format["open"]),
    }
    distinguishability = {
        fmt: pairwise_distinguishable_ratio(per_items, fmt)
        for fmt in ["mcqa4", "mcqa10", "open"]
    }
    floor_indicators = {
        fmt: {
            "all_wrong_item_rate": all_wrong_rate(per_items, fmt),
            "pairwise_distinguishable_ratio": distinguishability[fmt]["pairwise_distinguishable_ratio"],
        }
        for fmt in ["mcqa4", "mcqa10", "open"]
    }
    result = {
        "score_table": table,
        "rank_correlations": rank_correlations,
        "distinguishability": distinguishability,
        "floor_indicators": floor_indicators,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
