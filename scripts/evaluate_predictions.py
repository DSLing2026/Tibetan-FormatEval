#!/usr/bin/env python3
"""Score model predictions for Tibetan FormatEval.

Prediction JSONL schemas:

MCQA:
  {"id": "tfe-0001", "prediction": "B"}
  {"id": "tfe-0001", "prediction_index": 1}

Open short answer:
  {"id": "tfe-0001", "prediction": "..."}
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


LETTER_TO_INDEX = {chr(ord("A") + i): i for i in range(10)}
TIBETAN_DIGITS = str.maketrans("༠༡༢༣༤༥༦༧༨༩", "0123456789")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TIBETAN_DIGITS)
    text = re.sub(r"[\s\"'`.,;:!?，。；：！？་།༎\[\](){}<>]+", "", text)
    return text.lower()


def parse_mcqa_prediction(value: Any, num_choices: int) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value < num_choices else None
    if value is None:
        return None
    text = str(value).strip().translate(TIBETAN_DIGITS)
    if not text:
        return None
    upper = text.upper()

    # Prefer explicit answer markers.
    patterns = [
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([A-J])\b",
        r"(?:答案|选项)\s*(?:是|为|:)?\s*([A-J])\b",
        r"\b([A-J])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            idx = LETTER_TO_INDEX[match.group(1)]
            if idx < num_choices:
                return idx

    number_match = re.search(r"\b([1-9]|10)\b", upper)
    if number_match:
        idx = int(number_match.group(1)) - 1
        if 0 <= idx < num_choices:
            return idx
    return None


def bootstrap_ci(values: list[int], seed: int = 42, n_boot: int = 2000) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def score_mcqa(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], chance: float) -> dict[str, Any]:
    preds = {row["id"]: row for row in pred_rows}
    per_item = []
    missing = 0
    unparsable = 0
    pred_pos = Counter()

    for gold in gold_rows:
        pred = preds.get(gold["id"])
        if pred is None:
            missing += 1
            pred_idx = None
        elif "prediction_index" in pred:
            pred_idx = parse_mcqa_prediction(pred.get("prediction_index"), len(gold["choices"]))
        else:
            pred_idx = parse_mcqa_prediction(pred.get("prediction"), len(gold["choices"]))

        if pred_idx is None:
            unparsable += 1
            correct = 0
        else:
            pred_pos[pred_idx] += 1
            correct = int(pred_idx == gold["answer"])

        per_item.append(
            {
                "id": gold["id"],
                "correct": correct,
                "prediction_index": pred_idx,
                "answer": gold["answer"],
                "source_dataset": gold.get("source_dataset"),
                "source_subject": gold.get("source_subject"),
            }
        )

    values = [row["correct"] for row in per_item]
    acc = sum(values) / len(values) if values else math.nan
    ci_low, ci_high = bootstrap_ci(values)
    chance_norm = (acc - chance) / (1 - chance)
    return {
        "n": len(values),
        "accuracy": acc,
        "ci95": [ci_low, ci_high],
        "chance": chance,
        "chance_normalized_accuracy": chance_norm,
        "missing": missing,
        "unparsable": unparsable,
        "prediction_position_distribution": {str(k): pred_pos[k] for k in sorted(pred_pos)},
        "per_item": per_item,
    }


def score_open(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    preds = {row["id"]: row for row in pred_rows}
    per_item = []
    missing = 0
    not_attempted = 0
    abstain_patterns = re.compile(r"(i don't know|unknown|not sure|cannot answer|不知道|无法回答)", re.I)

    for gold in gold_rows:
        pred = preds.get(gold["id"])
        aliases = gold.get("aliases") or [gold["answer_text"]]
        if pred is None:
            missing += 1
            pred_text = ""
        else:
            pred_text = str(pred.get("prediction", ""))

        normalized_pred = normalize_text(pred_text)
        normalized_aliases = [normalize_text(str(alias)) for alias in aliases]
        attempted = bool(normalized_pred) and not abstain_patterns.search(pred_text)
        if not attempted:
            not_attempted += 1
        correct = int(attempted and any(alias and alias in normalized_pred for alias in normalized_aliases))
        per_item.append(
            {
                "id": gold["id"],
                "correct": correct,
                "attempted": int(attempted),
                "prediction": pred_text,
                "answer_text": gold["answer_text"],
                "source_dataset": gold.get("source_dataset"),
                "source_subject": gold.get("source_subject"),
            }
        )

    values = [row["correct"] for row in per_item]
    attempts = [row["attempted"] for row in per_item]
    acc = sum(values) / len(values) if values else math.nan
    attempted_rate = sum(attempts) / len(attempts) if attempts else math.nan
    ci_low, ci_high = bootstrap_ci(values)
    return {
        "n": len(values),
        "accuracy": acc,
        "ci95": [ci_low, ci_high],
        "attempted_rate": attempted_rate,
        "not_attempted_rate": 1 - attempted_rate,
        "missing": missing,
        "not_attempted": not_attempted,
        "per_item": per_item,
    }


def aggregate_by_dataset(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for row in per_item:
        groups.setdefault(row.get("source_dataset", "unknown"), []).append(row["correct"])
    return {
        key: {"n": len(vals), "accuracy": sum(vals) / len(vals)}
        for key, vals in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--task", choices=["mcqa4", "mcqa10", "open"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    preds = read_jsonl(args.predictions)
    if args.task == "mcqa4":
        result = score_mcqa(gold, preds, chance=0.25)
    elif args.task == "mcqa10":
        result = score_mcqa(gold, preds, chance=0.10)
    else:
        result = score_open(gold, preds)

    per_item = result.pop("per_item")
    result["by_source_dataset"] = aggregate_by_dataset(per_item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    per_item_path = args.output.with_suffix(".per_item.jsonl")
    with per_item_path.open("w", encoding="utf-8") as f:
        for row in per_item:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
