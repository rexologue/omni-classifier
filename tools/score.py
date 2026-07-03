#!/usr/bin/env python3
"""Score classifier predictions against gold labels from a manifest.

Reports confusion matrix, per-class precision/recall/F1, accuracy and macro-F1.
Optionally restricts to a split (--split dev) and/or slices by a difficulty CSV
(id,difficulty) produced by triage (--difficulty ... --only-difficulty hard).

Usage:
    python tools/score.py --pred WORKDIR/out.jsonl --manifest WORKDIR/manifest.csv --split dev
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

CLASSES = ["human", "answering_machine"]


def load_gold(manifest: Path) -> dict[str, dict[str, str]]:
    with manifest.open(encoding="utf-8") as handle:
        return {r["id"]: r for r in csv.DictReader(handle)}


def load_pred(pred_path: Path) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    with pred_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            preds[str(r["id"])] = r
    return preds


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score(gold: dict, preds: dict, *, split: str | None, difficulty: dict | None, only_difficulty: str | None):
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    n = 0
    invalid = 0
    missing = 0
    considered_ids = []

    for sid, meta in gold.items():
        if split and meta.get("split") != split:
            continue
        if only_difficulty and difficulty is not None and difficulty.get(sid) != only_difficulty:
            continue
        g = meta["gold"]
        pred = preds.get(sid)
        if pred is None:
            missing += 1
            continue
        p = pred.get("class")
        if p not in CLASSES:
            invalid += 1
            p = "INVALID"
        confusion[(g, p)] += 1
        n += 1
        considered_ids.append(sid)

    labels_pred = CLASSES + (["INVALID"] if any(k[1] == "INVALID" for k in confusion) else [])
    print(f"scored: {n}  (split={split or 'all'}, difficulty={only_difficulty or 'all'})  "
          f"missing_pred={missing} invalid_pred={invalid}")
    print("\nconfusion (rows=gold, cols=pred):")
    header = "  gold\\pred  " + "".join(f"{c[:16]:>18}" for c in labels_pred)
    print(header)
    for g in CLASSES:
        row = "".join(f"{confusion[(g, p)]:>18}" for p in labels_pred)
        print(f"  {g:<10}" + row)

    correct = sum(confusion[(c, c)] for c in CLASSES)
    acc = correct / n if n else 0.0
    print(f"\naccuracy: {correct}/{n} = {acc:.3f}")

    f1s = []
    print("\nper-class:")
    for c in CLASSES:
        tp = confusion[(c, c)]
        fp = sum(confusion[(g, c)] for g in CLASSES if g != c)
        fn = sum(confusion[(c, p)] for p in labels_pred if p != c)
        precision, recall, f1 = _prf(tp, fp, fn)
        f1s.append(f1)
        print(f"  {c:<18} P={precision:.3f} R={recall:.3f} F1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")
    print(f"\nmacro-F1: {sum(f1s) / len(f1s):.3f}")
    return considered_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default=None, help="train|dev|test (default: all)")
    parser.add_argument("--difficulty", type=Path, default=None, help="CSV id,difficulty")
    parser.add_argument("--only-difficulty", default=None, help="e.g. hard|easy")
    parser.add_argument("--list-errors", action="store_true", help="print misclassified ids")
    args = parser.parse_args()

    gold = load_gold(args.manifest)
    preds = load_pred(args.pred)

    difficulty = None
    if args.difficulty:
        with args.difficulty.open(encoding="utf-8") as handle:
            difficulty = {r["id"]: r["difficulty"] for r in csv.DictReader(handle)}

    considered = score(gold, preds, split=args.split, difficulty=difficulty, only_difficulty=args.only_difficulty)

    if args.list_errors:
        print("\nerrors (gold -> pred):")
        for sid in considered:
            g = gold[sid]["gold"]
            p = preds[sid].get("class")
            if p != g:
                print(f"  {sid}: {g} -> {p}   {Path(gold[sid]['audio_path']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
