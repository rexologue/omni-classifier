#!/usr/bin/env python3
"""Build a labeled manifest with a deterministic stratified train/dev/test split.

Scans dataset/<class-dir>/*.mp3, assigns the gold label from the directory, and
buckets each sample into train/dev/test by a hash of its id (stable across runs,
decorrelated from filename/source ordering, exact per-class proportions).

Usage:
    python tools/build_manifest.py --root dataset --out WORKDIR/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

# Directory name -> gold label.
LABEL_DIRS = {"human": "human", "ao": "answering_machine"}

# Cumulative split boundaries on a 0..1 hash.
SPLITS = [("train", 0.60), ("dev", 0.80), ("test", 1.00)]


def _hash_unit(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _split_for(rank: float) -> str:
    for name, upper in SPLITS:
        if rank <= upper:
            return name
    return SPLITS[-1][0]


def build(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dir_name, gold in LABEL_DIRS.items():
        files = sorted((root / dir_name).glob("*.mp3"))
        # Rank within class by hash so proportions are exact per class and the
        # split does not follow filename/date ordering.
        ranked = sorted(files, key=lambda p: _hash_unit(f"{gold}/{p.name}"))
        n = len(ranked)
        for idx, path in enumerate(ranked):
            sample_id = f"{gold[:1]}{idx:03d}"
            split = _split_for((idx + 1) / n)
            rows.append(
                {
                    "id": sample_id,
                    "audio_path": str(path.resolve()),
                    "gold": gold,
                    "split": split,
                    "text": "Определи тип ответа в звонке.",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="dataset", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "audio_path", "gold", "split", "text"])
        writer.writeheader()
        writer.writerows(rows)

    # Report the resulting distribution.
    from collections import Counter

    by_split_class = Counter((r["split"], r["gold"]) for r in rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    for split, _ in SPLITS:
        parts = [f"{gold}={by_split_class[(split, gold)]}" for gold in sorted(set(LABEL_DIRS.values()))]
        print(f"  {split:<5} " + " ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
