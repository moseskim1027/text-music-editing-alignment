#!/usr/bin/env python3
"""Aggregate per-example evaluation scores from a JSONL file."""

import argparse
import json
from pathlib import Path

SCORES = ("edit_success", "target_change", "preservation", "alignment", "preference", "quality")


def aggregate(records: list[dict]) -> dict:
    if not records:
        raise ValueError("at least one score record is required")
    for index, record in enumerate(records, 1):
        missing = {"example_id", "operation", *SCORES} - record.keys()
        if missing:
            raise ValueError(f"record {index}: missing {sorted(missing)}")
        for score in SCORES:
            if not 0 <= float(record[score]) <= 1:
                raise ValueError(f"record {index}: {score} must be between 0 and 1")
    means = {score: sum(float(r[score]) for r in records) / len(records) for score in SCORES}
    return {
        "records": len(records),
        "by_operation": {
            operation: aggregate([r for r in records if r["operation"] == operation])
            for operation in sorted({r["operation"] for r in records})
        } if len({r["operation"] for r in records}) > 1 else {},
        "mean": means,
        "preservation_adjusted_edit_success": means["edit_success"] * means["preservation"],
    }


def read_scores(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(read_scores(args.scores)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
