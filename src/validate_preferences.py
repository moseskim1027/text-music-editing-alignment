#!/usr/bin/env python3
"""Validate pairwise preference records used for alignment."""

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED = {"example_id", "instruction", "candidate_a", "candidate_b", "preferred", "criteria", "annotator_id"}
PREFERRED = {"a", "b", "tie"}
CRITERIA = {"requested_edit", "untouched_content", "audio_quality"}


def load_preferences(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON ({exc.msg})") from exc
        missing = REQUIRED - record.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing {sorted(missing)}")
        if record["preferred"] not in PREFERRED:
            raise ValueError(f"line {line_number}: preferred must be one of {sorted(PREFERRED)}")
        if not isinstance(record["criteria"], list) or not set(record["criteria"]).issubset(CRITERIA):
            raise ValueError(f"line {line_number}: unsupported criteria")
        if record["candidate_a"] == record["candidate_b"]:
            raise ValueError(f"line {line_number}: candidates must differ")
        records.append(record)
    if not records:
        raise ValueError("at least one preference record is required")
    return records


def summarize(path: Path) -> dict:
    records = load_preferences(path)
    return {"records": len(records), "by_preference": dict(sorted(Counter(r["preferred"] for r in records).items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preferences", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.preferences), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
