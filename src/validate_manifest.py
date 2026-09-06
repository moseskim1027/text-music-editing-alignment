#!/usr/bin/env python3
"""Validate and summarize the JSONL benchmark manifest."""

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED = {"example_id", "source_audio", "instruction", "operation", "target_stem", "split"}
OPERATIONS = {"add", "remove", "replace"}
SPLITS = {"train", "validation", "test"}


def load_records(path: Path):
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: record must be an object")
        missing = REQUIRED - record.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing {sorted(missing)}")
        if record["operation"] not in OPERATIONS:
            raise ValueError(f"line {line_number}: unsupported operation {record['operation']!r}")
        if record["split"] not in SPLITS:
            raise ValueError(f"line {line_number}: unsupported split {record['split']!r}")
        yield record


def summarize(path: Path) -> dict:
    records = list(load_records(path))
    ids = [record["example_id"] for record in records]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate example_id values: {duplicate_ids}")
    sources_by_split = {}
    for record in records:
        sources_by_split.setdefault(record["source_audio"], set()).add(record["split"])
    leaked_sources = sorted(source for source, splits in sources_by_split.items() if len(splits) > 1)
    if leaked_sources:
        raise ValueError(f"source audio appears in multiple splits: {leaked_sources}")
    return {
        "records": len(records),
        "by_operation": dict(sorted(Counter(r["operation"] for r in records).items())),
        "by_split": dict(sorted(Counter(r["split"] for r in records).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.manifest), indent=2, sort_keys=True))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
