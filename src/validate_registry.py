#!/usr/bin/env python3
"""Validate private audio provenance registry records."""

import argparse
import json
from pathlib import Path

REQUIRED = {"asset_id", "path", "source", "license", "attribution", "sha256", "consent_status"}
CONSENT = {"not_required", "documented", "pending", "denied"}


def validate(path: Path) -> list[dict]:
    records = []
    ids = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = REQUIRED - record.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing {sorted(missing)}")
        if record["asset_id"] in ids:
            raise ValueError(f"line {line_number}: duplicate asset_id")
        if len(record["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in record["sha256"].lower()):
            raise ValueError(f"line {line_number}: sha256 must be a 64-character hexadecimal digest")
        if record["consent_status"] not in CONSENT:
            raise ValueError(f"line {line_number}: invalid consent_status")
        if not all(str(record[field]).strip() for field in ("path", "source", "license", "attribution")):
            raise ValueError(f"line {line_number}: provenance fields cannot be empty")
        ids.add(record["asset_id"])
        records.append(record)
    if not records:
        raise ValueError("at least one registry record is required")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()
    records = validate(args.registry)
    print(json.dumps({"records": len(records), "valid": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
