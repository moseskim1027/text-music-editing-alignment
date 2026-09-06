#!/usr/bin/env python3
"""Prepare manifest-driven baseline editing jobs.

The model backend is intentionally isolated: ``--dry-run`` works without
AudioCraft and is the local smoke test for the pipeline contract.
"""

import argparse
import json
from pathlib import Path

from src.validate_manifest import load_records


def plan_jobs(manifest: Path, output_dir: Path, require_audio: bool = True) -> list[dict]:
    jobs = []
    for record in load_records(manifest):
        source = Path(record["source_audio"])
        if require_audio and not source.exists():
            raise FileNotFoundError(f"{record['example_id']}: source audio not found: {source}")
        jobs.append({
            "example_id": record["example_id"],
            "source_audio": str(source),
            "instruction": record["instruction"],
            "operation": record["operation"],
            "output_audio": str(output_dir / f"{record['example_id']}.wav"),
        })
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--dry-run", action="store_true", help="plan jobs without loading AudioCraft")
    args = parser.parse_args()
    jobs = plan_jobs(args.manifest, args.output_dir, require_audio=not args.dry_run)
    print(json.dumps({"jobs": len(jobs), "dry_run": args.dry_run, "plan": jobs}, indent=2))
    if not args.dry_run:
        raise RuntimeError("AudioCraft backend is not installed; use --dry-run until the backend is configured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
