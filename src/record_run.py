#!/usr/bin/env python3
"""Write reproducibility metadata for an experiment run."""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_metadata(config: Path, manifest: Path, device: str) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": str(config),
        "config_sha256": sha256(config),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "device": device,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metadata = build_metadata(args.config, args.manifest, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
