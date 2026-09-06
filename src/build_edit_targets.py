#!/usr/bin/env python3
"""Render deterministic metadata-recorded edit targets from stem audio."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _align(*arrays: np.ndarray) -> list[np.ndarray]:
    length = min(len(array) for array in arrays)
    return [array[:length] for array in arrays]


def render_target(mix: Path, target: Path, output: Path, operation: str, replacement: Path | None = None) -> None:
    """Render one target; source files are read only and output is explicit."""
    mix_audio, mix_sr = sf.read(mix, dtype="float32", always_2d=True)
    target_audio, target_sr = sf.read(target, dtype="float32", always_2d=True)
    if mix_sr != target_sr:
        raise ValueError(f"sample-rate mismatch: mix={mix_sr}, target={target_sr}")
    mix_audio, target_audio = _align(mix_audio, target_audio)
    if operation == "remove":
        rendered = mix_audio - target_audio
    elif operation == "add":
        rendered = mix_audio + target_audio
    elif operation == "replace":
        if replacement is None:
            raise ValueError("replace requires a replacement stem")
        replacement_audio, replacement_sr = sf.read(replacement, dtype="float32", always_2d=True)
        if replacement_sr != mix_sr:
            raise ValueError("sample-rate mismatch: replacement")
        mix_audio, target_audio, replacement_audio = _align(mix_audio, target_audio, replacement_audio)
        rendered = mix_audio - target_audio + replacement_audio
    else:
        raise ValueError(f"unsupported operation: {operation}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, np.clip(rendered, -1.0, 1.0), mix_sr, subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        records = records[:args.limit]
    for record in records:
        render_target(Path(record["source_audio"]), Path(record["target_stem_audio"]), Path(record["target_audio"]), record["operation"], Path(record["replacement_stem_audio"]) if record.get("replacement_stem_audio") else None)
    print(json.dumps({"rendered": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
