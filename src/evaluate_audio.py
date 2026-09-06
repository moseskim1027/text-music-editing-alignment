"""Compute reproducible audio comparison metrics for one edit artifact set."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _read(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1)


def _aligned(*arrays: np.ndarray) -> list[np.ndarray]:
    length = min(len(array) for array in arrays)
    return [array[:length] for array in arrays]


def compare(source: Path, target: Path, generated: Path) -> dict:
    source_audio, target_audio, generated_audio = _aligned(_read(source), _read(target), _read(generated))
    scale = max(float(np.mean(np.abs(source_audio - target_audio))), 1e-6)
    target_error = float(np.mean(np.abs(generated_audio - target_audio)))
    source_error = float(np.mean(np.abs(generated_audio - source_audio)))
    clipping = float(np.mean(np.abs(generated_audio) >= 0.999))
    return {
        "adherence_proxy": max(0.0, min(1.0, 1.0 - target_error / scale)),
        "preservation_proxy": max(0.0, min(1.0, 1.0 - source_error)),
        "quality_proxy": max(0.0, min(1.0, 1.0 - clipping)),
        "preference_win_rate": None,
        "preference_status": "unavailable_without_blinded_human_preferences",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.source, args.target, args.generated)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
