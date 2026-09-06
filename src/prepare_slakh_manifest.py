#!/usr/bin/env python3
"""Create metadata-only edit examples from a local Slakh-style directory."""

import argparse
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")


def build_records(dataset_root: Path, output_root: str = "data/derived") -> list[dict]:
    records = []
    has_official_splits = any((dataset_root / split).is_dir() for split in SPLITS)
    track_groups = ((split, sorted((dataset_root / split).glob("Track*"))) for split in SPLITS) if has_official_splits else (
        ("test" if int(track.name.removeprefix("Track")) % 10 == 0 else "validation" if int(track.name.removeprefix("Track")) % 10 == 1 else "train", [track])
        for track in sorted(dataset_root.glob("Track*"))
    )
    for split, tracks in track_groups:
        for track in tracks:
            mix = next((track / f"mix{extension}" for extension in (".flac", ".wav") if (track / f"mix{extension}").exists()), None)
            stems = sorted([* (track / "stems").glob("*.flac"), * (track / "stems").glob("*.wav")])
            if mix is None or not stems:
                continue
            for stem in stems:
                stem_name = stem.stem
                base = f"{track.name}_{stem_name}"
                records.extend([
                    {"example_id": f"{base}_remove", "source_audio": str(mix), "instruction": f"Remove the {stem_name} stem", "operation": "remove", "target_stem": stem_name, "target_audio": f"{output_root}/{split}/{base}_remove.wav", "untouched_stems": [s.stem for s in stems if s != stem], "split": split},
                    {"example_id": f"{base}_add", "source_audio": str(mix), "instruction": f"Add a {stem_name} part", "operation": "add", "target_stem": stem_name, "target_audio": f"{output_root}/{split}/{base}_add.wav", "untouched_stems": [s.stem for s in stems], "split": split},
                ])
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = build_records(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(record) for record in records) + ("\n" if records else ""))
    print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
