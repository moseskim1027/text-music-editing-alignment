#!/usr/bin/env python3
"""Check access to the selected MusicGen checkpoint without loading weights."""

import argparse
import json
import platform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/musicgen-small")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = {"model": args.model, "platform": platform.platform(), "ready": False}
    try:
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
        AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
        result.update({"architecture": config.architectures, "model_type": config.model_type, "ready": True})
    except Exception as exc:  # preflight should report actionable failure details
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
