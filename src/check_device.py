#!/usr/bin/env python3
"""Report available ML accelerators without requiring PyTorch."""

import json
import platform
import sys


def probe() -> dict:
    result = {"platform": platform.platform(), "python": sys.version.split()[0], "torch_available": False, "cuda": False, "mps": False, "recommended": "cpu"}
    try:
        import torch
    except ImportError:
        return result
    result["torch_available"] = True
    result["cuda"] = bool(torch.cuda.is_available())
    result["mps"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    result["recommended"] = "cuda" if result["cuda"] else "mps" if result["mps"] else "cpu"
    return result


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, sort_keys=True))
