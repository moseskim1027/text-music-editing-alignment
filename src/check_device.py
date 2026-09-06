#!/usr/bin/env python3
"""Report available ML accelerators without requiring PyTorch."""

import json
import platform
import subprocess
import sys


def probe() -> dict:
    result = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "apple_gpu_visible": False,
        "metal_supported": False,
        "torch_available": False,
        "mps_built": False,
        "cuda": False,
        "mps": False,
        "recommended": "cpu",
        "diagnostic": "No accelerator detected; CPU is the fallback.",
    }

    if sys.platform == "darwin":
        try:
            display_info = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            result["apple_gpu_visible"] = "Chipset Model:" in display_info
            result["metal_supported"] = "Metal: Supported" in display_info
        except (OSError, subprocess.SubprocessError):
            pass

    try:
        import torch
    except ImportError:
        return result
    result["torch_available"] = True
    result["cuda"] = bool(torch.cuda.is_available())
    result["mps_built"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    result["mps"] = bool(result["mps_built"] and torch.backends.mps.is_available())
    result["recommended"] = "cuda" if result["cuda"] else "mps" if result["mps"] else "cpu"
    if result["mps"]:
        result["diagnostic"] = "MPS is available and selected."
    elif result["apple_gpu_visible"] and result["metal_supported"] and result["mps_built"]:
        result["diagnostic"] = (
            "macOS sees the Apple GPU and Metal, and PyTorch includes MPS, "
            "but the MPS runtime is unavailable in this process. Run natively "
            "outside Docker or a restricted/sandboxed shell."
        )
    elif result["apple_gpu_visible"] and result["metal_supported"]:
        result["diagnostic"] = "macOS sees Metal, but this PyTorch build lacks MPS support."
    return result


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, sort_keys=True))
