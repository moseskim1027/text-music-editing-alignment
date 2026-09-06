#!/usr/bin/env python3
"""Run a tiny adapter-only smoke test without loading a generative model.

This validates the training/runtime contract. It is deliberately not a
MusicGen training implementation; the model backend will plug into this
entry point after the data loader and licensed checkpoint are selected.
"""

import argparse
import json
from pathlib import Path


def choose_device(requested: str, torch):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(config: Path, steps: int, requested_device: str) -> dict:
    import torch

    settings = json.loads(config.read_text())
    device = choose_device(requested_device, torch)
    torch.manual_seed(settings.get("seed", 42))
    parameter = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.SGD([parameter], lr=settings.get("learning_rate", 1e-4))
    losses = []
    for _ in range(steps):
        loss = (parameter - 1).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"smoke_test": True, "backend_ready": False, "device": device, "steps": steps, "initial_loss": losses[0], "final_loss": losses[-1], "config": str(config)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adapter_training.json"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    result = run(args.config, args.steps, args.device)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
