"""Explicit edit and preservation objectives for music-edit training."""

import torch
import torch.nn.functional as F


def edit_loss(predicted_target: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Measure reconstruction of the requested target edit."""
    return F.smooth_l1_loss(predicted_target, target)


def preservation_loss(predicted_untouched: torch.Tensor, source_untouched: torch.Tensor) -> torch.Tensor:
    """Penalize changes to content not named by the instruction."""
    return F.smooth_l1_loss(predicted_untouched, source_untouched)


def combined_loss(
    predicted_target: torch.Tensor,
    target: torch.Tensor,
    predicted_untouched: torch.Tensor,
    source_untouched: torch.Tensor,
    preservation_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return separately auditable edit, preservation, and total losses."""
    requested = edit_loss(predicted_target, target)
    preserved = preservation_loss(predicted_untouched, source_untouched)
    total = requested + preservation_weight * preserved
    return {"edit_loss": requested, "preservation_loss": preserved, "total_loss": total}
