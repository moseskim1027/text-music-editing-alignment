"""Lightweight audio/text fusion primitives for instruction-conditioned editing."""

import torch
from torch import nn


class SourceAudioFusion(nn.Module):
    """Turn source-audio features into a prefix token for text conditioning.

    The upstream audio encoder is injected by the training backend; this module
    only owns the trainable projection and fusion operation.
    """

    def __init__(self, audio_dim: int, text_dim: int):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, text_dim))

    def forward(self, audio_features: torch.Tensor, text_hidden_states: torch.Tensor) -> torch.Tensor:
        if audio_features.ndim != 3 or text_hidden_states.ndim != 3:
            raise ValueError("audio and text features must be rank-3 tensors")
        if audio_features.shape[0] != text_hidden_states.shape[0]:
            raise ValueError("audio and text batch sizes must match")
        pooled_audio = audio_features.mean(dim=1, keepdim=True)
        prefix = self.projection(pooled_audio)
        return torch.cat([prefix, text_hidden_states], dim=1)
