"""Policy and value heads for HexONet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Single-move policy head: trunk features -> [B, N] logits.

    Simple 1x1 conv to reduce channels, then flatten.
    Occupied cells are masked to -inf.
    """

    def __init__(self, trunk_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(trunk_channels, 1, 1)

    def forward(self, trunk_features: torch.Tensor,
                occupied_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            trunk_features: [B, C, H, W]
            occupied_mask: [B, H, W] bool, True where a stone exists

        Returns:
            logits: [B, H*W]
        """
        B, C, H, W = trunk_features.shape
        logits = self.conv(trunk_features).reshape(B, H * W)

        if occupied_mask is not None:
            flat_mask = occupied_mask.reshape(B, H * W)
            logits = logits.masked_fill(flat_mask, float("-inf"))

        return logits


class ValueHead(nn.Module):
    """Value head: trunk features -> scalar in [-1, 1].

    1x1 conv -> GN -> ReLU -> mean+max pool -> FC -> ReLU -> FC -> tanh
    """

    def __init__(self, trunk_channels: int, v_channels: int = 32,
                 gn_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(trunk_channels, v_channels, 1, bias=False)
        self.gn = nn.GroupNorm(gn_groups, v_channels)
        self.fc1 = nn.Linear(v_channels * 2, 256)  # mean + max
        self.fc2 = nn.Linear(256, 1)

    def forward(self, trunk_features: torch.Tensor) -> torch.Tensor:
        """Returns [B] scalar values in [-1, 1]."""
        v = F.relu(self.gn(self.conv(trunk_features)))
        v_mean = v.mean(dim=[2, 3])
        v_max = v.amax(dim=[2, 3])
        pooled = torch.cat([v_mean, v_max], dim=-1)
        return torch.tanh(self.fc2(F.relu(self.fc1(pooled)))).squeeze(-1)
