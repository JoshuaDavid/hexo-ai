"""Residual blocks for HexONet."""

import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-activation residual block with GroupNorm and circular padding."""

    def __init__(self, channels: int, gn_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1,
                               padding_mode='circular', bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1,
                               padding_mode='circular', bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, channels)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + x)
