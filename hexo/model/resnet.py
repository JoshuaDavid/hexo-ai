"""HexONet: ResNet with single-move policy + value heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexo.model.blocks import ResBlock
from hexo.model.heads import PolicyHead, ValueHead


class HexONet(nn.Module):
    """ResNet for HeXO: outputs move logits and position value.

    Architecture:
        - Conv stem (in_channels -> num_filters)
        - N residual blocks with GroupNorm + circular padding
        - Policy head: 1x1 conv -> [B, 1024] logits
        - Value head: 1x1 conv -> pool -> FC -> tanh -> [B] scalar
    """

    def __init__(self, in_channels: int = 3, num_blocks: int = 8,
                 num_filters: int = 64, gn_groups: int = 8,
                 v_channels: int = 32):
        super().__init__()

        # Stem
        self.stem_conv = nn.Conv2d(in_channels, num_filters, 3, padding=1,
                                    padding_mode='circular', bias=False)
        self.stem_gn = nn.GroupNorm(gn_groups, num_filters)

        # Residual trunk
        self.blocks = nn.Sequential(
            *[ResBlock(num_filters, gn_groups) for _ in range(num_blocks)]
        )

        # Heads
        self.policy_head = PolicyHead(num_filters)
        self.value_head = ValueHead(num_filters, v_channels, gn_groups)

    def forward(self, x: torch.Tensor,
                occupied_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, in_channels, H, W] board planes
            occupied_mask: [B, H, W] bool, True where a stone exists

        Returns:
            policy_logits: [B, H*W] (occupied cells masked to -inf)
            value: [B] scalar in [-1, 1]
        """
        trunk = F.relu(self.stem_gn(self.stem_conv(x)))
        trunk = self.blocks(trunk)

        policy = self.policy_head(trunk, occupied_mask)
        value = self.value_head(trunk)

        return policy, value
