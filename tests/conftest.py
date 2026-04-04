"""Shared test fixtures."""

import pytest
import torch

from hexo.game.state import GameState
from hexo.model.resnet import HexONet


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_model():
    """Small model for fast tests."""
    return HexONet(num_blocks=2, num_filters=16, gn_groups=4, v_channels=8)


@pytest.fixture
def game_with_moves():
    """Game state with a few moves played."""
    g = GameState()
    g.make_move(16, 16)  # A
    g.make_move(15, 15)  # B
    g.make_move(15, 16)  # B
    return g
