"""Tests for board encoding."""

import torch
from hexo.game.constants import BOARD_SIZE, Player
from hexo.game.state import GameState
from hexo.encoding.planes import board_to_planes


class TestBoardToPlanes:
    def test_empty_board(self):
        g = GameState()
        planes = board_to_planes(g)
        assert planes.shape == (3, BOARD_SIZE, BOARD_SIZE)
        assert planes[0].sum() == 0  # no current player stones
        assert planes[1].sum() == 0  # no opponent stones
        assert planes[2, 0, 0] == 0.5  # moves_left=1 -> 0.5

    def test_stones_on_correct_channels(self):
        g = GameState()
        g.make_move(16, 16)  # A places at (16,16), now B's turn
        planes = board_to_planes(g)
        # Current player is B, so A's stone should be on channel 1 (opponent)
        assert planes[1, 16, 16] == 1.0
        assert planes[0, 16, 16] == 0.0

    def test_moves_left_channel(self):
        g = GameState()
        # Initial: A has 1 move left
        planes = board_to_planes(g)
        assert planes[2, 0, 0] == 0.5  # 0.5 * 1

        g.make_move(16, 16)  # Now B has 2 moves left
        planes = board_to_planes(g)
        assert planes[2, 0, 0] == 1.0  # 0.5 * 2

        g.make_move(10, 10)  # B used 1, has 1 left
        planes = board_to_planes(g)
        assert planes[2, 0, 0] == 0.5  # 0.5 * 1

    def test_multiple_stones(self):
        g = GameState()
        g.make_move(16, 16)  # A
        g.make_move(10, 10)  # B
        g.make_move(10, 11)  # B
        # Now it's A's turn with 2 moves
        planes = board_to_planes(g)
        # A's stone at (16,16) is current player
        assert planes[0, 16, 16] == 1.0
        # B's stones at (10,10) and (10,11) are opponent
        assert planes[1, 10, 10] == 1.0
        assert planes[1, 10, 11] == 1.0
        assert planes[0].sum() == 1.0
        assert planes[1].sum() == 2.0
