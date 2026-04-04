"""Tests for heuristic policy."""

import numpy as np
from hexo.game.constants import BOARD_SIZE, Player
from hexo.game.state import GameState
from hexo.encoding.heuristic import compute_heuristic_values, heuristic_policy


class TestHeuristicValues:
    def test_empty_board_symmetric(self):
        """On an empty board, all cells should have equal value."""
        g = GameState()
        vals = compute_heuristic_values(g)
        # All cells should be equal (by torus symmetry)
        assert vals.shape == (BOARD_SIZE, BOARD_SIZE)
        assert np.allclose(vals, vals[0, 0])

    def test_occupied_cells_negative_inf(self):
        g = GameState()
        g.make_move(16, 16)
        vals = compute_heuristic_values(g)
        assert vals[16, 16] == float('-inf')

    def test_adjacent_to_friendly_stone_scores_higher(self):
        """Cells near a friendly stone should score higher than random distant cells."""
        g = GameState()
        g.board[(16, 16)] = Player.A
        g.move_count = 1
        g.current_player = Player.A
        g.moves_left_in_turn = 1

        vals = compute_heuristic_values(g)
        # Cell adjacent to A's stone in a hex direction
        adj_val = vals[17, 16]  # direction (1, 0)
        # Cell far away
        far_val = vals[0, 0]
        assert adj_val > far_val

    def test_extending_own_line_scores_highest(self):
        """Cells that extend a friendly 4-in-a-row should score very high."""
        g = GameState()
        for i in range(4):
            g.board[(10 + i, 10)] = Player.A
        g.move_count = 4
        g.current_player = Player.A
        g.moves_left_in_turn = 2

        vals = compute_heuristic_values(g)
        # Cells at ends of A's line should be the highest-scoring
        end1 = vals[14, 10]
        end2 = vals[9, 10]
        far = vals[0, 0]
        assert end1 > far
        assert end2 > far


class TestHeuristicPolicy:
    def test_sums_to_one(self):
        g = GameState()
        g.make_move(16, 16)
        policy = heuristic_policy(g)
        assert policy.shape == (BOARD_SIZE * BOARD_SIZE,)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_occupied_cells_near_zero(self):
        g = GameState()
        g.make_move(16, 16)
        policy = heuristic_policy(g)
        occupied_idx = 16 * BOARD_SIZE + 16
        assert policy[occupied_idx] < 1e-6

    def test_non_negative(self):
        g = GameState()
        g.make_move(16, 16)
        policy = heuristic_policy(g)
        assert (policy >= 0).all()
