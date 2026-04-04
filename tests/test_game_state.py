"""Tests for HeXO game state."""

import pytest
from hexo.game.constants import BOARD_SIZE, WIN_LENGTH, HEX_DIRECTIONS, Player
from hexo.game.state import GameState


class TestPlayer:
    def test_opponent(self):
        assert Player.A.opponent() == Player.B
        assert Player.B.opponent() == Player.A
        assert Player.NONE.opponent() == Player.NONE


class TestTurnOrder:
    """Turn order: A(1) B(2) A(2) B(2) A(2) ..."""

    def test_initial_state(self):
        g = GameState()
        assert g.current_player == Player.A
        assert g.moves_left_in_turn == 1
        assert g.move_count == 0

    def test_first_turn_single_stone(self):
        g = GameState()
        g.make_move(16, 16)
        # After A's single stone, it's B's turn with 2 moves
        assert g.current_player == Player.B
        assert g.moves_left_in_turn == 2
        assert g.move_count == 1

    def test_second_turn_two_stones(self):
        g = GameState()
        g.make_move(16, 16)  # A plays 1
        g.make_move(15, 15)  # B plays 1st of 2
        assert g.current_player == Player.B
        assert g.moves_left_in_turn == 1
        assert g.move_count == 2

        g.make_move(15, 16)  # B plays 2nd of 2
        assert g.current_player == Player.A
        assert g.moves_left_in_turn == 2
        assert g.move_count == 3

    def test_full_turn_sequence(self):
        """Verify: A(1) B(2) A(2) B(2) A(2) ..."""
        g = GameState()
        expected = [
            # (player_before_move, moves_left_before)
            (Player.A, 1),   # A's single stone
            (Player.B, 2),   # B's first
            (Player.B, 1),   # B's second
            (Player.A, 2),   # A's first
            (Player.A, 1),   # A's second
            (Player.B, 2),   # B's first
            (Player.B, 1),   # B's second
            (Player.A, 2),   # A's first
            (Player.A, 1),   # A's second
        ]
        for i, (exp_player, exp_left) in enumerate(expected):
            assert g.current_player == exp_player, f"Move {i}: wrong player"
            assert g.moves_left_in_turn == exp_left, f"Move {i}: wrong moves_left"
            g.make_move(i, 0)


class TestMakeMove:
    def test_basic_placement(self):
        g = GameState()
        assert g.make_move(5, 5)
        assert g.board[(5, 5)] == Player.A

    def test_cannot_place_on_occupied(self):
        g = GameState()
        g.make_move(5, 5)
        assert not g.make_move(5, 5)

    def test_wrapping_coordinates(self):
        g = GameState()
        g.make_move(BOARD_SIZE + 3, BOARD_SIZE + 7)
        assert (3, 7) in g.board

    def test_cannot_move_after_game_over(self):
        g = GameState()
        g.game_over = True
        assert not g.make_move(0, 0)


class TestUndoMove:
    def test_undo_restores_state(self):
        g = GameState()
        g.make_move(16, 16)  # A plays
        state = g.save_state()
        g.make_move(10, 10)  # B plays 1st
        g.undo_move(10, 10, state)

        assert g.current_player == Player.B
        assert g.moves_left_in_turn == 2
        assert g.move_count == 1
        assert (10, 10) not in g.board

    def test_undo_sequence(self):
        g = GameState()
        states = []
        moves = [(16, 16), (10, 10), (10, 11), (5, 5)]
        for q, r in moves:
            states.append(g.save_state())
            g.make_move(q, r)

        # Undo all in reverse
        for (q, r), state in zip(reversed(moves), reversed(states)):
            g.undo_move(q, r, state)

        assert g.move_count == 0
        assert g.current_player == Player.A
        assert g.moves_left_in_turn == 1
        assert len(g.board) == 0


class TestWinDetection:
    def _force_line(self, g, player, start_q, start_r, dq, dr, length):
        """Place a line of stones, ignoring turn order."""
        for i in range(length):
            q = (start_q + dq * i) % BOARD_SIZE
            r = (start_r + dr * i) % BOARD_SIZE
            g.board[(q, r)] = player
            g.move_count += 1

    def test_win_direction_1_0(self):
        g = GameState()
        self._force_line(g, Player.A, 10, 10, 1, 0, WIN_LENGTH)
        assert g._check_win(10, 10)

    def test_win_direction_0_1(self):
        g = GameState()
        self._force_line(g, Player.A, 10, 10, 0, 1, WIN_LENGTH)
        assert g._check_win(10, 10)

    def test_win_direction_1_neg1(self):
        g = GameState()
        self._force_line(g, Player.A, 10, 10, 1, -1, WIN_LENGTH)
        assert g._check_win(10, 10)

    def test_no_win_with_5(self):
        g = GameState()
        self._force_line(g, Player.A, 10, 10, 1, 0, WIN_LENGTH - 1)
        assert not g._check_win(10, 10)

    def test_win_checked_from_middle(self):
        """Win detection should work when checking from any cell in the line."""
        g = GameState()
        self._force_line(g, Player.A, 10, 10, 1, 0, WIN_LENGTH)
        # Check from middle of line
        assert g._check_win(13, 10)

    def test_wrapping_win(self):
        """A line that wraps around the torus edge should count."""
        g = GameState()
        # Place stones crossing the q=0 boundary
        start_q = BOARD_SIZE - 3  # e.g., 29
        self._force_line(g, Player.A, start_q, 10, 1, 0, WIN_LENGTH)
        # Stones at q=29,30,31,0,1,2 r=10
        assert g._check_win(start_q, 10)

    def test_wrapping_win_diagonal(self):
        """Diagonal line wrapping around edges."""
        g = GameState()
        start_q = BOARD_SIZE - 2
        start_r = 3
        self._force_line(g, Player.B, start_q, start_r, 1, -1, WIN_LENGTH)
        assert g._check_win(start_q, start_r)

    def test_different_players_dont_connect(self):
        g = GameState()
        self._force_line(g, Player.A, 10, 10, 1, 0, 3)
        self._force_line(g, Player.B, 13, 10, 1, 0, 3)
        assert not g._check_win(10, 10)
        assert not g._check_win(13, 10)

    def test_win_via_make_move(self):
        """Integration: win detected through make_move."""
        g = GameState()
        self._force_line(g, Player.A, 10, 10, 1, 0, WIN_LENGTH - 1)
        g.current_player = Player.A
        g.moves_left_in_turn = 1
        last_q = (10 + WIN_LENGTH - 1) % BOARD_SIZE
        g.make_move(last_q, 10)
        assert g.game_over
        assert g.winner == Player.A


class TestCloneAndSerialize:
    def test_clone_independent(self):
        g = GameState()
        g.make_move(16, 16)
        g2 = g.clone()
        g2.make_move(10, 10)
        assert (10, 10) not in g.board
        assert (10, 10) in g2.board

    def test_roundtrip_dict(self):
        g = GameState()
        g.make_move(16, 16)
        g.make_move(10, 10)
        d = g.to_dict()
        g2 = GameState.from_dict(d)
        assert g2.board == g.board
        assert g2.current_player == g.current_player
        assert g2.moves_left_in_turn == g.moves_left_in_turn
        assert g2.move_count == g.move_count
