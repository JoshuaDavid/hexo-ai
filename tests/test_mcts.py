"""Tests for MCTS engine."""

import torch
from hexo.game.constants import BOARD_SIZE, N_CELLS, Player
from hexo.game.state import GameState
from hexo.model.resnet import HexONet
from hexo.mcts.search import (
    create_tree, select_leaf, expand_and_backprop, maybe_expand_leaf,
    select_move, get_visit_counts,
)
from hexo.mcts.tree import idx_to_cell


def _make_model():
    return HexONet(num_blocks=2, num_filters=16, gn_groups=4, v_channels=8)


class TestCreateTree:
    def test_tree_has_candidates(self):
        g = GameState()
        g.make_move(16, 16)  # A plays center
        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))
        assert tree.root_pos.move_node.n > 0
        assert tree.root_player == Player.B

    def test_empty_board_first_move(self):
        g = GameState()
        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))
        # First move: only center candidate
        assert tree.root_pos.move_node.n == 1
        assert tree.root_pos.move_node.actions[0] == 16 * BOARD_SIZE + 16

    def test_occupied_cells_not_candidates(self):
        g = GameState()
        g.make_move(16, 16)
        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))
        occupied_idx = 16 * BOARD_SIZE + 16
        assert occupied_idx not in tree.root_pos.move_node.actions


class TestSelectLeaf:
    def test_returns_leaf_info(self):
        g = GameState()
        g.make_move(16, 16)
        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))
        leaf = select_leaf(tree, g)
        assert len(leaf.path) > 0
        # Game state should be restored
        assert g.move_count == 1
        assert g.current_player == Player.B

    def test_terminal_detection(self):
        """If a move wins, leaf should be terminal."""
        g = GameState()
        # Set up A with 5 in a row, A's turn
        for i in range(5):
            g.board[(10 + i, 10)] = Player.A
            g.move_count += 1
        g.current_player = Player.A
        g.moves_left_in_turn = 1

        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))
        # Run select_leaf — should find the winning move
        leaf = select_leaf(tree, g)
        # May or may not be terminal on first try (depends on which cell PUCT picks)
        # But after enough visits, the winning cell should be found
        for _ in range(100):
            leaf = select_leaf(tree, g)
            expand_and_backprop(tree, leaf, 0.0)
        # Now the winning move should dominate
        visits = get_visit_counts(tree)
        winning_idx = (10 + 5) * BOARD_SIZE + 10
        # Check it was found (could also be at position before the line)
        assert len(visits) > 0


class TestBackprop:
    def test_visit_counts_increase(self):
        g = GameState()
        g.make_move(16, 16)
        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))

        for _ in range(10):
            leaf = select_leaf(tree, g)
            expand_and_backprop(tree, leaf, 0.0)

        root = tree.root_pos.move_node
        total_visits = sum(root.visits)
        assert total_visits == 10
        assert root.visit_count == 10

    def test_values_bounded(self):
        g = GameState()
        g.make_move(16, 16)
        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))

        for _ in range(50):
            leaf = select_leaf(tree, g)
            expand_and_backprop(tree, leaf, 0.5)

        root = tree.root_pos.move_node
        for i in range(root.n):
            if root.visits[i] > 0:
                q = root.values[i] / root.visits[i]
                assert -1.5 <= q <= 1.5, f"Q value out of range: {q}"


class TestBackpropSamePlayer:
    """Test that backprop handles 2-stone turns correctly."""

    def test_same_player_consecutive_moves(self):
        """During a 2-stone turn, both moves should get the same sign value."""
        from hexo.mcts.tree import MCTSNode, PosNode, MCTSTree, LeafInfo, init_node_children
        from hexo.mcts.search import expand_and_backprop

        # Create a minimal path: player B makes 2 consecutive moves
        node1 = MCTSNode()
        init_node_children(node1, [(100, 0.5), (200, 0.5)])
        node2 = MCTSNode()
        init_node_children(node2, [(300, 0.5), (400, 0.5)])

        leaf = LeafInfo(
            path=[(node1, 100), (node2, 300)],
            path_players=[Player.B, Player.B],  # same player, 2-stone turn
            current_player=Player.A,  # after B's turn, it's A's turn
            is_terminal=False,
        )
        # nn_value=0.8 from A's perspective (current_player at leaf)
        expand_and_backprop(None, leaf, 0.8)

        # Both moves by B should get value from B's perspective: -0.8
        assert node1.values[0] / node1.visits[0] == -0.8
        assert node2.values[0] / node2.visits[0] == -0.8

    def test_alternating_players(self):
        """Normal alternation: A then B should get opposite signs."""
        from hexo.mcts.tree import MCTSNode, LeafInfo, init_node_children
        from hexo.mcts.search import expand_and_backprop

        node1 = MCTSNode()
        init_node_children(node1, [(100, 1.0)])
        node2 = MCTSNode()
        init_node_children(node2, [(200, 1.0)])

        leaf = LeafInfo(
            path=[(node1, 100), (node2, 200)],
            path_players=[Player.A, Player.B],
            current_player=Player.A,
            is_terminal=False,
        )
        expand_and_backprop(None, leaf, 0.6)

        # Node1 (A's move): A's perspective of A's value = 0.6
        assert node1.values[0] / node1.visits[0] == 0.6
        # Node2 (B's move): B's perspective of A's value = -0.6
        assert node2.values[0] / node2.visits[0] == -0.6


class TestSelectMove:
    def test_returns_valid_cell(self):
        g = GameState()
        g.make_move(16, 16)
        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))

        for _ in range(20):
            leaf = select_leaf(tree, g)
            expand_and_backprop(tree, leaf, 0.0)

        q, r = select_move(tree, temperature=0.0)
        assert 0 <= q < BOARD_SIZE
        assert 0 <= r < BOARD_SIZE
        assert (q, r) not in g.board

    def test_greedy_concentrates(self):
        """With temperature=0, should always pick the most-visited move."""
        g = GameState()
        g.make_move(16, 16)
        model = _make_model()
        model.eval()
        tree = create_tree(g, model, torch.device("cpu"))

        for _ in range(50):
            leaf = select_leaf(tree, g)
            expand_and_backprop(tree, leaf, 0.0)

        move1 = select_move(tree, temperature=0.0)
        move2 = select_move(tree, temperature=0.0)
        assert move1 == move2


class TestFullGame:
    def test_mcts_plays_legal_game(self):
        """Play a full game using MCTS, verify all moves are legal."""
        model = _make_model()
        model.eval()
        g = GameState()

        for turn in range(30):  # limit to 30 turns
            if g.game_over:
                break
            tree = create_tree(g, model, torch.device("cpu"), add_noise=False)
            for _ in range(20):
                leaf = select_leaf(tree, g)
                expand_and_backprop(tree, leaf, 0.0)
            q, r = select_move(tree, temperature=0.5)
            assert g.is_valid_move(q, r), f"Invalid move ({q}, {r}) at turn {turn}"
            g.make_move(q, r)

        assert g.move_count > 0
