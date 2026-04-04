"""Head-to-head evaluation between agents."""

import logging
import random

import torch

from hexo.game.constants import BOARD_SIZE, N_CELLS, Player
from hexo.game.state import GameState
from hexo.mcts.search import create_tree, select_leaf, expand_and_backprop, select_move

logger = logging.getLogger(__name__)


class MCTSBot:
    """Bot that uses MCTS + neural network to select moves."""

    def __init__(self, model, device, n_sims=100):
        self.model = model
        self.device = device
        self.n_sims = n_sims

    def get_move(self, game: GameState) -> tuple[int, int]:
        tree = create_tree(game, self.model, self.device, add_noise=False)
        for _ in range(self.n_sims):
            leaf = select_leaf(tree, game)
            if leaf.is_terminal or not leaf.deltas:
                expand_and_backprop(tree, leaf, 0.0)
            else:
                planes = tree.root_planes.clone()
                if leaf.player_flipped:
                    planes[[0, 1]] = planes[[1, 0]]
                for gq, gr, ch in leaf.deltas:
                    actual_ch = (1 - ch) if leaf.player_flipped else ch
                    planes[actual_ch, gq, gr] = 1.0
                with torch.no_grad():
                    x = planes.unsqueeze(0).to(self.device)
                    pol, val = self.model(x)
                expand_and_backprop(tree, leaf, val[0].item())
        return select_move(tree, temperature=0.0)


class RandomBot:
    """Bot that plays random legal moves."""

    def get_move(self, game: GameState) -> tuple[int, int]:
        empty = [(q, r) for q in range(BOARD_SIZE) for r in range(BOARD_SIZE)
                 if (q, r) not in game.board]
        return random.choice(empty)


def play_match(bot_a, bot_b, n_games: int = 20,
               max_moves: int = 150) -> dict:
    """Play n_games between bot_a (player A) and bot_b (player B).

    Returns dict with wins_a, wins_b, draws, and game lengths.
    """
    wins_a = wins_b = draws = 0
    total_moves = 0

    for game_idx in range(n_games):
        game = GameState()
        game.make_move(BOARD_SIZE // 2, BOARD_SIZE // 2)  # First move at center

        while not game.game_over and game.move_count < max_moves:
            if game.current_player == Player.A:
                q, r = bot_a.get_move(game)
            else:
                q, r = bot_b.get_move(game)
            game.make_move(q, r)

        total_moves += game.move_count
        if game.winner == Player.A:
            wins_a += 1
        elif game.winner == Player.B:
            wins_b += 1
        else:
            draws += 1

        logger.info(f"Game {game_idx+1}/{n_games}: "
                     f"winner={game.winner.name} moves={game.move_count}")

    avg_moves = total_moves / max(n_games, 1)
    return {
        'wins_a': wins_a, 'wins_b': wins_b, 'draws': draws,
        'win_rate_a': wins_a / n_games,
        'avg_moves': avg_moves,
        'n_games': n_games,
    }
