"""Cross-evaluation: HeXO MCTS bot vs HexTicTacToe minimax bot.

Bridges the coordinate systems: HexTicTacToe uses infinite axial coords,
our bot uses a 32x32 torus centered at (16, 16) = origin (0, 0).
"""

import sys
import os
import logging

import torch

# Add HexTicTacToe to path
sys.path.insert(0, "/workspace/HexTicTacToe")

from game import HexGame, Player as HTTPlayer
from ai import MinimaxBot

from hexo.game.constants import BOARD_SIZE, Player as HexoPlayer
from hexo.game.state import GameState
from hexo.model.resnet import HexONet
from hexo.mcts.search import (
    create_tree, select_leaf, expand_and_backprop, select_move,
    maybe_expand_leaf,
)

logger = logging.getLogger(__name__)

# Torus center = origin in HexTicTacToe coordinates
CENTER = BOARD_SIZE // 2


def htt_to_torus(q: int, r: int) -> tuple[int, int]:
    """Convert HexTicTacToe infinite coords to torus coords."""
    return (q + CENTER) % BOARD_SIZE, (r + CENTER) % BOARD_SIZE


def torus_to_htt(tq: int, tr: int) -> tuple[int, int]:
    """Convert torus coords back to HexTicTacToe infinite coords."""
    # Undo the centering, keeping values near 0
    q = tq - CENTER
    r = tr - CENTER
    return q, r


class MCTSBotAdapter:
    """Wraps our MCTS bot to play against HexTicTacToe bots.

    Maintains a shadow GameState on the torus that mirrors the HexGame.
    """

    def __init__(self, model, device, n_sims=200):
        self.model = model
        self.device = device
        self.n_sims = n_sims
        self.pair_moves = True  # match MinimaxBot interface
        self.shadow = GameState()
        self.last_depth = 0
        self.last_score = 0.0

    def reset(self):
        self.shadow = GameState()

    def sync_from_htt_game(self, htt_game):
        """Rebuild shadow state from HexTicTacToe game."""
        self.shadow = GameState()
        # Copy board
        for (q, r), player in htt_game.board.items():
            tq, tr = htt_to_torus(q, r)
            if player == HTTPlayer.A:
                self.shadow.board[(tq, tr)] = HexoPlayer.A
            else:
                self.shadow.board[(tq, tr)] = HexoPlayer.B
        self.shadow.move_count = htt_game.move_count
        self.shadow.current_player = HexoPlayer.A if htt_game.current_player == HTTPlayer.A \
            else HexoPlayer.B
        self.shadow.moves_left_in_turn = htt_game.moves_left_in_turn
        self.shadow.game_over = htt_game.game_over
        if htt_game.winner == HTTPlayer.A:
            self.shadow.winner = HexoPlayer.A
        elif htt_game.winner == HTTPlayer.B:
            self.shadow.winner = HexoPlayer.B
        else:
            self.shadow.winner = HexoPlayer.NONE

    def get_move(self, htt_game) -> list[tuple[int, int]]:
        """Return moves in HexTicTacToe coordinates."""
        self.sync_from_htt_game(htt_game)

        moves_htt = []
        moves_to_make = self.shadow.moves_left_in_turn

        for _ in range(moves_to_make):
            if self.shadow.game_over:
                break

            tree = create_tree(self.shadow, self.model, self.device,
                               add_noise=False)
            for _ in range(self.n_sims):
                leaf = select_leaf(tree, self.shadow)
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
                    if leaf.needs_expansion:
                        maybe_expand_leaf(tree, leaf, pol[0].cpu())

            tq, tr = select_move(tree, temperature=0.0)
            q, r = torus_to_htt(tq, tr)
            moves_htt.append((q, r))
            self.shadow.make_move(tq, tr)

        return moves_htt


def play_cross_game(mcts_bot, minimax_bot, mcts_is_a=True,
                    max_moves=200) -> dict:
    """Play one game between MCTS bot and minimax bot."""
    game = HexGame(win_length=6)

    move_count = 0
    while not game.game_over and move_count < max_moves:
        player = game.current_player
        is_a = (player == HTTPlayer.A)

        if (is_a and mcts_is_a) or (not is_a and not mcts_is_a):
            bot = mcts_bot
        else:
            bot = minimax_bot

        moves = bot.get_move(game)
        for q, r in moves:
            if not game.game_over:
                ok = game.make_move(q, r)
                if ok:
                    move_count += 1
                else:
                    logger.warning(f"Invalid move ({q},{r}) by {type(bot).__name__}")
                    # Force end the game to avoid infinite loop
                    move_count = max_moves
                    break

    winner_str = "MCTS" if (
        (game.winner == HTTPlayer.A and mcts_is_a) or
        (game.winner == HTTPlayer.B and not mcts_is_a)
    ) else "Minimax" if game.winner != HTTPlayer.NONE else "Draw"

    return {
        'winner': winner_str,
        'winner_raw': game.winner,
        'moves': move_count,
        'game_over': game.game_over,
    }


def run_cross_eval(model_path: str, n_games: int = 10, n_sims: int = 200,
                   minimax_time: float = 0.5, device=None):
    """Run cross-evaluation between trained MCTS model and minimax bot."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HexONet().to(device)
    from hexo.training.checkpoint import load_checkpoint
    load_checkpoint(model_path, model, device=device)
    model.eval()

    mcts_bot = MCTSBotAdapter(model, device, n_sims=n_sims)
    minimax_bot = MinimaxBot(
        time_limit=minimax_time,
        pattern_path='/workspace/HexTicTacToe/learned_eval/results_baseline_8k/pattern_values.json',
    )

    results = {'mcts_wins': 0, 'minimax_wins': 0, 'draws': 0}

    for i in range(n_games):
        mcts_is_a = (i % 2 == 0)
        mcts_bot.reset()
        result = play_cross_game(mcts_bot, minimax_bot, mcts_is_a=mcts_is_a)
        logger.info(f"Game {i+1}/{n_games}: MCTS={'A' if mcts_is_a else 'B'} "
                     f"winner={result['winner']} moves={result['moves']}")

        if result['winner'] == 'MCTS':
            results['mcts_wins'] += 1
        elif result['winner'] == 'Minimax':
            results['minimax_wins'] += 1
        else:
            results['draws'] += 1

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--minimax-time", type=float, default=0.5)
    args = parser.parse_args()

    results = run_cross_eval(args.model, args.games, args.sims,
                             args.minimax_time)
    print(f"\nResults: MCTS {results['mcts_wins']} - "
          f"{results['minimax_wins']} Minimax "
          f"({results['draws']} draws)")
