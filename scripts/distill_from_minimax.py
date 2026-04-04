#!/usr/bin/env python3
"""Generate training data by having the minimax bot play against itself.

The NN learns to imitate the minimax bot's move choices (policy) and
game outcomes (value). This bootstraps the NN to minimax-level play
much faster than pure self-play.
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, "/workspace/HexTicTacToe")

import torch
import numpy as np

from game import HexGame, Player as HTTPlayer
from ai import MinimaxBot

from hexo.game.constants import BOARD_SIZE, N_CELLS, Player as HexoPlayer
from hexo.encoding.planes import board_to_planes_from_dict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CENTER = BOARD_SIZE // 2
PATTERN_PATH = "/workspace/HexTicTacToe/learned_eval/results_baseline_8k/pattern_values.json"


def htt_to_torus(q, r):
    return (q + CENTER) % BOARD_SIZE, (r + CENTER) % BOARD_SIZE


def generate_distillation_data(n_games=200, time_limit=0.5):
    """Generate training examples from minimax self-play.

    Returns list of dicts with 'planes', 'visit_counts', 'value_target',
    matching the SelfPlayDataset format.
    """
    bot_a = MinimaxBot(time_limit=time_limit, pattern_path=PATTERN_PATH)
    bot_b = MinimaxBot(time_limit=time_limit, pattern_path=PATTERN_PATH)

    all_examples = []
    wins_a = wins_b = draws = 0

    for game_idx in range(n_games):
        game = HexGame(win_length=6)
        examples = []

        while not game.game_over and game.move_count < 150:
            cp = game.current_player
            bot = bot_a if cp == HTTPlayer.A else bot_b

            # Record pre-move state
            torus_board = {}
            for (q, r), player in game.board.items():
                tq, tr = htt_to_torus(q, r)
                if player == HTTPlayer.A:
                    torus_board[(tq, tr)] = HexoPlayer.A
                else:
                    torus_board[(tq, tr)] = HexoPlayer.B

            cp_hexo = HexoPlayer.A if cp == HTTPlayer.A else HexoPlayer.B
            planes = board_to_planes_from_dict(
                torus_board, cp_hexo, game.moves_left_in_turn)

            # Get minimax moves
            moves = bot.get_move(game)

            # Create visit_counts from the move (one-hot)
            visit_counts = {}
            for q, r in moves:
                tq, tr = htt_to_torus(q, r)
                idx = tq * BOARD_SIZE + tr
                visit_counts[idx] = 1.0 / len(moves)

            examples.append({
                "planes": planes,
                "visit_counts": visit_counts,
                "current_player": int(cp_hexo),
                "value_target": 0.0,  # backfilled
                "move_count": game.move_count,
                "moves_left": 0,  # backfilled
                "game_drawn": False,
                "game_id": game_idx,
                "round_id": 0,
            })

            # Apply moves
            for q, r in moves:
                if not game.game_over:
                    game.make_move(q, r)

        # Backfill values
        winner = game.winner
        total_moves = game.move_count
        is_drawn = (winner == HTTPlayer.NONE)
        for ex in examples:
            ex["moves_left"] = total_moves - ex["move_count"]
            ex["game_drawn"] = is_drawn
            if is_drawn:
                ex["value_target"] = 0.0
            else:
                cp = HexoPlayer(ex["current_player"])
                winner_hexo = HexoPlayer.A if winner == HTTPlayer.A else HexoPlayer.B
                ex["value_target"] = 1.0 if cp == winner_hexo else -1.0

        all_examples.extend(examples)

        if winner == HTTPlayer.A:
            wins_a += 1
        elif winner == HTTPlayer.B:
            wins_b += 1
        else:
            draws += 1

        if (game_idx + 1) % 10 == 0:
            logger.info(f"Game {game_idx+1}/{n_games}: "
                        f"A={wins_a} B={wins_b} draws={draws} "
                        f"examples={len(all_examples)}")

    logger.info(f"Done: {n_games} games, {len(all_examples)} examples, "
                f"A={wins_a} B={wins_b} draws={draws}")
    return all_examples


def train_on_distillation(examples, num_epochs=20, lr=1e-3, batch_size=128,
                          model_config=None):
    """Train a fresh model on distillation data."""
    from hexo.model.resnet import HexONet
    from hexo.training.dataset import SelfPlayDataset
    from hexo.training.trainer import Trainer
    from hexo.training.checkpoint import save_checkpoint
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_config:
        model = HexONet(**model_config).to(device)
    else:
        model = HexONet().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params on {device}")

    trainer = Trainer(model, lr=lr, use_amp=(device.type == 'cuda'))

    planes = [ex["planes"] for ex in examples]
    visits = [ex["visit_counts"] for ex in examples]
    values = [ex["value_target"] for ex in examples]
    round_ids = [0] * len(examples)

    ds = SelfPlayDataset(planes, visits, values, round_ids,
                         current_round=0, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=True)

    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dl)
        logger.info(f"Epoch {epoch}: loss={metrics['total_loss']:.4f} "
                     f"policy={metrics['policy_loss']:.4f} "
                     f"value={metrics['value_loss']:.4f}")

    os.makedirs("checkpoints_distill", exist_ok=True)
    save_checkpoint(model, trainer.optimizer, None, trainer.scaler,
                    0, "checkpoints_distill/distilled.pt")
    logger.info("Saved distilled model")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    examples = generate_distillation_data(args.games, args.time_limit)
    model = train_on_distillation(examples, num_epochs=args.epochs, lr=args.lr)
