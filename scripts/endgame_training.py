#!/usr/bin/env python3
"""Retrograde endgame training from minimax losses.

Takes recorded games where minimax beat our model, and trains the NN
backwards from the endgame:
1. Take the position where minimax delivered the winning blow
2. Swap sides: have our NN play from minimax's winning position
3. Train on that until the NN consistently finds the win
4. Step back one move and repeat
5. This teaches the NN the tactical patterns it's missing

"We're not Google, we don't care about proving that it's possible to
learn from non-curated data when we have the ability to learn hundreds
of times more efficiently with curated data."
"""

import json
import logging
import os
import sys

sys.path.insert(0, "/workspace/HexTicTacToe")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from game import HexGame, Player as HTTPlayer
from ai import MinimaxBot

from hexo.game.constants import BOARD_SIZE, N_CELLS, Player as HexoPlayer
from hexo.game.state import GameState
from hexo.encoding.planes import board_to_planes_from_dict
from hexo.encoding.symmetry import apply_symmetry_planes, PERMS
from hexo.model.resnet import HexONet
from hexo.mcts.search import (
    create_tree, select_leaf, expand_and_backprop, maybe_expand_leaf,
    select_move, get_visit_counts,
)
from hexo.training.dataset import SelfPlayDataset
from hexo.training.trainer import Trainer
from hexo.training.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CENTER = BOARD_SIZE // 2
PATTERN_PATH = "/workspace/HexTicTacToe/learned_eval/results_baseline_8k/pattern_values.json"


def htt_to_torus(q, r):
    return (q + CENTER) % BOARD_SIZE, (r + CENTER) % BOARD_SIZE


def load_game_logs(path):
    """Load recorded games."""
    with open(path) as f:
        return json.load(f)


def reconstruct_game_state(history_entry):
    """Reconstruct a GameState from a history entry."""
    board_raw = history_entry['board']
    cp = history_entry['cp']
    ml = history_entry['ml']

    torus_board = {}
    for key, player_val in board_raw.items():
        q, r = (int(x) for x in key.split(','))
        tq, tr = htt_to_torus(q, r)
        torus_board[(tq, tr)] = HexoPlayer(player_val)

    g = GameState()
    g.board = torus_board
    g.current_player = HexoPlayer(cp)
    g.moves_left_in_turn = ml
    g.move_count = len(torus_board)
    return g


def generate_endgame_examples(games_log, model, device, n_sims=200):
    """Generate training examples from endgame positions.

    For each minimax win:
    - Work backwards from the last position
    - At each position where it was minimax's turn (the winner),
      have the NN play from that position and record visit counts
    - At positions where it was our bot's turn, this is where we
      failed to defend — record minimax's move as the target

    This creates examples that teach both:
    1. How to take a win when you have one (from winner's positions)
    2. How to defend when opponent has threats (from loser's positions)
    """
    examples = []

    mm = MinimaxBot(time_limit=0.1, pattern_path=PATTERN_PATH)

    for game_data in games_log:
        if game_data['winner_type'] != 'minimax':
            continue

        history = game_data['history']
        winner_val = game_data['winner']  # 1 or 2
        mcts_is_a = game_data['mcts_is_a']

        # Work backwards through the last N positions
        for step_back in range(min(10, len(history))):
            pos_idx = len(history) - 1 - step_back
            entry = history[pos_idx]

            g = reconstruct_game_state(entry)
            cp_hexo = g.current_player
            cp_is_winner = (cp_hexo.value == winner_val)

            # Generate planes
            planes = board_to_planes_from_dict(
                g.board, cp_hexo, g.moves_left_in_turn)

            if cp_is_winner:
                # Winner's turn: use MCTS to find the winning move
                tree = create_tree(g, model, device, add_noise=False)
                for _ in range(n_sims):
                    leaf = select_leaf(tree, g)
                    if leaf.is_terminal or not leaf.deltas:
                        expand_and_backprop(tree, leaf, 0.0)
                    else:
                        lp = tree.root_planes.clone()
                        if leaf.player_flipped:
                            lp[[0, 1]] = lp[[1, 0]]
                        for gq, gr, ch in leaf.deltas:
                            actual_ch = (1 - ch) if leaf.player_flipped else ch
                            lp[actual_ch, gq, gr] = 1.0
                        with torch.no_grad():
                            x = lp.unsqueeze(0).to(device)
                            pol, val = model(x)
                        expand_and_backprop(tree, leaf, val[0].item())

                visits = get_visit_counts(tree)
                total = sum(visits.values())
                if total > 0:
                    visit_norm = {k: v/total for k, v in visits.items()}
                    examples.append({
                        "planes": planes,
                        "visit_counts": visit_norm,
                        "value_target": 1.0,  # winner's perspective
                        "round_id": 0,
                    })
            else:
                # Loser's turn: use minimax's actual move as target
                moves = entry['moves']
                visit_counts = {}
                for q, r in moves:
                    tq, tr = htt_to_torus(q, r)
                    idx = tq * BOARD_SIZE + tr
                    visit_counts[idx] = 1.0 / len(moves)
                examples.append({
                    "planes": planes,
                    "visit_counts": visit_counts,
                    "value_target": -1.0,  # loser's perspective
                    "round_id": 0,
                })

    return examples


def train_on_endgames(model, examples, device, epochs=30, lr=5e-4):
    """Train model on endgame examples."""
    trainer = Trainer(model, lr=lr, use_amp=(device.type == 'cuda'),
                      value_weight=2.0)  # extra weight on value

    planes = [ex["planes"] for ex in examples]
    visits = [ex["visit_counts"] for ex in examples]
    values = [ex["value_target"] for ex in examples]
    round_ids = [0] * len(examples)

    ds = SelfPlayDataset(planes, visits, values, round_ids,
                         current_round=0, augment=True)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        metrics = trainer.train_epoch(dl)
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: loss={metrics['total_loss']:.4f} "
                         f"policy={metrics['policy_loss']:.4f} "
                         f"value={metrics['value_loss']:.4f}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-log", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", default="checkpoints_endgame/endgame.pt")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gn = 16 if args.filters >= 128 else 8
    model = HexONet(num_blocks=args.blocks, num_filters=args.filters,
                    gn_groups=gn).to(device)
    load_checkpoint(args.model_checkpoint, model, device=device)
    model.eval()

    logger.info(f"Loaded model from {args.model_checkpoint}")

    games_log = load_game_logs(args.games_log)
    logger.info(f"Loaded {len(games_log)} games")

    examples = generate_endgame_examples(games_log, model, device,
                                          n_sims=args.sims)
    logger.info(f"Generated {len(examples)} endgame examples")

    if examples:
        model.train()
        model = train_on_endgames(model, examples, device, epochs=args.epochs)

        os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
        save_checkpoint(model, None, None, None, 0, args.output_checkpoint)
        logger.info(f"Saved to {args.output_checkpoint}")
