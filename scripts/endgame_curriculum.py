#!/usr/bin/env python3
"""Endgame curriculum training: scale up retrograde analysis.

Generates 500+ endgame examples from MCTS-vs-minimax games,
mixes with self-play data, and trains iteratively.

Strategy:
1. Record many games of MCTS vs minimax (both sides)
2. For each minimax win, step back up to 20 positions
3. At each position, record:
   - Winner's turn: MCTS visit counts as policy target, value=+1
   - Loser's turn: minimax's move as policy target, value=-1
4. Mix these with self-play data (50/50) to prevent forgetting
5. Train with low LR, iterate
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, "/workspace/HexTicTacToe")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from game import HexGame, Player as HTTPlayer
from ai import MinimaxBot

from hexo.game.constants import BOARD_SIZE, N_CELLS, Player as HexoPlayer
from hexo.game.state import GameState
from hexo.encoding.planes import board_to_planes, board_to_planes_from_dict
from hexo.model.resnet import HexONet
from hexo.mcts.search import (
    create_tree, select_leaf, expand_and_backprop, maybe_expand_leaf,
    select_move, get_visit_counts,
)
from hexo.training.dataset import SelfPlayDataset
from hexo.training.trainer import Trainer, compute_loss
from hexo.training.checkpoint import save_checkpoint, load_checkpoint
from hexo.training.self_play import SelfPlayManager
from hexo.eval.cross_eval import MCTSBotAdapter, run_cross_eval

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CENTER = BOARD_SIZE // 2
PATTERN_PATH = "/workspace/HexTicTacToe/learned_eval/results_baseline_8k/pattern_values.json"


def htt_to_torus(q, r):
    return (q + CENTER) % BOARD_SIZE, (r + CENTER) % BOARD_SIZE


def record_games(model, device, n_games=20, mcts_sims=200, mm_time=0.05):
    """Record MCTS vs minimax games with full history."""
    import signal

    model.eval()
    mcts = MCTSBotAdapter(model, device, n_sims=mcts_sims)

    games_log = []
    t0 = time.monotonic()

    for gi in range(n_games):
        mcts_is_a = (gi % 2 == 0)
        mcts.reset()
        game = HexGame()
        history = []
        # Fresh minimax per game to avoid memory buildup
        mm = MinimaxBot(time_limit=mm_time, pattern_path=PATTERN_PATH)
        game_broken = False

        while not game.game_over and game.move_count < 80 and not game_broken:
            cp = game.current_player
            board_snap = {f"{q},{r}": p.value for (q, r), p in game.board.items()}

            if (cp == HTTPlayer.A and mcts_is_a) or (cp == HTTPlayer.B and not mcts_is_a):
                mcts.sync_from_htt_game(game)
                moves = mcts.get_move(game)
                who = 'mcts'
            else:
                # Timeout protection for minimax
                move_t0 = time.monotonic()
                moves = mm.get_move(game)
                move_elapsed = time.monotonic() - move_t0
                who = 'minimax'
                mm._tt.clear()

                # Validate minimax moves
                if move_elapsed > 5.0:
                    logger.warning(f'Minimax took {move_elapsed:.1f}s, skipping game')
                    game_broken = True
                    break
                for mq, mr in moves:
                    if (mq, mr) in game.board:
                        logger.warning(f'Minimax returned occupied cell ({mq},{mr}), skipping game')
                        game_broken = True
                        break

            if game_broken:
                break

            history.append({
                'board': board_snap,
                'cp': cp.value,
                'ml': game.moves_left_in_turn,
                'who': who,
                'moves': [[q, r] for q, r in moves],
            })

            for q, r in moves:
                if not game.game_over:
                    ok = game.make_move(q, r)
                    if not ok:
                        game_broken = True
                        break

        if game_broken:
            logger.info(f'Game {gi+1}/{n_games}: BROKEN at move {game.move_count}')
            continue

        winner = game.winner.value if game.winner != HTTPlayer.NONE else 0
        winner_type = 'minimax' if (
            (winner == 1 and not mcts_is_a) or (winner == 2 and mcts_is_a)
        ) else 'mcts' if winner != 0 else 'draw'

        games_log.append({
            'history': history,
            'winner': winner,
            'winner_type': winner_type,
            'mcts_is_a': mcts_is_a,
            'move_count': game.move_count,
        })

        elapsed = time.monotonic() - t0
        logger.info(f'Game {gi+1}/{n_games}: mcts={"A" if mcts_is_a else "B"} '
                     f'winner={winner_type} moves={game.move_count} '
                     f'({elapsed:.0f}s elapsed)')

    return games_log


def extract_endgame_examples(games_log, model, device, n_sims=200,
                              max_steps_back=20):
    """Extract endgame training examples from recorded games.

    For each minimax win, step back up to max_steps_back positions.
    Returns list of training example dicts.
    """
    model.eval()
    examples = []

    for game_data in games_log:
        if game_data['winner_type'] != 'minimax':
            continue

        history = game_data['history']
        winner_val = game_data['winner']

        n_positions = min(max_steps_back, len(history))

        for step_back in range(n_positions):
            pos_idx = len(history) - 1 - step_back
            entry = history[pos_idx]

            # Reconstruct torus board
            board_raw = entry['board']
            torus_board = {}
            for key, player_val in board_raw.items():
                q, r = (int(x) for x in key.split(','))
                tq, tr = htt_to_torus(q, r)
                torus_board[(tq, tr)] = HexoPlayer(player_val)

            cp_hexo = HexoPlayer(entry['cp'])
            ml = entry['ml']
            planes = board_to_planes_from_dict(torus_board, cp_hexo, ml)

            cp_is_winner = (cp_hexo.value == winner_val)

            if cp_is_winner:
                # Winner's turn: use MCTS to find the winning continuation
                g = GameState()
                g.board = dict(torus_board)
                g.current_player = cp_hexo
                g.moves_left_in_turn = ml
                g.move_count = len(torus_board)

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
                        if leaf.needs_expansion:
                            maybe_expand_leaf(tree, leaf, pol[0].cpu())

                visits = get_visit_counts(tree)
                total = sum(visits.values())
                if total > 0:
                    visit_norm = {k: v / total for k, v in visits.items()}
                    examples.append({
                        "planes": planes,
                        "visit_counts": visit_norm,
                        "value_target": 1.0,
                        "round_id": 0,
                    })
            else:
                # Loser's turn: use minimax's actual move as target
                moves = entry['moves']
                visit_counts = {}
                for mq, mr in moves:
                    tq, tr = htt_to_torus(mq, mr)
                    idx = tq * BOARD_SIZE + tr
                    visit_counts[idx] = 1.0 / max(len(moves), 1)

                examples.append({
                    "planes": planes,
                    "visit_counts": visit_counts,
                    "value_target": -1.0,
                    "round_id": 0,
                })

    return examples


def generate_selfplay_examples(model, device, n_games=32, n_sims=100):
    """Generate fresh self-play examples to mix with endgame data."""
    model.eval()
    sp = SelfPlayManager(model, device, batch_size=16, n_sims=n_sims,
                         use_fp16=True)
    examples, draw_rate = sp.generate(round_id=0, target_games=n_games)
    logger.info(f'Self-play: {len(examples)} examples, draw_rate={draw_rate:.2f}')
    return examples


def train_mixed(model, endgame_examples, selfplay_examples, device,
                epochs=10, lr=2e-4):
    """Train on mixed endgame + self-play data."""
    trainer = Trainer(model, lr=lr, use_amp=(device.type == 'cuda'),
                      value_weight=2.0, entropy_weight=0.005)

    # Build datasets
    eg_planes = [ex["planes"] for ex in endgame_examples]
    eg_visits = [ex["visit_counts"] for ex in endgame_examples]
    eg_values = [ex["value_target"] for ex in endgame_examples]
    eg_rids = [0] * len(endgame_examples)

    sp_planes = [ex["planes"] for ex in selfplay_examples]
    sp_visits = [ex["visit_counts"] for ex in selfplay_examples]
    sp_values = [ex["value_target"] for ex in selfplay_examples]
    sp_rids = [0] * len(selfplay_examples)

    # Combine
    all_planes = eg_planes + sp_planes
    all_visits = eg_visits + sp_visits
    all_values = eg_values + sp_values
    all_rids = eg_rids + sp_rids

    ds = SelfPlayDataset(all_planes, all_visits, all_values, all_rids,
                         current_round=0, augment=True)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                    pin_memory=True)

    logger.info(f'Training on {len(endgame_examples)} endgame + '
                f'{len(selfplay_examples)} self-play = {len(all_planes)} total')

    for epoch in range(epochs):
        metrics = trainer.train_epoch(dl)
        if epoch % 3 == 0 or epoch == epochs - 1:
            logger.info(f'  Epoch {epoch}: loss={metrics["total_loss"]:.4f} '
                        f'policy={metrics["policy_loss"]:.4f} '
                        f'value={metrics["value_loss"]:.4f}')

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best model
    model_kwargs = {'num_blocks': 12, 'num_filters': 128, 'gn_groups': 16}
    model = HexONet(**model_kwargs).to(device)
    load_checkpoint('/workspace/hexo-ai/checkpoints_big_long/round_0100.pt',
                    model, device=device)
    logger.info(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} params)")

    os.makedirs('/workspace/hexo-ai/checkpoints_endgame_v2', exist_ok=True)
    os.makedirs('/workspace/hexo-ai/data', exist_ok=True)

    all_endgame_examples = []
    all_selfplay_examples = []

    for iteration in range(10):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")

        # 1. Record games against minimax
        t0 = time.monotonic()
        games_log = record_games(model, device, n_games=8,
                                  mcts_sims=200, mm_time=0.05)
        t_record = time.monotonic() - t0

        minimax_wins = sum(1 for g in games_log if g['winner_type'] == 'minimax')
        mcts_wins = sum(1 for g in games_log if g['winner_type'] == 'mcts')
        draws = sum(1 for g in games_log if g['winner_type'] == 'draw')
        logger.info(f'Recording: {t_record:.0f}s, MCTS={mcts_wins} MM={minimax_wins} D={draws}')

        # Save game log
        with open(f'/workspace/hexo-ai/data/games_iter_{iteration}.json', 'w') as f:
            json.dump(games_log, f)

        # 2. Extract endgame examples
        t0 = time.monotonic()
        new_endgame = extract_endgame_examples(
            games_log, model, device, n_sims=200, max_steps_back=20)
        t_extract = time.monotonic() - t0
        all_endgame_examples.extend(new_endgame)
        logger.info(f'Endgame: +{len(new_endgame)} examples '
                    f'(total={len(all_endgame_examples)}) in {t_extract:.0f}s')

        # 3. Generate self-play data (to prevent forgetting)
        t0 = time.monotonic()
        new_selfplay = generate_selfplay_examples(
            model, device, n_games=16, n_sims=50)
        t_sp = time.monotonic() - t0
        all_selfplay_examples = new_selfplay  # Fresh each iteration
        logger.info(f'Self-play: {len(new_selfplay)} examples in {t_sp:.0f}s')

        # 4. Train on mixed data
        model.train()
        n_epochs = 15 if iteration < 3 else 10
        lr = 2e-4 if iteration < 5 else 1e-4
        model = train_mixed(model, all_endgame_examples, all_selfplay_examples,
                            device, epochs=n_epochs, lr=lr)

        # 5. Save checkpoint
        ckpt_path = f'/workspace/hexo-ai/checkpoints_endgame_v2/iter_{iteration:02d}.pt'
        save_checkpoint(model, None, None, None, iteration, ckpt_path)
        logger.info(f'Saved {ckpt_path}')

        # 6. Quick eval every 2 iterations
        if iteration % 2 == 1:
            model.eval()
            results = run_cross_eval(
                model_path=ckpt_path, n_games=6, n_sims=500,
                minimax_time=0.1, model_kwargs=model_kwargs)
            logger.info(f'EVAL: MCTS {results["mcts_wins"]} - '
                        f'{results["minimax_wins"]} MM '
                        f'({results["draws"]} draws)')

    logger.info("\n=== DONE ===")
    logger.info(f'Total endgame examples: {len(all_endgame_examples)}')


if __name__ == "__main__":
    main()
