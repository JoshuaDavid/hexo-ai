"""Batched self-play game generation for MCTS training.

Runs batch_size games in lockstep: all slots search simultaneously,
batch NN evals on GPU, then advance games together.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import tqdm

from hexo.game.constants import BOARD_SIZE, N_CELLS, Player
from hexo.game.state import GameState
from hexo.encoding.planes import board_to_planes
from hexo.mcts.search import (
    create_trees_batched, select_leaf, expand_and_backprop,
    maybe_expand_leaf, get_visit_counts, select_move,
)

logger = logging.getLogger(__name__)

MAX_GAME_MOVES = 150
_CENTER = BOARD_SIZE // 2


@dataclass
class SelfPlaySlot:
    game: GameState
    tree: object = None  # MCTSTree
    turn_number: int = 0
    game_id: int = 0
    examples: list[dict] = field(default_factory=list)


class SelfPlayManager:
    def __init__(self, model, device, batch_size=256, n_sims=200,
                 late_temperature=0.3, early_temp_turns=20,
                 draw_penalty=0.1, use_fp16=True):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.n_sims = n_sims
        self.late_temperature = late_temperature
        self.early_temp_turns = early_temp_turns
        self.draw_penalty = draw_penalty
        self.use_fp16 = use_fp16 and device.type == 'cuda'
        self.model_dtype = torch.float16 if self.use_fp16 else torch.float32

    def generate(self, round_id: int, target_games: int = 256
                 ) -> tuple[list[dict], float]:
        """Generate target_games completed games. Returns (examples, draw_rate)."""
        model = self.model
        device = self.device
        n_sims = self.n_sims

        all_examples: list[dict] = []
        games_completed = 0
        wins_a = wins_b = draws = 0
        next_game_id = 0

        # Create initial slots
        slots = []
        for _ in range(self.batch_size):
            slots.append(self._new_slot(next_game_id))
            next_game_id += 1

        # Timing
        t_tree = t_search = t_move = 0.0
        n_turns = 0

        pbar = tqdm(total=target_games, desc="Games", unit="game")

        while games_completed < target_games:
            # Phase 1: Create trees for slots that need them
            needs_tree = [i for i, s in enumerate(slots) if s.tree is None]
            if needs_tree:
                t0 = time.monotonic()
                active = [i for i in needs_tree
                          if not slots[i].game.game_over
                          and slots[i].game.move_count < MAX_GAME_MOVES]
                if active:
                    games = [slots[i].game for i in active]
                    trees = create_trees_batched(games, model, device)
                    for i, tree in zip(active, trees):
                        slots[i].tree = tree
                t_tree += time.monotonic() - t0

            # Phase 2: Run n_sims with batched NN eval
            t0 = time.monotonic()
            active_slots = [s for s in slots if s.tree is not None]

            for _sim in range(n_sims):
                # Select leaves for all active slots
                leaves = []
                for slot in active_slots:
                    leaf = select_leaf(slot.tree, slot.game)
                    leaves.append(leaf)

                # Collect leaves needing NN eval
                eval_indices = []
                eval_planes = []
                for i, (slot, leaf) in enumerate(zip(active_slots, leaves)):
                    if leaf.is_terminal or not leaf.deltas:
                        continue
                    # Build leaf planes from root + deltas
                    planes = slot.tree.root_planes.clone()
                    if leaf.player_flipped:
                        planes[[0, 1]] = planes[[1, 0]]
                    for gq, gr, ch in leaf.deltas:
                        actual_ch = (1 - ch) if leaf.player_flipped else ch
                        planes[actual_ch, gq, gr] = 1.0
                    # moves_left channel: we need the leaf game state
                    # Since select_leaf undoes moves, we reconstruct from deltas
                    # For now, use a reasonable default
                    planes[2] = 0.5  # approximate
                    eval_indices.append(i)
                    eval_planes.append(planes)

                # Batched forward pass
                if eval_planes:
                    batch = torch.stack(eval_planes).to(
                        device=device, dtype=self.model_dtype)
                    with torch.no_grad(), torch.amp.autocast(
                            'cuda', enabled=self.use_fp16):
                        pol_batch, val_batch = model(batch)
                    pol_cpu = pol_batch.float().cpu()
                    val_cpu = val_batch.float().cpu()

                # Backprop all leaves
                eval_map = {}
                for j, idx in enumerate(eval_indices):
                    eval_map[idx] = j

                for i, (slot, leaf) in enumerate(zip(active_slots, leaves)):
                    if leaf.is_terminal:
                        expand_and_backprop(slot.tree, leaf, 0.0)
                    elif i in eval_map:
                        j = eval_map[i]
                        nn_value = val_cpu[j].item()
                        expand_and_backprop(slot.tree, leaf, nn_value)
                        if leaf.needs_expansion:
                            maybe_expand_leaf(slot.tree, leaf, pol_cpu[j])
                    else:
                        expand_and_backprop(slot.tree, leaf, 0.0)

            t_search += time.monotonic() - t0

            n_turns += 1

            # Phase 3: Pick moves, record examples, advance games
            t0 = time.monotonic()
            for slot in slots:
                if slot.tree is None:
                    continue

                temperature = 1.0 if slot.turn_number < self.early_temp_turns \
                    else self.late_temperature

                q, r = select_move(slot.tree, temperature=temperature)

                # Record training example
                visit_counts = get_visit_counts(slot.tree)
                total_visits = sum(visit_counts.values())
                visits_normalized = {k: v / total_visits for k, v in visit_counts.items()} \
                    if total_visits > 0 else {}

                example = {
                    "planes": board_to_planes(slot.game),
                    "visit_counts": visits_normalized,
                    "current_player": int(slot.game.current_player),
                    "value_target": 0.0,  # backfilled
                    "move_count": slot.game.move_count,
                    "moves_left": 0,  # backfilled
                    "game_drawn": False,  # backfilled
                    "game_id": slot.game_id,
                    "round_id": round_id,
                }
                slot.examples.append(example)

                slot.game.make_move(q, r)
                slot.turn_number += 1
                slot.tree = None

            t_move += time.monotonic() - t0

            # Phase 4: Check for finished games
            for i, slot in enumerate(slots):
                done = slot.game.game_over or slot.game.move_count >= MAX_GAME_MOVES

                if done:
                    winner = slot.game.winner if slot.game.game_over else Player.NONE
                    is_drawn = (winner == Player.NONE)
                    total_moves = slot.game.move_count

                    for ex in slot.examples:
                        ex["moves_left"] = total_moves - ex["move_count"]
                        ex["game_drawn"] = is_drawn
                        if is_drawn:
                            ex["value_target"] = -self.draw_penalty
                        else:
                            cp = Player(ex["current_player"])
                            ex["value_target"] = 1.0 if cp == winner else -1.0

                    all_examples.extend(slot.examples)

                    if winner == Player.A:
                        wins_a += 1
                    elif winner == Player.B:
                        wins_b += 1
                    else:
                        draws += 1
                    games_completed += 1
                    pbar.update(1)

                    slots[i] = self._new_slot(next_game_id)
                    next_game_id += 1

        pbar.close()

        # Log timing
        t_total = t_tree + t_search + t_move
        if t_total > 0:
            logger.info(
                f"Timing ({n_turns} turns, {t_total:.1f}s): "
                f"tree {t_tree/t_total*100:.0f}% "
                f"search {t_search/t_total*100:.0f}% "
                f"move {t_move/t_total*100:.0f}%"
            )

        total_games = wins_a + wins_b + draws
        draw_rate = draws / max(total_games, 1)
        logger.info(f"Games: A={wins_a} B={wins_b} draws={draws} "
                     f"draw_rate={draw_rate:.2f}")
        return all_examples, draw_rate

    def _new_slot(self, game_id: int) -> SelfPlaySlot:
        game = GameState()
        game.make_move(_CENTER, _CENTER)  # First move at center
        return SelfPlaySlot(game=game, game_id=game_id)
