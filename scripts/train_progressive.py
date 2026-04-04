#!/usr/bin/env python3
"""Progressive training: start with low sims, scale up as model strengthens."""

import logging
import os
import sys
import time

import torch

from hexo.config import PipelineConfig, ModelConfig, MCTSConfig, TrainConfig
from hexo.training.pipeline import TrainingPipeline
from hexo.training.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def run_phase(pipeline, num_rounds, start_round, n_sims, games_per_round,
              cold_start_games=None):
    """Run a training phase with specific settings."""
    pipeline.config.mcts.n_sims = n_sims
    pipeline.config.games_per_round = games_per_round
    if cold_start_games is not None:
        pipeline.config.cold_start_games = cold_start_games

    pipeline.run(num_rounds=num_rounds, start_round=start_round)
    return start_round + num_rounds


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = PipelineConfig(
        model=ModelConfig(num_blocks=8, num_filters=64),
        mcts=MCTSConfig(n_sims=50),
        train=TrainConfig(batch_size=128, lr=1e-3, epochs_per_round=2),
        self_play_batch_size=32,
        games_per_round=64,
        cold_start_games=64,
        checkpoint_dir="checkpoints_prog",
    )

    pipeline = TrainingPipeline(config, device)
    t0 = time.monotonic()
    round_id = 0

    # Phase 1: Bootstrap with low sims (fast, gets past random play)
    logger.info("=== PHASE 1: Bootstrap (50 sims, 30 rounds) ===")
    round_id = run_phase(pipeline, num_rounds=30, start_round=round_id,
                         n_sims=50, games_per_round=64, cold_start_games=64)

    # Phase 2: Strengthen with medium sims
    logger.info("=== PHASE 2: Strengthen (100 sims, 30 rounds) ===")
    round_id = run_phase(pipeline, num_rounds=30, start_round=round_id,
                         n_sims=100, games_per_round=64)

    # Phase 3: Polish with high sims
    logger.info("=== PHASE 3: Polish (200 sims, 20 rounds) ===")
    round_id = run_phase(pipeline, num_rounds=20, start_round=round_id,
                         n_sims=200, games_per_round=32)

    total = time.monotonic() - t0
    logger.info(f"DONE: {total:.0f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
