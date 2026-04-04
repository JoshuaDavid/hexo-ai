#!/usr/bin/env python3
"""CLI entry point for HeXO training."""

import argparse
import logging
import sys
import time

import torch

from hexo.config import PipelineConfig, ModelConfig, MCTSConfig, TrainConfig
from hexo.training.pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Train HeXO AI agent")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sp-batch", type=int, default=64,
                        help="Self-play batch size")
    parser.add_argument("--games", type=int, default=64,
                        help="Games per round")
    parser.add_argument("--cold-start", type=int, default=128,
                        help="Games for cold start")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/train_{int(time.time())}.log"),
        ],
    )

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = PipelineConfig(
        model=ModelConfig(num_blocks=args.blocks, num_filters=args.filters),
        mcts=MCTSConfig(n_sims=args.sims),
        train=TrainConfig(batch_size=args.batch_size, lr=args.lr,
                          use_amp=not args.no_amp),
        self_play_batch_size=args.sp_batch,
        games_per_round=args.games,
        cold_start_games=args.cold_start,
    )

    pipeline = TrainingPipeline(config, device)
    pipeline.run(num_rounds=args.rounds)


if __name__ == "__main__":
    main()
