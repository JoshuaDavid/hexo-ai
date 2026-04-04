"""Full training pipeline: generate -> train -> evaluate -> repeat."""

import logging
import os
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from hexo.config import PipelineConfig
from hexo.model.resnet import HexONet
from hexo.training.self_play import SelfPlayManager
from hexo.training.dataset import SelfPlayDataset
from hexo.training.trainer import Trainer
from hexo.training.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config: PipelineConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        mc = config.model
        self.model = HexONet(
            in_channels=mc.in_channels,
            num_blocks=mc.num_blocks,
            num_filters=mc.num_filters,
            gn_groups=mc.gn_groups,
            v_channels=mc.v_channels,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {n_params:,} parameters on {self.device}")

        # Create trainer
        tc = config.train
        self.trainer = Trainer(
            self.model, lr=tc.lr, weight_decay=tc.weight_decay,
            use_amp=tc.use_amp, policy_weight=tc.policy_weight,
            value_weight=tc.value_weight, entropy_weight=tc.entropy_weight,
        )

        # Ring buffer of training data
        self.data_rounds: list[list[dict]] = []

        # Create dirs
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def run(self, num_rounds: int, start_round: int = 0):
        """Main training loop."""
        for round_id in range(start_round, start_round + num_rounds):
            logger.info(f"=== Round {round_id} ===")
            t0 = time.monotonic()

            # 1. Generate self-play data
            self.model.eval()
            sp = SelfPlayManager(
                self.model, self.device,
                batch_size=self.config.self_play_batch_size,
                n_sims=self.config.mcts.n_sims,
                late_temperature=self.config.late_temperature,
                early_temp_turns=self.config.early_temp_turns,
                draw_penalty=self.config.draw_penalty,
            )
            target = self.config.cold_start_games if round_id == 0 \
                else self.config.games_per_round
            examples, draw_rate = sp.generate(round_id, target_games=target)
            logger.info(f"Generated {len(examples)} examples "
                        f"({target} games, draw_rate={draw_rate:.2f})")

            # 2. Add to ring buffer
            self.data_rounds.append(examples)
            if len(self.data_rounds) > self.config.max_data_rounds:
                self.data_rounds.pop(0)

            # 3. Build dataset from all rounds
            all_planes = []
            all_visits = []
            all_values = []
            all_round_ids = []
            for ex_list in self.data_rounds:
                for ex in ex_list:
                    all_planes.append(ex["planes"])
                    all_visits.append(ex["visit_counts"])
                    all_values.append(ex["value_target"])
                    all_round_ids.append(ex["round_id"])

            dataset = SelfPlayDataset(
                all_planes, all_visits, all_values, all_round_ids,
                current_round=round_id,
            )

            weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights, num_samples=len(dataset), replacement=True)
            dataloader = DataLoader(
                dataset, batch_size=self.config.train.batch_size,
                sampler=sampler, num_workers=0, pin_memory=True,
            )

            # 4. Train
            for epoch in range(self.config.train.epochs_per_round):
                metrics = self.trainer.train_epoch(dataloader)
                logger.info(
                    f"  Epoch {epoch}: "
                    f"loss={metrics['total_loss']:.4f} "
                    f"policy={metrics['policy_loss']:.4f} "
                    f"value={metrics['value_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f}")

            # 5. Save checkpoint
            ckpt_path = os.path.join(
                self.config.checkpoint_dir, f"round_{round_id:04d}.pt")
            save_checkpoint(
                self.model, self.trainer.optimizer, None,
                self.trainer.scaler, round_id, ckpt_path)

            elapsed = time.monotonic() - t0
            logger.info(f"Round {round_id} complete in {elapsed:.1f}s "
                        f"({len(all_planes)} total examples)")
