"""Training loop: loss computation, optimization, AMP."""

import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from hexo.game.constants import N_CELLS
from hexo.training.dataset import SelfPlayDataset

logger = logging.getLogger(__name__)


def compute_loss(policy_logits, value_pred, visit_targets, value_targets,
                 policy_weight=1.0, value_weight=1.0, entropy_weight=0.01):
    """Compute combined loss.

    Args:
        policy_logits: [B, N_CELLS]
        value_pred: [B]
        visit_targets: [B, N_CELLS] (probability distribution)
        value_targets: [B]

    Returns:
        total_loss, {component losses}
    """
    # Policy loss: KL divergence between visit distribution and policy
    log_policy = F.log_softmax(policy_logits, dim=-1)
    # Mask out -inf entries to avoid 0 * -inf = nan
    log_policy = log_policy.clamp(min=-100.0)
    policy_loss = -(visit_targets * log_policy).sum(dim=-1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred, value_targets)

    # Entropy bonus (encourages exploration)
    policy_probs = F.softmax(policy_logits, dim=-1)
    entropy = -(policy_probs * log_policy).sum(dim=-1).mean()

    total = (policy_weight * policy_loss +
             value_weight * value_loss -
             entropy_weight * entropy)

    return total, {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'total_loss': total.item(),
    }


class Trainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, use_amp=True,
                 policy_weight=1.0, value_weight=1.0, entropy_weight=0.01):
        self.model = model
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.amp.GradScaler('cuda') if use_amp and self.device.type == 'cuda' else None
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train one epoch. Returns average metrics."""
        self.model.train()
        total_metrics = {}
        n_batches = 0

        for planes, visits, values in dataloader:
            planes = planes.to(self.device)
            visits = visits.to(self.device)
            values = values.to(self.device)

            # Build occupied mask from planes
            occupied = (planes[:, 0] + planes[:, 1]) > 0.5  # [B, H, W]

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    policy, value = self.model(planes, occupied_mask=occupied)
                    loss, metrics = compute_loss(
                        policy, value, visits, values,
                        self.policy_weight, self.value_weight,
                        self.entropy_weight)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy, value = self.model(planes, occupied_mask=occupied)
                loss, metrics = compute_loss(
                    policy, value, visits, values,
                    self.policy_weight, self.value_weight,
                    self.entropy_weight)
                loss.backward()
                self.optimizer.step()

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            n_batches += 1

        return {k: v / n_batches for k, v in total_metrics.items()}

    def validate(self, dataloader: DataLoader) -> dict:
        """Validate. Returns average metrics."""
        self.model.eval()
        total_metrics = {}
        n_batches = 0

        with torch.no_grad():
            for planes, visits, values in dataloader:
                planes = planes.to(self.device)
                visits = visits.to(self.device)
                values = values.to(self.device)

                occupied = (planes[:, 0] + planes[:, 1]) > 0.5
                policy, value = self.model(planes, occupied_mask=occupied)
                _, metrics = compute_loss(
                    policy, value, visits, values,
                    self.policy_weight, self.value_weight,
                    self.entropy_weight)

                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
