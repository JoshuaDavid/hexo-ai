"""Tests for training pipeline components."""

import torch
from hexo.game.constants import BOARD_SIZE, N_CELLS
from hexo.model.resnet import HexONet
from hexo.training.trainer import Trainer, compute_loss
from hexo.training.dataset import SelfPlayDataset


class TestComputeLoss:
    def test_loss_finite(self):
        policy_logits = torch.randn(4, N_CELLS)
        value_pred = torch.tanh(torch.randn(4))
        visit_targets = torch.zeros(4, N_CELLS)
        visit_targets[:, 500] = 1.0  # all examples target cell 500
        value_targets = torch.tensor([1.0, -1.0, 1.0, -1.0])

        loss, metrics = compute_loss(policy_logits, value_pred,
                                     visit_targets, value_targets)
        assert torch.isfinite(loss)
        assert all(v > 0 or k == 'entropy' for k, v in metrics.items()
                   if k != 'total_loss')

    def test_perfect_prediction_low_loss(self):
        # Create "perfect" predictions
        policy_logits = torch.zeros(2, N_CELLS)
        policy_logits[:, 500] = 100.0  # very confident on cell 500
        value_pred = torch.tensor([1.0, -1.0])
        visit_targets = torch.zeros(2, N_CELLS)
        visit_targets[:, 500] = 1.0
        value_targets = torch.tensor([1.0, -1.0])

        loss, metrics = compute_loss(policy_logits, value_pred,
                                     visit_targets, value_targets)
        assert metrics['policy_loss'] < 0.1
        assert metrics['value_loss'] < 0.01


class TestDataset:
    def test_basic_dataset(self):
        planes = [torch.randn(3, BOARD_SIZE, BOARD_SIZE) for _ in range(10)]
        visits = [{100: 0.5, 200: 0.3, 300: 0.2} for _ in range(10)]
        values = [1.0, -1.0] * 5
        round_ids = [0] * 10

        ds = SelfPlayDataset(planes, visits, values, round_ids,
                             current_round=0, augment=False)
        assert len(ds) == 10

        p, v, val = ds[0]
        assert p.shape == (3, BOARD_SIZE, BOARD_SIZE)
        assert v.shape == (N_CELLS,)
        assert abs(v.sum().item() - 1.0) < 1e-5

    def test_augmentation_changes_planes(self):
        planes = [torch.zeros(3, BOARD_SIZE, BOARD_SIZE) for _ in range(1)]
        planes[0][0, 5, 10] = 1.0  # single stone
        visits = [{5 * BOARD_SIZE + 10: 1.0}]
        values = [1.0]
        round_ids = [0]

        ds = SelfPlayDataset(planes, visits, values, round_ids,
                             current_round=0, augment=True)
        # With augmentation, the stone should sometimes be at a different position
        positions = set()
        for _ in range(50):
            p, _, _ = ds[0]
            pos = tuple(p[0].nonzero().squeeze().tolist())
            positions.add(pos)
        # Should have multiple different positions due to D6 symmetry
        assert len(positions) > 1


class TestTrainer:
    def test_loss_decreases_overfit(self):
        """Training on a single batch should decrease loss."""
        model = HexONet(num_blocks=2, num_filters=16, gn_groups=4, v_channels=8)
        device = torch.device('cpu')
        model = model.to(device)

        trainer = Trainer(model, lr=1e-3, use_amp=False)

        # Create valid planes: mostly empty board with a few stones
        planes_list = []
        for _ in range(8):
            p = torch.zeros(3, BOARD_SIZE, BOARD_SIZE)
            p[0, 16, 16] = 1.0  # one friendly stone
            p[1, 15, 15] = 1.0  # one opponent stone
            p[2] = 1.0  # moves_left = 2
            planes_list.append(p)

        # Target: cell 500 is not occupied
        visits = [{500: 1.0} for _ in range(8)]
        values = [1.0] * 4 + [-1.0] * 4
        round_ids = [0] * 8
        ds = SelfPlayDataset(planes_list, visits, values, round_ids,
                             current_round=0, augment=False)
        dl = torch.utils.data.DataLoader(ds, batch_size=8)

        # Train for a few epochs
        losses = []
        for _ in range(10):
            metrics = trainer.train_epoch(dl)
            losses.append(metrics['total_loss'])

        # Loss should decrease
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
