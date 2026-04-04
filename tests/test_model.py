"""Tests for HexONet neural network."""

import torch
from hexo.game.constants import BOARD_SIZE, N_CELLS
from hexo.model.resnet import HexONet


class TestHexONet:
    def test_output_shapes(self):
        model = HexONet()
        x = torch.randn(4, 3, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(x)
        assert policy.shape == (4, N_CELLS)
        assert value.shape == (4,)

    def test_value_range(self):
        model = HexONet()
        x = torch.randn(8, 3, BOARD_SIZE, BOARD_SIZE)
        _, value = model(x)
        assert (value >= -1.0).all() and (value <= 1.0).all()

    def test_occupied_masking(self):
        model = HexONet()
        x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE)
        mask = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.bool)
        mask[0, 5, 5] = True
        mask[0, 10, 10] = True
        policy, _ = model(x, occupied_mask=mask)
        assert policy[0, 5 * BOARD_SIZE + 5] == float("-inf")
        assert policy[0, 10 * BOARD_SIZE + 10] == float("-inf")
        # Unmasked cells should be finite
        assert torch.isfinite(policy[0, 0])
        # Second batch item should have no masked cells
        assert torch.isfinite(policy[1]).all()

    def test_param_count(self):
        model = HexONet(num_blocks=8, num_filters=64)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be in the ~1-5M range
        assert 500_000 < n_params < 10_000_000, f"Unexpected param count: {n_params:,}"
        print(f"Parameter count: {n_params:,}")

    def test_batch_size_1(self):
        model = HexONet()
        x = torch.randn(1, 3, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(x)
        assert policy.shape == (1, N_CELLS)
        assert value.shape == (1,)

    def test_gradient_flows(self):
        model = HexONet(num_blocks=2, num_filters=16, gn_groups=4, v_channels=8)
        x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE)
        policy, value = model(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"

    def test_single_sample_overfit(self):
        """Model should be able to overfit a single sample."""
        model = HexONet(num_blocks=2, num_filters=16, gn_groups=4, v_channels=8)
        x = torch.randn(1, 3, BOARD_SIZE, BOARD_SIZE)
        target_policy = torch.zeros(1, N_CELLS)
        target_policy[0, 500] = 1.0  # target: move at cell 500
        target_value = torch.tensor([0.8])

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(200):
            policy, value = model(x)
            policy_loss = torch.nn.functional.cross_entropy(policy, target_policy)
            value_loss = torch.nn.functional.mse_loss(value, target_value)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        policy, value = model(x)
        assert policy.argmax(dim=-1).item() == 500
        assert abs(value.item() - 0.8) < 0.2
