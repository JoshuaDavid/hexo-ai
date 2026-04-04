"""Integration tests: end-to-end pipeline verification."""

import torch
from hexo.config import PipelineConfig, ModelConfig, MCTSConfig, TrainConfig
from hexo.training.pipeline import TrainingPipeline
from hexo.training.self_play import SelfPlayManager
from hexo.model.resnet import HexONet
from hexo.eval.arena import MCTSBot, RandomBot, play_match


class TestSelfPlaySmoke:
    def test_generates_examples(self):
        """Self-play generates valid training examples."""
        model = HexONet(num_blocks=2, num_filters=16, gn_groups=4, v_channels=8)
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()

        sp = SelfPlayManager(model, device, batch_size=4, n_sims=5,
                             use_fp16=False)
        examples, draw_rate = sp.generate(round_id=0, target_games=4)

        assert len(examples) > 0
        for ex in examples:
            assert "planes" in ex
            assert "visit_counts" in ex
            assert "value_target" in ex
            assert ex["planes"].shape == (3, 32, 32)
            assert -1.0 <= ex["value_target"] <= 1.0


class TestPipelineSmoke:
    def test_one_round(self):
        """A single training round runs without error."""
        config = PipelineConfig(
            model=ModelConfig(num_blocks=2, num_filters=16, gn_groups=4, v_channels=8),
            mcts=MCTSConfig(n_sims=5),
            train=TrainConfig(batch_size=8, use_amp=False, epochs_per_round=1),
            self_play_batch_size=4,
            games_per_round=4,
            cold_start_games=4,
            data_dir="/tmp/hexo_test_data",
            checkpoint_dir="/tmp/hexo_test_ckpt",
            log_dir="/tmp/hexo_test_logs",
        )
        pipeline = TrainingPipeline(config, torch.device('cpu'))

        # Save initial weights
        initial_params = {n: p.clone() for n, p in pipeline.model.named_parameters()}

        pipeline.run(num_rounds=1)

        # Parameters should have changed
        changed = False
        for n, p in pipeline.model.named_parameters():
            if not torch.equal(p, initial_params[n]):
                changed = True
                break
        assert changed, "Model parameters did not change after training"


class TestArenaSmoke:
    def test_bot_plays_legal_game(self):
        """MCTSBot plays a legal game against RandomBot."""
        model = HexONet(num_blocks=2, num_filters=16, gn_groups=4, v_channels=8)
        device = torch.device('cpu')
        model.eval()

        bot = MCTSBot(model, device, n_sims=5)
        rbot = RandomBot()
        result = play_match(bot, rbot, n_games=2, max_moves=50)
        assert result['n_games'] == 2
