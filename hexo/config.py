"""Configuration dataclasses for HeXO training."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    in_channels: int = 3
    num_blocks: int = 8
    num_filters: int = 64
    gn_groups: int = 8
    v_channels: int = 32


@dataclass
class MCTSConfig:
    puct_c: float = 2.0
    n_sims: int = 200
    dirichlet_alpha: float = 0.15
    dirichlet_frac: float = 0.25
    non_root_top_k: int = 64
    expand_visits: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_round: int = 2
    value_weight: float = 1.0
    policy_weight: float = 1.0
    entropy_weight: float = 0.01
    use_amp: bool = True


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    self_play_batch_size: int = 256
    games_per_round: int = 256
    cold_start_games: int = 512
    max_data_rounds: int = 20
    late_temperature: float = 0.3
    early_temp_turns: int = 20
    draw_penalty: float = 0.1
    max_game_moves: int = 150
    data_dir: str = "data/selfplay"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
