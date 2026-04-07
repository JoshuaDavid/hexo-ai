# Artifacts

Model checkpoints, game logs, and training summaries from the HeXO AI project.

## Model Checkpoints

All models use the architecture defined in `hexo/model/resnet.py` (`HexONet`).

### `small_model_r19.pt`
- **Architecture**: 8 blocks, 64 filters, GroupNorm(8) — 613K params
- **Training**: 20 rounds of self-play, 50 MCTS sims, batch_size=32
- **Duration**: ~6 minutes on RTX 5080
- **Eval**: Beats random 20/20 (avg 42 moves). Loses to minimax 0-6.
- **Load with**: `HexONet(num_blocks=8, num_filters=64, gn_groups=8)`

### `big_model_r399.pt`
- **Architecture**: 12 blocks, 128 filters, GroupNorm(16) — 3.6M params
- **Training**: 200 rounds of self-play @ 500 sims, starting from a base model that had 100 prior rounds at progressive sims (50/200/400). Total effective training: ~400 rounds, 7.5 hours.
- **LR**: 1e-4, ring buffer 40 rounds
- **Eval**: Beats random 10/10 (avg 13 moves). Beats earlier self (R215) 10/10. Loses to HexTicTacToe minimax 0-6 at 0.05s.
- **Load with**: `HexONet(num_blocks=12, num_filters=128, gn_groups=16)`

### `endgame_best_iter07.pt`
- **Architecture**: 12 blocks, 128 filters, GroupNorm(16) — 3.6M params
- **Training**: Started from big model R100, then 7 iterations of endgame curriculum training. Each iteration: record 8 games vs minimax, extract ~120 endgame positions (stepping back 20 positions from each minimax win), mix with fresh self-play data, train 15 epochs.
- **Total endgame examples**: ~696 (from iterations 0-7)
- **Eval**: Created positions that caused the minimax to crash in some games. In clean games, loses 0-10.
- **Load with**: `HexONet(num_blocks=12, num_filters=128, gn_groups=16)`

## Game Logs

### `mcts_vs_minimax_games.json`
4 recorded games of the big model (R100, 200 sims) vs HexTicTacToe minimax (0.1s). All minimax wins. Each game includes full board history with move-by-move snapshots. Used as seed data for endgame curriculum training.

### `minimax_games_iter0.json`
8 recorded games from endgame curriculum iteration 0. Same format. All minimax wins (avg 40 moves).

## Training Summaries

### `train_100r_summary.log`
Extracted round headers, epoch losses, and game stats from the 100-round small model training run (50 sims). Shows policy loss progression from 6.39 to 2.00.

### `train_big_200r_summary.log`
Extracted from the 200-round big model long training run (500 sims). Policy loss from 2.96 to 2.57.

### `endgame_v2_summary.log`
Extracted iteration summaries, endgame example counts, and cross-eval results from the endgame curriculum v2 run (10 iterations). Shows the oscillation pattern: best at iter 7 (0-3 with 3 "draws" that were minimax crashes).

## How to Load a Checkpoint

```python
from hexo.model.resnet import HexONet
from hexo.training.checkpoint import load_checkpoint
import torch

device = torch.device('cuda')

# Small model
model = HexONet(num_blocks=8, num_filters=64, gn_groups=8).to(device)
load_checkpoint('artifacts/small_model_r19.pt', model, device=device)

# Big model
model = HexONet(num_blocks=12, num_filters=128, gn_groups=16).to(device)
load_checkpoint('artifacts/big_model_r399.pt', model, device=device)
```
