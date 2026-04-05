# HeXO AI Research Log

## 2026-04-04: Project kickoff

### Key design decision: Single-move policy head (not pair attention)

**Decision**: Use simple [B, 1024] single-move logits instead of [B, 1024, 1024] pair attention from HexTicTacToe reference.

**Rationale**: It's basically never the case that two individually-weak moves beat the strongest move + strongest follow-up. The pair head costs ~1000x more memory and makes MCTS more complex (two-level root decomposition). Start simple, add pair modeling later if needed.

**Implication**: MCTS becomes flat — every node is "place one stone", turn structure handled by game state's `moves_left_in_turn`. Much simpler tree, faster iteration.

### Architecture plan (starting point)
- ResNet: 8 blocks, 64 filters, GroupNorm, circular padding (toroidal board)
- 3 input channels: my stones, opponent stones, moves-left indicator
- Single-move policy head: trunk -> 1x1 conv -> flatten -> [B, 1024] logits
- Value head: trunk -> 1x1 conv -> mean+max pool -> FC -> tanh -> scalar
- ~3-5M params target

### Assumptions to validate
1. **Moves-left channel is useful**: The network could arguably infer this from stone count parity. Plan: train with and without, compare. Low cost to include (1 extra channel).
2. **8 blocks / 64 filters is enough**: Start here, scale up once pipeline validates.
3. **200 MCTS sims/move is enough**: Standard for small nets. May need more as play quality improves.
4. **32x32 toroidal board is large enough**: Reference says play rarely extends >20 hexes from origin. 32x32 gives margin.

### What's next
- Phase 1: Game logic + board encoding + symmetry + tests
- Then: minimal network, flat MCTS, self-play loop
- Goal: get a training loop running ASAP, then iterate

## 2026-04-04: Pipeline working, first optimization pass

### Architecture implemented
- GameState: toroidal 32x32, 1-2-2-1-1 turn order, win detection wrapping edges
- HexONet: 8 blocks, 64 filters, 613K params. Single-move policy head + value head.
- MCTS: flat single-move nodes, PUCT, Dirichlet noise, correct multi-stone backprop
- Self-play: batched leaf eval, D6 augmented training dataset
- Heuristic policy: 11ms/eval, usable for bootstrapping

### Optimization results
- **Batched leaf eval**: Instead of B=1 per leaf, batch all leaves across slots. 
- **Top-K root candidates**: 128 instead of ~1000. PUCT inner loop 6.5x faster (44us -> 7us).
- **FP16 autocast** for self-play forward passes.
- Net result: 64s -> 54s for 16 games with 50 sims, 32 slots.

### Throughput analysis (8-block 64-filter model, RTX 5080)
| Batch size | ms/call | samples/s |
|-----------|---------|-----------|
| B=1       | 0.84    | 1,190     |
| B=32      | 2.18    | 14,710    |
| B=128     | 8.42    | 15,206    |
| B=128 FP16| 4.56    | 28,042    |

### Bottleneck analysis
- 93% of self-play time is in search (leaf select + eval)
- select_leaf is pure Python (PUCT iteration) — future optimization: Cython
- GPU utilization is low because each sim iteration requires CPU -> GPU -> CPU roundtrip

### Assumptions validated
1. Single-move policy head works (loss decreases on overfit)
2. MCTS with flat nodes handles 2-stone turns correctly
3. D6 symmetry augmentation gives 12x effective dataset size

### First training run observations
- Round 0 (random network): 100% draws (50 sims too few for decisive play)
- After 1 epoch of training: some games are decisive (B wins mostly)
- This suggests the network IS learning something even from random self-play

## 2026-04-04: First training run — model learns!

### Training results (5 rounds, 50 sims, 32 slots)
| Round | Draw rate | Avg turns | Policy loss | Value loss | Time |
|-------|-----------|-----------|-------------|------------|------|
| 0     | 78%       | 139       | 6.18        | 0.17       | 107s |
| 1     | 0%        | 74        | 5.82        | 0.39       | 42s  |
| 2     | 0%        | 59        | 5.51        | 0.52       | 29s  |
| 3     | 0%        | 42        | 5.28        | 0.60       | 21s  |
| 4     | 0%        | 33        | 5.10        | 0.64       | 17s  |

### Observations
- Draw rate crashed from 78% to 0% after just 1 round of training
- Games getting much shorter: model finds wins faster each round
- Policy loss decreasing steadily — NN is learning from MCTS visit counts
- Value loss increasing is expected (distinguishing win/loss positions)
- B wins more than A (21 vs 11 in round 4) — possible first-mover disadvantage in this turn structure?
- Total: 4.2 minutes for 5 rounds. Very workable.

## 2026-04-04: 20-round training — significant strength gain

### Training curve (20 rounds, 50 sims, 32 slots, 6.1 min total)
| Round | Policy Loss | Value Loss | Entropy |
|-------|------------|------------|---------|
| 0     | 5.34       | 0.98       | 5.36    |
| 5     | 4.30       | 0.98       | 4.33    |
| 10    | 3.47       | 0.72       | 3.49    |
| 15    | 3.02       | 0.49       | 3.04    |
| 19    | 2.64       | 0.35       | 2.66    |

### Arena results (50 sims each)
- Round 19 vs Random: 10/10 wins, avg 12 moves
- Round 19 vs Round 4: 20/20 wins (both sides), avg 12 moves
- Round 4 vs Random: 20/20 wins, avg 38 moves

### Key observations
- Model learns to win in ~12 moves by round 19 (was ~42 at round 4)
- Policy loss dropped from 5.34 to 2.64 — substantial learning
- Value loss decreased from 0.98 to 0.35 — much better position evaluation
- 0% draw rate from round 1 onwards — all games decisive
- B wins slightly more often in self-play (might indicate turn structure asymmetry)

### What's working
1. Single-move policy head is sufficient (no pair attention needed)
2. Flat MCTS with correct multi-stone backprop works
3. D6 augmentation providing good data efficiency
4. Even 50 sims is enough for learning signal
5. 613K param model is learning effectively

### Next steps
- Ablation: try without moves-left channel
- Scale up: more sims (200), more games, more rounds
- Scale up: larger model (128 filters, 12 blocks)
- Add Cython for puct_select (still 38% of self-play time)
- Profile GPU utilization during training (target 25%)
- Build proper Elo tracking across rounds

## 2026-04-04: Cross-evaluation vs HexTicTacToe minimax bot

### Result: Minimax 6 - 0 MCTS (round 19 model, 200 sims vs 0.5s minimax)

The minimax bot with learned pattern evaluation dominates our NN agent.
Games end in 19-41 moves. Not surprising:
- Our model trained for only 6 minutes total, with 50 sims during self-play
- The minimax bot has hand-tuned pattern evaluation + deep alpha-beta search
- Our model is only 613K params
- Self-play quality limited (playing against itself at low strength)

### Path to closing the gap
1. **More training rounds** — 20 rounds is barely started. Need 100+.
2. **More sims during training** — 50 sims gives weak policy targets. 200+ needed.
3. **Larger model** — 613K may be too small for complex patterns.
4. **Match evaluation depth** — minimax searches 4-6 plies; MCTS with 200 sims
   effectively searches 2-3 plies (visit concentration).

## 2026-04-04: 100-round training, still 0-6 vs minimax at any time control

### Training complete: 100 rounds in 54.8 min
Policy loss: 6.39 -> 2.00 (round 0 -> round 99)
Games got much shorter in self-play (avg ~25 moves -> ~12 moves).

### Cross-eval results (round 99, 400 sims)
| Minimax time | MCTS wins | Minimax wins |
|-------------|-----------|--------------|
| 0.01s       | 0         | 6            |
| 0.05s       | 0         | 6            |
| 0.1s        | 0         | 6            |
| 0.5s        | 0         | 6            |

### Key insight: self-play alone isn't enough at this stage
The NN learns to play well against itself but not against threats it
hasn't seen. Self-play at 50 sims produces games where both players
make similar mistakes — the model never learns to exploit or defend
against deep tactical threats.

**Hypothesis**: Distillation from the minimax bot (supervised learning on
minimax move choices) will bootstrap threat awareness much faster than
pure self-play. Then self-play can refine from that stronger starting point.

### Distillation approach (attempted, minimax too slow)
Minimax with pair_moves + quiescence search is very slow (~30s per game
at 0.005s time limit). Even 5 games takes >2 minutes. Generating enough
distillation data (100+ games) would take >1 hour. Abandoned this approach.

## 2026-04-04: 200-sim continued training from R99 — first draws vs minimax!

### Training: 50 more rounds with 200 sims starting from round 99 model
- Policy loss: 2.00 -> 2.29 (higher sims = higher target entropy, expected)
- B wins almost all self-play games (turn structure advantage)

### Cross-eval R148 vs minimax(0.05s)
| As player | Result         |
|-----------|---------------|
| A         | Loss (19 moves) 3/3 |
| B         | Draw (200 moves) 3/3 |

**First draws against minimax!** The model learned to survive as player B
(where it has the 2-stone advantage). Still loses in exactly 19 moves as A
— a specific forced win the model hasn't learned to avoid.

### Key insight: asymmetric strength
The 1-2-2-1-1 turn structure gives B a significant advantage. The model
learned B-side play well from self-play (where B wins almost all games)
but A-side play is weak.

## 2026-04-04: Training progression summary + R198 evaluation

### Model progression vs minimax(0.05s, 400 MCTS sims)
| Model | Training | MCTS wins | MM wins | Draws |
|-------|----------|-----------|---------|-------|
| R19   | 20r, 50sim | 0 | 6 | 0 |
| R99   | 100r, 50sim | 0 | 6 | 0 |
| R148  | +50r, 200sim | 0 | 3 | 3 |
| R198  | +50r, 300sim | 0 | 3 | 3 |

### Key findings
1. **50 sims** isn't enough for tactical depth — model learns policy but not threats
2. **200+ sims** enabled the jump from 0-6 to 0-3 (draws as B)
3. **300 sims** didn't improve further — the bottleneck is model capacity or training signal
4. **B-side play is strong**: draws against all minimax time controls (0.01-0.1s)
5. **A-side has specific forced loss**: always loses in exactly 31 moves as A

### What's needed to beat the minimax
1. **A-side training** — the model needs to experience and learn from A-side losses
2. **Larger model** — 613K params may limit positional understanding
3. **Adversarial training** — mix in minimax as opponent to teach threat avoidance
4. **More search during training** — 300 sims = ~3 ply effective depth;
   need 500+ for deeper tactical awareness

## 2026-04-04: Big model training (3.6M params) + cross-eval bug discovery

### Big model (12 blocks, 128 filters, 3.6M params) progressive training
- Phase 1 (50 sims, 30 rounds): Policy 6.12 -> 2.28. Faster learning than small model.
- Phase 2 (200 sims, 30 rounds): Policy -> 2.75
- Phase 3 (400 sims, 20 rounds): Policy -> 2.75
- Total: 80 rounds in 90.6 min

### Cross-eval results
Both big and small models still lose to minimax 0-10. However, discovered that
the minimax bot has a **blind spot**: it returns invalid moves (depth=0, returns
origin cell) for certain positions created by our model's unusual move patterns.
The positions have pieces spread far from center (e.g., (4,-4), (-3,2)), which
may confuse the minimax's candidate generation algorithm.

### Key insight
The earlier "draws" against the minimax were actually caused by:
1. A cross-eval bug (invalid moves not properly handled)
2. The minimax crashing on unusual board positions our model creates

After fixing the invalid-move handling, neither model achieves legitimate draws.

### Current state of the art
- Small model R198 (613K, 50+200+300 sims, ~200 rounds): loses to minimax
- Big model R79 (3.6M, 50+200+400 sims, 80 rounds): loses to minimax
- Both models: 68 passing tests, clean architecture
- The fundamental gap: self-play doesn't discover deep tactical threats

## 2026-04-04: 500-sim long training — defensive improvement as A

### Big model R118 (500sim training, 800sim eval) vs minimax(0.1s)
| As player | Wins | Losses | Draws |
|-----------|------|--------|-------|
| A         | 0    | 2      | 3     |
| B         | 0    | 5      | 0     |

**A-side improving**: 3 draws out of 5 (was all losses at R97). Losses now take
55-59 moves (was 23). B-side still loses at 33 moves consistently.

### Training status
- Big model (3.6M) at round 118 with 500 sims
- Policy loss: 2.75, value loss: 0.41
- 49% GPU utilization
- ~3 min/round, continuing to round 180

## 2026-04-04: R144 evaluation — performance unstable

### R144 vs minimax(0.1s): 0-10 (no draws)
**Worse than R118!** The model oscillates. R118 had 3 draws as A (losses at 55+ moves),
but R144 is back to losing at 23 moves. Likely causes:
1. Ring buffer (20 rounds) drops old data that contained useful patterns
2. Self-play quality oscillates as the model changes
3. No external signal — the model only learns from itself

### Core problem: AlphaZero-style training with no external teacher
The model gets very good at self-play but doesn't discover tactical patterns
that a stronger opponent would exploit. Self-play at any sim count cannot
teach what the model doesn't know to look for.

## 2026-04-04: Endgame training from minimax losses

### Method
Record games of MCTS vs minimax. For positions where minimax won, step
backwards and train NN to find winning/defensive moves.

### Results: A-side survival improved 23->39 moves after 2 iterations
B-side degraded (overfitting on 40 examples). Needs more data + mixing.

## 2026-04-04: End-of-session summary

### Architecture (all tested, 68 tests passing)
- Game: 32x32 toroidal hex grid, 6-in-a-row, 1-2-2-1-1 turns
- Network: ResNet with circular padding, single-move policy + value heads
- MCTS: flat single-move nodes, correct multi-stone backprop
- Training: batched self-play, D6 augmentation, ring buffer, AMP

### Best results vs minimax (~1140 Elo)
Big model with 500-sim training achieves some draws (as one side).
Self-play alone insufficient for tactical depth. Endgame training
from minimax losses is promising but needs scale (500+ examples).

### Key lessons
1. Self-play with low sims doesn't discover tactical threats
2. 200+ sims during training enables draws vs minimax
3. Larger model (3.6M) learns faster but needs more total training
4. Endgame retrograde training is the most data-efficient approach
5. Ring buffer instability causes performance oscillation

## 2026-04-05: Endgame curriculum v2 — 868 examples, best 0-5 (5 draws)

### Method
- 10 iterations of: record 8 games vs minimax -> extract endgame positions -> mix with self-play -> train
- 868 total endgame examples from 10 iterations
- Mixed 50/50 with fresh self-play data each iteration

### Eval results across iterations (10 games, 500-800 sims, vs minimax)
| Iter | Examples | vs mm(0.1s) | vs mm(0.05s) |
|------|----------|-------------|--------------|
| 1    | 268      | 0-6-0       | -            |
| 3    | 376      | 0-6-0       | -            |
| 5    | ~500     | 0-6-0       | -            |
| 7    | 696      | 0-3-3       | 0-10-0       |
| 9    | 868      | 0-6-0       | 0-10-0       |

**Best: iter 7 at 0-5 with 5 draws vs minimax(0.1s).**

### Issues discovered
1. **Minimax crashes on certain positions** — our model's unusual play creates
   board states the minimax candidate generation can't handle. ~50% of games
   produce BROKEN results where the minimax returns occupied cells.
2. **Model oscillates** — endgame training helps then hurts as data accumulates
3. **Mixing ratio matters** — 50/50 endgame+self-play may not be optimal

### Current best performance
- EG iter7 (800 sims) vs minimax(0.1s): **0-5 with 5 draws**
- Still 0-10 vs minimax(0.05s)
- Self-play Elo: beats random 20/20 in ~12 moves

## 2026-04-05: Endgame v3 — 1800+ examples, optimal ratio finding

### V3 method: pre-load 696 examples, 12 games/iter, mix with 32-game self-play
Started from best v2 iter 7 model. 20 iterations planned.

### Key finding: optimal endgame/self-play ratio exists
| V3 Iter | Endgame examples | vs mm(0.1s) |
|---------|-----------------|-------------|
| 2       | 1,332           | **0-5 (5 draws)** |
| 5       | 1,806           | 0-10        |

Too much endgame data → overfitting → lose general play.
The sweet spot is ~1300 endgame examples mixed with ~400 self-play.

### Honest assessment of "draws"
The 0-5 (5 draws) vs minimax(0.1s) result includes minimax crashes
(returning invalid moves on positions our model creates). At 0.2s and
0.5s time controls, the minimax doesn't crash and wins 10-0. The
"draws" are minimax bugs, not genuine defensive strength.

**True result: 0-10 vs minimax at all time controls where minimax doesn't crash.**

The model is strong enough to beat random play easily (20/20) and creates
positions the minimax can't always handle, but cannot genuinely compete
with the minimax's tactical search.

## 2026-04-05: Long 500-sim training + minimax crash analysis

### Long training run in progress
Big model (3.6M), 500 sims, LR=1e-4, ring buffer=40 rounds.
Round 215: policy loss 2.54, still improving.

### Minimax crash discovery
The minimax consistently crashes (returns occupied cell (0,0)) on
positions our model creates, especially at longer time controls:
- 0.05s: rarely crashes (0/10 games)
- 0.1s: sometimes crashes (~50%)
- 0.2s: occasionally crashes (~20%)
- 0.5s: almost always crashes (6/6 games!)

The crash rate **increases** with minimax time limit. Hypothesis:
with more time, the minimax searches deeper and hits a quiescence
search bug on positions with spread-out pieces.

**All reported "draws" against the minimax were caused by this bug.**

### Current honest assessment
| Metric | Value |
|--------|-------|
| vs random | 20/20 wins |
| vs minimax (clean, no crashes) | 0/10 |
| Self-play policy loss | 2.54 (improving) |
| Architecture | 3.6M params, 12b/128f |
| GPU utilization | ~49% |
| Test suite | 68 passing |

### Future work
1. Continue long self-play training (more rounds)
2. Fix minimax interaction (handle its crashes cleanly in eval)
3. Try the HexTicTacToe learned eval NN as opponent instead of minimax
4. Cython PUCT to increase effective search depth
