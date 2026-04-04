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

### Next steps
- Build arena evaluation (model vs model, model vs random)
- Run longer training (20+ rounds)
- Profile GPU utilization during training
- Consider scaling model (currently 613K params — very small)
