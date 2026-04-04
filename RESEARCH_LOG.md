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
