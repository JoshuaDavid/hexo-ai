"""Heuristic policy for bootstrapping before neural network is trained.

For each empty cell, for each of the 3 hex directions, examine the 11-cell
window centered on that cell (-5 to +5). For each of the 6 possible
6-in-a-row sub-windows containing the cell, compute:
  p(cell participates in 6-in-a-row for current player) -
  p(cell participates in 6-in-a-row for opponent)

Sum across all sub-windows and all directions.

The probability is approximated as: if a sub-window has 0 opponent stones
and c current-player stones among the 6 cells, the remaining (5-c) empty
cells all need to be filled by current player. We weight this by (c+1)^2
as a proxy for probability (more existing stones = more likely to complete).
"""

import numpy as np

from hexo.game.constants import BOARD_SIZE, WIN_LENGTH, HEX_DIRECTIONS, Player


def compute_heuristic_values(game) -> np.ndarray:
    """Compute heuristic value for each cell from current player's perspective.

    Returns [BOARD_SIZE, BOARD_SIZE] array. Higher = better for current player.
    Empty cells get heuristic scores, occupied cells get -inf.
    """
    N = BOARD_SIZE
    values = np.zeros((N, N), dtype=np.float32)
    cp = game.current_player
    opp = cp.opponent()
    board = game.board

    for q in range(N):
        for r in range(N):
            if (q, r) in board:
                values[q, r] = float('-inf')
                continue

            score = 0.0
            for dq, dr in HEX_DIRECTIONS:
                # Look at 11-cell window centered on (q, r): offsets -5 to +5
                # There are 6 sub-windows of length 6 containing (q, r):
                # starting at offsets -5, -4, -3, -2, -1, 0
                for start_offset in range(-WIN_LENGTH + 1, 1):
                    cp_count = 0
                    opp_count = 0
                    for i in range(WIN_LENGTH):
                        wq = (q + (start_offset + i) * dq) % N
                        wr = (r + (start_offset + i) * dr) % N
                        cell_player = board.get((wq, wr))
                        if cell_player == cp:
                            cp_count += 1
                        elif cell_player == opp:
                            opp_count += 1

                    # Score for current player: unblocked window value
                    if opp_count == 0:
                        score += (cp_count + 1) ** 2
                    # Score for opponent (negative)
                    if cp_count == 0:
                        score -= (opp_count + 1) ** 2

            values[q, r] = score

    return values


def heuristic_policy(game) -> np.ndarray:
    """Convert heuristic values to a probability distribution over cells.

    Returns [N_CELLS] array (flattened, softmax-normalized).
    """
    values = compute_heuristic_values(game)
    flat = values.reshape(-1)

    # Replace -inf with very negative (for softmax)
    mask = np.isfinite(flat)
    if not mask.any():
        # All cells occupied (shouldn't happen)
        return np.ones_like(flat) / len(flat)

    # Temperature-scaled softmax
    temperature = 1.0
    valid_max = flat[mask].max()
    flat_shifted = np.where(mask, flat - valid_max, -1e9)
    exp_vals = np.exp(flat_shifted / temperature)
    total = exp_vals.sum()
    if total > 0:
        return exp_vals / total
    return np.ones_like(flat) / len(flat)
