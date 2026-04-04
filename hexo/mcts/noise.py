"""Exploration noise for MCTS."""

import numpy as np

from hexo.mcts.tree import MCTSNode

DIRICHLET_ALPHA = 0.15  # ~10 / sqrt(N_legal) for ~1000 candidates
DIRICHLET_FRAC = 0.25


def add_exploration_noise(node: MCTSNode, alpha: float = DIRICHLET_ALPHA,
                          frac: float = DIRICHLET_FRAC):
    """Add Dirichlet noise to priors (standard AlphaZero)."""
    if node.actions is None or node.n == 0:
        return
    n = node.n
    dirichlet = np.random.dirichlet([alpha] * n)
    keep = 1.0 - frac
    priors = node.priors
    node.priors = [keep * priors[i] + frac * float(dirichlet[i])
                   for i in range(n)]
