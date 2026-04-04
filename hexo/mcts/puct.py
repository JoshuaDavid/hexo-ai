"""PUCT selection for MCTS nodes."""

import math

from hexo.mcts.tree import MCTSNode

PUCT_C = 2.0


def puct_select(node: MCTSNode, c: float = PUCT_C) -> int:
    """Select child with highest PUCT score. Returns action index."""
    c_sqrt = c * math.sqrt(node.visit_count)
    best = -1e30
    best_a = -1
    actions = node.actions
    priors = node.priors
    visits = node.visits
    values = node.values

    if node._has_terminal:
        terminals = node.terminals
        term_vals = node.term_vals
        for i in range(node.n):
            vc = visits[i]
            if terminals[i]:
                q = term_vals[i]
            elif vc > 0:
                q = values[i] / vc
            else:
                q = 0.0
            s = q + c_sqrt * priors[i] / (1 + vc)
            if s > best:
                best = s
                best_a = actions[i]
    else:
        for i in range(node.n):
            vc = visits[i]
            q = values[i] / vc if vc > 0 else 0.0
            s = q + c_sqrt * priors[i] / (1 + vc)
            if s > best:
                best = s
                best_a = actions[i]
    return best_a
