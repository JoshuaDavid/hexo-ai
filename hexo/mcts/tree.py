"""MCTS data structures for flat single-move tree search."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hexo.game.constants import BOARD_SIZE, N_CELLS, Player


class MCTSNode:
    """MCTS node with parallel-list children for fast PUCT.

    Each node represents one stone placement. The game state's
    moves_left_in_turn handles the 1-2-2-1-1 turn order naturally.
    """
    __slots__ = ('visit_count', 'n', 'actions', 'priors', 'visits',
                 'values', 'terminals', 'term_vals', 'action_map',
                 '_has_terminal')

    def __init__(self):
        self.visit_count: int = 0
        self.n: int = 0
        self.actions: list[int] | None = None     # [K] cell indices
        self.priors: list[float] | None = None    # [K] normalized priors
        self.visits: list[int] | None = None      # [K] visit counts
        self.values: list[float] | None = None    # [K] total value
        self.terminals: list[bool] | None = None  # [K] is terminal?
        self.term_vals: list[float] | None = None # [K] terminal values
        self.action_map: dict[int, int] | None = None  # action -> local idx
        self._has_terminal: bool = False


class PosNode:
    """An expanded position in the MCTS tree."""
    __slots__ = ('move_node', 'children', 'player', 'value')

    def __init__(self):
        self.move_node: MCTSNode = MCTSNode()
        self.children: dict[int, PosNode] | None = None  # action -> child
        self.player: Player | None = None
        self.value: float = 0.0


def init_node_children(node: MCTSNode, actions_priors: list[tuple[int, float]]):
    """Initialize list-based children from (action, prior) pairs."""
    n = len(actions_priors)
    node.n = n
    node.actions = [a for a, _ in actions_priors]
    priors = [p for _, p in actions_priors]
    total = sum(priors)
    if total > 0:
        node.priors = [p / total for p in priors]
    else:
        u = 1.0 / n if n > 0 else 0.0
        node.priors = [u] * n
    node.visits = [0] * n
    node.values = [0.0] * n
    node.terminals = [False] * n
    node.term_vals = [0.0] * n
    node.action_map = {a: i for i, a in enumerate(node.actions)}


@dataclass
class LeafInfo:
    """Info returned by select_leaf for batched NN eval."""
    path: list[tuple[MCTSNode, int]] = field(default_factory=list)
    # Player who selected each action in path (for backprop sign)
    path_players: list[Player] = field(default_factory=list)
    current_player: Player | None = None
    is_terminal: bool = False
    terminal_value: float = 0.0
    # The player who made the last move (for terminal value attribution)
    terminal_mover: Player | None = None
    # Delta from root: cells placed as (q, r, channel)
    # channel: 0 = root player's stones, 1 = opponent's
    deltas: list[tuple[int, int, int]] = field(default_factory=list)
    player_flipped: bool = False
    needs_expansion: bool = False
    expand_parent: PosNode | None = None
    expand_action: int = -1


@dataclass
class MCTSTree:
    root_pos: PosNode
    root_planes: torch.Tensor | None = None   # [3, BOARD_SIZE, BOARD_SIZE]
    root_player: Player | None = None
    root_value: float = 0.0
    root_occupied: frozenset | None = None


def cell_to_idx(q: int, r: int) -> int:
    return q * BOARD_SIZE + r


def idx_to_cell(idx: int) -> tuple[int, int]:
    return idx // BOARD_SIZE, idx % BOARD_SIZE
