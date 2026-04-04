"""MCTS search: select_leaf, expand, backprop, create_tree."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hexo.game.constants import BOARD_SIZE, N_CELLS, Player
from hexo.encoding.planes import board_to_planes
from hexo.mcts.tree import (
    MCTSNode, PosNode, MCTSTree, LeafInfo,
    init_node_children, cell_to_idx, idx_to_cell,
)
from hexo.mcts.puct import puct_select
from hexo.mcts.noise import add_exploration_noise

MAX_DEPTH = 100
EXPAND_VISITS = 1
NON_ROOT_TOP_K = 64  # candidates for non-root nodes

_ALL_CELLS = frozenset((q, r) for q in range(BOARD_SIZE) for r in range(BOARD_SIZE))


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _build_tree_from_eval(
    game,
    root_value: float,
    policy_logits: torch.Tensor,   # [N_CELLS]
    root_planes: torch.Tensor,
    add_noise: bool = True,
) -> MCTSTree:
    """Build an MCTSTree from pre-computed NN outputs."""
    pos = PosNode()
    pos.value = root_value
    pos.player = game.current_player if isinstance(game.current_player, Player) \
        else Player(game.current_player)

    occupied = game.get_occupied_set()
    if game.move_count > 0:
        cands = _ALL_CELLS - occupied
    else:
        cands = {(BOARD_SIZE // 2, BOARD_SIZE // 2)}

    # Build (action, prior) list from policy logits
    probs = F.softmax(policy_logits, dim=0)
    cand_indices = [cell_to_idx(q, r) for q, r in cands]
    cand_values = probs[cand_indices].tolist()
    cand_priors = sorted(zip(cand_indices, cand_values),
                         key=lambda x: x[1], reverse=True)

    init_node_children(pos.move_node, cand_priors)

    if add_noise:
        add_exploration_noise(pos.move_node)

    return MCTSTree(
        root_pos=pos,
        root_planes=root_planes,
        root_player=pos.player,
        root_value=root_value,
        root_occupied=frozenset(occupied),
    )


def create_tree(
    game,
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = True,
) -> MCTSTree:
    """Create a single MCTS tree with one B=1 NN forward pass."""
    planes = board_to_planes(game)
    x = planes.unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = model(x)

    return _build_tree_from_eval(
        game, value[0].item(), policy_logits[0].cpu(), planes, add_noise)


@torch.no_grad()
def create_trees_batched(
    games: list,
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = True,
) -> list[MCTSTree]:
    """Create trees for multiple games in one batched forward pass."""
    B = len(games)
    if B == 0:
        return []

    planes_list = [board_to_planes(g) for g in games]
    batch = torch.stack(planes_list).to(device)
    policy_logits, values = model(batch)

    trees = []
    for i, game in enumerate(games):
        tree = _build_tree_from_eval(
            game, values[i].item(), policy_logits[i].cpu(),
            planes_list[i], add_noise)
        trees.append(tree)
    return trees


# ---------------------------------------------------------------------------
# Select leaf
# ---------------------------------------------------------------------------

def select_leaf(tree: MCTSTree, game) -> LeafInfo:
    """Select a leaf via PUCT descent. Makes/undoes moves on game."""
    path: list[tuple[MCTSNode, int]] = []
    path_players: list[Player] = []
    states: list[tuple[int, int, object]] = []
    deltas: list[tuple[int, int, int]] = []
    pos = tree.root_pos
    root_cp = tree.root_player
    depth = 0

    while depth < MAX_DEPTH:
        action_idx = puct_select(pos.move_node)
        q, r = idx_to_cell(action_idx)

        prev_player = game.current_player
        path.append((pos.move_node, action_idx))
        path_players.append(prev_player)

        state = game.save_state()
        states.append((q, r, state))
        game.make_move(q, r)

        # Track delta for plane construction
        ch = 0 if prev_player == root_cp else 1
        deltas.append((q, r, ch))

        # Terminal?
        if game.game_over:
            local = pos.move_node.action_map[action_idx]
            pos.move_node.terminals[local] = True
            pos.move_node.term_vals[local] = 1.0
            pos.move_node._has_terminal = True
            _undo_all(game, states)
            return LeafInfo(path=path, path_players=path_players,
                            is_terminal=True, terminal_value=1.0,
                            terminal_mover=prev_player)

        # Check for child PosNode
        child = (pos.children or {}).get(action_idx)
        if child is not None:
            pos = child
            depth += 1
            continue

        # Leaf — check expansion threshold
        local = pos.move_node.action_map[action_idx]
        needs_exp = (pos.move_node.visits[local] + 1 >= EXPAND_VISITS)

        cp = game.current_player
        player_flipped = (cp != root_cp)
        _undo_all(game, states)
        return LeafInfo(
            path=path, path_players=path_players,
            current_player=cp, deltas=deltas,
            player_flipped=player_flipped,
            needs_expansion=needs_exp,
            expand_parent=pos, expand_action=action_idx,
        )

    # MAX_DEPTH reached
    cp = game.current_player
    _undo_all(game, states)
    return LeafInfo(path=path, path_players=path_players,
                    current_player=cp, deltas=deltas,
                    player_flipped=(cp != root_cp))


def _undo_all(game, states: list):
    """Undo all moves in reverse order."""
    for q, r, state in reversed(states):
        game.undo_move(q, r, state)


# ---------------------------------------------------------------------------
# Backprop
# ---------------------------------------------------------------------------

def expand_and_backprop(
    tree: MCTSTree,
    leaf: LeafInfo,
    nn_value: float,
):
    """Backpropagate value through the path.

    Value convention:
    - terminal: terminal_value (1.0 = the terminal_mover won)
    - non-terminal: nn_value is from current_player's perspective at leaf

    Each path entry records which player selected that action. We assign
    value to each entry from that player's perspective. In a 2-stone turn,
    consecutive entries may be by the same player — no sign flip between them.
    """
    if not leaf.path:
        return

    if leaf.is_terminal:
        # terminal_mover won with value 1.0
        ref_player = leaf.terminal_mover
        ref_value = leaf.terminal_value
    else:
        # nn_value is from leaf's current_player perspective
        ref_player = leaf.current_player
        ref_value = nn_value

    for (node, action_idx), selector in zip(leaf.path, leaf.path_players):
        # Value from selector's perspective
        v = ref_value if selector == ref_player else -ref_value
        local = node.action_map[action_idx]
        node.visits[local] += 1
        node.values[local] += v
        node.visit_count += 1


# ---------------------------------------------------------------------------
# Child expansion
# ---------------------------------------------------------------------------

def maybe_expand_leaf(
    tree: MCTSTree,
    leaf: LeafInfo,
    policy_logits: torch.Tensor,  # [N_CELLS]
):
    """Create a child PosNode at the leaf if expansion conditions met."""
    if not leaf.needs_expansion or leaf.is_terminal:
        return
    if leaf.expand_parent is None:
        return

    parent = leaf.expand_parent
    action = leaf.expand_action

    if parent.children is not None and action in parent.children:
        return

    # Get occupied cells at leaf position
    occupied_idx = {cell_to_idx(q, r) for q, r in tree.root_occupied}
    for q, r, _ch in leaf.deltas:
        occupied_idx.add(cell_to_idx(q, r))

    # Top-K candidates from policy, excluding occupied
    probs = F.softmax(policy_logits, dim=0)
    actions_priors = []
    top_indices = policy_logits.argsort(descending=True)
    for idx in top_indices.tolist():
        if idx not in occupied_idx:
            actions_priors.append((idx, probs[idx].item()))
            if len(actions_priors) >= NON_ROOT_TOP_K:
                break

    if not actions_priors:
        return

    child = PosNode()
    child.player = leaf.current_player
    init_node_children(child.move_node, actions_priors)

    if parent.children is None:
        parent.children = {}
    parent.children[action] = child


# ---------------------------------------------------------------------------
# Move selection
# ---------------------------------------------------------------------------

def get_visit_counts(tree: MCTSTree) -> dict[int, int]:
    """Get visit counts for root actions."""
    root = tree.root_pos.move_node
    if root.actions is None:
        return {}
    visits = {}
    for i in range(root.n):
        vc = root.visits[i]
        if vc > 0:
            visits[root.actions[i]] = vc
    return visits


def select_move(tree: MCTSTree, temperature: float = 1.0) -> tuple[int, int]:
    """Select a move from visit counts. Returns (q, r)."""
    root = tree.root_pos.move_node
    if root.actions is None:
        raise RuntimeError("No actions in tree")

    counts = torch.tensor([root.visits[i] for i in range(root.n)],
                          dtype=torch.float32)

    if temperature < 0.05:
        best_local = counts.argmax().item()
    else:
        logits = counts.log().clamp(min=-100) / temperature
        probs = F.softmax(logits, dim=0)
        best_local = torch.multinomial(probs, 1).item()

    return idx_to_cell(root.actions[best_local])
