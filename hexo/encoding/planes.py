"""Board-to-tensor encoding for the neural network."""

import torch

from hexo.game.constants import BOARD_SIZE, Player


def board_to_planes(game) -> torch.Tensor:
    """Convert game state to [3, BOARD_SIZE, BOARD_SIZE] tensor.

    Channel 0: current player's stones (1.0)
    Channel 1: opponent's stones (1.0)
    Channel 2: moves-left indicator (0.5 if 1 move left, 1.0 if 2 moves left)
    """
    planes = torch.zeros(3, BOARD_SIZE, BOARD_SIZE)
    cp = game.current_player
    for (q, r), player in game.board.items():
        if player == cp:
            planes[0, q, r] = 1.0
        else:
            planes[1, q, r] = 1.0
    planes[2] = 0.5 * game.moves_left_in_turn
    return planes


def board_to_planes_from_dict(board_dict: dict, current_player: int,
                               moves_left: int) -> torch.Tensor:
    """Convert raw board dict to planes tensor."""
    planes = torch.zeros(3, BOARD_SIZE, BOARD_SIZE)
    for (q, r), player in board_dict.items():
        p = player.value if hasattr(player, 'value') else int(player)
        cp = current_player.value if hasattr(current_player, 'value') else int(current_player)
        if p == cp:
            planes[0, q, r] = 1.0
        else:
            planes[1, q, r] = 1.0
    planes[2] = 0.5 * moves_left
    return planes
