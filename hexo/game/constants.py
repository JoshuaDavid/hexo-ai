"""Game constants for HeXO."""

from enum import IntEnum

BOARD_SIZE = 32
WIN_LENGTH = 6
N_CELLS = BOARD_SIZE * BOARD_SIZE  # 1024

# Axial hex directions: (dq, dr). The third axis is implicit (ds = -dq - dr).
HEX_DIRECTIONS = [(1, 0), (0, 1), (1, -1)]


class Player(IntEnum):
    NONE = 0
    A = 1
    B = 2

    def opponent(self) -> "Player":
        if self == Player.A:
            return Player.B
        if self == Player.B:
            return Player.A
        return Player.NONE
