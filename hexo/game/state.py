"""Toroidal game state for HeXO (6-in-a-row on hex grid)."""

from hexo.game.constants import BOARD_SIZE, WIN_LENGTH, HEX_DIRECTIONS, Player


class GameState:
    """HeXO game on a BOARD_SIZE x BOARD_SIZE toroidal hex grid.

    Turn order: A places 1, then alternating 2-stone turns (B2, A2, B2, ...).
    First to get WIN_LENGTH in a row along any hex axis wins.
    """
    __slots__ = (
        "board", "current_player", "moves_left_in_turn",
        "move_count", "winner", "game_over",
    )

    def __init__(self):
        self.board: dict[tuple[int, int], Player] = {}
        self.current_player: Player = Player.A
        self.moves_left_in_turn: int = 1  # A gets 1 stone on first turn
        self.move_count: int = 0
        self.winner: Player = Player.NONE
        self.game_over: bool = False

    def reset(self):
        self.board = {}
        self.current_player = Player.A
        self.moves_left_in_turn = 1
        self.move_count = 0
        self.winner = Player.NONE
        self.game_over = False

    def is_valid_move(self, q: int, r: int) -> bool:
        if self.game_over:
            return False
        return (q % BOARD_SIZE, r % BOARD_SIZE) not in self.board

    def save_state(self) -> tuple:
        return (
            self.current_player,
            self.moves_left_in_turn,
            self.winner,
            self.game_over,
        )

    def undo_move(self, q: int, r: int, state: tuple):
        wq, wr = q % BOARD_SIZE, r % BOARD_SIZE
        del self.board[(wq, wr)]
        self.move_count -= 1
        (self.current_player, self.moves_left_in_turn,
         self.winner, self.game_over) = state

    def make_move(self, q: int, r: int) -> bool:
        wq, wr = q % BOARD_SIZE, r % BOARD_SIZE
        if self.game_over or (wq, wr) in self.board:
            return False

        self.board[(wq, wr)] = self.current_player
        self.move_count += 1

        if self._check_win(wq, wr):
            self.winner = self.current_player
            self.game_over = True
            return True

        self.moves_left_in_turn -= 1
        if self.moves_left_in_turn <= 0:
            self._switch_player()

        return True

    def _switch_player(self):
        self.current_player = self.current_player.opponent()
        self.moves_left_in_turn = 2

    def _check_win(self, q: int, r: int) -> bool:
        player = self.board[(q, r)]
        N = BOARD_SIZE

        for dq, dr in HEX_DIRECTIONS:
            count = 1
            for i in range(1, WIN_LENGTH):
                nq = (q + dq * i) % N
                nr = (r + dr * i) % N
                if self.board.get((nq, nr)) == player:
                    count += 1
                else:
                    break
            for i in range(1, WIN_LENGTH):
                nq = (q - dq * i) % N
                nr = (r - dr * i) % N
                if self.board.get((nq, nr)) == player:
                    count += 1
                else:
                    break
            if count >= WIN_LENGTH:
                return True

        return False

    def get_occupied_set(self) -> frozenset:
        return frozenset(self.board.keys())

    def clone(self) -> "GameState":
        g = GameState()
        g.board = dict(self.board)
        g.current_player = self.current_player
        g.moves_left_in_turn = self.moves_left_in_turn
        g.move_count = self.move_count
        g.winner = self.winner
        g.game_over = self.game_over
        return g

    def to_dict(self) -> dict:
        return {
            "board": {f"{q},{r}": int(p) for (q, r), p in self.board.items()},
            "current_player": int(self.current_player),
            "moves_left_in_turn": self.moves_left_in_turn,
            "move_count": self.move_count,
            "winner": int(self.winner),
            "game_over": self.game_over,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        g = cls()
        g.board = {
            tuple(int(x) for x in k.split(",")): Player(v)
            for k, v in d["board"].items()
        }
        g.current_player = Player(d["current_player"])
        g.moves_left_in_turn = d["moves_left_in_turn"]
        g.move_count = d["move_count"]
        g.winner = Player(d["winner"])
        g.game_over = d["game_over"]
        return g
