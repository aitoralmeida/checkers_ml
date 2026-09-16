#!/usr/bin/env python3
"""
A simple implementation of the checkers machine learning system described in Mitchell, T. M. (1997). Machine learning (Vol. 1, No. 9). New York: McGraw-hill.

he program implements four modules:

* ExperimentGenerator: creates a standard initial board.
* PerformanceSystem: selects legal moves with a learned evaluation function.
* Critic: turns self-play histories into successor-state training targets.
* Generalizer: updates a linear value function with the LMS rule.

The value function is

 V_hat(b) = w0 + w1*x1 + ... + w6*x6

where x1..x6 count black pieces, red pieces, black kings, red kings,
black pieces threatened by red, and red pieces threatened by black.

The learning cycle is:

1. Play a complete game against the current policy (self-play).
2. Let the Critic turn visited positions into training examples.
3. Use the value of a later position as the target for an earlier position.
4. Apply the LMS rule to move the weights toward those targets.
5. Repeat, occasionally choosing a random move to explore new positions.

Values are always expressed from Black's point of view. A large positive
value favours Black, while a large negative value favours Red. Consequently,
Black maximises the value function and Red minimises it.

No third-party packages are required. Run python checkers_ml.py --help

"""


# ---------------------------------------------------------------------------
# Board encoding and game constants
# ---------------------------------------------------------------------------
#
# A board is stored as a flat tuple of 64 integers. Positive integers belong
# to Black and negative integers belong to Red. The absolute value tells us
# whether the piece is a man (1) or a king (2). This makes owner(piece) and
# king tests simple, while the tuple keeps board states immutable and hashable.
EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
RED_MAN = -1
RED_KING = -2
BLACK = 1
RED = -1
BOARD_SIZE = 8
WIN_VALUE = 100.0
LOSE_VALUE = -100.0
DRAW_VALUE = 0.0

@dataclass(frozen=True)
class Move:
    """One complete move, including every jump in a multiple capture."""

    # path contains the start square followed by every landing square.
    # captured lists the jumped squares so apply_move can remove those pieces.
    path: tuple[tuple[int, int], ...]
    captured: tuple[tuple[int, int], ...] = ()

    def __str__(self) -> str:
        separator = "x" if self.captured else "-"
        return separator.join(square_name(row, col) for row, col in self.path)


@dataclass
class GameResult:
    """The outcome and trace of a played game."""

    winner: int  # BLACK, RED, or 0 for a draw
    # Each history item is (board, player_to_move). Keeping the player matters
    # because the same arrangement of pieces can mean a different game state.
    history: list[tuple[tuple[int, ...], int]]
    moves: int


Board = tuple[int, ...]

