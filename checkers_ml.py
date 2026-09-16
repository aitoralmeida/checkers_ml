#!/usr/bin/env python3
"""
A simple implementation of the checkers machine learning system described 
in Mitchell, T. M. (1997). Machine learning (Vol. 1, No. 9). New York: 
McGraw-hill.

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

Students: In the code you will find 'Additional question' comments, proposing 
changes to try or questions to answer.

"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


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

# This part is not for the Machine Learning part, just for the game codign
# Search 'The machine learning part starts here' to find where it starts

# ---------------------------------------------------------------------------
# Checkers rules and move generation
# ---------------------------------------------------------------------------
# These methods define the rules for american checkers, and are used to generate
# legal moves from a board state.


def index(row: int, col: int) -> int:
    """Convert a two-dimensional square into an index in the flat board."""
    return row * BOARD_SIZE + col


def inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def owner(piece: int) -> int:
    return BLACK if piece > 0 else RED if piece < 0 else 0


def square_name(row: int, col: int) -> str:
    return f"{chr(ord('a') + col)}{BOARD_SIZE - row}"


def initial_board() -> Board:
    """Return the standard American-checkers starting position."""
    cells = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)
    for row in range(3):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:
                cells[index(row, col)] = BLACK_MAN
    for row in range(5, 8):
        for col in range(BOARD_SIZE):
            if (row + col) % 2 == 1:
                cells[index(row, col)] = RED_MAN
    return tuple(cells)


def directions(piece: int) -> tuple[tuple[int, int], ...]:
    """Return the diagonal directions in which this piece may move."""
    # Kings move both up and down. Men move toward the opposite side only.
    # All pieces move only a single square if they are not capturing another
    # piece.
    if abs(piece) == 2:
        return ((-1, -1), (-1, 1), (1, -1), (1, 1))
    step = 1 if piece > 0 else -1
    return ((step, -1), (step, 1))


def capture_sequences(
    board: Board,
    start: tuple[int, int],
    piece: int,
) -> list[Move]:
    """Generate all complete jump sequences for one piece.

    Multiple jumps form a small search tree: after each capture there may be
    zero, one, or several possible next captures. The nested search function
    performs a depth-first traversal and records a Move only at a leaf. This
    prevents the player from stopping halfway through a required jump chain.
    """
    results: list[Move] = []

    def search(
        cells: Board,
        row: int,
        col: int,
        path: tuple[tuple[int, int], ...],
        captured: tuple[tuple[int, int], ...],
    ) -> None:
        # extended tells us whether at least one further capture was possible.
        extended = False
        for dr, dc in directions(piece):
            # A capture jumps over `middle` and lands two squares away.
            middle = (row + dr, col + dc)
            landing = (row + 2 * dr, col + 2 * dc)
            if not inside(*landing) or not inside(*middle):
                continue
            jumped = cells[index(*middle)]
            if owner(jumped) != -owner(piece) or cells[index(*landing)] != EMPTY:
                continue

            # Build a new board for this branch of the search tree. We never
            # mutate the board used by a sibling branch.
            updated = list(cells)
            updated[index(row, col)] = EMPTY
            updated[index(*middle)] = EMPTY
            updated[index(*landing)] = piece
            next_cells = tuple(updated)
            next_path = path + (landing,)
            next_captured = captured + (middle,)
            extended = True

            # In American checkers, reaching the king row ends a man's jump.
            reaches_king_row = abs(piece) == 1 and (
                (piece > 0 and landing[0] == 7)
                or (piece < 0 and landing[0] == 0)
            )
            if reaches_king_row:
                results.append(Move(next_path, next_captured))
            else:
                search(next_cells, *landing, next_path, next_captured)

        # No continuation means we have found one complete legal jump chain.
        if not extended and captured:
            results.append(Move(path, captured))

    search(board, *start, (start,), ())
    return results


def legal_moves(board: Board, player: int) -> list[Move]:
    """Return legal moves, enforcing compulsory capture."""
    captures: list[Move] = []
    ordinary: list[Move] = []

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[index(row, col)]
            if owner(piece) != player:
                continue
            captures.extend(capture_sequences(board, (row, col), piece))
            for dr, dc in directions(piece):
                destination = (row + dr, col + dc)
                if inside(*destination) and board[index(*destination)] == EMPTY:
                    ordinary.append(Move(((row, col), destination)))
    # American checkers requires a capture whenever one is available.
    return captures if captures else ordinary


def apply_move(board: Board, move: Move) -> Board:
    """Apply a previously validated move and return an immutable new board."""
    # A temporary list is convenient for mutation. Returning a tuple keeps the
    # public Board representation immutable and usable as a dictionary key.
    cells = list(board)
    start = move.path[0]
    destination = move.path[-1]
    piece = cells[index(*start)]
    cells[index(*start)] = EMPTY
    for square in move.captured:
        cells[index(*square)] = EMPTY
    if piece == BLACK_MAN and destination[0] == 7:
        piece = BLACK_KING
    elif piece == RED_MAN and destination[0] == 0:
        piece = RED_KING
    cells[index(*destination)] = piece
    return tuple(cells)


def terminal_winner(board: Board, player_to_move: int) -> Optional[int]:
    """Return the winner if the current player has no pieces or legal moves."""
    # In either case, the opponent of the blocked/eliminated player wins.
    if not any(owner(piece) == player_to_move for piece in board):
        return -player_to_move
    if not legal_moves(board, player_to_move):
        return -player_to_move
    return None


def threatened_squares(board: Board, attacker: int) -> set[tuple[int, int]]:
    """Return unique enemy squares the attacker can capture next."""
    # A set avoids double-counting a piece that appears in more than one legal
    # capture sequence.
    threatened: set[tuple[int, int]] = set()
    for move in legal_moves(board, attacker):
        threatened.update(move.captured)
    return threatened


def features(board: Board) -> tuple[float, ...]:
    """Extract the six board features from slide 28, plus a bias feature.

    The leading constant 1.0 is x0. Its weight w0 acts as an intercept, so the
    model need not predict zero when all six measured features are zero.
    """
    black_pieces = sum(piece > 0 for piece in board)
    red_pieces = sum(piece < 0 for piece in board)
    black_kings = board.count(BLACK_KING)
    red_kings = board.count(RED_KING)
    black_threatened = len(threatened_squares(board, RED))
    red_threatened = len(threatened_squares(board, BLACK))
    return (
        1.0,  # x0: bias/intercept feature
        float(black_pieces),
        float(red_pieces),
        float(black_kings),
        float(red_kings),
        float(black_threatened),
        float(red_threatened),
    )

# The machine learning part starts here

# ---------------------------------------------------------------------------
# Learned value function (target-function representation + Generalizer)
# ---------------------------------------------------------------------------
class LinearEvaluator:
    """Linear approximation V_hat(b) and its LMS generalizer.

    This class represents the hypothesis learned by the program. It replaces
    an enormous table of board values with only seven trainable parameters.
    That compact representation generalises from visited boards to unseen ones,
    but it cannot express every possible checkers strategy.
    """
    # The features in our approximation to the function
    # Additional question: how useful are all the features? Do removing each
    # one of them affect the results the same way? Are there features that are
    # more important than others?
    FEATURE_NAMES = (
        "bias",
        "black_pieces",
        "red_pieces",
        "black_kings",
        "red_kings",
        "black_threatened",
        "red_threatened",
    )

    # Instead of initializing the weight randomly, we start at 0.
    # Additional question: Does changing the weight initialization chage anything?
    def __init__(self, weights: Optional[Sequence[float]] = None) -> None:
        # Starting from zero means the learner initially has no preference.
        self.weights = list(weights) if weights is not None else [0.0] * 7 # Number of features
        if len(self.weights) != 7:
            raise ValueError("The evaluator requires exactly seven weights.")

    # Here we are computing our approximation function V_hat(b) = w0 + w1*x1 + ... + w6*x6
    def value(self, board: Board) -> float:
        """Compute the dot product w*x for one board state."""
        return sum(weight * feature for weight, feature in zip(self.weights, features(board)))

    # Here we compute the new weights based on the results
    def update(self, board: Board, target: float, learning_rate: float) -> float:
        """Apply the LMS rule: wi <- wi + eta * (target - prediction) * xi.

        `error` is positive when the current model underestimates the target and
        negative when it overestimates it. Multiplying by xi assigns more of the
        correction to features that are strongly present in this board.
        """
        xs = features(board)
        prediction = sum(weight * feature for weight, feature in zip(self.weights, xs))
        error = target - prediction
        # This is one stochastic-gradient step on squared prediction error.
        for i, feature in enumerate(xs):
            self.weights[i] += learning_rate * error * feature
        return error

    def save(self, path: Path) -> None:
        data = {
            "model": "linear_checkers_value_function",
            "perspective": "black",
            "feature_names": list(self.FEATURE_NAMES),
            "weights": self.weights,
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinearEvaluator":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["weights"])

