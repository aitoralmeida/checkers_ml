#!/usr/bin/env python3
"""
Author: Aitor Almeida aitor.almeida@deusto.es

A simple implementation of the checkers machine learning system described 
in Mitchell, T. M. (1997). Machine learning (Vol. 1, No. 9). New York: 
McGraw-hill.

The program implements four modules:

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

- Verify the program: 
python3 checkers_ml.py self-test

- Train through 1,000 self-play games (see the at the end the arguments to control 
the hyperparameters):
python3 checkers_ml.py train --games 1000 --output weights.json

- Evaluate the trained model:
python3 checkers_ml.py evaluate weights.json --games 200

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

# This is the Experiment Generator in the slides.
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

    # Instead of initializing the weight randomly, we start at 0. This are the parameters
    # that the model is learning.
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

    # Here we compute the new weights based on the results.
    # This is the Generalizer in the slides.
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

# The interesting part of this method is the exploration vs explotation
# We discussed this concept in the first unit. We control that with the epsilon 
# parameter. An epsilon of 0 selects the best possible move (exploitation), otherwise
# we add some probability of selecting a move randomly, allowing our learner to explore 
# other possible branches. 
# # Additional question: How does changing epsilon modify the results?
def select_move(
    board: Board,
    player: int,
    evaluator: LinearEvaluator,
    rng: random.Random,
    epsilon: float = 0.0,
) -> Move:
    """Choose an epsilon-greedy one-ply move using Black's perspective.

    With probability epsilon, choose a random legal move (exploration).
    Otherwise choose the move with the best predicted successor value
    (exploitation). Black maximises; Red minimises.
    """
    moves = legal_moves(board, player)
    if not moves:
        raise ValueError("Cannot select a move from a terminal position.")
    if rng.random() < epsilon:
        return rng.choice(moves)

    # This is a one-ply search: evaluate the board immediately after each move.
    scored = [(evaluator.value(apply_move(board, move)), move) for move in moves]
    best_score = (max if player == BLACK else min)(score for score, _ in scored)
    best_moves = [move for score, move in scored if abs(score - best_score) < 1e-12]
    # Random tie-breaking prevents a fixed board-order bias.
    return rng.choice(best_moves)

def play_game(
    black_evaluator: LinearEvaluator,
    red_evaluator: LinearEvaluator,
    rng: random.Random,
    epsilon: float = 0.0,
    max_moves: int = 200,
) -> GameResult:
    """Run one game and record states before every turn and at termination.

    In the slides' architecture this function is the Performance System. During
    self-play both colours use the same learned evaluator, producing the
    solution trace that the Critic will later convert into examples.
    """
    board = initial_board()
    player = BLACK
    history: list[tuple[Board, int]] = []

    # A repeated state with the same player or a long game is scored as a draw.
    occurrences: dict[tuple[Board, int], int] = {}
    for move_number in range(max_moves):
        state_key = (board, player)
        history.append(state_key)
        winner = terminal_winner(board, player)
        if winner is not None:
            return GameResult(winner, history, move_number)

        occurrences[state_key] = occurrences.get(state_key, 0) + 1
        if occurrences[state_key] >= 3:
            return GameResult(0, history, move_number)

        # Separate arguments allow evaluation against another policy, although
        # self-play passes the same evaluator for both colours.
        evaluator = black_evaluator if player == BLACK else red_evaluator
        move = select_move(board, player, evaluator, rng, epsilon)
        board = apply_move(board, move)
        player = -player

    history.append((board, player))
    winner = terminal_winner(board, player)
    return GameResult(winner or 0, history, max_moves)

def critic_examples(
    result: GameResult,
    evaluator: LinearEvaluator,
) -> Iterable[tuple[Board, float]]:
    """Yield (board, target) pairs using the deck's successor-state rule.

    This is the Critic in the slides.

    Successor(b) is the position after the learner's move and the opponent's
    reply, so trace[i + 2] has the same player to move as trace[i]. Positions
    immediately preceding termination receive the observed game result.

    This is a temporal-difference idea: most targets are estimates produced by
    the current model rather than labels supplied by a teacher. Final outcomes
    anchor the chain with +100 for a Black win, -100 for a Red win, and 0 for a
    draw. Repeated training propagates that information toward earlier states.
    """
    terminal_value = WIN_VALUE * result.winner
    history = result.history
    for i in range(len(history) - 1):
        board, _player = history[i]
        if i + 2 < len(history) - 1:
            # Two entries ahead is after this player's move and the opponent's
            # response, so it is again the same player's turn.
            successor_board, _ = history[i + 2]
            target = evaluator.value(successor_board)
        else:
            # Near the end there is no non-terminal two-ply successor. Use the
            # known game result instead of another model estimate.
            target = terminal_value
        yield board, target

# The full Actor-Critic cycle
def train(
    evaluator: LinearEvaluator,
    games: int,
    learning_rate: float,
    epsilon: float,
    max_moves: int,
    seed: int,
    report_every: int,
) -> None:
    """Train one shared value function through self-play.

    This function connects the Experiment Generator (initial_board),
    Performance System (play_game), Critic (critic_examples), and Generalizer
    (LinearEvaluator.update) into one learning loop.
    """
    rng = random.Random(seed)
    window = {BLACK: 0, RED: 0, 0: 0}
    absolute_error = 0.0
    example_count = 0

    for game in range(1, games + 1):
        # Experiment Generator: every experiment starts at the standard board.
        # Performance System: play a game using the current hypothesis.
        result = play_game(evaluator, evaluator, rng, epsilon, max_moves)
        window[result.winner] += 1

        # Critic: freeze all targets before changing the weights. Otherwise an
        # update early in the loop would change targets later in the same game.
        examples = list(critic_examples(result, evaluator))
        rng.shuffle(examples)
        for board, target in examples:
            # Generalizer: one online LMS update per training example.
            absolute_error += abs(evaluator.update(board, target, learning_rate))
            example_count += 1

        if game % report_every == 0 or game == games:
            covered = report_every if game % report_every == 0 else game % report_every
            mean_error = absolute_error / max(example_count, 1)
            print(
                f"games {game - covered + 1:>6}-{game:<6} "
                f"black={window[BLACK]:>4} red={window[RED]:>4} "
                f"draw={window[0]:>4} mean_abs_td_error={mean_error:.3f}"
            )
            window = {BLACK: 0, RED: 0, 0: 0}
            absolute_error = 0.0
            example_count = 0

def random_move(
    board: Board,
    player: int,
    _evaluator: LinearEvaluator,
    rng: random.Random,
) -> Move:
    return rng.choice(legal_moves(board, player))


def evaluation_game(
    evaluator: LinearEvaluator,
    learner_color: int,
    rng: random.Random,
    max_moves: int,
) -> int:
    """Play the learned policy against a uniformly random legal policy."""
    board = initial_board()
    player = BLACK
    occurrences: dict[tuple[Board, int], int] = {}
    for _ in range(max_moves):
        winner = terminal_winner(board, player)
        if winner is not None:
            return winner
        key = (board, player)
        occurrences[key] = occurrences.get(key, 0) + 1
        if occurrences[key] >= 3:
            return 0
        if player == learner_color:
            move = select_move(board, player, evaluator, rng)
        else:
            move = random_move(board, player, evaluator, rng)
        board = apply_move(board, move)
        player = -player
    return 0

# Additional question: can you modify the code to make one model play against other?
def evaluate(evaluator: LinearEvaluator, games: int, max_moves: int, seed: int) -> None:
    """Estimate performance while alternating the learner's colour."""
    rng = random.Random(seed)
    wins = losses = draws = 0
    for game in range(games):
        # Alternating colours reduces first-player/colour bias in the metric.
        learner_color = BLACK if game % 2 == 0 else RED
        winner = evaluation_game(evaluator, learner_color, rng, max_moves)
        if winner == learner_color:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    decisive = wins + losses
    print(f"learned policy vs random over {games} games")
    print(f"wins={wins} losses={losses} draws={draws}")
    print(f"win_rate={wins / games:.1%} decisive_win_rate={wins / decisive:.1%}" if decisive else "all games drawn")

# Some sanity check. Students, don't do this, use unit testing.
def run_self_tests() -> None:
    """Check core rule and learning behaviour without an external test library."""
    board = initial_board()
    assert board.count(BLACK_MAN) == 12
    assert board.count(RED_MAN) == 12
    assert len(legal_moves(board, BLACK)) == 7
    assert len(legal_moves(board, RED)) == 7

    # Compulsory capture and board update.
    cells = [EMPTY] * 64
    cells[index(2, 1)] = BLACK_MAN
    cells[index(3, 2)] = RED_MAN
    cells[index(2, 5)] = BLACK_MAN
    test_board = tuple(cells)
    moves = legal_moves(test_board, BLACK)
    assert len(moves) == 1 and moves[0].captured == ((3, 2),)
    after = apply_move(test_board, moves[0])
    assert after[index(4, 3)] == BLACK_MAN and after[index(3, 2)] == EMPTY

    # Multiple capture.
    cells = [EMPTY] * 64
    cells[index(1, 0)] = BLACK_MAN
    cells[index(2, 1)] = RED_MAN
    cells[index(4, 3)] = RED_MAN
    multi = legal_moves(tuple(cells), BLACK)
    assert len(multi) == 1 and len(multi[0].captured) == 2

    # Promotion.
    cells = [EMPTY] * 64
    cells[index(6, 1)] = BLACK_MAN
    promoted = apply_move(tuple(cells), Move(((6, 1), (7, 0))))
    assert promoted[index(7, 0)] == BLACK_KING

    # An LMS step must reduce error on a small, non-terminal example.
    evaluator = LinearEvaluator()
    before = abs(1.0 - evaluator.value(test_board))
    evaluator.update(test_board, 1.0, 0.001)
    after_error = abs(1.0 - evaluator.value(test_board))
    assert after_error < before
    print("All self-tests passed.")


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def probability(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value

# Console commands
def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface used by students."""
    parser = argparse.ArgumentParser(
        description="Train a Samuel-style checkers learner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Additional question: These are our hyperparameters. The ML system is not learning these,
    # we configure them manually. Can you find an optimal configuration? 
    train_parser = subparsers.add_parser("train", help="learn by self-play")
    train_parser.add_argument("--games", type=positive_int, default=1000)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--epsilon", type=probability, default=0.10)
    train_parser.add_argument("--max-moves", type=positive_int, default=200)
    train_parser.add_argument("--seed", type=int, default=7)
    # hyperparameters end here
    train_parser.add_argument("--report-every", type=positive_int, default=100)
    train_parser.add_argument("--load", type=Path, help="continue from saved weights")
    train_parser.add_argument("--output", type=Path, default=Path("checkers_weights.json"))

    evaluate_parser = subparsers.add_parser("evaluate", help="test saved weights against random play")
    evaluate_parser.add_argument("weights", type=Path)
    evaluate_parser.add_argument("--games", type=positive_int, default=200)
    evaluate_parser.add_argument("--max-moves", type=positive_int, default=200)
    evaluate_parser.add_argument("--seed", type=int, default=11)

    subparsers.add_parser("self-test", help="run deterministic engine and learning checks")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Dispatch the chosen command. Supplying argv also makes testing easier."""
    args = build_parser().parse_args(argv)
    if args.command == "self-test":
        run_self_tests()
    elif args.command == "train":
        if args.learning_rate <= 0:
            raise SystemExit("--learning-rate must be positive")
        evaluator = LinearEvaluator.load(args.load) if args.load else LinearEvaluator()
        train(
            evaluator,
            args.games,
            args.learning_rate,
            args.epsilon,
            args.max_moves,
            args.seed,
            args.report_every,
        )
        evaluator.save(args.output)
        print(f"saved weights to {args.output}")
        for name, weight in zip(LinearEvaluator.FEATURE_NAMES, evaluator.weights):
            print(f"  {name:>18}: {weight: .6f}")
    elif args.command == "evaluate":
        evaluate(LinearEvaluator.load(args.weights), args.games, args.max_moves, args.seed)

if __name__ == "__main__":
    main()

