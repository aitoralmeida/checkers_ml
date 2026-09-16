#!/usr/bin/env python3
"""Graphical player for models trained by ``checkers_ml.py``.

Keep this file in the same folder as ``checkers_ml.py``. Train a model first:

    python checkers_ml.py train --games 1000 --output weights.json

Then launch the graphical player:

    python checkers_gui.py weights.json

If the JSON path is omitted, the program opens a file chooser. The interface
uses only Tkinter, which is included with most standard Python installations.
No third-party Python packages or image files are required.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Sequence

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - depends on the Python install
    raise SystemExit(
        "Tkinter is not installed. Install your operating system's Tk package "
        "(often named python3-tk) and try again."
    ) from exc

try:
    # Reuse the rules and learned value function instead of duplicating them.
    from checkers_ml import (
        BLACK,
        BLACK_KING,
        BLACK_MAN,
        EMPTY,
        RED,
        RED_KING,
        RED_MAN,
        Board,
        LinearEvaluator,
        Move,
        apply_move,
        index,
        initial_board,
        legal_moves,
        select_move,
        square_name,
        terminal_winner,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import checkers_ml.py. Put checkers_gui.py and "
        "checkers_ml.py in the same folder."
    ) from exc


SQUARE_SIZE = 76
BOARD_PIXELS = 8 * SQUARE_SIZE
MAX_MOVES = 200

# Colours are kept in one place so students can easily restyle the interface.
LIGHT_SQUARE = "#ead9b6"
DARK_SQUARE = "#7a5230"
SELECTED_SQUARE = "#f2c94c"
LEGAL_START = "#56b4e9"
LEGAL_DESTINATION = "#52b788"
BLACK_PIECE = "#20242a"
RED_PIECE = "#c73e3a"
PIECE_OUTLINE = "#f3eee5"
KING_MARK = "#ffd166"


class CheckersGUI:
    """Tkinter interface that lets a human play against a saved evaluator."""

    def __init__(
        self,
        root: tk.Tk,
        evaluator: LinearEvaluator,
        model_path: Path,
        human_color: int = RED,
        seed: int = 13,
    ) -> None:
        self.root = root
        self.evaluator = evaluator
        self.model_path = model_path
        self.rng = random.Random(seed)

        self.board: Board = initial_board()
        self.player = BLACK
        self.human_color = human_color
        self.move_count = 0
        self.occurrences: dict[tuple[Board, int], int] = {}
        self.selected_path: tuple[tuple[int, int], ...] = ()
        self.game_over = False
        self.computer_busy = False
        # Store Tkinter's scheduled-callback ID so restarting cannot leave an
        # old computer turn waiting in the event queue.
        self.computer_job: Optional[str] = None

        self.color_choice = tk.StringVar(
            value="Red" if human_color == RED else "Black"
        )
        self.status_text = tk.StringVar()
        self.model_text = tk.StringVar()
        self.info_text = tk.StringVar()

        self._build_window()
        self._update_model_label()
        self.new_game()

    # ------------------------------------------------------------------ UI
    def _build_window(self) -> None:
        self.root.title("ML Checkers: Play the Learned Model")
        self.root.resizable(False, False)

        outer = ttk.Frame(self.root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")

        title = ttk.Label(
            outer,
            text="ML Checkers",
            font=("TkDefaultFont", 18, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        controls = ttk.Frame(outer)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 8))

        ttk.Label(controls, text="Play as:").grid(row=0, column=0, padx=(0, 4))
        color_box = ttk.Combobox(
            controls,
            textvariable=self.color_choice,
            values=("Red", "Black"),
            width=8,
            state="readonly",
        )
        color_box.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(controls, text="New game", command=self.new_game).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(controls, text="Load model…", command=self.load_model).grid(
            row=0, column=3
        )

        self.canvas = tk.Canvas(
            outer,
            width=BOARD_PIXELS,
            height=BOARD_PIXELS,
            highlightthickness=1,
            highlightbackground="#4b3a2b",
        )
        self.canvas.grid(row=2, column=0)
        self.canvas.bind("<Button-1>", self.on_board_click)

        status = ttk.Label(
            outer,
            textvariable=self.status_text,
            font=("TkDefaultFont", 11, "bold"),
            anchor="center",
        )
        status.grid(row=3, column=0, sticky="ew", pady=(10, 2))

        ttk.Label(outer, textvariable=self.info_text, anchor="center").grid(
            row=4, column=0, sticky="ew"
        )
        ttk.Label(outer, textvariable=self.model_text, anchor="center").grid(
            row=5, column=0, sticky="ew", pady=(2, 0)
        )

    def _update_model_label(self) -> None:
        self.model_text.set(f"Model: {self.model_path.name}")

    # ---------------------------------------------------------- Coordinates
    def display_to_board(self, display_row: int, display_col: int) -> tuple[int, int]:
        """Map a displayed square to the engine's board coordinates.

        The board rotates when the human plays Black, keeping the human pieces
        at the bottom of the window.
        """
        if self.human_color == BLACK:
            return 7 - display_row, 7 - display_col
        return display_row, display_col

    def board_to_display(self, row: int, col: int) -> tuple[int, int]:
        """Inverse of display_to_board."""
        if self.human_color == BLACK:
            return 7 - row, 7 - col
        return row, col

    # --------------------------------------------------------------- Drawing
    def draw_board(self) -> None:
        self.canvas.delete("all")

        legal = legal_moves(self.board, self.player) if not self.game_over else []
        legal_starts = {move.path[0] for move in legal}
        next_squares = self._next_path_squares(legal)

        for display_row in range(8):
            for display_col in range(8):
                row, col = self.display_to_board(display_row, display_col)
                x0 = display_col * SQUARE_SIZE
                y0 = display_row * SQUARE_SIZE
                x1 = x0 + SQUARE_SIZE
                y1 = y0 + SQUARE_SIZE
                colour = DARK_SQUARE if (row + col) % 2 else LIGHT_SQUARE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=colour, width=0)

                square = (row, col)
                if square in self.selected_path:
                    self.canvas.create_rectangle(
                        x0 + 3,
                        y0 + 3,
                        x1 - 3,
                        y1 - 3,
                        outline=SELECTED_SQUARE,
                        width=5,
                    )
                elif (
                    self.player == self.human_color
                    and not self.computer_busy
                    and square in legal_starts
                ):
                    self.canvas.create_rectangle(
                        x0 + 5,
                        y0 + 5,
                        x1 - 5,
                        y1 - 5,
                        outline=LEGAL_START,
                        width=3,
                    )

                if square in next_squares:
                    centre_x = x0 + SQUARE_SIZE / 2
                    centre_y = y0 + SQUARE_SIZE / 2
                    radius = 10
                    self.canvas.create_oval(
                        centre_x - radius,
                        centre_y - radius,
                        centre_x + radius,
                        centre_y + radius,
                        fill=LEGAL_DESTINATION,
                        outline="white",
                        width=2,
                    )

                piece = self.board[index(row, col)]
                if piece != EMPTY:
                    self._draw_piece(display_row, display_col, piece)

        # Small coordinate labels help students connect the GUI to printed
        # move names such as b6-a5 and c3xe5xg7.
        for display_col in range(8):
            row, col = self.display_to_board(7, display_col)
            self.canvas.create_text(
                display_col * SQUARE_SIZE + SQUARE_SIZE - 9,
                BOARD_PIXELS - 9,
                text=chr(ord("a") + col),
                fill="#f8f3e7" if (row + col) % 2 else "#3a2a1e",
                font=("TkDefaultFont", 8, "bold"),
            )
        for display_row in range(8):
            row, col = self.display_to_board(display_row, 0)
            self.canvas.create_text(
                9,
                display_row * SQUARE_SIZE + 10,
                text=str(8 - row),
                fill="#f8f3e7" if (row + col) % 2 else "#3a2a1e",
                font=("TkDefaultFont", 8, "bold"),
            )

        value = self.evaluator.value(self.board)
        self.info_text.set(
            f"Moves: {self.move_count}    Model value: {value:+.2f} "
            "(positive favours Black)"
        )

    def _draw_piece(self, display_row: int, display_col: int, piece: int) -> None:
        padding = 10
        x0 = display_col * SQUARE_SIZE + padding
        y0 = display_row * SQUARE_SIZE + padding
        x1 = (display_col + 1) * SQUARE_SIZE - padding
        y1 = (display_row + 1) * SQUARE_SIZE - padding
        fill = BLACK_PIECE if piece > 0 else RED_PIECE

        # Two slightly offset discs give each piece some visual depth.
        self.canvas.create_oval(
            x0 + 2, y0 + 5, x1 + 2, y1 + 5, fill="#3b2b22", outline=""
        )
        self.canvas.create_oval(
            x0, y0, x1, y1, fill=fill, outline=PIECE_OUTLINE, width=2
        )
        self.canvas.create_oval(
            x0 + 8, y0 + 8, x1 - 8, y1 - 8, outline=PIECE_OUTLINE, width=1
        )

        if piece in (BLACK_KING, RED_KING):
            self.canvas.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text="K",
                fill=KING_MARK,
                font=("TkDefaultFont", 21, "bold"),
            )

    def _next_path_squares(self, moves: list[Move]) -> set[tuple[int, int]]:
        """Return the squares the user may click next in the selected move."""
        if not self.selected_path:
            return set()
        prefix_length = len(self.selected_path)
        return {
            move.path[prefix_length]
            for move in moves
            if len(move.path) > prefix_length
            and move.path[:prefix_length] == self.selected_path
        }

    # ----------------------------------------------------------- Interaction
    def on_board_click(self, event: tk.Event) -> None:
        """Build a legal move path from the human's sequence of clicks."""
        if self.game_over or self.computer_busy or self.player != self.human_color:
            return

        display_col = event.x // SQUARE_SIZE
        display_row = event.y // SQUARE_SIZE
        if not (0 <= display_row < 8 and 0 <= display_col < 8):
            return
        clicked = self.display_to_board(display_row, display_col)
        moves = legal_moves(self.board, self.player)
        legal_starts = {move.path[0] for move in moves}

        if not self.selected_path:
            if clicked in legal_starts:
                self.selected_path = (clicked,)
                self.status_text.set("Choose a highlighted destination.")
                self.draw_board()
            return

        # Clicking another movable piece changes the selection.
        if clicked in legal_starts and clicked != self.selected_path[0]:
            self.selected_path = (clicked,)
            self.status_text.set("Choose a highlighted destination.")
            self.draw_board()
            return

        next_index = len(self.selected_path)
        matching = [
            move
            for move in moves
            if len(move.path) > next_index
            and move.path[:next_index] == self.selected_path
            and move.path[next_index] == clicked
        ]
        if not matching:
            # An invalid destination clears the selection. This gives the user
            # an easy way to cancel without adding another control.
            self.selected_path = ()
            self.status_text.set("Select a blue-outlined piece.")
            self.draw_board()
            return

        self.selected_path += (clicked,)
        completed = [move for move in matching if len(move.path) == len(self.selected_path)]
        if completed:
            self._finish_move(completed[0])
        else:
            # A multiple capture is still in progress; show the next landings.
            self.status_text.set("Capture continues: choose the next green square.")
            self.draw_board()

    def _finish_move(self, move: Move) -> None:
        self.board = apply_move(self.board, move)
        self.player = -self.player
        self.move_count += 1
        self.selected_path = ()
        self._begin_turn(last_move=f"You played {move}.")

    def _computer_move(self) -> None:
        self.computer_job = None
        if self.game_over or self.player == self.human_color:
            self.computer_busy = False
            return

        move = select_move(self.board, self.player, self.evaluator, self.rng)
        self.board = apply_move(self.board, move)
        self.player = -self.player
        self.move_count += 1
        self.computer_busy = False
        self._begin_turn(last_move=f"Model played {move}.")

    # -------------------------------------------------------------- Game flow
    def new_game(self) -> None:
        if self.computer_job is not None:
            self.root.after_cancel(self.computer_job)
            self.computer_job = None
        self.human_color = BLACK if self.color_choice.get() == "Black" else RED
        self.board = initial_board()
        self.player = BLACK
        self.move_count = 0
        self.occurrences = {}
        self.selected_path = ()
        self.game_over = False
        self.computer_busy = False
        self._begin_turn()

    def _begin_turn(self, last_move: str = "") -> None:
        """Check the outcome, record repetitions, then start the next turn."""
        winner = terminal_winner(self.board, self.player)
        if winner is not None:
            self._end_game(winner, last_move)
            return
        if self.move_count >= MAX_MOVES:
            self._end_game(0, last_move + " Move limit reached.")
            return

        state = (self.board, self.player)
        self.occurrences[state] = self.occurrences.get(state, 0) + 1
        if self.occurrences[state] >= 3:
            self._end_game(0, last_move + " Position repeated three times.")
            return

        if self.player == self.human_color:
            self.status_text.set(
                f"{last_move} Your turn: select a blue-outlined piece.".strip()
            )
            self.draw_board()
        else:
            self.computer_busy = True
            self.status_text.set(f"{last_move} Model is thinking…".strip())
            self.draw_board()
            # A short delay lets the interface repaint before the model moves.
            self.computer_job = self.root.after(450, self._computer_move)

    def _end_game(self, winner: int, reason: str = "") -> None:
        if self.computer_job is not None:
            self.root.after_cancel(self.computer_job)
            self.computer_job = None
        self.game_over = True
        self.computer_busy = False
        if winner == 0:
            result = "Draw."
        elif winner == self.human_color:
            result = "You win!"
        else:
            result = "The model wins."
        self.status_text.set(f"{reason} {result}".strip())
        self.draw_board()

    def load_model(self) -> None:
        """Replace the evaluator with another JSON file and start a new game."""
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Choose trained checkers weights",
            filetypes=(("JSON model", "*.json"), ("All files", "*.*")),
        )
        if not filename:
            return
        try:
            evaluator = LinearEvaluator.load(Path(filename))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            messagebox.showerror(
                "Could not load model",
                f"The selected file is not a compatible weights JSON file.\n\n{exc}",
                parent=self.root,
            )
            return
        self.evaluator = evaluator
        self.model_path = Path(filename)
        self._update_model_label()
        self.new_game()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play graphically against a model trained by checkers_ml.py."
    )
    parser.add_argument(
        "weights",
        nargs="?",
        type=Path,
        help="weights JSON produced by checkers_ml.py (file chooser if omitted)",
    )
    parser.add_argument(
        "--color",
        choices=("red", "black"),
        default="red",
        help="the human player's starting colour",
    )
    parser.add_argument("--seed", type=int, default=13, help="tie-breaking seed")
    return parser


def choose_initial_model(root: tk.Tk, requested: Optional[Path]) -> Optional[Path]:
    """Use the command-line path or ask the user to select a model."""
    if requested is not None:
        return requested
    filename = filedialog.askopenfilename(
        parent=root,
        title="Choose trained checkers weights",
        filetypes=(("JSON model", "*.json"), ("All files", "*.*")),
    )
    return Path(filename) if filename else None


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    root = tk.Tk()
    root.withdraw()

    model_path = choose_initial_model(root, args.weights)
    if model_path is None:
        root.destroy()
        return
    try:
        evaluator = LinearEvaluator.load(model_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        messagebox.showerror(
            "Could not load model",
            f"The selected file is not a compatible weights JSON file.\n\n{exc}",
            parent=root,
        )
        root.destroy()
        return

    human_color = BLACK if args.color == "black" else RED
    CheckersGUI(root, evaluator, model_path, human_color, args.seed)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
