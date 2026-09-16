# Machine-Learning Checkers

This project is a small, dependency-free implementation of the checkers learning system described in Tom M. Mitchell's *Machine Learning* (1997). It includes:

- a command-line program that trains a checkers evaluator through self-play;
- an evaluation command that tests a saved evaluator against random play;
- deterministic self-tests for the game engine and learning update; and
- a Tkinter GUI for playing against a trained model.

The code is intended to be educational. It demonstrates an experiment generator, performance system, critic, and generalizer using a seven-weight linear value function. It is not a full-strength checkers engine.

## How it works

The learner estimates the value of a board from Black's perspective:

```text
V_hat(b) = w0 + w1*x1 + ... + w6*x6
```

The seven features are:

1. bias/intercept;
2. number of Black pieces;
3. number of Red pieces;
4. number of Black kings;
5. number of Red kings;
6. number of Black pieces threatened by Red; and
7. number of Red pieces threatened by Black.

Positive values favor Black and negative values favor Red. During training, the program plays games against itself, creates successor-state targets from each game, and updates the weights with the least-mean-squares (LMS) rule. Move selection is one-ply and epsilon-greedy: the model usually chooses its best evaluated move, but sometimes explores with a random legal move.

The engine implements standard American-checkers behavior used by this project, including compulsory captures, multiple jumps, kings, promotion, a threefold-repetition draw, and a configurable move limit.

## Requirements

- Python 3.9 or newer
- Tkinter for the graphical player

The training and evaluation program uses only the Python standard library. Tkinter is included with many Python installations; on some Linux systems it must be installed separately (for example, the package is commonly named `python3-tk`).

## Setup

Place both scripts in the same directory and use these exact filenames:

```text
checkers_ml.py
checkers_gui.py
```

No `pip install` step is required.

## Quick start

Run the built-in checks:

```bash
python3 checkers_ml.py self-test
```

Train a model with 1,000 self-play games:

```bash
python3 checkers_ml.py train --games 1000 --output weights.json
```

Evaluate it over 200 games against a random player:

```bash
python3 checkers_ml.py evaluate weights.json --games 200
```

Play against it in the GUI:

```bash
python3 checkers_gui.py weights.json
```

On Windows, use `py` instead of `python3` if that is how Python is installed.

## Command-line reference

Show the main help screen:

```bash
python3 checkers_ml.py --help
```

### `train`

```bash
python3 checkers_ml.py train [OPTIONS]
```

Trains one shared evaluator through self-play and saves the learned weights as JSON.

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--games N` | `1000` | Number of self-play games. Must be a positive integer. |
| `--learning-rate RATE` | `0.001` | LMS update step size. Must be greater than zero. Smaller values learn more gradually; large values may make training unstable. |
| `--epsilon P` | `0.10` | Exploration probability from `0` to `1`. `0` always exploits the current evaluator; `1` always chooses a random legal move. |
| `--max-moves N` | `200` | Maximum moves per game before it is recorded as a draw. Must be a positive integer. |
| `--seed N` | `7` | Integer seed controlling exploration and tie-breaking, allowing reproducible runs with the same inputs. |
| `--report-every N` | `100` | Print a training summary after every `N` games. Must be a positive integer. |
| `--load PATH` | none | Load an existing weights JSON file and continue training it. |
| `--output PATH` | `checkers_weights.json` | File in which to save the final model. An existing file at this path is overwritten. |
| `-h`, `--help` | — | Show help for the training command. |

Example with custom hyperparameters:

```bash
python3 checkers_ml.py train \
  --games 5000 \
  --learning-rate 0.0005 \
  --epsilon 0.15 \
  --max-moves 250 \
  --seed 42 \
  --report-every 250 \
  --output weights-5000.json
```

Continue training a saved model:

```bash
python3 checkers_ml.py train \
  --load weights-5000.json \
  --games 1000 \
  --output weights-6000.json
```

Training progress reports the numbers of Black wins, Red wins, and draws in the latest reporting window, plus the mean absolute temporal-difference error. When training finishes, the command prints the learned weights and saves them to the requested output file.

### `evaluate`

```bash
python3 checkers_ml.py evaluate WEIGHTS [OPTIONS]
```

Tests a saved model against a player that selects uniformly from the legal moves. The learner alternates between Black and Red to reduce color/first-move bias.

| Argument | Default | Meaning |
| --- | ---: | --- |
| `WEIGHTS` | required | Path to a weights JSON file produced by `train`. |
| `--games N` | `200` | Number of evaluation games. Must be a positive integer. |
| `--max-moves N` | `200` | Maximum moves per game before declaring a draw. Must be a positive integer. |
| `--seed N` | `11` | Integer seed for reproducible random opponent moves and model tie-breaking. |
| `-h`, `--help` | — | Show help for the evaluation command. |

Example:

```bash
python3 checkers_ml.py evaluate weights.json \
  --games 1000 \
  --max-moves 250 \
  --seed 42
```

The output includes wins, losses, draws, overall win rate, and decisive-game win rate.

### `self-test`

```bash
python3 checkers_ml.py self-test
```

Runs deterministic sanity checks for initial piece placement, legal opening moves, compulsory captures, multiple captures, promotion, board updates, and the LMS learning step. It takes no additional arguments.

## Graphical player

```bash
python3 checkers_gui.py [WEIGHTS] [OPTIONS]
```

| Argument | Default | Meaning |
| --- | ---: | --- |
| `WEIGHTS` | file chooser | Optional path to a model JSON file. If omitted, a file chooser opens at startup. |
| `--color {red,black}` | `red` | Human player's color. Black always moves first, and the board rotates when the human plays Black. |
| `--seed N` | `13` | Integer seed used when the model must break a tie between equally valued moves. |
| `-h`, `--help` | — | Show help for the graphical player. |

Examples:

```bash
# Choose a model with the file chooser and play as Red
python3 checkers_gui.py

# Open a specific model and play as Black
python3 checkers_gui.py weights.json --color black

# Use a reproducible tie-breaking seed
python3 checkers_gui.py weights.json --color red --seed 99
```

During play, blue outlines mark pieces that can move and green dots mark valid destinations. Captures are compulsory. For a multiple capture, keep selecting the next green destination until the sequence is complete. Use **New game** to restart or change color, and **Load model…** to switch weights files.

The GUI displays the current move count and model value. Its game limit is fixed at 200 moves; a repeated position occurring three times is also a draw.

## Saved model format

Training writes a human-readable JSON file similar to:

```json
{
  "model": "linear_checkers_value_function",
  "perspective": "black",
  "feature_names": [
    "bias",
    "black_pieces",
    "red_pieces",
    "black_kings",
    "red_kings",
    "black_threatened",
    "red_threatened"
  ],
  "weights": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

The loader expects a `weights` array containing exactly seven numeric values. Use files generated by this program unless you intentionally want to experiment with hand-written weights.

## Project files

| File | Purpose |
| --- | --- |
| `checkers_ml.py` | Rules engine, feature extraction, linear evaluator, self-play training, evaluation, persistence, and self-tests. |
| `checkers_gui.py` | Tkinter interface for playing against a saved evaluator. |
| `*.json` | Trained model files created by the `train` command. |

## Notes and limitations

- The evaluator looks only one move ahead and uses seven simple board features.
- Training is self-play with temporal-difference-style successor targets; it does not use a dataset of expert games.
- Evaluation against a random opponent is a useful basic benchmark, not proof of strong checkers play.
- Results depend on the hyperparameters, random seed, and number of training games.
- Keep model JSON files from untrusted sources out of important output paths; training overwrites the file supplied to `--output`.