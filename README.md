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
