# Multi-armed bandit demo

An interactive Matplotlib demo for LINMA2222. Pull individual arms or compare
epsilon-greedy and upper confidence bound (UCB) action selection across trials.
The default environment has five arms with rewards from 1 to 5.

## Setup and launch

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.10
or newer, and a graphical desktop with a Matplotlib GUI backend. From the
repository root:

```sh
uv sync --locked
uv run bandit-demo
```

uv creates and manages `.venv` automatically; no manual activation is needed.
Dependencies are pinned in `uv.lock`, including the default development group.
For a reproducible sequence of actions and rewards:

```sh
uv run bandit-demo --seed 42
```

Use the same controls in the same order to reproduce a seeded trial.
Alternatively, launch with `uv run python -m bandit_demo`.
Importing the package does not open a window.

## Controls and algorithms

- **Arm buttons:** sample a reward from that arm's weighted distribution.
- **New trial:** reset statistics and retain earlier average-reward curves for
  comparison. Parameters stay unchanged; the random generator is not reseeded.
- **Epsilon-greedy:** run 1,000 steps, exploring with probability epsilon and
  otherwise choosing the arm with the highest estimated reward.
- **UCB:** run 1,000 steps with the original demo's score:
  `estimate + sqrt(reward_range * log(max(total_pulls, 1)) / max(arm_pulls, 1e-9))`.
  This is the teaching demo's variant, rather than a change to a textbook UCB1
  implementation. At zero and one total pulls, the exploration bonus is zero.
- **Epsilon slider:** adjust exploration probability from 0 to 1.
- **Baseline slider:** adjust the additive offset in each reward estimate.

The original baseline semantics are preserved: an unvisited arm has estimate
`baseline`; after `n` pulls its estimate is `(reward_sum + baseline) / n`.
This is an additive offset, not a prior observation. Average rewards use only
actual sampled rewards. Ties select the first arm.

## Development

```sh
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
```

Tests use Matplotlib's noninteractive Agg backend, so they can run without a desktop.

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for
development tools. To update the pinned versions, run `uv lock --upgrade`, then
`uv sync --locked`. Commit `pyproject.toml` and `uv.lock` together when dependencies
change. CI installs from the lockfile on Python 3.10 and 3.14.

- `src/bandit_demo/model.py`: reward distributions, trial state, and policies.
- `src/bandit_demo/plotting.py`: widgets and plot updates.
- `src/bandit_demo/main.py`: default environment and command-line entry point.
- `tests/`: algorithm regression checks and headless widget integration tests.

Change `DEFAULT_REWARDS` and `DEFAULT_WEIGHTS` in `src/bandit_demo/main.py` to customize the
environment. Each weight row describes one arm and must have one nonnegative
weight per reward, with a positive total.
