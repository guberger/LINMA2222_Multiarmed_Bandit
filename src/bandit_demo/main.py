"""Launch the interactive multi-armed bandit demo."""

import argparse
import random

import matplotlib.pyplot as plt

from .model import BanditSimulation
from .plotting import Window

DEFAULT_REWARDS = (1, 2, 3, 4, 5)
DEFAULT_WEIGHTS = (
    (5, 1, 3, 2, 1),
    (1, 1, 2, 5, 5),
    (1, 1, 5, 1, 1),
    (1, 3, 1, 3, 1),
    (1, 2, 3, 2, 1),
)


def main() -> None:
    """Launch a window with an optional reproducible random seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Seed for reward and action sampling")
    args = parser.parse_args()
    simulation = BanditSimulation(
        DEFAULT_REWARDS, DEFAULT_WEIGHTS, rng=random.Random(args.seed)
    )
    window = Window(simulation)
    plt.show()
    # Keep widget callbacks alive throughout the event loop.
    plt.close(window.figure)


if __name__ == "__main__":
    main()
