"""Reward sampling and action selection, independent of the GUI."""

import math
import random
from collections.abc import Sequence
from typing import Literal

Policy = Literal["epsilon_greedy", "ucb"]


class BanditSimulation:
    """A trial using the original demo's baseline and UCB formulas.

    After n pulls, an estimate is (reward_sum + baseline) / n.
    Unvisited arms use baseline. Ties favor the first arm.
    """

    def __init__(
        self,
        rewards: Sequence[float],
        weights: Sequence[Sequence[float]],
        *,
        baseline: float = 0.0,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        self.rewards = tuple(float(value) for value in rewards)
        self.weights = tuple(tuple(float(value) for value in row) for row in weights)
        if not self.rewards or not all(math.isfinite(x) for x in self.rewards):
            raise ValueError("Rewards must be nonempty and finite.")
        if not self.weights:
            raise ValueError("At least one arm is required.")
        for row in self.weights:
            if len(row) != len(self.rewards):
                raise ValueError("Each arm needs one weight per reward.")
            if any(not math.isfinite(x) or x < 0 for x in row):
                raise ValueError("Weights must be finite and nonnegative.")
            if not math.isfinite(sum(row)) or sum(row) <= 0:
                raise ValueError("Each arm needs a positive finite total weight.")
        self.rng = rng if rng is not None else random.Random()
        self.baseline = baseline
        self.epsilon = epsilon
        self.reset()

    @property
    def arm_count(self) -> int:
        return len(self.weights)

    @property
    def baseline(self) -> float:
        return self._baseline

    @baseline.setter
    def baseline(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("Baseline must be finite.")
        self._baseline = float(value)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("Epsilon must lie between 0 and 1.")
        self._epsilon = float(value)

    @property
    def estimates(self) -> list[float]:
        return [
            (total + self.baseline) / max(count, 1)
            for total, count in zip(self.reward_sums, self.counts, strict=True)
        ]

    def reset(self) -> None:
        """Start a trial without changing parameters or reseeding the RNG."""
        self.counts = [0] * self.arm_count
        self.reward_sums = [0.0] * self.arm_count
        self.average_rewards: list[float] = []
        self.total_reward = 0.0

    def pull(self, arm: int) -> float:
        """Sample an arm and update running totals in constant time."""
        if not 0 <= arm < self.arm_count:
            raise IndexError("Arm index is out of range.")
        reward = self.rng.choices(self.rewards, weights=self.weights[arm])[0]
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        self.total_reward += reward
        self.average_rewards.append(self.total_reward / (len(self.average_rewards) + 1))
        return reward

    def select_arm(self, policy: Policy) -> int:
        """Choose an arm using epsilon-greedy or the demo's UCB variant."""
        estimates = self.estimates
        if policy == "epsilon_greedy":
            arm = max(range(self.arm_count), key=estimates.__getitem__)
            if self.rng.random() < self.epsilon:
                arm = self.rng.choices(range(self.arm_count))[0]
            return arm
        if policy == "ucb":
            scale = (max(self.rewards) - min(self.rewards)) * math.log(
                max(len(self.average_rewards), 1)
            )
            scores = [
                estimate + math.sqrt(scale / max(count, 1e-9))
                for estimate, count in zip(estimates, self.counts, strict=True)
            ]
            return max(range(self.arm_count), key=scores.__getitem__)
        raise ValueError(f"Unknown policy: {policy}")

    def run(self, policy: Policy, steps: int = 1000) -> None:
        """Run a batch of actions without plotting overhead."""
        if policy not in ("epsilon_greedy", "ucb"):
            raise ValueError(f"Unknown policy: {policy}")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("Steps must be a nonnegative integer.")
        for _ in range(steps):
            self.pull(self.select_arm(policy))
