"""Regression checks for trial bookkeeping and the original policies."""

import math
import random
import unittest

from bandit_demo.model import BanditSimulation


class SimulationTests(unittest.TestCase):
    def test_rewards_baseline_and_reset(self):
        simulation = BanditSimulation([1, 5], [[1, 0], [0, 1]], baseline=2)
        for arm in (0, 1, 1):
            simulation.pull(arm)
        self.assertEqual(simulation.counts, [1, 2])
        self.assertEqual(simulation.estimates, [3, 6])
        self.assertEqual(simulation.average_rewards, [1, 3, 11 / 3])
        simulation.baseline = 4
        self.assertEqual(simulation.estimates, [5, 7])
        simulation.reset()
        self.assertEqual(simulation.counts, [0, 0])
        self.assertEqual(simulation.estimates, [4, 4])
        self.assertEqual(simulation.average_rewards, [])

    def test_seeded_policies_match_original_formulas(self):
        rewards = [1, 2, 5]
        weights = [[5, 1, 1], [1, 1, 5], [1, 3, 1]]
        for policy in ("epsilon_greedy", "ucb"):
            with self.subTest(policy=policy):
                rng = random.Random(42)
                simulation = BanditSimulation(
                    rewards, weights, baseline=2, rng=random.Random(42)
                )
                histories = [[], [], []]
                averages = []
                for _ in range(150):
                    estimates = [
                        (sum(history) + 2) / max(len(history), 1)
                        for history in histories
                    ]
                    if policy == "ucb":
                        scale = 4 * math.log(max(len(averages), 1))
                        scores = [
                            estimate + math.sqrt(scale / max(len(history), 1e-9))
                            for estimate, history in zip(
                                estimates, histories, strict=True
                            )
                        ]
                        arm = max(range(3), key=scores.__getitem__)
                    else:
                        arm = max(range(3), key=estimates.__getitem__)
                        if rng.random() < 0.1:
                            arm = rng.choices(range(3))[0]
                    reward = rng.choices(rewards, weights[arm])[0]
                    histories[arm].append(reward)
                    averages.append(sum(map(sum, histories)) / (len(averages) + 1))
                    self.assertEqual(simulation.select_arm(policy), arm)
                    self.assertEqual(simulation.pull(arm), reward)
                self.assertEqual(simulation.average_rewards, averages)
                self.assertEqual(simulation.counts, list(map(len, histories)))

    def test_batch_matches_individual_steps(self):
        for policy in ("epsilon_greedy", "ucb"):
            batch = BanditSimulation([1, 5], [[1, 1], [1, 4]], rng=random.Random(8))
            manual = BanditSimulation([1, 5], [[1, 1], [1, 4]], rng=random.Random(8))
            batch.run(policy, 100)
            for _ in range(100):
                manual.pull(manual.select_arm(policy))
            self.assertEqual(batch.counts, manual.counts)
            self.assertEqual(batch.average_rewards, manual.average_rewards)

    def test_invalid_inputs(self):
        for rewards, weights in (
            ([], [[1]]),
            ([1], []),
            ([1], [[0]]),
            ([1], [[-1]]),
            ([1], [[1, 2]]),
            ([float("nan")], [[1]]),
            ([1], [[float("inf")]]),
        ):
            with self.subTest(rewards=rewards, weights=weights):
                with self.assertRaises(ValueError):
                    BanditSimulation(rewards, weights)
        simulation = BanditSimulation([1], [[1]])
        with self.assertRaises(ValueError):
            simulation.epsilon = 2
        with self.assertRaises(ValueError):
            simulation.baseline = float("nan")
        with self.assertRaises(IndexError):
            simulation.pull(-1)
        with self.assertRaises(ValueError):
            simulation.run("unknown", 0)
        with self.assertRaises(ValueError):
            simulation.run("ucb", -1)


if __name__ == "__main__":
    unittest.main()
