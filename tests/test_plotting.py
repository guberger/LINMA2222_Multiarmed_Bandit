"""Exercise widget callbacks without opening a desktop window."""

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from bandit_demo.model import BanditSimulation  # noqa: E402
from bandit_demo.plotting import Window  # noqa: E402


class WindowTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_controls_and_trial_comparison(self):
        simulation = BanditSimulation([1, 5], [[1, 0], [0, 1]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            window = Window(simulation, batch_steps=10)
            window.pull_arm(1)
            self.assertEqual(window.estimate_bars[1].get_height(), 5)
            window.baseline_slider.set_val(2)
            self.assertEqual(window.estimate_bars[1].get_height(), 7)
            window.epsilon_slider.set_val(0)
            self.assertEqual(simulation.epsilon, 0)
            window.run_policy("epsilon_greedy")
            self.assertEqual(sum(simulation.counts), 11)
            previous_curve = window.current_curve
            window.new_trial()
            self.assertEqual(len(previous_curve.get_ydata()), 11)
            self.assertEqual(len(window.current_curve.get_ydata()), 0)
            self.assertEqual(simulation.counts, [0, 0])
            window.run_policy("ucb")
            self.assertEqual(sum(simulation.counts), 10)
            window.figure.canvas.draw()


if __name__ == "__main__":
    unittest.main()
