"""Matplotlib controls and plots for the bandit simulation."""

from functools import partial

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from .model import BanditSimulation, Policy

# Coordinates are fractions of the figure size, except SLIDER_LABEL_X,
# which is relative to each slider's axes. Adjust the controls here.
FIGURE_SIZE = (15, 9)
PLOT_LAYOUT = dict(left=0.30, right=0.97, top=0.93, hspace=0.45)
BUTTON_LEFT = 0.035
BUTTON_WIDTH = 0.20
BUTTON_HEIGHT = 0.045
NEW_TRIAL_BOTTOM = 0.88
ARM_TOP = 0.80
ARM_SPACING = 0.065
ARM_AREA_HEIGHT = 0.40
ARM_HEIGHT_RATIO = 0.8
UCB_BOTTOM = 0.05
GREEDY_BOTTOM = 0.12
EPSILON_SLIDER_RECT = (0.08, 0.22, 0.12, 0.03)
BASELINE_SLIDER_RECT = (0.08, 0.29, 0.12, 0.03)
SLIDER_LABEL_X = -0.10
REWARD_TEXT_POSITION = (0.04, 0.36)


class Window:
    """Display a simulation, retaining earlier trial curves for comparison."""

    def __init__(
        self, simulation: BanditSimulation, *, batch_steps: int = 1000
    ) -> None:
        if not isinstance(batch_steps, int) or batch_steps < 1:
            raise ValueError("Batch steps must be a positive integer.")
        self.simulation = simulation
        self.batch_steps = batch_steps
        self.figure = plt.figure(figsize=FIGURE_SIZE)
        self.figure.suptitle("Multi-armed bandit")
        self.average_axes, self.estimate_axes, self.count_axes = self.figure.subplots(3)
        self.figure.subplots_adjust(**PLOT_LAYOUT)
        self.average_axes.set(xlabel="Step", ylabel="Average reward")
        self.estimate_axes.set(ylabel="Estimated reward", xlabel="Arm")
        self.count_axes.set(ylabel="Number of pulls", xlabel="Arm")
        arms = range(1, simulation.arm_count + 1)
        self.estimate_bars = self.estimate_axes.bar(arms, simulation.estimates)
        self.count_bars = self.count_axes.bar(arms, simulation.counts)
        for axes in (self.estimate_axes, self.count_axes):
            axes.set_xticks(list(arms))
        (self.current_curve,) = self.average_axes.plot([], [])

        # Keep widgets referenced: Matplotlib stores weak callback references.
        self.buttons: list[Button] = []
        self._add_button("New trial", NEW_TRIAL_BOTTOM, self.new_trial)
        # Fit manual controls above the policy buttons, even with many arms.
        spacing = min(ARM_SPACING, ARM_AREA_HEIGHT / simulation.arm_count)
        for arm in range(simulation.arm_count):
            self._add_button(
                f"Arm {arm + 1}",
                ARM_TOP - arm * spacing,
                partial(self.pull_arm, arm),
                height=min(BUTTON_HEIGHT, spacing * ARM_HEIGHT_RATIO),
            )
        self._add_button(
            f"{batch_steps} steps UCB", UCB_BOTTOM, partial(self.run_policy, "ucb")
        )
        self._add_button(
            f"{batch_steps} steps epsilon-greedy",
            GREEDY_BOTTOM,
            partial(self.run_policy, "epsilon_greedy"),
        )
        self.epsilon_slider = Slider(
            self.figure.add_axes(EPSILON_SLIDER_RECT),
            "Epsilon",
            0.0,
            1.0,
            valinit=simulation.epsilon,
        )
        self.epsilon_slider.label.set_x(SLIDER_LABEL_X)
        self.epsilon_slider.on_changed(self.set_epsilon)
        self.baseline_slider = Slider(
            self.figure.add_axes(BASELINE_SLIDER_RECT),
            "Baseline",
            min(0, min(simulation.rewards), simulation.baseline),
            max(1, max(simulation.rewards), simulation.baseline),
            valinit=simulation.baseline,
        )
        self.baseline_slider.label.set_x(SLIDER_LABEL_X)
        self.baseline_slider.on_changed(self.set_baseline)
        self.reward_text = self.figure.text(*REWARD_TEXT_POSITION, "Last reward: —")
        self.refresh()

    def _add_button(self, label, bottom, callback, *, height=BUTTON_HEIGHT) -> None:
        axes = self.figure.add_axes((BUTTON_LEFT, bottom, BUTTON_WIDTH, height))
        button = Button(axes, label)
        button.on_clicked(callback)
        self.buttons.append(button)

    def refresh(self) -> None:
        """Update all artists once after a manual pull or policy batch."""
        for bar, estimate in zip(
            self.estimate_bars, self.simulation.estimates, strict=True
        ):
            bar.set_height(estimate)
        for bar, count in zip(self.count_bars, self.simulation.counts, strict=True):
            bar.set_height(count)
        rewards = self.simulation.average_rewards
        self.current_curve.set_data(range(1, len(rewards) + 1), rewards)
        for axes in (self.average_axes, self.estimate_axes, self.count_axes):
            axes.relim()
            axes.autoscale_view()
        self.figure.canvas.draw_idle()

    def new_trial(self, _event=None) -> None:
        self.simulation.reset()
        (self.current_curve,) = self.average_axes.plot([], [])
        self.reward_text.set_text("Last reward: —")
        self.refresh()

    def pull_arm(self, arm: int, _event=None) -> None:
        reward = self.simulation.pull(arm)
        self.reward_text.set_text(f"Last reward: {reward:g}")
        self.refresh()

    def run_policy(self, policy: Policy, _event=None) -> None:
        self.simulation.run(policy, self.batch_steps)
        self.reward_text.set_text(f"Completed {self.batch_steps} steps")
        self.refresh()

    def set_epsilon(self, value: float) -> None:
        self.simulation.epsilon = value

    def set_baseline(self, value: float) -> None:
        self.simulation.baseline = value
        self.refresh()
