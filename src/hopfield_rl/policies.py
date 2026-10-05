"""Causal recurrent interface and small diagnostic reference policies, without training."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from hopfield_rl.env import COLORS, CUE, LEFT_COLOR, Action


@dataclass(frozen=True)
class PolicyInput:
    """Only data already available at decision time; info and simulator are excluded."""

    observation: dict[str, np.ndarray]
    previous_action: int | None = None
    previous_reward: float = 0.0
    episode_start: bool = False


class RecurrentPolicy(Protocol):
    def initial_state(self) -> Any: ...

    def act(self, inputs: PolicyInput, state: Any) -> tuple[int, Any]: ...


def observed_colors(observation: dict[str, np.ndarray]) -> tuple[int | None, int]:
    """Read the visible cue and left card from current features or current pixels.

    The visual decoder is a hand-written solvability check, not a learned encoder.
    Its regions cover every split's layout without looking up a layout identifier.
    """
    if "features" in observation:
        features = observation["features"]
        cue = int(features[CUE].argmax()) if features[CUE].any() else None
        return cue, int(features[LEFT_COLOR].argmax())
    image = observation["image"]
    cue = None
    left = None
    for color, rgb in enumerate(COLORS):
        if np.any(np.all(image[12:32] == rgb, axis=-1)):
            cue = color
        if np.any(np.all(image[40:68, 8:32] == rgb, axis=-1)):
            left = color
    if left is None:
        raise ValueError("left card is missing from the current image")
    return cue, left


class CueMemoryPolicy:
    """Store the cue in recurrent state; receive no simulator labels or info."""

    def initial_state(self) -> int | None:
        return None

    def act(self, inputs: PolicyInput, state: int | None) -> tuple[int, int | None]:
        if inputs.episode_start:
            state = self.initial_state()
        cue, left = observed_colors(inputs.observation)
        if cue is not None:
            state = cue
        if inputs.observation["action_mask"][Action.WAIT]:
            return int(Action.WAIT), state
        if state is None:
            raise ValueError("the decision needs a cue from an earlier observation")
        return int(Action.LEFT if state == left else Action.RIGHT), state


class ReactivePolicy:
    """Use only current visible information, choosing left when the cue is absent."""

    def initial_state(self) -> None:
        return None

    def act(self, inputs: PolicyInput, state: None) -> tuple[int, None]:
        if inputs.observation["action_mask"][Action.WAIT]:
            return int(Action.WAIT), None
        cue, left = observed_colors(inputs.observation)
        action = Action.LEFT if cue is None or cue == left else Action.RIGHT
        return int(action), None


class RandomPolicy:
    """Uniform over the current legal actions; seeded independently of simulator RNG."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def initial_state(self) -> None:
        return None

    def act(self, inputs: PolicyInput, state: None) -> tuple[int, None]:
        actions = np.flatnonzero(inputs.observation["action_mask"])
        return int(self.rng.choice(actions)), None


def transition_masks(terminated: bool, truncated: bool) -> tuple[float, float]:
    """Return (bootstrap, carry_state): time limits bootstrap but end rollout history."""
    return float(not terminated), float(not (terminated or truncated))
