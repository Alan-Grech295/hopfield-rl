"""A forced-wait cue-matching task with independent memory and reward delays."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hopfield_rl.splits import LAYOUTS, SEED_RANGES, Split

Variant = Literal["full_state", "delayed_cue", "visual"]
COLORS = np.array([[230, 65, 65], [65, 125, 230]], dtype=np.uint8)
# Symbolic schema: phase(5), cue(2), left(2), right(2), choice(3), clocks(2).
PHASE = slice(0, 5)
CUE = slice(5, 7)
LEFT_COLOR = slice(7, 9)
RIGHT_COLOR = slice(9, 11)
CHOICE = slice(11, 14)
CLOCKS = slice(14, 16)


class Action(IntEnum):
    WAIT = 0
    LEFT = 1
    RIGHT = 2


class Phase(IntEnum):
    CUE = 0
    DELAY = 1
    DECISION = 2
    PAYOFF = 3
    TERMINAL = 4


@dataclass(frozen=True)
class TaskConfig:
    variant: Variant = "delayed_cue"
    split: Split = "pilot"
    cue_delay: int = 4
    reward_delay: int = 0
    time_limit: int | None = None

    def __post_init__(self):
        if self.variant not in ("full_state", "delayed_cue", "visual"):
            raise ValueError("unknown observation variant")
        if self.split not in LAYOUTS:
            raise ValueError("unknown split")
        for name, lower in (("cue_delay", 1), ("reward_delay", 0)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= 64:
                raise ValueError(f"{name} must be an integer in [{lower}, 64]")
        if self.time_limit is not None and (
            type(self.time_limit) is not int or self.time_limit < 1
        ):
            raise ValueError("time_limit must be a positive integer or None")

    @property
    def episode_length(self) -> int:
        return self.cue_delay + 1 + self.reward_delay


class CueTaskEnv(gym.Env):
    """Match the earlier red/blue cue to the left/right card exactly once.

    Only the diagnostic_action method exposes the correct answer. Ordinary
    observations and nonterminal info never contain diagnostic labels.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, config: TaskConfig | None = None, render_mode: str | None = None):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("only rgb_array rendering is supported")
        config = config or TaskConfig()
        self.config = config
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        observation = {"action_mask": spaces.MultiBinary(3)}
        if config.variant == "visual":
            observation["image"] = spaces.Box(0, 255, (96, 96, 3), dtype=np.uint8)
        else:
            observation["features"] = spaces.Box(0, 1, (16,), dtype=np.float32)
        self.observation_space = spaces.Dict(observation)
        self._done = True
        self._initialized = False

    @property
    def phase(self) -> Phase:
        if self._terminated:
            return Phase.TERMINAL
        if self._t == 0:
            return Phase.CUE
        if self._t < self.config.cue_delay:
            return Phase.DELAY
        if self._t == self.config.cue_delay:
            return Phase.DECISION
        return Phase.PAYOFF

    def reset(self, *, seed=None, options=None):
        if seed is not None and (
            not isinstance(seed, (int, np.integer)) or seed not in SEED_RANGES[self.config.split]
        ):
            raise ValueError("episode seed is outside the configured split")
        options = options or {}
        if set(options) - {"layout_id", "cue"}:
            raise ValueError("reset options support only layout_id and diagnostic cue override")
        layouts = LAYOUTS[self.config.split]
        layout_id = options.get("layout_id")
        if layout_id is not None and layout_id not in {layout.id for layout in layouts}:
            raise ValueError("layout is outside the configured split")
        cue = options.get("cue")
        if cue is not None and (type(cue) is not int or cue not in (0, 1)):
            raise ValueError("cue override must be 0 or 1")
        super().reset(seed=seed)
        self._layout = (
            next(layout for layout in layouts if layout.id == layout_id)
            if layout_id is not None
            else layouts[int(self.np_random.integers(len(layouts)))]
        )
        self._cue = int(self.np_random.integers(2)) if cue is None else cue
        self._t = 0
        self._choice = Action.WAIT  # WAIT means no choice has been committed yet.
        self._terminated = False
        self._done = False
        self._initialized = True
        return self._observation(), {}

    def _action_mask(self) -> np.ndarray:
        if self.phase == Phase.TERMINAL:
            return np.zeros(3, dtype=np.int8)
        if self.phase == Phase.DECISION:
            return np.array([0, 1, 1], dtype=np.int8)
        return np.array([1, 0, 0], dtype=np.int8)

    def diagnostic_action(self) -> int:
        """Privileged simulator-state policy; never call this from a learned policy."""
        if not self._initialized or self._done:
            raise RuntimeError("reset before requesting an action")
        if self.phase != Phase.DECISION:
            return int(Action.WAIT)
        return int(Action.LEFT if self._cue == self._layout.left_color else Action.RIGHT)

    def step(self, action):
        if self._done:
            raise RuntimeError("reset before stepping or after an episode boundary")
        if not self.action_space.contains(action) or not self._action_mask()[int(action)]:
            raise ValueError("action is outside the current action mask")
        if self.phase == Phase.DECISION:
            self._choice = Action(int(action))
        self._t += 1
        self._terminated = self._t == self.config.episode_length
        # A true ending takes precedence if the time limit coincides with payout.
        truncated = bool(
            not self._terminated
            and self.config.time_limit is not None
            and self._t >= self.config.time_limit
        )
        self._done = self._terminated or truncated
        reward = 0.0
        info = {}
        if self._terminated:
            correct = Action.LEFT if self._cue == self._layout.left_color else Action.RIGHT
            success = self._choice == correct
            reward = float(success)
            info["success"] = bool(success)
        return self._observation(), reward, self._terminated, truncated, info

    def _features(self, *, full_state: bool) -> np.ndarray:
        features = np.zeros(16, dtype=np.float32)
        features[int(self.phase)] = 1
        if full_state or self.phase == Phase.CUE:
            features[CUE.start + self._cue] = 1
        features[LEFT_COLOR.start + self._layout.left_color] = 1
        features[RIGHT_COLOR.start + 1 - self._layout.left_color] = 1
        features[CHOICE.start + int(self._choice)] = 1
        features[CLOCKS] = (
            max(self.config.cue_delay - self._t, 0) / 64,
            min(self.config.reward_delay, max(self.config.episode_length - self._t, 0)) / 64,
        )
        return features

    def _observation(self) -> dict[str, np.ndarray]:
        observation = {"action_mask": self._action_mask()}
        if self.config.variant == "visual":
            observation["image"] = self.render()
        else:
            observation["features"] = self._features(full_state=self.config.variant == "full_state")
        return observation

    def render(self) -> np.ndarray:
        """Render only delayed-cue information, including when used as a full-state preview."""
        if not self._initialized:
            raise RuntimeError("reset before rendering")
        frame = np.full((96, 96, 3), self._layout.background, dtype=np.uint8)
        frame[2:8, 4 + int(self.phase) * 18 : 16 + int(self.phase) * 18] = 240
        if self.phase == Phase.CUE:
            x = self._layout.cue_x
            frame[12:32, x : x + 20] = COLORS[self._cue]
        y = self._layout.cards_y
        frame[y : y + 20, 8:32] = COLORS[self._layout.left_color]
        frame[y : y + 20, 64:88] = COLORS[1 - self._layout.left_color]
        if self._choice != Action.WAIT:
            x = 8 if self._choice == Action.LEFT else 64
            frame[70:74, x : x + 24] = 240
        # Fixed-scale bars encode exact remaining step counts in [0, 64].
        clocks = np.rint(self._features(full_state=False)[CLOCKS] * 64).astype(int)
        frame[80:84, 8 : 8 + clocks[0]] = 240
        frame[88:92, 8 : 8 + clocks[1]] = 240
        return frame
