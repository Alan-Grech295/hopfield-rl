"""Fixed, disjoint layout and episode-seed partitions; never split individual frames."""

from dataclasses import dataclass
from typing import Literal

Split = Literal["pilot", "train", "held_out"]


@dataclass(frozen=True)
class Layout:
    id: str
    left_color: int
    cue_x: int
    cards_y: int
    background: int


# Each partition contains both color orders, but different visual arrangements.
LAYOUTS: dict[Split, tuple[Layout, ...]] = {
    "pilot": (Layout("pilot-a", 0, 32, 40, 16), Layout("pilot-b", 1, 40, 44, 20)),
    "train": (Layout("train-a", 0, 24, 48, 24), Layout("train-b", 1, 48, 40, 28)),
    "held_out": (Layout("test-a", 0, 28, 44, 32), Layout("test-b", 1, 44, 48, 36)),
}
SEED_RANGES: dict[Split, range] = {
    "pilot": range(0, 1_000),
    "train": range(10_000, 100_000),
    "held_out": range(1_000_000, 1_100_000),
}


def episode_seeds(split: Split, count: int) -> list[int]:
    """Return the declared evaluation budget without reusing another split's seeds."""
    seeds = SEED_RANGES[split]
    if not 1 <= count <= len(seeds):
        raise ValueError(f"count must be between 1 and {len(seeds)}")
    return list(seeds[:count])
