"""Reproducible M01 gate checks and per-episode experiment records.

Run with: python -m hopfield_rl.evaluate --output experiments/m01
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from hopfield_rl.env import Action, CueTaskEnv, TaskConfig
from hopfield_rl.policies import (
    CueMemoryPolicy,
    PolicyInput,
    RandomPolicy,
    ReactivePolicy,
    RecurrentPolicy,
)
from hopfield_rl.splits import LAYOUTS, episode_seeds

GAMMA = 0.99
POLICY_SEEDS = (101, 202, 303)  # Policy sampling streams, not independent training runs.
DELAYS = ((1, 0), (4, 0), (4, 8), (16, 8))


def wilson_interval(successes: int, count: int) -> list[float]:
    """95% binomial Wilson interval; episodes do not count as training runs."""
    if count < 1 or not 0 <= successes <= count:
        raise ValueError("require 0 <= successes <= count and count >= 1")
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [
        0.0 if successes == 0 else max(0.0, center - radius),
        1.0 if successes == count else min(1.0, center + radius),
    ]


def run_episode(
    env: CueTaskEnv,
    policy: RecurrentPolicy | None,
    seed: int,
    *,
    layout_id: str,
) -> dict:
    """None selects the explicitly privileged simulator oracle, only for diagnostics."""
    observation, _ = env.reset(seed=seed, options={"layout_id": layout_id})
    state = policy.initial_state() if policy else None
    inputs = PolicyInput(observation, episode_start=True)
    rewards = []
    decision_action = None
    while True:
        if policy is None:
            action = env.diagnostic_action()
        else:
            action, state = policy.act(inputs, state)
        if action != Action.WAIT:
            decision_action = action
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
        inputs = PolicyInput(observation, previous_action=action, previous_reward=reward)
    return {
        "episode_seed": seed,
        "layout_id": layout_id,
        "steps": len(rewards),
        "decision_action": decision_action,
        "rewards": rewards,
        "return": sum(rewards),
        "discounted_return": sum(GAMMA**t * reward for t, reward in enumerate(rewards)),
        "terminated": terminated,
        "truncated": truncated,
        "success": info.get("success", False),
    }


def audit_memory_requirement() -> dict:
    """Exhaust both cues and layouts, comparing *all* pre-choice policy inputs."""
    cases = 0
    steps = 0
    for split, layouts in LAYOUTS.items():
        for variant in ("delayed_cue", "visual"):
            for delay in (1, 4, 16, 64):
                for layout in layouts:
                    snapshots = []
                    for cue in (0, 1):
                        env = CueTaskEnv(TaskConfig(variant=variant, split=split, cue_delay=delay))
                        obs, info = env.reset(
                            seed=episode_seeds(split, 1)[0],
                            options={"layout_id": layout.id, "cue": cue},
                        )
                        for _ in range(delay):
                            obs, reward, terminated, truncated, info = env.step(Action.WAIT)
                            steps += 1
                            if reward or terminated or truncated or info:
                                raise AssertionError("pre-decision reward or diagnostic leakage")
                        snapshots.append((obs, env.diagnostic_action()))
                    first, second = snapshots
                    if first[1] == second[1] or any(
                        not np.array_equal(first[0][key], second[0][key]) for key in first[0]
                    ):
                        raise AssertionError(
                            "history is not necessary: cue leaked or choice unchanged"
                        )
                    # Both also have previous_action=WAIT, previous_reward=0, episode_start=False.
                    cases += 1
    return {
        "paired_cases": cases,
        "passed": True,
        "balanced_reactive_success_bound": 0.5,
        "episodes": 2 * cases,
        "environment_steps": steps,
    }


def audit_reward_delay() -> dict:
    """Enumerate decisions: delay changes discounted magnitude, never the maximizer."""
    cases = 0
    steps = 0
    for split, layouts in LAYOUTS.items():
        for layout in layouts:
            for cue in (0, 1):
                best_actions = []
                for delay in (0, 1, 8, 64):
                    returns = []
                    for action in (Action.LEFT, Action.RIGHT):
                        env = CueTaskEnv(TaskConfig(split=split, reward_delay=delay))
                        env.reset(
                            seed=episode_seeds(split, 1)[0],
                            options={"layout_id": layout.id, "cue": cue},
                        )
                        rewards = []
                        for t in range(env.config.episode_length):
                            _, reward, term, trunc, _ = env.step(
                                action if t == env.config.cue_delay else Action.WAIT
                            )
                            rewards.append(reward)
                            steps += 1
                        if not term or trunc:
                            raise AssertionError("reward-delay audit did not finish")
                        returns.append(sum(GAMMA**t * r for t, r in enumerate(rewards)))
                    expected_action = Action.LEFT if cue == layout.left_color else Action.RIGHT
                    expected = [0.0, 0.0]
                    expected[int(expected_action) - 1] = GAMMA ** (env.config.episode_length - 1)
                    np.testing.assert_allclose(returns, expected, rtol=1e-12, atol=0)
                    best_actions.append(int(np.argmax(returns)) + 1)
                    cases += 1
                if len(set(best_actions)) != 1:
                    raise AssertionError("reward delay changes the best decision")
    return {
        "cue_layout_delay_cases": cases,
        "passed": True,
        "gamma": GAMMA,
        "episodes": 2 * cases,
        "environment_steps": steps,
    }


def source_revision() -> dict:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    root = Path(__file__).parent
    files = sorted(root.glob("*.py"))
    return {
        "git_head": result.stdout.strip(),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }


def validate(output: Path, episodes: int = 200, splits: tuple[str, ...] = ("held_out",)) -> dict:
    # Validate the entire budget before creating files or running a partial experiment.
    seeds_by_split = {split: episode_seeds(split, episodes) for split in splits}
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    audits = {"memory": audit_memory_requirement(), "reward_delay": audit_reward_delay()}
    summaries = []
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for split, seeds in seeds_by_split.items():
            for variant in ("full_state", "delayed_cue", "visual"):
                for cue_delay, reward_delay in DELAYS:
                    config = TaskConfig(
                        variant=variant,
                        split=split,
                        cue_delay=cue_delay,
                        reward_delay=reward_delay,
                    )
                    for name in ("simulator_oracle", "cue_memory", "random", "reactive"):
                        sampling_seeds = POLICY_SEEDS if name == "random" else (None,)
                        for policy_seed in sampling_seeds:
                            policy = {
                                "simulator_oracle": lambda: None,
                                "cue_memory": CueMemoryPolicy,
                                "random": lambda seed=policy_seed: RandomPolicy(seed),
                                "reactive": ReactivePolicy,
                            }[name]()
                            env = CueTaskEnv(config)
                            rows = []
                            for i, seed in enumerate(seeds):
                                layout = LAYOUTS[split][i % len(LAYOUTS[split])]
                                row = run_episode(env, policy, seed, layout_id=layout.id)
                                row.update(
                                    config=asdict(config), policy=name, policy_seed=policy_seed
                                )
                                stream.write(json.dumps(row) + "\n")
                                rows.append(row)
                            env.close()
                            successes = sum(row["success"] for row in rows)
                            returns = [row["discounted_return"] for row in rows]
                            summaries.append(
                                {
                                    "config": asdict(config),
                                    "policy": name,
                                    "policy_seed": policy_seed,
                                    "episodes": episodes,
                                    "successes": successes,
                                    "success_rate": successes / episodes,
                                    "success_ci95": wilson_interval(successes, episodes),
                                    "mean_discounted_return": mean(returns),
                                    "discounted_return_standard_error": (
                                        stdev(returns) / sqrt(episodes) if episodes > 1 else None
                                    ),
                                    "environment_steps": sum(row["steps"] for row in rows),
                                    "truncations": sum(row["truncated"] for row in rows),
                                }
                            )
    gate = all(
        row["success_rate"] >= 0.95
        for row in summaries
        if row["policy"] in ("simulator_oracle", "cue_memory")
    )
    report = {
        "milestone": "M01",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "hypothesis": "History is needed for hidden cues; payout delay preserves the best choice.",
        "gamma": GAMMA,
        "success_target": 0.95,
        "gate_passed": gate,
        "decision": "continue to M02/M03" if gate else "stop: resolve task solvability",
        "training_runs": 0,
        "parameter_updates": 0,
        "retrieval_iterations": 0,
        "policy_sampling_seeds": list(POLICY_SEEDS),
        "stopping_rule": "fixed episode budget per condition; no adaptive retries or exclusions",
        "reset_rules": "reset policy state, previous action/reward at termination or truncation",
        "privileged_access": "simulator_oracle and exhaustive audits only",
        "observation_window": "one current frame; cue_memory retains only observed past cue",
        "audits": audits,
        "conditions": summaries,
        "episodes": sum(row["episodes"] for row in summaries),
        "evaluation_environment_steps": sum(row["environment_steps"] for row in summaries),
        "audit_episodes": sum(audit["episodes"] for audit in audits.values()),
        "audit_environment_steps": sum(audit["environment_steps"] for audit in audits.values()),
        "environment_steps": (
            sum(row["environment_steps"] for row in summaries)
            + sum(audit["environment_steps"] for audit in audits.values())
        ),
        "wall_seconds": time.perf_counter() - started,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {name: version(name) for name in ("gymnasium", "numpy", "hopfield-rl")},
        },
        "source": source_revision(),
        "episodes_sha256": hashlib.sha256((output / "episodes.jsonl").read_bytes()).hexdigest(),
    }
    (output / "record.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/m01"))
    parser.add_argument(
        "--episodes", type=int, default=200, help="episodes per policy and condition"
    )
    parser.add_argument("--splits", nargs="+", choices=list(LAYOUTS), default=["held_out"])
    args = parser.parse_args()
    report = validate(args.output, args.episodes, tuple(args.splits))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "gate_passed",
                    "decision",
                    "episodes",
                    "environment_steps",
                    "wall_seconds",
                )
            },
            indent=2,
        )
    )
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
