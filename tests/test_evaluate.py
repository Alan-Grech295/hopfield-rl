import hashlib
import json

import numpy as np
import pytest

from hopfield_rl.env import Action, CueTaskEnv, TaskConfig
from hopfield_rl.evaluate import (
    audit_memory_requirement,
    audit_reward_delay,
    run_episode,
    validate,
    wilson_interval,
)
from hopfield_rl.policies import (
    CueMemoryPolicy,
    PolicyInput,
    RandomPolicy,
    ReactivePolicy,
    transition_masks,
)


@pytest.mark.parametrize("variant", ["full_state", "delayed_cue", "visual"])
def test_memory_policy_uses_observations_and_clears_state_at_episode_start(variant):
    policy = CueMemoryPolicy()
    env = CueTaskEnv(TaskConfig(variant=variant, cue_delay=1, reward_delay=2))
    state = policy.initial_state()
    for cue in (0, 1, 0):
        obs, _ = env.reset(seed=0, options={"cue": cue, "layout_id": "pilot-a"})
        action, state = policy.act(PolicyInput(obs, episode_start=True), state)
        assert action == Action.WAIT and state == cue
        obs, *_ = env.step(action)
        action, state = policy.act(PolicyInput(obs, previous_action=action), state)
        assert action == (Action.LEFT if cue == 0 else Action.RIGHT)


@pytest.mark.parametrize("variant", ["full_state", "delayed_cue", "visual"])
def test_reactive_policy_on_balanced_opposite_cues(variant):
    env = CueTaskEnv(TaskConfig(variant=variant, cue_delay=1))
    policy = ReactivePolicy()
    successes = 0
    for cue in (0, 1):
        obs, _ = env.reset(seed=0, options={"cue": cue, "layout_id": "pilot-a"})
        obs, *_ = env.step(Action.WAIT)
        action, _ = policy.act(PolicyInput(obs, previous_action=0), None)
        _, _, term, _, info = env.step(action)
        assert term
        successes += info["success"]
    assert successes == (2 if variant == "full_state" else 1)


def test_runner_resets_recurrent_state_and_does_not_pass_info_to_policy():
    class SpyPolicy(CueMemoryPolicy):
        def __init__(self):
            self.resets = 0
            self.inputs = []

        def initial_state(self):
            self.resets += 1
            return super().initial_state()

        def act(self, inputs, state):
            self.inputs.append(inputs)
            assert set(vars(inputs)) == {
                "observation",
                "previous_action",
                "previous_reward",
                "episode_start",
            }
            assert inputs.previous_reward == 0.0
            return super().act(inputs, state)

    policy = SpyPolicy()
    env = CueTaskEnv(TaskConfig(cue_delay=4, reward_delay=3, time_limit=5))
    for seed in (0, 1):
        row = run_episode(env, policy, seed, layout_id="pilot-a")
        assert row["truncated"] and not row["terminated"]
        assert not row["success"] and row["return"] == 0
    starts = [inputs for inputs in policy.inputs if inputs.episode_start]
    assert len(starts) == 2 and policy.resets >= 2
    assert all(inputs.previous_action is None for inputs in starts)


@pytest.mark.parametrize(
    "terminated,truncated,expected",
    [(False, False, (1, 1)), (True, False, (0, 0)), (False, True, (1, 0)), (True, True, (0, 0))],
)
def test_bootstrap_and_state_masks(terminated, truncated, expected):
    assert transition_masks(terminated, truncated) == expected


def test_random_policy_is_masked_and_reproducible():
    first, second = RandomPolicy(101), RandomPolicy(101)
    inputs = PolicyInput({"action_mask": np.array([0, 1, 1])})
    actions = [first.act(inputs, None)[0] for _ in range(100)]
    assert actions == [second.act(inputs, None)[0] for _ in range(100)]
    assert set(actions) == {1, 2}


def test_exhaustive_audits():
    memory = audit_memory_requirement()
    assert memory["paired_cases"] == 48 and memory["passed"]
    assert memory["environment_steps"] == 2040
    reward = audit_reward_delay()
    assert reward["cue_layout_delay_cases"] == 48 and reward["passed"]
    assert reward["environment_steps"] == 2232


def test_record_has_raw_episodes_budgets_provenance_and_uncertainty(tmp_path):
    report = validate(tmp_path, episodes=2, splits=("pilot",))
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert report["gate_passed"] and report["training_runs"] == 0
    assert len(report["conditions"]) == 72
    assert report["episodes"] == len(rows) == 144
    assert report["environment_steps"] == sum(row["steps"] for row in rows) + 4272
    assert (
        report["episodes_sha256"]
        == hashlib.sha256((tmp_path / "episodes.jsonl").read_bytes()).hexdigest()
    )
    assert json.loads((tmp_path / "record.json").read_text()) == report
    for condition in report["conditions"]:
        assert len(condition["success_ci95"]) == 2
        assert condition["truncations"] == 0
    assert report["source"]["source_sha256"]


def test_wilson_interval_is_finite_even_at_boundaries():
    assert wilson_interval(0, 200)[0] == 0
    assert wilson_interval(200, 200)[1] == 1
    low, high = wilson_interval(100, 200)
    assert low == pytest.approx(1 - high)
    assert 0.4 < low < 0.5 < high < 0.6
