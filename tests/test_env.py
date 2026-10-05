import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_reset_seed_determinism
from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker

from hopfield_rl.env import CUE, Action, CueTaskEnv, TaskConfig
from hopfield_rl.splits import LAYOUTS, SEED_RANGES, episode_seeds


@pytest.mark.parametrize("variant", ["full_state", "delayed_cue", "visual"])
@pytest.mark.parametrize("cue_delay,reward_delay", [(1, 0), (4, 3), (64, 64)])
def test_timing_rewards_and_gymnasium_contract(variant, cue_delay, reward_delay):
    env = CueTaskEnv(TaskConfig(variant=variant, cue_delay=cue_delay, reward_delay=reward_delay))
    check_reset_seed_determinism(env)
    obs, info = env_reset_passive_checker(env, seed=12)
    assert info == {}
    rewards = []
    for t in range(env.config.episode_length):
        assert env.observation_space.contains(obs)
        expected_mask = [0, 1, 1] if t == cue_delay else [1, 0, 0]
        np.testing.assert_array_equal(obs["action_mask"], expected_mask)
        obs, reward, terminated, truncated, info = env_step_passive_checker(
            env, env.diagnostic_action()
        )
        rewards.append(reward)
        assert truncated is False
        assert terminated == (t == env.config.episode_length - 1)
        assert info == ({"success": True} if terminated else {})
    assert env.observation_space.contains(obs)
    assert rewards == [0.0] * (cue_delay + reward_delay) + [1.0]
    assert not obs["action_mask"].any()
    with pytest.raises(RuntimeError):
        env.step(Action.WAIT)


@pytest.mark.parametrize("variant", ["delayed_cue", "visual"])
@pytest.mark.parametrize("split", list(LAYOUTS))
def test_opposite_cues_have_identical_history_free_decision_inputs(variant, split):
    for layout in LAYOUTS[split]:
        observations = []
        actions = []
        for cue in (0, 1):
            env = CueTaskEnv(TaskConfig(variant=variant, split=split))
            initial, _ = env.reset(
                seed=SEED_RANGES[split].start, options={"layout_id": layout.id, "cue": cue}
            )
            obs = initial
            for _ in range(env.config.cue_delay):
                obs, _, _, _, info = env.step(Action.WAIT)
                assert info == {}
            observations.append((initial, obs))
            actions.append(env.diagnostic_action())
        key = "image" if variant == "visual" else "features"
        assert not np.array_equal(observations[0][0][key], observations[1][0][key])
        for key in observations[0][1]:
            np.testing.assert_array_equal(observations[0][1][key], observations[1][1][key])
        assert set(actions) == {Action.LEFT, Action.RIGHT}


@pytest.mark.parametrize("variant", ["full_state", "delayed_cue", "visual"])
def test_wrong_choice_and_no_pre_payout_correctness_leak(variant):
    histories = []
    for cue in (0, 1):
        env = CueTaskEnv(TaskConfig(variant=variant, cue_delay=1, reward_delay=3))
        env.reset(seed=0, options={"layout_id": "pilot-a", "cue": cue})
        env.step(Action.WAIT)
        history = []
        for action in (Action.LEFT, Action.WAIT, Action.WAIT, Action.WAIT):
            obs, reward, term, trunc, info = env.step(action)
            assert not trunc
            if term:
                assert reward == float(cue == 0)
                assert info == {"success": cue == 0}
            else:
                assert reward == 0 and info == {}
                history.append(obs)
        histories.append(history)
    if variant != "full_state":
        for first, second in zip(*histories, strict=True):
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize("limit", [1, 4, 5, 7, 8, 9])
def test_time_limit_retains_final_observation_and_true_end_takes_precedence(limit):
    env = CueTaskEnv(TaskConfig(cue_delay=4, reward_delay=3, time_limit=limit))
    reference = CueTaskEnv(TaskConfig(cue_delay=4, reward_delay=3))
    env.reset(seed=17)
    reference.reset(seed=17)
    for _ in range(min(limit, 8)):
        action = env.diagnostic_action()
        obs, reward, term, trunc, info = env.step(action)
        ref_obs, ref_reward, ref_term, _, ref_info = reference.step(action)
        for key in obs:
            np.testing.assert_array_equal(obs[key], ref_obs[key])
        assert (reward, term, info) == (ref_reward, ref_term, ref_info)
    assert term == (limit >= 8)
    assert trunc == (limit < 8)
    if trunc:
        assert obs["action_mask"].any()  # Valid continuation for value bootstrapping.
        assert info == {}  # An interrupted success is not a completed success.
    with pytest.raises(RuntimeError):
        env.step(Action.WAIT)
    obs, _ = env.reset(seed=17)
    assert obs["features"][CUE].sum() == 1


def test_mask_errors_do_not_advance_state_and_observations_are_not_mutated():
    env = CueTaskEnv()
    with pytest.raises(RuntimeError):
        env.step(Action.WAIT)
    obs, _ = env.reset(seed=0)
    saved = {key: value.copy() for key, value in obs.items()}
    for action in (Action.LEFT, -1, 3, 0.5):
        with pytest.raises(ValueError):
            env.step(action)
    assert env.diagnostic_action() == Action.WAIT
    env.step(Action.WAIT)
    for key in saved:
        np.testing.assert_array_equal(obs[key], saved[key])


def test_split_layouts_and_seeds_are_disjoint_and_enforced():
    for a, first in enumerate(LAYOUTS):
        for second in list(LAYOUTS)[a + 1 :]:
            assert set(LAYOUTS[first]).isdisjoint(LAYOUTS[second])
            assert set(SEED_RANGES[first]).isdisjoint(SEED_RANGES[second])
    env = CueTaskEnv(TaskConfig(split="train"))
    with pytest.raises(ValueError):
        env.reset(seed=0)
    with pytest.raises(ValueError):
        env.reset(seed=10_000, options={"layout_id": "test-a"})
    with pytest.raises(ValueError):
        env.reset(seed=10_000, options={"cue": 2})
    with pytest.raises(ValueError):
        episode_seeds("pilot", 1001)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variant": "invalid"},
        {"split": "invalid"},
        {"cue_delay": 0},
        {"cue_delay": 65},
        {"reward_delay": -1},
        {"reward_delay": 1.5},
        {"time_limit": 0},
        {"time_limit": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TaskConfig(**kwargs)


def test_vector_environment_accepts_all_three_observation_types():
    for variant in ("full_state", "delayed_cue", "visual"):
        env = gym.vector.SyncVectorEnv(
            [
                lambda variant=variant: CueTaskEnv(TaskConfig(variant=variant, cue_delay=1))
                for _ in range(2)
            ]
        )
        obs, _ = env.reset(seed=[11, 12])
        obs, _, _, _, _ = env.step(np.array([0, 0]))
        obs, _, term, _, _ = env.step(np.array([1, 2]))
        assert term.all()
        env.close()
