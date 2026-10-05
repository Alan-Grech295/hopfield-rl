import pytest

# The viewer is optional: core task tests still work without Matplotlib installed.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hopfield_rl.env import CueTaskEnv  # noqa: E402
from hopfield_rl.preview import preview  # noqa: E402


def test_complete_episode_returns_reward_and_releases_figure(monkeypatch):
    monkeypatch.setattr(plt, "pause", lambda interval: plt.gcf().canvas.draw())
    was_interactive = plt.isinteractive()
    assert preview(interval=0.01, keep_open=False) == 1.0
    assert not plt.get_fignums()
    assert plt.isinteractive() == was_interactive


def test_closing_window_before_first_step_does_not_reopen_it(monkeypatch):
    monkeypatch.setattr(plt, "pause", lambda interval: plt.close("all"))

    def unexpected_step(self, action):
        pytest.fail("closed viewer must not advance the environment")

    def unexpected_show(*args, **kwargs):
        pytest.fail("closed viewer must not reopen a window")

    monkeypatch.setattr(CueTaskEnv, "step", unexpected_step)
    monkeypatch.setattr(plt, "show", unexpected_show)
    assert preview(interval=0.01) is None
    assert not plt.get_fignums()


def test_playback_error_still_closes_environment_and_window(monkeypatch):
    closed = []
    monkeypatch.setattr(plt, "pause", lambda interval: None)
    monkeypatch.setattr(CueTaskEnv, "close", lambda self: closed.append(True))

    def broken_step(self, action):
        raise RuntimeError("playback failed")

    monkeypatch.setattr(CueTaskEnv, "step", broken_step)
    with pytest.raises(RuntimeError, match="playback failed"):
        preview(interval=0.01, keep_open=False)
    assert closed == [True]
    assert not plt.get_fignums()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_interval_is_rejected_before_creating_window(interval):
    with pytest.raises(ValueError, match="finite positive"):
        preview(interval=interval)
    assert not plt.get_fignums()
