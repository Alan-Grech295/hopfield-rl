"""Watch one automatic, correct episode using the simulator's diagnostic policy."""

import argparse
from math import isfinite
from secrets import choice

from hopfield_rl.env import Action, CueTaskEnv, TaskConfig
from hopfield_rl.splits import SEED_RANGES


def preview(
    *,
    cue_delay: int = 4,
    reward_delay: int = 3,
    seed: int | None = None,
    interval: float = 1.0,
    keep_open: bool = True,
) -> float | None:
    """Return the final reward, or None if the viewer was closed before payout."""
    if not isfinite(interval) or interval <= 0:
        raise ValueError("interval must be a finite positive number of seconds")
    # Import the optional dependency only when someone actually opens the viewer.
    import matplotlib.pyplot as plt

    env = CueTaskEnv(
        TaskConfig(
            variant="visual",
            cue_delay=cue_delay,
            reward_delay=reward_delay,
        )
    )
    fig = None
    try:
        if seed is None:
            seed = choice(SEED_RANGES["pilot"])
        env.reset(seed=seed)
        print(f"Episode seed: {seed}")
        with plt.ion():
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.canvas.manager.set_window_title(f"Cue-matching game | Seed: {seed}")
            image = ax.imshow(env.render(), interpolation="nearest")
            ax.axis("off")
            ax.set_title("CUE: remember this color")
            plt.pause(1.5 * interval)

            reward = None
            while plt.fignum_exists(fig.number):
                action = env.diagnostic_action()  # Privileged demo, not a trained policy.
                _, step_reward, terminated, truncated, _ = env.step(action)
                image.set_data(env.render())
                ax.set_title(
                    f"{env.phase.name} | Action: {Action(action).name} | Reward: {step_reward:g}"
                )
                fig.canvas.draw_idle()
                plt.pause(interval)
                if terminated or truncated:
                    reward = step_reward
                    break

        # Keep the result visible until the user closes it; never reopen a closed window.
        if keep_open and plt.fignum_exists(fig.number):
            plt.show(block=True)
        return reward
    finally:
        env.close()
        if fig is not None:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cue-delay", type=int, default=4, help="wait steps before the choice (1-64)"
    )
    parser.add_argument(
        "--reward-delay", type=int, default=3, help="wait steps after the choice (0-64)"
    )
    parser.add_argument(
        "--seed", type=int, help="fixed pilot episode seed (0-999); random when omitted"
    )
    parser.add_argument("--interval", type=float, default=1.0, help="seconds per animation step")
    parser.add_argument("--no-wait", action="store_true", help="close automatically after payout")
    args = parser.parse_args()
    try:
        reward = preview(
            cue_delay=args.cue_delay,
            reward_delay=args.reward_delay,
            seed=args.seed,
            interval=args.interval,
            keep_open=not args.no_wait,
        )
    except ValueError as error:
        parser.error(str(error))
    print("Viewer closed before payout." if reward is None else f"Final reward: {reward:g}")


if __name__ == "__main__":
    main()
