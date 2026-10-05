# Cue-matching game viewer

`preview.py` opens a Matplotlib window and animates one complete visual episode.
It runs on CPU and uses the existing task renderer. The simulator's diagnostic
policy chooses the correct card automatically; this is a demonstration of the
task, not a trained policy or an interactive game you control.

## Run it

From the repository root in PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run --extra preview python preview.py
```

uv installs the locked optional Matplotlib dependency and its dependencies in
the selected environment. No manual activation is needed. Core training/task
commands do not require the `preview` extra.

The default episode shows a cue, waits four steps, selects its matching card,
then waits three steps for reward. Each step is visible for one second; the
initial cue stays visible for 1.5 seconds. The final reward is printed to the
terminal after the window is closed.

Close the window at any time to stop playback. At the end, the final frame stays
open until you close it. To close automatically after payout, use `--no-wait`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--cue-delay` | 4 | Environment wait steps before choosing; integer 1–64 |
| `--reward-delay` | 3 | Environment wait steps after choosing; integer 0–64 |
| `--seed` | 0 | Reproducible pilot episode seed; integer 0–999 |
| `--interval` | 1.0 | Positive, finite seconds per animation step |
| `--no-wait` | off | Close automatically after the terminal frame |

```powershell
# Longer memory and reward delays, with faster playback.
uv run --extra preview python preview.py --cue-delay 8 --reward-delay 5 --interval 0.5

# A different cue/layout sample and immediate reward on the choice.
uv run --extra preview python preview.py --seed 12 --reward-delay 0

# List all options.
uv run --extra preview python preview.py --help
```

Animation seconds are separate from simulated environment steps. Changing
`--interval` changes viewing speed, not memory demand, reward delay, or rewards.

## What you see

- The colored square near the top is the cue. Remember it; it disappears after
  the first observation.
- The left and right colored cards remain visible throughout the episode.
- The title shows the phase, the action just taken, and the reward just received.
  DECISION means the next action will select a card.
- A white marker below a card indicates the committed choice.
- The top row of white markers shows the phase.
- The lower two bars encode remaining cue and payout steps, on a fixed 64-step
  scale. Short delays therefore appear as short bars.
- TERMINAL with reward 1 means the automatic policy matched the cue correctly.

The display enlarges the original 96×96 RGB frame without smoothing it.
The title is viewer metadata, not an additional policy observation. The viewer
always uses the visual, delayed-cue variant and the pilot partition. To inspect
other observation schemas, use the environment API in [the M01 guide](m01.md).

## Verification and troubleshooting

```powershell
# All task and viewer tests, plus formatting and lint checks.
uv run --extra preview pytest -q
uv run --extra preview ruff check src tests preview.py
uv run --extra preview ruff format --check src tests preview.py

# Short real-window smoke run; closes itself after reward.
uv run --extra preview python preview.py --interval 0.05 --no-wait
```

Viewer tests use Matplotlib's non-GUI Agg backend. They check successful payout,
early window closure, resource cleanup after playback errors, and invalid timing
arguments. These tests are skipped when the optional dependency is absent.
The real-window smoke command separately exercises the desktop GUI backend.

Verified on Windows 11 with Python 3.13.5 and Matplotlib 3.11.2: all 57 task and
viewer tests passed, formatting and lint checks passed, and the real-window smoke
run used TkAgg and completed with `Final reward: 1`.

If a window does not appear, run from a local desktop terminal and check that
`MPLBACKEND` is not set to `Agg`, which cannot open windows. Matplotlib needs a
working GUI backend such as TkAgg. See its
[interactive figure documentation](https://matplotlib.org/stable/users/explain/figure/interactive.html).
The viewer's pause calls process GUI events between frames, as described in the
[pause documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pause.html).

The small root script calls `src/hopfield_rl/preview.py`, which is also runnable
as `uv run --extra preview python -m hopfield_rl.preview`. Importing the module
does not start playback or import Matplotlib; the optional dependency is loaded
when the viewer function is called.
