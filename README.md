# Hopfield RL

A research project for studying memory, retrieval, and delayed credit assignment
in recurrent RL controllers. Milestone M01 implements a controlled cue-matching
task suite and reproducible diagnostic evaluation.

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required. On PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv sync --locked
uv run pytest -q
uv run ruff check src tests
```

The environment uses Gymnasium and NumPy; M01 runs on CPU.

## Watch the game

```powershell
uv run --extra preview python preview.py
```

This opens a window showing one automatic episode: cue, waiting, choice, then
delayed reward. The final frame stays open until you close the window. Matplotlib
is installed only when the optional `preview` extra is selected.

Each launch uses a random pilot seed, which is printed for reproducibility.
Add `--seed 12` (or another integer from 0–999) to replay a fixed episode.

Change the timing with, for example:

```powershell
uv run --extra preview python preview.py --cue-delay 8 --reward-delay 5 --interval 0.5
```

See [the viewer guide](docs/preview.md) for options, visual markers, and checks.

## Run M01

```powershell
uv run python -m hopfield_rl.evaluate --episodes 200 --splits pilot held_out --output experiments/m01-reproduction
```

This writes a JSON experiment record and compressed per-episode trajectories.
Use a new output directory to preserve earlier runs. Without `--splits`, the
runner evaluates only the held-out partition.

See [the comprehensive M01 guide](docs/m01.md) for task semantics, policy inputs,
state resets, timing conventions, results, and limitations. The retained run is
in [experiments/m01/record.json](experiments/m01/record.json). Use
[the experiment record template](docs/experiment_record_template.md) for later
milestones. The original research plan is at
`literature_review/implementation_plan.md`.
