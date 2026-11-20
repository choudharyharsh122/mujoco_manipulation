# Running mission_mujoco: commands and options

## Environment and working directory

Create a dedicated environment once:

```bash
conda create -n mission-mujoco python=3.11 -y
conda activate mission-mujoco
cd /home/harsh/mujoco_source_only/mission_mujoco
python -m pip install -r requirements.lock
python -m pip install -e .
```

For subsequent sessions:

```bash
conda activate mission-mujoco
cd /home/harsh/mujoco_source_only/mission_mujoco
```

Run with `python -m mission_mujoco.cli` to use this environment's Python and the
local package. Relative output paths are relative to this working directory.
`mission-sim` is an installed shortcut for the same CLI, but another environment
may contain an older installation of that shortcut.

## Start here: scheduled kicks and joint plots

```bash
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli \
  --task color \
  --rounds 1 \
  --seed 0 \
  --epsilon 0 \
  --impratio 10 \
  --support-solref .01 \
  --verbose \
  --out results/local/color_debug \
  --plot-mode joint-prediction \
  --kick-timestamps 2,5,7 \
  --kick-magnitude "(1e-3,0,0), (1e-3,1e-4,0)"
```

This runs the color task with exact observations by default. At 2 seconds it
adds `(0.001, 0, 0, 0, 0, 0)` radians to the arm joints. At 5 seconds it adds
`(0.001, 0.0001, 0, 0, 0, 0)` radians. The missing third vector is zero, so the
7-second event causes no displacement. No random kicks occur in this experiment.
It saves action-level joint prediction and one-step error figures, plus a debug log.

`OPENBLAS_NUM_THREADS=1` is an environment variable, not a CLI option: it limits
OpenBLAS to one thread. A trailing `\` continues the shell command on the next line.

## Other useful commands

No physical perturbations, with final grounding plots:

```bash
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli \
  --task color --epsilon 0 --impratio 10 --support-solref .01 \
  --out results/local/color_baseline --plot-mode final-grounding
```

Cartesian XYZ positions of all six joint anchors, with a scheduled kick:

```bash
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli \
  --task color --epsilon 0 --impratio 10 --support-solref .01 \
  --out results/local/color_cartesian --cartisian-prediction \
  --kick-timestamps 2 --kick-magnitude "(1e-3,0,0,0,0,0)"
```

Combine angular and Cartesian prediction figures with final grounding:

```bash
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli \
  --task color --epsilon 0 --impratio 10 --support-solref .01 \
  --out results/local/color_combined \
  --plot-mode joint-prediction cartesian-prediction final-grounding \
  --kick-timestamps 2,5 --kick-magnitude "(1e-3,0,0), (0,-1e-3,0)"
```

Observation noise only (this changes readings, not physical joint positions):

```bash
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli \
  --task color --epsilon 0 --impratio 10 --support-solref .01 \
  --joint-observation-noise 1e-4 --joint-observation-bias 0 \
  --out results/local/color_sensor_noise --plot-mode joint-prediction
```

Optional ongoing uniform kicks, instead of a schedule:

```bash
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli \
  --task color --epsilon 0 --impratio 10 --support-solref .01 \
  --joint-kick-half-range 1e-3 --joint-kick-bias 0 \
  --out results/local/color_random --plot-mode joint-prediction
```

Video on a Linux machine with EGL support:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli \
  --task color --epsilon 0 --impratio 10 --support-solref .01 \
  --video --out results/local/color_video --plot-mode final-grounding
```

Video needs the optional dependencies: `python -m pip install -e '.[video]'`.
Without `--video`, physics and saved plots do not need a display.

## All command-line options

Defaults below match the current source. Values shown in examples can differ
from defaults; for instance, the examples explicitly choose `--impratio 10`.

### Task, output, and reproducibility

| Option | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Print the CLI options and exit. |
| `--task color\|letter\|b0\|b1` | `color` | Color task, letter task, or fixed B0/B1 benchmark fixture. |
| `--rounds N` | `1` | Number of task rounds. B0/B1 require exactly one round. |
| `--seed N` | `0` | Seed for random observations, optional random kicks, and planning/exploration randomness. |
| `--epsilon P` | `0.1` | Probability of restricting a round to a randomly selected candidate goal for exploration. Use `0` to disable this exploration. It is not a solver tolerance. |
| `--out PATH` | `results/run` | Directory for run data, plots, logs, and optional video. |
| `--verbose` | Off | Write detailed function/MPC/planning traces to `<out>/debug_trace.log`. |
| `--video` | Off | Render and save `<out>/rollout.mp4` and `<out>/final.png`. |

### Plot selection

| Option | Default | Meaning |
|---|---|---|
| `--plot-mode MODE [MODE ...]` | `all` | Select plots using the modes listed below. Multiple selections can be combined. |
| `--cartisian-prediction` | Off | Shortcut selecting only `cartesian-prediction`; overrides `--plot-mode`. |
| `--cartesian-prediction` | Off | Correctly spelled alias of the shortcut above. |
| `--completion-plots` | Off | Add waypoint completion figures showing position error, joint speed, and consecutive completion checks. |
| `--no-plots` | Off | Disable plot logging, including completion plots. Run JSON data is still saved. |

| Plot mode | Output |
|---|---|
| `all` | Detailed per-MPC forecasts, execution/constraint plots, action summaries, and grounding plots. This is the existing detailed mode; select the dedicated prediction modes below for their compact action-level layouts. |
| `final-grounding` | Grounding figures at final action/round checks, without per-MPC figures. |
| `executed-cartesian` | Two action-level figures: actual tool XYZ position and actual tool XYZ velocity. No MPC prediction overlays. |
| `joint-prediction` | One six-panel angular trajectory figure per action (orange MPC horizons, blue actual robot), plus one six-panel next-step angular prediction-error figure when completed intervals exist. |
| `cartesian-prediction` | One 18-panel figure per action: six joint anchors × XYZ world coordinates. Orange shows MPC horizons; blue shows execution. These are hinge locations, not the tool tip. |
| `none` | Disable plots. Cannot be combined with other modes. |

Use `--plot-mode joint-prediction final-grounding` to combine modes. Selecting
`all` takes precedence over the other mode names; it does not additionally enable
the dedicated compact prediction layouts. Avoid the Cartesian shortcut when you
want a combination, because the shortcut overrides `--plot-mode`.

### Scheduled physical kicks

| Option | Default | Meaning |
|---|---|---|
| `--kick-timestamps T1,T2,...` | No events | Nonnegative finite elapsed simulation times, in seconds after initialization. |
| `--kick-magnitude "(d1,d2,...), (d1,d2,...)"` | Zero vectors | One vector per event, with 1–6 finite arm joint-angle increments in radians. Missing components and missing event vectors become zero. More vectors than timestamps is an error. |

Quote the magnitude expression: shell parentheses are special characters.
Timestamps apply across the whole run, not separately for each action or round.
They are simulation times, not elapsed wall-clock time.

Each event is applied once after integration, at the first control-step endpoint
at or after its requested time. The default interval is `0.05` s, so `2.02` s
becomes `2.05` s. A timestamp of `0` applies at the first endpoint, not during
initialization. Unsorted timestamps are processed chronologically while preserving
each timestamp's associated vector. Events landing at the same endpoint add together.
Events beyond the end of the run are not applied.

These kicks change arm **joint positions**, not measurements. For example,
`(1e-3,0,0)` affects joint 1; it does not translate the tool by 1 mm along X.
Joint velocities, finger positions, and block positions are not directly rewritten.
Predictions assume no future kicks. Existing contact guards can still stop a run
if a kick produces an invalid contact or loses a grasp.

### Optional ongoing physical perturbations

| Option | Default | Meaning |
|---|---|---|
| `--joint-kick-half-range A` | `0` | Independent uniform angular kick in `[-A,A]` rad per arm joint per executed control interval. Must be finite and nonnegative. |
| `--joint-kick-bias B` | `0` | Add `B` radians to each joint's interval kick. With half-range zero and nonzero bias, this causes deterministic repeated increments. |

These legacy options are used only when there is **no timestamp schedule**.
A schedule replaces ongoing random/bias kicks, even if these options are nonzero.
With all kick options omitted, no physical kicks are applied.

### Observation noise

| Option | Default | Meaning |
|---|---|---|
| `--joint-observation-noise S` | `0` | Gaussian joint-angle observation standard deviation in radians, independently per arm joint. |
| `--joint-observation-bias B` | `0` | Fixed observation offset in radians applied to each arm joint. |
| `--observation-noise S` | `0` | Gaussian block-position observation standard deviation in metres for XYZ. |
| `--observation-bias B` | `0` | Fixed X-coordinate offset in metres for block observations. |

Noise standard deviations must be finite and nonnegative; biases must be finite.
These options affect readings rather than directly moving the robot or blocks.
Keep all four zero to isolate the effect of physical kicks. Velocity, rotation,
and contact channels remain simulator-derived.

### Physical parameters

| Option | Default | Meaning |
|---|---|---|
| `--mass-scale M` | `1.0` | Scale the physical block masses (nominal block mass is `0.08` kg). |
| `--friction F` | `1.0` | Set the sliding friction coefficient on blocks and gripper pads. |
| `--impratio I` | `1.0` | MuJoCo tangential-to-normal contact constraint impedance ratio. |
| `--support-solref S` | `0.02` | Cube contact time constant in seconds. CLI requires at least `0.004` s (twice the default integration timestep). |

Other parameters live in `mission_mujoco/physics.py` → `Config`, including the
control interval (`dt=0.05` s), physics timestep (`0.002` s), and command limits.
MPC horizon and solver iteration defaults are in `mission_mujoco/controller.py`.
Those are not CLI options.

## Where results go

For `--out results/local/color_debug`:

```text
results/local/color_debug/
  manifest.json                 # full configuration and source hashes
  summary.json                  # run status and round results
  trajectory.json               # actual states, commands, applied kicks/events
  mpc.json                      # optimizer records
  scene.xml                     # physical MuJoCo scene
  debug_trace.log               # with --verbose
  rollout.mp4                   # with --video
  final.png                     # with --video
  plots/<session-timestamp>/
    moveEmpty/action_0000_red_block0_to_None/
      action.json
      red_block0_joint_prediction.png
      red_block0_one_step_joint_error.png
      joint_prediction.json
    pickLift/action_0001_red_block0_to_p0c0/
      ...
```

Files depend on the selected plot modes. Cartesian prediction uses
`<color>_block<N>_cartesian_prediction.png` and `cartesian_prediction.json`.
Grounding figures include `<color>_block<N>_area_height_orientation.png`,
`<color>_block<N>_combined_memberships.png`, and `grounding.json`.
The printed run summary includes `plot_directory` with the exact session path.
Plot sessions have separate timestamp directories; use a distinct `--out` per
experiment to preserve top-level JSON files, which otherwise get overwritten.

The next-step angular error is `actual[k+1] - predicted[1|k]`, at equal physical
times after the kick. It includes sensor error and model mismatch if those are
present. Orange horizon tails are forecasts under a held command; later replanning
can make execution differ from those unused tails even without noise.

## Check available options

```bash
python -m mission_mujoco.cli --help
```

If an installed `mission-sim` command does not recognize an option, run the
`python -m` form from the project directory in your activated environment.

Action-level angular prediction, one-step angular error, Cartesian joint-anchor
prediction, and executed Cartesian plots mark nonzero scheduled kicks with **red
dashed vertical lines**. Markers use the actual application time relative to the
start of the plotted action; requested global run timestamps may therefore differ
from the displayed X coordinate. Zero-magnitude events are not marked. Prediction
JSON files include the applied `kick_events`. Existing saved PNGs need a new run
to show these markers.
