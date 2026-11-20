# Running the simulation and choosing plots

Run the `mission_mujoco` project explicitly: the environment may have the
separate `mission_mujoco_linear` project installed under the same package name.
Using `python -m` from the directory below selects the local project.

## Save only executed Cartesian position and velocity

```bash
cd /home/harsh/mujoco_source_only/mission_mujoco

MUJOCO_GL=glfw OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mission-mpl \
/home/harsh/anaconda3/envs/mujoco/bin/python -m mission_mujoco.cli \
  --task color --rounds 1 --epsilon 0 --video \
  --out results/local/color_executed \
  --plot-mode executed-cartesian final-grounding
```

This saves **two PNGs per action**, each with three panels:

- `<color>_block<ID>_executed_position_xyz.png`: the whole executed XYZ position history.
- `<color>_block<ID>_executed_cartesian_velocity_xyz.png`: the whole executed XYZ velocity history.

“Final executed” means the accumulated motion over the action, not just its last
position or velocity value. These figures are written when the action completes
or fails; during execution the samples are accumulated in memory. A failure
inside a physics interval retains its partial motion. An action with no executed
motion produces no trajectory PNGs. A forcibly killed process may not get to
write its action summaries.

There are no per-MPC plot folders, predicted curves, joint plots, noise plots,
or grounding plots in this mode. Normal action metadata and the run's numerical
JSON results remain enabled. XYZ velocities use the TCP Jacobian and simulated
joint velocities. Cartesian position targets are references; this controller
has no hard Cartesian position or Cartesian velocity component bounds.

You can combine selections in one option:

```bash
--plot-mode executed-cartesian final-grounding
```

This saves executed XYZ position/velocity summaries **and** final grounding
plots for each action, without per-MPC plot folders. `final-grounding` is one
value (not separate `final` and `grounding` values). `none` cannot be combined
with other modes; `all` includes all plot types. For Python calls, pass
`plot_mode=["executed-cartesian", "final-grounding"]`.

## Plot modes

Append one of these options to the command:

| Option | Saved plots |
|---|---|
| `--plot-mode all` | Every MPC solve, action summaries, and grounding evaluations; default |
| `--plot-mode executed-cartesian` | Only executed Cartesian position and velocity summaries for each action |
| `--plot-mode final-grounding` | Only final grounding for each action and each completed round |
| `--plot-mode none` | No diagnostic plots |
| `--no-plots` | Also disables diagnostic plots; overrides `--plot-mode` |

Plot selection does not disable MPC, sensor noise, physical guards, or symbolic
planning. The two summary-only modes skip the additional shadow forecasts used
by the full plot logger.

## Output folders and saving times

For the example command:

```text
results/local/color_executed/plots/<session>/
  moveEmpty/action_0000_red_block0_to_None/
    action.json
    red_block0_executed_position_xyz.png
    red_block0_executed_cartesian_velocity_xyz.png
  pickLift/action_0001_red_block0_to_p0c0/
    ...
  moveHolding/...
  placeRelease/...
```

`moveEmpty` is an action type. Each `action_*` directory identifies a separate
execution and block. `None` means MoveEmpty has no destination-support argument.
Session directories preserve results from previous runs. In `all` mode each
`mpc_*` subfolder represents one feedback solve inside its action, not a separate
action. Full plots save during execution; summary-only plots save at action end.
The consolidated `summary.json`, `trajectory.json`, `mpc.json`, etc. save when
the run ends normally or through a caught error. Video saves at run end.

## Noise parameters

To enable measurement mismatch, append:

```bash
--joint-observation-bias .004 --joint-observation-noise .00015 \
--observation-bias .001 --observation-noise .0001
```

| Parameter | Default | Meaning |
|---|---|---|
| `--joint-observation-bias` | `0` | Fixed observation bias in radians, applied to each of the six arm joints |
| `--joint-observation-noise` | `0` | Gaussian joint-angle noise standard deviation in radians |
| `--observation-bias` | `0` | Fixed block-position observation bias along x, in metres |
| `--observation-noise` | `0` | Gaussian block-position noise standard deviation for XYZ, in metres |

Joint noise is sampled once per MPC solve; block noise once per grounding call.
These scalar settings apply uniformly within each channel. They change measured
states, not physical states. Joint velocities, block orientations, contacts,
and completion guards still use simulator truth. Change CLI defaults in
`mission_mujoco/cli.py`; Python `World` defaults are in `Config` in
`mission_mujoco/physics.py`. CLI options are passed explicitly to `Config`.
Restart the process to use changes.

## Other run parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--task` | `color` | `color`, `letter`, `b0`, or `b1` |
| `--rounds` | `1` | Number of rounds; B0/B1 require one |
| `--seed` | `0` | Reproducible random seed |
| `--epsilon` | `0.1` | Planning exploration probability; use `0` for no exploration |
| `--out` | `results/run` | Output directory; relative paths are relative to the launch directory |
| `--video` | Off | Save video; requires a working rendering backend |
| `--verbose` | Off | Save labeled function/debug traces to `<out>/debug_trace.log` |
| `--mass-scale` | `1` | Physical cube mass scale; predictor remains nominal |
| `--friction` | `1` | Physical scene friction parameter; predictor remains nominal |
| `--impratio` | `1` | Contact impedance ratio |
| `--support-solref` | `0.02` | Cube support-contact time constant in seconds; CLI requires at least `0.004` |

`MUJOCO_GL=glfw` selects rendering with a desktop display. Other backends require
appropriate system libraries. For plots without video, remove `--video` and the
`MUJOCO_GL` assignment. `OPENBLAS_NUM_THREADS=1` controls numerical-library
threading, and `MPLCONFIGDIR` selects a writable Matplotlib cache directory.

Show all options:

```bash
/home/harsh/anaconda3/envs/mujoco/bin/python -m mission_mujoco.cli --help
```

For Python calls:

```python
from mission_mujoco.cli import run
run(out="results/local/executed", plot_mode="executed-cartesian")

# Or for an existing World and Models instance:
from mission_mujoco.execution import Executor
executor = Executor(world, models, plot_dir="results/local/plots",
                    plot_mode="executed-cartesian")
```

## Verify waypoint completion and timeouts

Append `--completion-plots` to any enabled plot selection, for example:

```bash
--plot-mode executed-cartesian final-grounding --completion-plots
```

Each waypoint attempt saves a three-panel completion plot and JSON in the
action folder, on reaching that waypoint or on timeout. The panels show actual
position error (mm), actual joint-speed norm (rad/s), and consecutive accepted
checks. The defaults in `MPC.reach` in `mission_mujoco/controller.py` are:

- Position error strictly below 2.5 mm.
- Joint-speed norm strictly below 0.035 rad/s.
- Both conditions met for three consecutive control steps.
- A 12-second timeout (240 intervals at dt=0.05 s).

These are hard completion checks, not fuzzy membership functions. A failed
check resets the consecutive count to zero. They use simulator truth, while
MPC optimizes from noisy joint measurements. Completion metrics are also stored
in each reach solve's record in `mpc.json`, even when completion plotting is off.
The flag adds completion images to the selected plot types; `--no-plots` or
`--plot-mode none` disables them too. It does not change any acceptance threshold.

## Simplified two-link notebooks: one trajectory to above the block

- [Closed-loop tracking](toy_problems/two_link_mpc_toy.ipynb)
- [Open-loop tracking](toy_problems/two_link_open_loop_mismatch_toy.ipynb)

Both now follow the **same single Cartesian reference** from `(0.50, 1.10)` m
straight through the supplied smooth curved path to `(target_x, hover_y)`, directly
above the block center. There is no separate horizontal alignment problem, no
measurement-defined height reference, and no alignment phase. The endpoint is
repeated for `hold_steps` using the same tracking objective. Default reference
length is 100 samples followed by 20 endpoint repetitions, with dt=0.1 s.

Open-loop makes **one optimization call** for the complete 119-command schedule,
starting from its noisy initial observation, then applies all fixed commands to
its true state. Closed-loop repeatedly solves the **same tracking problem** with
an 8-step rolling horizon and a fresh measurement, applying only the first
command of each solution. It also retains a separate noise-free baseline.
These strategies have different horizons; they are not two identical optimization
problems differing only in whether future controls are applied.

Both use identical geometry, bounds, references, and seeded measurement-error
schedules. Measurements do not relocate the physical robot. Current sensor
settings were preserved from the closed-loop notebook and copied to open-loop.
If you want a large first error followed by small errors, set the four arrays
as in the section below; the notebooks do not force that relationship.

Run top to bottom with NumPy, SciPy, Matplotlib and IPython. MuJoCo is not needed.
The full open-loop solve may take a few minutes. Figures show actual-minus-reference
tracking errors, endpoint x/y with fixed final-goal lines, joint/Cartesian velocities,
measurement errors, final grounding, and optional animation. Final distance and a
1 mm goal check are printed; completing the fixed execution schedule alone is
not a claim that the goal has been reached within tolerance. The same solver
checks constraints; actual motion is diagnosed separately and is not projected
onto the model prediction.

## Toy notebooks: large first measurement error, smaller later errors

Both two-link notebooks now expose the same parameters:

```python
initial_state_bias = np.array([0.02, -0.015])  # first reading only [rad]
initial_noise_std = np.array([0.001, 0.001])   # first-reading noise std [rad]
state_bias = np.array([0.0001, -0.0001])       # later readings [rad]
noise_std = np.array([0.00015, 0.00015])       # later noise std [rad]
mismatch_seed = 7
```

The first observation uses `initial_state_bias` and `initial_noise_std`. Every
later observation uses `state_bias` and `noise_std`. No physical state is moved
by these measurement errors. Both notebooks generate the same observation-error
samples at matching time indices; the different horizons can give different total
schedule lengths. The closed-loop baseline remains noise-free.

Closed-loop MPC uses each fresh reading to replan. Open-loop
optimizes once using the large first-reading error and ignores the improved
readings for control. Its later readings remain available in the plots and logs.
Both final grounding observations use the pre-drawn sample at the final executed time index.

Restart the notebook kernel and run from the top after changing these parameters.
Both use one reference schedule and the same stage cost and constraint families.
The default later noise was reduced to make the requested initial-outlier experiment
explicit; you can edit all four arrays independently. Because full-horizon and
rolling-horizon controllers have different horizons, use actual reference/goal
tracking errors to assess recovery rather than treating every difference from a
nominal prediction as the same performance metric.

### Physical arm-position kicks

`mission_mujoco/physics.py` → `Config` defines `joint_kick_half_range` (default
`0.0` rad) and `joint_kick_bias` (default `0.0` rad). Random kicks are disabled
by default. If explicitly enabled, each completed
`World.step()` control interval receives one independent uniform position
increment per arm joint in `[bias - half_range, bias + half_range]`.
For half-range `0.001` and zero bias this is `[-0.001, 0.001]` rad; its standard
deviation is `0.001 / sqrt(3)` rad. These are position
increments per interval, not velocity noise; they are not multiplied by `dt`.
For example, add `--joint-kick-half-range 0.001 --joint-kick-bias 0.0` to the usual CLI
command. Set both to zero for undisturbed execution. The CLI defaults come from
`Config`; the run manifest records the chosen values, and each physical trace
entry records its six-component `joint_kick`.

Kicks change only the six arm joint positions after physical integration, then
refresh MuJoCo kinematics and contacts. They do not directly change joint velocities,
finger positions, or block positions. They are not applied during initialization,
IK, forecasts, or optimizer rollouts. The independent kick RNG uses the run seed,
so solver evaluation counts do not alter the kick sequence. No projection or
clipping hides a kick; existing contact guards also check the post-kick state.
Diagnostics include a post-kick sample at the same timestamp as the pre-kick
endpoint. Instantaneous jumps have no finite modeled Cartesian velocity.

This is closed-loop MPC: `MPC.reach()` and `MPC.hold()` repeatedly observe,
optimize, and execute one interval. A single velocity vector is optimized and
held across the short prediction horizon (move blocking); after execution, a new
solve starts from the latest actual state. Predictions assume zero future kicks.
Joint and block observation noise remain separately configurable and default to
zero. Leave those settings at zero for the exact-observation kick experiment.
MuJoCo servo/contact dynamics still apply, so the plant is not the toy model's
exact joint-position integrator.
