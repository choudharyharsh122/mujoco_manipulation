# Mission-aligned manipulation in MuJoCo

For step-by-step Git initialization and GitHub upload commands, see [GIT_README.md](GIT_README.md).

For commands, scheduled kicks, plot modes, and every CLI option, see [CLI_README.md](CLI_README.md).

See [Running and plot parameters](RUNNING_AND_PLOTS.md) for commands, noise settings, and plot modes.

A self-contained simulation repository for the manuscript's four action types,
classical replanning, fuzzy support grounding and upward reward propagation.
Objects remain dynamically free and are held by physical finger contacts.

**Read [CONFORMANCE.md](CONFORMANCE.md) before treating this as a reproduction
of the complete manuscript.** The executable pilot retains the task rules and
rigid-body kinematics, but uses regularized MuJoCo contacts and a restricted
shooting MPC. The full hard-contact OCP, horizon-wide wrench/servo certificates,
contact-sensitive KKT derivatives and hardware calibration are not implemented.

**Verification status: the full four-task protocol currently fails.** See `VERIFICATION.md` for per-task outcomes and the retained failures. Do not treat this release as a fully verified manuscript reproduction.

## Install and run

Python 3.11+ is required; the recorded environment uses Python 3.12. For the
verified dependency versions:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
python -m pytest -q
```

MuJoCo runs without a display if video is omitted. For video on a Linux machine
with EGL support:

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 mission-sim --task color --rounds 3 --epsilon 0 --video --out results/local/color
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 mission-sim --task letter --rounds 2 --epsilon 0 --impratio 10 --support-solref .01 --video --out results/local/letter
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 mission-sim --task b0 --epsilon 0 --impratio 10 --support-solref .01 --video --out results/local/b0
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 mission-sim --task b1 --epsilon 0 --impratio 10 --support-solref .01 --video --out results/local/b1
```

Add `--verbose` to any `mission-sim` command to save function entry/exit,
compact input/output summaries, symbolic state handovers, grounding, rewards,
and each MPC receding-horizon iteration to `<out>/debug_trace.log` instead of
printing the debug trace to the terminal. Entries label the function, file, task,
passed arguments and their types, current attributes, returns, and errors. For example:

```sh
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli --task color --epsilon 0 --impratio 10 --support-solref .01 --verbose --out results/local/b0_debug
```

For Python calls, use `mission_mujoco.cli.run(..., verbose=True)`.
The single `verbose` flag defaults to `False`; omit `--verbose` to disable the
additional tracing. Each verbose run overwrites `debug_trace.log` in its output
folder; choose a different `--out` folder to preserve an earlier trace. Existing
progress and summary output remains in the terminal. Trace values
are abbreviated for readability. Each MPC iteration predicts three intervals
by default and applies one; it is not a new symbolic action. Trace files can
be large and writing them can affect wall-clock timings.

## Incremental MPC and grounding plots

Plots are enabled by default, including without `--verbose` or `--video`.
Every motion and hold solve saves PNG figures and a `solve.json` record below
`<out>/plots/<session>/`. A fresh session preserves earlier results when an
output directory is reused. Python `Executor` calls default to `results/plots`;
pass `plot_dir=...` to choose another directory or `plot_dir=None` to disable.
Use `--no-plots` for runs where logging overhead is unwanted.

You can combine selections in one option:

```bash
--plot-mode executed-cartesian final-grounding
```

This saves executed XYZ position/velocity summaries **and** final grounding
plots for each action, without per-MPC plot folders. `final-grounding` is one
value (not separate `final` and `grounding` values). `none` cannot be combined
with other modes; `all` includes all plot types. For Python calls, pass
`plot_mode=["executed-cartesian", "final-grounding"]`.

Choose the amount of plotting with `--plot-mode`:

- `--plot-mode all` (default): per-MPC plots, action summaries, and grounding plots.
- `--plot-mode final-grounding`: only the final grounding plots for each action
  and at each round's terminal check. No MPC, trajectory, initial-grounding or
  intermediate dwell plots are generated. Action plots are saved directly inside
  the action folder, without `mpc_*` or intermediate `grounding_*` subfolders.
- `--plot-mode executed-cartesian`: only the full executed XYZ position and velocity
  summaries, saved as two PNGs per action at completion or failure. No MPC or
  grounding plot folders are generated.
- `--plot-mode none` or `--no-plots`: disable all diagnostic plots.

For example, append `--plot-mode final-grounding` to your existing command.
For Python calls, use `run(..., plot_mode="final-grounding")` or
`Executor(..., plot_mode="final-grounding")`. Failed actions retain a final
state grounding when available, clearly labeled by their failed action record.

A symbolic action contains many feedback solves, sometimes across several
phases (retreat, transit, pregrasp, etc.). `moveEmpty` groups that action type;
`action_0000_red_block0_to_None` identifies one action instance and block;
`mpc_000000_retreat` identifies one feedback solve. Repeated actions for different
blocks need distinct action folders to preserve their results. `None` means that
MoveEmpty has no destination support argument. The final-grounding mode avoids
the per-solve folder expansion.

Artifacts are grouped as:

```text
plots/<session>/
  initial/grounding_000000_initial/
  moveEmpty/action_0000_red_block0_to_None/
    action.json
    mpc_000000_retreat/
      solve.json
      red_block0_position_xyz.png
      red_block0_cartesian_velocity_xyz.png
      red_block0_joint_angles_1-3.png
      red_block0_joint_angles_4-6.png
      red_block0_joint_velocity_1-3.png
      red_block0_joint_velocity_4-6.png
      red_block0_joint_commands_1-3.png
      red_block0_joint_commands_4-6.png
      red_block0_constraint_status.png
      red_block0_observation_perturbation.png
      red_block0_one_step_joint_error.png
      red_block0_one_step_cartesian_error.png
    grounding_000001_pregrasp/
      grounding.json
      red_block0_area_height_orientation.png
      red_block0_combined_memberships.png
      ... (every unheld block)
    red_block0_executed_position_xyz.png
    ... (action trajectories, commands, perturbations, prediction errors)
  pickLift/...
  moveHolding/...
  placeRelease/...
  terminal/...
```

Names use the rendered block color and identity, so repeated colors and repeated
actions stay distinct. All figures have at most three panels. Solves are saved
immediately; after applying the first interval, their figures and JSON are
updated with executed motion sampled at every physics substep. Partial intervals
and failed actions are retained. Action summaries are saved at completion or
failure, with phase boundaries marked; all grounding evaluations, including
released dwell checks and rejection cases, save their memberships immediately.

Motion curves distinguish **executed true**, **MPC estimated forecast** (nominal
predictor from the observed joint angles), and **true-start forecast** (a copy of
the plant model starting at the true state, under the same optimized command).
The last curve is not an independently optimized baseline-policy run. Cartesian
velocities use the MuJoCo TCP Jacobian and simulated joint velocities, not a
finite difference of plotted positions. `solve.json` includes samples, true minus
predicted errors at equal physical times, optimizer status, and worst named
constraint residuals with their units and sample/joint index. Nonconverged
iteration-limit commands retain their actual solver status.

Joint plots mark the model's limited-joint ranges and the 1 rad/s actual-speed
guard. Command plots mark the intersection of the 0.7 rad/s command bound and
last-command slew bounds. The constraint-status image distinguishes violated,
active and inactive residuals (within 1e-5 of the boundary in that constraint's
units is marked active). Collision diagnostics use the action's intended-contact
exceptions. **This controller has no hard Cartesian position or velocity
component bounds:** position targets are dotted reference lines, not constraints.
The figures and JSON explicitly mark this distinction. Grasp/contact and support
acceptance still use the existing physical guards; plotting does not certify a
full hard-contact OCP.

Grounding figures plot the actual area-ratio, signed height-error, and orientation
taper curves, with markers for true and noisy observed geometry for every support
candidate. Combined-membership plots show entry/retention thresholds, and JSON
retains every factor, assignment gap, result and rejection reason. When a matching
one-step MPC forecast exists at the grounding time, its predicted block geometry
is included as a third marker. Plotting reuses the exact observation passed to
grounding; it never takes an extra noisy observation. Contact, velocity, forest
and COM acceptance remain separate from these fuzzy degrees.

For nonzero measurement mismatch:

```sh
OPENBLAS_NUM_THREADS=1 mission-sim --task color --epsilon 0 \
  --joint-observation-bias .004 --joint-observation-noise .00015 \
  --observation-bias .001 --observation-noise .0001 \
  --out results/local/noisy_plots
```

Joint settings are radians (bias is applied to each of the six arm joints);
block settings are metres (bias affects block x, noise all three components).
Both channels default to zero. Joint noise uses a separate seeded generator and
is sampled once per solve; block noise is sampled once per grounding call.
Velocity, block orientation, contact evidence and completion guards still use
simulator truth. Predictor mass/friction remain nominal under plant mismatch.
Plotting and shadow sampling add wall-clock time and can produce many files;
optimizer latency records exclude plotting time. No independent full baseline
experiment is launched by the logger.

Remove `--video` if EGL is unavailable. A failed or unresolved run exits with a
nonzero status and writes the failure record. These are offline runs; complex
search and numerical MPC can take minutes. No acados, ROS, proprietary robot
assets, external planner binary, account or network access is needed after
installation. B1 planning can take longer than the five-block tasks.

Optional diagnostic and mismatch examples:

```sh
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.translator --out results/local/translator.json
OPENBLAS_NUM_THREADS=1 mission-sim --task color --rounds 1 --mass-scale 1.3 --friction .7 --out results/local/mismatch
```

The translator command returns a bounded proposed correction after a local
finite-difference check. It does not silently enable adaptation in the task runs.
Block IDs 0–4 denote identities A–E in the letter task; B1 extends this
to A–H. Rendered colors identify objects, while the symbolic rules and feedback
use the logged identity IDs. The default exploration probability is 0.1; the verification commands explicitly
set it to zero to isolate deterministic task execution. Synthetic preference
ratings are generated only after verified complete rounds.

## What the code does

- `planning.py`: weighted best-first search over supports, payload and arm
  anchors; all four action types are explicit and costed. Search receives goals,
  not a prescribed action sequence. Each action boundary triggers a new search.
- `execution.py`: retreat/transit belongs to MoveEmpty; descent, preload and lift
  belong to PickLift; loaded transit belongs to MoveHolding; support transfer,
  opening and released dwell belong to PlaceRelease, which ends at the low pose.
- `controller.py`: six-variable, three-step move-blocked nonlinear shooting
  through a separate full MuJoCo predictor. Only the first command is applied.
- `physics.py` / `model.py`: UR10e chain and inertias, idealized parallel gripper,
  saturation, mechanical integration, free-joint cubes, physical contacts and
  oracle-state observation. Predictor mass/friction remain nominal under plant
  mismatch. Only initialization and disposable prediction/query states set poses.
- `grounding.py`: projected polygon overlap, height/orientation tapers, support
  forests, exclusivity, assignment gap, hysteresis and substack COM. Support
  contact is corroborated. An unaccepted state cannot acquire intended effects.
- `grasp.py`: finite-pad closure and static bounded-load LPs, plus observed
  per-jaw contact, force and relative-position slip. These are distinct checks.
- `learning.py`: realized execution costs update planning cost estimates;
  successful-round feedback updates color means or constrained letter regression.
  The hidden preference evaluator is separate from planning features.
- `geometry.py`: disposable candidate geometry queries reject blocked low poses
  and sampled vertical approach/retreat configurations. Runtime robot-contact
  checks remain necessary; finite samples are not a swept-clearance proof.
- `benchmarks.py`: B0 target support forest, B1's eight-block relational rules,
  and an optional history-monitor unit.
- `translator.py`: an isolated free-space fuzzy-sensitivity diagnostic that
  checks its gradient with a second finite-difference step and verifies that
  shadow rollouts leave the real plant unchanged.

## Declared numerical choices

| Quantity | Value / convention |
|---|---|
| MuJoCo | 3.3.5; `implicitfast`, elliptic friction cone, multi-contact enabled |
| Tangential impedance ratio | 1 by default; tall-stack diagnostics use `--impratio 10`, recorded in each manifest |
| Cube support-contact time constant | 0.02 s by default; tall-stack runs declare `--support-solref .01` |
| Integration / control interval | 0.002 s / 0.05 s |
| MPC horizon / decision variables | 3 intervals / 6 move-blocked arm velocity commands |
| Local optimizer | SciPy L-BFGS-B, at most 6 iterations; every status is logged |
| Arm velocity command / slew | 0.7 rad/s / 0.20 rad/s per control interval |
| Velocity servo | Gain 100 with gravity/Coriolis feedforward and torque saturation |
| Arm torque limits | 330, 330, 150, 56, 56, 56 Nm in joint order |
| Gripper | Two slider joints; 0–0.049 m travel; position gain 2000, damping 20; 25 N force bounds |
| Cubes / nominal mass | 0.05 m side / 0.08 kg each |
| Grasp LP | 8 friction rays at each of 4 corners on each pad; conservative coefficient 0.6 |
| Static capacity set | 12 vertices along ±axes of `[F, M/0.05]`, radius 0.15; 2 N minimum jaw preload |
| Platforms | 3 fixed supports, 6 cells each; platform-qualified identifiers |
| Color candidates | 44 derangements × 6 distinct cell subsets |
| Letter buffers | `p0c2`, `p0c5`, `p1c2`, `p1c5`; roots `p0c0`, `p1c0` |
| Grounding | Entry 0.80, retention 0.70, score gap 0.10; cell half-width 0.04 m |
| Height taper | Core 0.002 m, cutoff 0.009 m |
| Orientation taper | Core 0.035 rad, cutoff 0.17 rad |
| Supported-body velocity limits | 0.01 m/s linear, 0.10 rad/s angular; released object during dwell, all objects at terminal acceptance |
| Substack COM margin | 0.003 m |
| Released dwell | 0.35 consecutive accepted seconds after opening, checked every 0.05 s; 10 s timeout; goal-completing releases require all objects to settle |
| Search | Heuristic multiplier 2; 100,000 expanded-state budget; no optimality guarantee |
| Reward signs | Action `R_o` is cost; `R=-R_o`, `C_hat=R_o`; terminal rating is counted once |

Tracking cost combines joint-reference error, Cartesian error and rotation
error. Executed stage costs include time, tracking, squared velocity command
and actual joint speed; the final tracking term is counted once per action.
The optimizer also has a small command-change regularizer. Squared command
cost is not mechanical energy. Full coefficients and contact parameters are
visible in the source and every generated `scene.xml`.

Robot collision primitives are inherited from the upstream asset; they have
not been certified to enclose every piece of a real robot and tool. The arm
kinematic chain is preserved; commercial-gripper equivalence cannot be asserted
because the attachment's tool assets were absent.

## Evidence and reproducibility

The retained B1 checkpoint experiment can be reproduced with
`python scripts/resume_b1.py results/b1_release results/local/b1_resume`.
It replays 43 previously executed actions with plant/contact guards, then runs
the last release and terminal planning decision live. It is explicitly not a
fresh 44-action optimization run; its manifest preserves that provenance.

The complete declared pilot protocol can be rerun with
`python scripts/run_verification.py --out results/local/fresh --video`.
It runs the unit checks, four physical cases, independent audits and deterministic
plant replay with an additional three-second terminal stability check. This
long-running protocol retains failures and returns a nonzero status if any
case or audit fails. CI runs the focused unit checks only.

See `VERIFICATION.md` and `results/verification.json` for the actual runs in this
release, including failures and limits. Each completed CLI run writes:

- `manifest.json`: versions, parameters, source/model hashes and observation scope;
- `summary.json`: status, completed rounds, action counts, latency and slip;
- `actions.json`: phases, intended/observed supports, fuzzy maps and cost updates;
- `planner.json`, `mpc.json`, `grasp_observations.json`: raw decision/controller data;
- `trajectory.json`: executed joint states, actuator forces and object positions;
- `scene.xml`; and, when requested, `rollout.mp4` and `final.png`.

Some preserved development traces predate later checks; they are not release
success evidence. Video, a controller return code, and an LP result are each
insufficient alone: terminal supports, hard mission rules, released dwell,
observed grasp behavior and action ownership are checked separately.

The original archive review is in `REVIEW_OF_ATTACHED_REPO.md`. The accompanying
LaTeX repository documents the implementation gap without changing the mission
rules or ideal mechanical equations.

For action-level six-panel joint-angle comparisons, use
`--plot-mode joint-prediction`. Orange segments show every short MPC prediction
from its observed starting state; blue shows actual execution, including physical
kicks. All motion phases of an action share one trajectory figure and one error figure; no per-solve figures or
grounding plots are generated in this mode. Files are saved under
`<out>/plots/<session>/<action-type>/action_*/<color>_block<N>_joint_prediction.png`,
with the underlying samples in `joint_prediction.json`. To isolate physical kicks,
leave joint observation noise and bias at zero. With observation noise enabled,
the difference includes both sensing error and plant perturbation. Combine with
`--plot-mode joint-prediction final-grounding` if grounding figures are also needed.

`joint-prediction` also saves `<color>_block<N>_one_step_joint_error.png`,
a six-panel action-level figure of post-kick actual joint angle minus the previous
solve's one-step prediction at the same physical time. Only completed control
intervals are included. With sensing noise or plant-model mismatch, this residual
includes those effects too. Numeric errors and injected kicks are recorded per
solve in `joint_prediction.json`.

Use `--cartisian-prediction` (also spelled `--cartesian-prediction`) for only
one 18-panel Cartesian prediction figure per complete action. Each row is one
of the six arm joint anchors, with X/Y/Z columns in world-frame metres. Actual
motion is blue and each MPC horizon is a separate orange segment. The flag
selects this mode instead of `--plot-mode`; alternatively use
`--plot-mode cartesian-prediction` to combine it with other plot modes.
Files are saved beside `action.json` as
`<color>_block<N>_cartesian_prediction.png` and `cartesian_prediction.json`.
Joint anchors are the hinge locations: some coordinates (especially the first
joint at the base) are stationary. They are different from tool-tip positions.
These forecasts use full nonlinear MuJoCo dynamics and forward kinematics;
the MPC optimization is not a convex problem, and joint angles need not be
linear in time under the simulated servo dynamics.


Scheduled physical kicks are now the default experiment interface; random kicks
are disabled by default. Example (from the project directory):

```bash
OPENBLAS_NUM_THREADS=1 python -m mission_mujoco.cli \
  --task color --epsilon 0 --impratio 10 --support-solref .01 \
  --out results/local/color_debug --plot-mode joint-prediction \
  --kick-timestamps 2,5,7 \
  --kick-magnitude "(1e-3,0,0), (1e-3,1e-4,0)"
```

Times are elapsed **simulation seconds after initialization**, across the entire
run (they do not restart for each action). At 2 s the first three joint angles
receive `(0.001, 0, 0)` rad; at 5 s they receive `(0.001, 0.0001, 0)` rad; at 7 s
the omitted magnitude is zero, so there is no displacement. Supply up to six
components for all arm joints; omitted components are zero. These are joint-angle
increments, not Cartesian XYZ displacements. Quote the vectors because parentheses
have a special meaning in the shell. More magnitudes than timestamps is an error.

Kicks occur after integration at the first control endpoint at or after the
requested time. For example, with `dt=0.05`, a requested `2.02` s kick occurs at
`2.05` s. A zero timestamp occurs at the first endpoint (`0.05` s), not during
initialization. Events at the same endpoint add together. Schedules are never
consumed by predictions or optimizer evaluations. The manifest records the padded
schedule, and physical trace entries contain requested/applied times and the
actual six-joint kick. Existing plots and observation-noise settings are unchanged.
When a schedule is provided it replaces ongoing random/bias kicks, even if the
legacy `--joint-kick-half-range` or `--joint-kick-bias` options are also supplied.
Without a schedule those legacy options can still explicitly enable random kicks.
