# Verification of the executable pilot

Date: September 30, 2026. This is a development verification record, not a
statistical task-success study or a complete reproduction of the principal
ideal-contact OCP. The full constrained solver, horizon-wide grasp/servo
certificates, contact-sensitive KKT adaptation, hardware calibration and
comparative experiments remain unimplemented. See `CONFORMANCE.md`.

**The complete four-task protocol does not pass.** The failures below remain blocking verification results.

- `color_settled`: RuntimeError: released support/velocity dwell timed out (19/20 actions accepted; 0 complete rounds).
- `b1_settled_resume`: RuntimeError: released support/velocity dwell timed out (43/44 actions accepted; 0 complete rounds).

## Physical runs

All runs use seed 0, oracle simulator observations, nominal mass/friction and
zero exploration. Color uses contact impedance ratio 1 and cube contact time constant 0.02 s.
B0, letter and B1 use 10 and 0.01 s. These are explicitly
different numerical contact settings; do not treat the table as a controlled
comparison between tasks.

| Run | Status | Complete rounds | Successful/attempted actions | MPC solves | p95 solve time (ms) | Final-source replay |
|---|---|---:|---:|---:|---:|---|
| b0_settled | success | 1 | 16/16 | 1942 | 247.1 | pass |
| color_settled | failed | 0 | 19/20 | 2908 | 310.6 | not passed |
| letter_settled | success | 2 | 48/48 | 5442 | 327.5 | pass |
| b1_settled_resume | failed | 0 | 43/44 | 5442 | 548.8 | not passed |

The color protocol requests the complete three-platform cycle; a failed run
does not establish that cycle. The letter protocol
requests two alternating-platform rounds. B0 is a three-block restack; B1 is an
eight-block, two-tower relational task. Plans are generated from goals, and only
the first action is dispatched before the next measured-state replan.

The independent audit checks action status against observed support graphs,
reward signs, replanning boundaries, action-phase ownership, finite/bounded
commands, round memory and terminal task rules. For completed runs, plant replay
applies every saved command using the released source and independently
re-grounds every action boundary. It then continues free-body simulation for
three seconds and checks that the terminal support graph remains accepted.
Replay is an additional plant/grounding check; it does not replace the original
closed-loop MPC execution or prove the full optimization formulation.

The raw manifests preserve source/model hashes at execution. The final-source
replay also verifies all seven control samples in each released dwell, the
released object's linear speed (0.01 m/s) and angular speed (0.10 rad/s), and
all object speeds throughout each goal-completing release dwell and at
completed-round boundaries. The executor resets the dwell
clock if support or speed acceptance is lost, and aborts if 0.35 consecutive
accepted seconds are not obtained within 10 seconds after opening. Goal-completing releases apply that dwell
check to all supported objects.

Earlier fixed-dwell color, letter, B0 and B1 trials reached measured goals but
were superseded after a stricter action-contract review: some early released
samples still had angular motion. They are not used as final conformance passes.
The primary runs above execute physical commands from initialization. B0,
color and letters run the optimizer live throughout. B1 is explicitly a
checkpoint verification: 43 previously executed closed-loop actions are
replayed under current plant/contact guards, then the final release runs live
with the corrected MPC executor. This checkpoint fails during that release,
before its terminal planning decision can run. Its
manifest identifies the source trace and maximum replay error. Its wall time
covers replay plus the live tail, and its latency distribution includes the
original prefix solves. It is not a fresh 44-action optimizer run. The original
B1 trial completed 44 actions but failed the all-object terminal speed check.
Use the complete verification command below for a fresh B1 experiment. The
color and letter trials with a two-second settling limit are also retained:
color failed the second round's all-object speed check after 40 successful
actions, and letters timed out on release 48 after 47 successful actions. A
separate continuation retained both support graphs and found later settled
intervals; the released executor therefore permits ten seconds and enforces
all-object dwell on goal-completing releases. Acceptance thresholds were not
relaxed.

The additional three-second continuation uses zero arm velocity commands after
the task's MPC execution has ended. Support/contact graphs must remain valid.
Speed invariance in that extra interval is separately measured below; **False
is retained as a numerical limitation**, even when the sampled release dwells
and terminal acceptance passed. Small position drift can coexist with brief
contact-solver velocity spikes. These data do not establish a stable invariant
set or the full hard-contact formulation.

| Run | Maximum later linear speed (m/s) | Maximum later angular speed (rad/s) | Later speed limits maintained |
|---|---:|---:|---|
| b0_settled | 0.01256 | 0.23634 | False |
| letter_settled | 0.02121 | 0.50179 | False |

## Focused checks

- 14 tests pass: terminal candidate counts over all previous permutations,
  explicit four-action planning from empty/held states, support exclusivity and
  cycles, finite-pad closure and impossible-capacity rejection, reward signs,
  missing feedback, constrained identity-position regression, prediction-state
  isolation, floating-object rejection, entire-substack COM, B1 rule directions,
  hysteresis at pick/release, released-velocity rejection, TCP Jacobian/cube inertia, and a ten-second free
  tall-stack stability check.
- The wheel builds and loads its vendored robot model outside the source tree.
- Static undefined-name/import checks pass. CI runs the focused tests; no remote
  CI execution is claimed.
- The isolated free-space translator diagnostic reduces its fuzzy loss from
  0.25454 to zero, with a bounded 3 mm correction. Half-step gradient disagreement
  is 14.92%, and the plant snapshot is unchanged. This is finite differencing,
  not a KKT derivative, and is not active in the manipulation experiments.

## Failures and numerical sensitivity

Earlier letter-stack and B1 runs failed support grounding. In the ratio-10,
0.02 s letter run, the newly placed top block penetrated its support by about
4.15 mm; continuing the saved-state simulation led to collapse. Reducing the
cube contact time constant to 0.01 s in a disposable diagnostic preserved the
stack. The complete new runs above test that setting from initialization. The
task rules, physical kinematics and membership thresholds were not relaxed.
The diagnostic is numerical evidence, not identification of real material
properties or proof of a MuJoCo engine defect.

The mass-1.3/friction-0.7 stress case **fails** the per-jaw force guard during
transfer: the observed force reaches 26.35 N against a 25.5 N acceptance bound.
Thus mismatch robustness is not established. The deliberately low-friction
case (0.01) correctly aborts after measured relative-position slip exceeds
8 mm. These aborts detect violations; they do not prove proactive constraint
enforcement or provide a hardware recovery policy.

`development_failures.json` preserves failed and interrupted development-run
summaries. They are not independent trials and are not pooled into a success
rate. The selected contact-failure trace is retained for inspection. Any final
run that fails remains a failure, with no terminal rating or round-memory update.

## Performance and limits

The control interval is 50 ms, while the reported solve times exceed that
interval. Simulation time pauses during optimization; these are offline runs.
The solver commonly reaches its six-iteration limit. The recorded convergence
fractions are in `verification.json`; a usable iterate is not a hard-constraint
feasibility or optimality certificate. Timing was measured while other
development work was running on shared CPU resources.

The action-cost model pools observations by action type. The terminal learners
and reward path are exercised, but these few rounds do not establish learning
convergence, improved task ranking, generalization or superiority to a baseline.
Optional B2/B3 physical studies are not included.

## Reproduce

Install the pinned dependencies and this package as described in `README.md`.
Run `python scripts/run_verification.py --out results/local/fresh --video` for
the complete declared pilot protocol. Omit video if EGL is unavailable. Expect
long offline runs. The command retains logs, audits all four cases, replays
successful cases and exits nonzero on a run or audit failure.

The corrected LaTeX retains the planning and mechanical specification and adds
an implementation-scope section; it does not redefine the full formulation to
make this pilot appear complete.

## Timestep diagnostic

A disposable continuation of the failed color state compared numerical settings
without changing task thresholds. With the original 2 ms step, 75/200 all-object
settling samples passed and the longest accepted interval was 11 samples. A
1 ms integration step passed 198/200 samples with a longest interval of 114.
The half-step setting was **not validated in a fresh closed-loop mission** and
is not silently substituted into the released run evidence. Impedance, NoSlip
and CCD-tolerance alternatives are recorded in `contact_settling_options.json`.
These are diagnostics from one saved state, not independent trials or proof of
a universal fix.
