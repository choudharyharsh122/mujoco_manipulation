# Review of `full-main.zip`

**Grade: 4/10 — useful research prototype, not a reproducible manuscript implementation.**

This grade applies to the supplied archive, not to code or assets that may exist
elsewhere. All 45 Python source files parse. That is a syntax check, not evidence
that the integrated experiment runs.

| Area | Score | Finding |
|---|---:|---|
| Organization and research direction | 6/10 | Planning, learning, execution and physics are separated; there is real receding-horizon and contact-model work. |
| Reproducibility | 2/10 | The entry point requires a missing controller; robot XML and meshes are missing. |
| Mechanical implementation | 5/10 | Rigid-body dynamics and a small contact-wrench OCP are substantive, but do not implement the complete manuscript certificates. |
| Mission and grounding fidelity | 3/10 | The task and observed-effect semantics differ materially from the manuscript. |
| Verification and evidence | 2/10 | No runnable tests are supplied; all five included final-summary files report `Successful: 0/1`. |

## Blocking findings

1. **The supplied integration cannot launch.**
   `codice/integration/manipulation_bridge.py`, `load_manipulation_controller`,
   explicitly loads `mujoco-manipulation/catkin_ws/test_grasp_physics.py`. That file
   is absent. Importing the bridge with `PYTHONPATH=codice` and calling the loader
   reproduced `FileNotFoundError`. The existing `run_ur10e_acados_grasp.py` is a
   different interface and cannot simply be renamed. Both robot configurations
   also reference an absent `models/ur10e/ur10e.xml` and its assets.

2. **Symbolic effects are not grounded from measured block states.**
   `codice/integration/episode_runner.py`, around lines 465–480, applies the
   intended task action after the controller success flag. The grasp/release
   wrapper reports success when the call returns without an exception. Its
   terminal error is set to zero. Tool-position membership does not prove that
   the correct object was lifted or that released support is valid. The release
   wrapper also clears `_grasp_state` before the release call succeeds.

3. **The mission is different.**
   The block-table-cycle progression and its `come_back` feedback are not the
   manuscript's five-object spatial derangement with destination-leading-color
   preference and variable spread. There is no five-letter restacking executor
   with the new-bottom rule, finite buffers and verified stack support.

4. **Fuzzy grounding and reward propagation are incomplete.**
   `core/planning/fuzzy_state.py` primarily evaluates planar tool position. It
   lacks projected object footprints, height/orientation agreement, globally
   consistent occupancy/support selection, ambiguity handling, stack COM and
   released dwell. The feedback path is tied to the return action rather than
   the required measured execution-cost channel plus successful-round terminal
   feedback. The quantity reported as `distance_to_true` compares estimates with
   accumulated observed rewards, not an independent hidden ground truth.

5. **The mechanical code is stronger than its documentation, but narrower than
   the manuscript.** `unified_grasp_mpc.py` includes arm torque, gripper-rate and
   soft-finger wrench variables, rigid-body/object equations and contact
   inequalities. However, it freezes local dynamics/contact geometry and does
   not supply the manuscript's full spatial closure, disturbance-polytope
   capacity and horizon-wide servo-attainability system. A controller status is
   not a certificate for those conditions. The README's double-integrator,
   simplex-estimator, weld and convergence descriptions are stale relative to
   the actual branches. I did **not** find evidence that the newer manipulation
   run physically welds its object; it explicitly reports no kinematic attachment.

6. **Sensitivity and performance claims need qualification.** The target-update
   branch uses restored-state finite-difference probes, not implicit KKT
   derivatives. That can be useful, but must be named and validated accordingly.
   The supplied summaries do not support convergence or task-success claims.
   Five files containing unsuccessful episode summaries are not five independent
   controlled trials and should not be pooled as such.

## What is worth preserving

The separation of planner/controller/learner, replanning after each action,
snapshot-based controller probes, MuJoCo-derived mass/bias terms, and explicit
grasp contact variables provide a reasonable starting point. The replacement
repository preserves the manuscript's four action types, mission rules, rigid
UR10e chain and free-object manipulation while making its numerical scope
explicit. It is a separately implemented simulation pilot, not an assertion
that the supplied archive already validated the paper.
