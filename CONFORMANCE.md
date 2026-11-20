# Manuscript conformance and limits

The simulation is a physical pilot of the mission-aligned architecture. It is
**not a complete implementation of the principal ideal-contact OCP**. The
manuscript's planning rules and mechanical equations remain the specification;
the paper is not weakened to make a partial solver appear equivalent.

| Manuscript element | Executable status |
|---|---|
| Four typed actions, phase ownership, low release endpoint | Implemented in `execution.py`; every boundary returns to search. |
| Classical search using current learned costs, no embedded future MPC | Implemented; weighted best-first search, finite budget, no optimality claim. |
| Color spatial derangement, common source, three-platform cycle | Implemented; 44 orders and six possible five-cell subsets per destination. |
| Destination-leading-color utility and spread | Implemented; feedback source separate from planner; measured completion gates memory. |
| Letter alternate platform, new bottom neither old boundary | Implemented; 72 candidate orders and four finite buffers. |
| Identity-position ridge regression with zero row/column sums | Implemented; no vowel features passed to learner. |
| B0 and eight-block B1 rules | Implemented; generated plans, rather than executing the manuscript witness. |
| Optional B2/B3 comparisons | Milestone logic has a unit check; full physical temporal/alphabetical experiment not included. |
| Fuzzy overlap, height/orientation, support forest, gap/hysteresis | Implemented with contact corroboration, supported-body speed limits and entire-substack COM. |
| Joint fuzzy description of arm, holding and support atoms | Incomplete: runtime support memberships are fuzzy; arm and holding evidence use separate endpoint/contact guards rather than one joint fuzzy-state vector. |
| Realized execution cost upward, terminal feedback once | Implemented; logged `C_bar`, `R_o`, `R_e`, `C_hat`, residual and model version. |
| Transition learning | Fixed expert transition specialization; observed effects checked independently. No learned failure-probability penalty. |
| Translator adaptation | Separate finite-difference free-space fuzzy-sensitivity diagnostic; not enabled in the manipulation runs. KKT and contact-mode sensitivity propagation not implemented. |
| Rigid UR10e kinematics and inertia | Vendored Menagerie chain; original archive's missing custom tool cannot be recovered. |
| Physical objects and finger contact | Free-joint cubes, dynamic two-slider gripper, no plant welds or object-pose resets after initialization. |
| Coupled mechanics | Full MuJoCo rigid-body simulation in both predictor and plant, with distinct model/data objects. |
| Velocity-command servo | Explicit saturated simulation servo and gripper actuators; not calibrated UR firmware or a commercial gripper. |
| Finite-pad closure and bounded static load | Polyhedral finite-pad closure LP and 12-vertex static disturbance test, jaw preload/force and arm torque limits. |
| Full horizon capacity / internal preload allocation / servo attainability | **Not implemented as hard OCP constraints.** The static certificate and live contact guards are narrower. |
| Rigid sticking, exact complementarity, projected multiple shooting | **Not implemented.** MuJoCo regularized contact and six-variable move-blocked shooting are used. |
| Collision constraints | Finite candidate pose checks, a soft prediction penalty and substep aborts. Cube–cube and cube–support contacts are broadly allowed by the guard; no continuous-time or robust clearance certificate. |
| Model mismatch | Separate nominal predictor; mass/friction stress inputs and explicit oracle-state condition. No realistic perception estimator. |
| Real-time operation and recovery | Offline synchronous simulation; wall-clock solve time does not advance simulation. Aborts terminate the experiment; no certified hardware fallback. |
| Full comparative study | Not done. Pilot runs are not 30 independent trials, convergence evidence, or hardware validation. |

The action-cost learner pools samples by action type. This deliberately small
feature set cannot learn all source/destination-dependent costs. Successful
feedback propagation is not proof that the resulting ranking improves.

Geometry uses upright canonical candidate objects and a fixed IK branch.
Measured poses initialize each control solve; finite candidate checks do not
cover arbitrary continuous grasp configurations. Arm anchors distinguish
pregrasp, lift, preplace and low release states within this restricted executor.

The finite-pad LP uses an ideal patch, not the simulator's instantaneous
contact-point distribution. It is evaluated prospectively at preload. Observed
force, slip, contact and support checks supply separate evidence during motion.
The distinction is retained in the manuscript's implementation appendix.

The tall-stack regularization runs declare tangential impedance ratio 10 and
cube contact time constant 0.01 s; the color run uses 1 and 0.02 s. B0 uses the same 10 and 0.01 s settings as the taller stacks.
The parameter change addresses observed contact penetration and drift in a
five-block stack; it does not change the kinematics, task rules or acceptance
thresholds. The earlier failed runs remain in the evidence. These task-specific
numerical settings preclude a matched-parameter comparison between tasks.

The post-goal continuation separately logs speed peaks. The three-block pilot
retains its support graph but exhibits brief speed excursions after task
completion under zero velocity commands. Successful sampled release/terminal
checks are not an invariant-set or continuous-time speed guarantee.
