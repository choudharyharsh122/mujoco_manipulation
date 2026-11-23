"""Four action instances with physical guards and measured symbolic effects."""

import numpy as np
import sys
from .controller import MPC
from .debug import traced, log
from .grounding import Grounder, conjunction
from .grasp import closure, capacity, measured, prospective_capacity
from .model import cells, SIDE
from .diagnostics import PlotLogger, normalize_plot_mode


class Executor:
    def __init__(self, world, models, render=False, plot_dir="results/plots", plot_mode="all", completion_plots=False):
        plot_mode = normalize_plot_mode(plot_mode)
        self.w = world
        self.models = models
        self.diagnostics = PlotLogger(plot_dir, mode=plot_mode, completion_plots=completion_plots) if plot_dir is not None and plot_mode != 'none' else None
        motion_diagnostics = self.diagnostics if (plot_mode == 'all' or 'executed-cartesian' in plot_mode or 'joint-prediction' in plot_mode or 'cartesian-prediction' in plot_mode or completion_plots) else None
        world.diagnostics = motion_diagnostics
        self.mpc = MPC(world, diagnostics=motion_diagnostics)
        self.grounder = Grounder(diagnostics=self.diagnostics if plot_mode != "executed-cartesian" else None)
        self.held = None
        self.anchor = ("home", None)
        self.reference = None
        self.render = render
        self.actions = []
        self.phases = []
        self.cruise = 0.70
        initial = self.grounder.measure(world)
        if not initial.accepted:
            raise RuntimeError("initial grounding: " + initial.reason)
        self.supports = initial.supports
        self.monitor = []
        self.mpc.monitor = self.check
        self.monitored_block = None
        self.allowed_block = None
        self.w.substep_guard = self.check

    def check(self):
        w = self.w
        for c in w.data.contact[: w.data.ncon]:
            names = [w.model.geom(int(g)).name for g in (c.geom1, c.geom2)]
            intended = (
                all(n.startswith("cube") for n in names)
                or any(n.startswith("platform") or n == "floor" for n in names)
                and any(n.startswith("cube") for n in names)
                or self.allowed_block is not None
                and f"cube{self.allowed_block}" in names
                and any(n.endswith("_pad") for n in names)
            )
            if not intended and c.dist < -0.00005:
                raise RuntimeError(f"unintended contact: {names}, penetration {c.dist}")
        if np.max(np.abs(w.data.qvel[:6])) > 1.0:
            raise RuntimeError("actual joint speed limit")
        if self.monitored_block is not None:
            evidence = measured(w, self.monitored_block, self.reference)
            if w.steps % w.nstep == 0:
                self.monitor.append(evidence)
            if not evidence["accepted"]:
                raise RuntimeError("grasp evidence lost: " + str(evidence))

    @traced("ACTION-to-MPC mapping: Cartesian target in metres, gripper command, internal phase and held identity")
    def reach(self, target, grip, phase, held=None):
        if self.diagnostics is not None:
            self.diagnostics.phase = phase
        self.mpc.allowed_block = self.allowed_block
        self.phases.append(
            dict(
                name=phase,
                time=float(self.w.data.time),
                target=np.asarray(target).tolist(),
            )
        )
        target = np.asarray(target)
        start = self.w.pose()[0]
        cost = 0.0
        if (
            np.linalg.norm(target[:2] - start[:2]) < 0.01
            and abs(target[2] - start[2]) > 0.045
        ):
            count = int(np.ceil(abs(target[2] - start[2]) / 0.035))
            for k in range(1, count + 1):
                cost += self.mpc.reach(
                    start + (target - start) * k / count, grip, held, render=self.render
                )
            return cost
        return self.mpc.reach(target, grip, held, render=self.render)

    def hold(self, duration, grip, held=None):
        if self.diagnostics is not None:
            self.diagnostics.phase = self.phases[-1]['name'] if self.phases else 'hold'
        self.mpc.allowed_block = self.allowed_block
        return self.mpc.hold(duration, grip, held, self.render)

    def support_center(self, d):
        return (
            cells()[d].copy()
            if isinstance(d, str)
            else self.w.blockpos(d) + [0, 0, SIDE]
        )

    def move_empty_targets(self, block, start=None):
        """Return the named waypoints for approaching an unheld block."""
        target = self.w.blockpos(block) + [0, 0, 0.12]
        start = self.w.pose()[0] if start is None else start
        return (
            ("retreat", np.array([*start[:2], self.cruise])),
            ("transit", np.array([*target[:2], self.cruise])),
            ("pregrasp", target),
        )

    def pick_lift_descent_target(self, block):
        return self.w.blockpos(block) + [0, 0, 0.003]

    def move_holding_targets(self, destination, start=None):
        center = self.support_center(destination)
        start = self.w.pose()[0] if start is None else start
        return (
            ("lifted transit height", np.array([*start[:2], self.cruise])),
            ("carry", np.array([*center[:2], self.cruise])),
            ("preplace", center + [0, 0, 0.12]),
        )

    @traced("SYMBOLIC action (type, block, destination) becomes physical phases and MPC targets")
    def execute(self, action, settle_all=False):
        kind, b, d = action
        w = self.w
        cost = 0.0
        start = float(w.data.time)
        solve_start = len(self.mpc.records)
        phase_start = len(self.phases)
        cost_start = self.mpc.total_cost
        before = self.supports.copy()
        intended = before.copy()
        expected_held = self.held
        record = dict(
            id=len(self.actions),
            kind=kind,
            block=b,
            destination=d,
            start=start,
            status="running",
            source_anchor=self.anchor,
            settle_all=bool(settle_all),
        )
        self.actions.append(record)
        if self.diagnostics is not None:
            self.diagnostics.begin_action(w, record)
            record['plot_directory'] = str(self.diagnostics.directory)
        try:
            if kind == "MoveEmpty":
                if self.held is not None:
                    raise RuntimeError("MoveEmpty precondition")
                for phase, target in self.move_empty_targets(b):
                    cost += self.reach(target, 0, phase)
                new_anchor = ("pre", b)
            elif kind == "PickLift":
                if (
                    self.held is not None
                    or self.anchor != ("pre", b)
                    or b in self.supports.values()
                ):
                    raise RuntimeError("PickLift precondition")
                self.allowed_block = b
                target = self.pick_lift_descent_target(b)
                certificate = closure()
                load = capacity()
                record["closure"] = certificate
                record["capacity"] = load
                if not certificate["accepted"] or not load["accepted"]:
                    raise RuntimeError("prospective grasp certificate failed")
                cost += self.reach(target, 0, "descend")
                self.phases.append(dict(name="preload", time=float(w.data.time)))
                for grip in [0.01, 0.02, 0.03, 0.038]:
                    cost += self.hold(0.15, grip, b)
                evidence = measured(w, b)
                if not evidence["accepted"]:
                    raise RuntimeError("preload contact evidence: " + str(evidence))
                record["capacity"] = prospective_capacity(w, b)
                if not record["capacity"]["accepted"]:
                    raise RuntimeError(
                        "unsupported static capacity with arm torque bounds failed"
                    )
                self.reference = np.asarray(evidence["relative"])
                self.monitored_block = b
                cost += self.reach([*w.pose()[0][:2], self.cruise], 0.038, "lift", b)
                self.held = b
                intended.pop(b)
                expected_held = b
                new_anchor = ("lift", b)
            elif kind == "MoveHolding":
                if self.held != b:
                    raise RuntimeError("MoveHolding precondition")
                for phase, target in self.move_holding_targets(d):
                    cost += self.reach(target, 0.038, phase, b)
                new_anchor = ("above", d)
            elif kind == "PlaceRelease":
                if (
                    self.held != b
                    or self.anchor != ("above", d)
                    or d in self.supports.values()
                ):
                    raise RuntimeError("PlaceRelease precondition")
                center = self.support_center(d)
                R = w.pose()[1]
                target = center - R @ self.reference
                cost += self.reach(target, 0.038, "descent", b)
                support_geom = (
                    w.model.geom("platform" + d[1]).id
                    if isinstance(d, str)
                    else w.blockgeoms[d]
                )
                for attempt in range(6):
                    if any(
                        c["other"] == support_geom and c["normal_force"] > 0.03
                        for c in w.contacts(b)
                    ):
                        break
                    target[2] -= 0.0015
                    cost += self.reach(target, 0.038, "support search", b)
                else:
                    raise RuntimeError(
                        "destination support not established before release"
                    )
                self.monitored_block = None
                self.phases.append(
                    dict(name="open and released dwell", time=float(w.data.time))
                )
                for grip in [0.025, 0.012, 0.0]:
                    cost += self.hold(0.15, grip, b)

                stable_samples = 0

                def released_dwell_check():
                    nonlocal stable_samples
                    observation = self.grounder.measure(
                        w,
                        require_settled=True,
                        settled_blocks=None if settle_all else [b],
                    )
                    if observation.accepted and observation.supports == {
                        **intended,
                        b: d,
                    }:
                        stable_samples += 1
                    else:
                        stable_samples = 0
                    return stable_samples >= round(0.35 / w.cfg.dt)

                cost += self.mpc.hold(
                    10.0, 0.0, render=self.render, sample_check=released_dwell_check
                )
                if stable_samples < round(0.35 / w.cfg.dt):
                    raise RuntimeError("released support/velocity dwell timed out")
                if any(c["pad"] and c["normal_force"] > 0.05 for c in w.contacts(b)):
                    raise RuntimeError("fingers still loaded after release")
                self.held = None
                self.reference = None
                self.allowed_block = None
                intended[b] = d
                expected_held = None
                new_anchor = ("low", d)
            else:
                raise ValueError(kind)
            observed = self.grounder.measure(
                w,
                self.held,
                require_settled=kind == "PlaceRelease",
                settled_blocks=None if settle_all else [b],
                plot_final=True,
            )
            record["grounding"] = dict(
                accepted=observed.accepted,
                reason=observed.reason,
                gap=observed.gap,
                mu={
                    str(i): {str(j): v for j, v in x.items()}
                    for i, x in observed.memberships.items()
                },
                intended=intended,
                observed=observed.supports,
                E_psi=conjunction(observed.memberships, observed.supports or {})
                - conjunction(observed.memberships, intended),
            )
            if not observed.accepted:
                raise RuntimeError("unknown grounding: " + observed.reason)
            # Preserve an accepted unexpected effect; never overwrite it with prediction.
            self.supports = observed.supports
            if self.supports != intended or self.held != expected_held:
                raise RuntimeError("observed transition differs from intended")
            self.anchor = new_anchor
            log("Handover to planner: observed SYMBOLIC state; continuous World state is preserved", supports=self.supports, held=self.held, anchor=self.anchor)
            float(
                np.linalg.norm(w.pose()[0] - self.phases[-1].get("target", w.pose()[0]))
            )
            cost = self.mpc.total_cost - cost_start + 4 * self.mpc.current_tracking()
            record.update(
                status="success",
                end=float(w.data.time),
                duration=float(w.data.time) - start,
                feedback=self.models.execution(kind, cost),
                target_anchor=new_anchor,
                solves=len(self.mpc.records) - solve_start,
                phases=self.phases[phase_start:],
            )
            return record
        except Exception as e:
            if (self.diagnostics is not None and 'final-grounding' in self.diagnostics.mode
                    and not self.diagnostics.final_grounding_saved):
                try:
                    self.grounder.measure(w, self.held, plot_final=True)
                except Exception as grounding_error:
                    e.add_note(f"Final grounding logging also failed: {grounding_error}")
            record.update(
                status="failed",
                error=str(e),
                end=float(w.data.time),
                duration=float(w.data.time) - start,
                partial_cost=self.mpc.total_cost - cost_start,
                solves=len(self.mpc.records) - solve_start,
                phases=self.phases[phase_start:],
            )
            raise
        finally:
            original_error = sys.exception()
            try:
                if self.diagnostics is not None:
                    self.diagnostics.end_action(w, record)
            except Exception as logging_error:
                if original_error is None:
                    raise
                original_error.add_note(f"Action plot logging also failed: {logging_error}")
