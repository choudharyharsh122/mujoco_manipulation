"""Move-blocked direct-shooting MPC through the full MuJoCo predictor.

Six joint velocity commands are optimized; the same command is held over the
short prediction horizon. Finger commands are set by the action phase. This is
an explicitly documented numerical approximation of the manuscript OCP.
"""

from .debug import traced, log

import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


class MPC:
    def __init__(self, world, horizon=3, maxiter=6, diagnostics=None):
        self.w = world
        self.horizon = horizon
        self.maxiter = maxiter
        self.records = []
        self.target_bias = np.zeros(3)
        self.monitor = None
        self.total_cost = 0.0
        self.current_qref = None
        self.current_target = None
        self.diagnostics = diagnostics
        self.pending_plot = None
        self.allowed_block = None

    @traced("MPC solve: continuous target and gripper command; held is a symbolic identity; returns six joint velocity commands")
    def solve(self, target, grip, held=None):
        self.pending_plot = None
        try:
            return self._solve(target, grip, held)
        except Exception as exc:
            if self.diagnostics is not None and self.diagnostics.mode == 'all':
                self.diagnostics.failed_solve(self.w, exc)
            raise

    def _solve(self, target, grip, held=None):
        w = self.w
        start = time.perf_counter()
        true_state = w.snapshot()
        state, observed_q, observation_noise = w.measured_joint_state()
        previous_command = w.last_u.copy()
        log("Measured CONTINUOUS state; optimize one command repeated across prediction horizon", q=observed_q, velocity=w.data.qvel[:6], horizon_steps=self.horizon, dt=w.cfg.dt)
        target = np.asarray(target) + self.target_bias
        ## How is qref calculated from the taraget (cartisian reference)?
        ## What problem does w.ik solve?
        qref, ikerr = w.ik(target, observed_q)
        if ikerr > 0.004:
            raise RuntimeError(f"IK target unreachable: {target}, residual={ikerr}")
        lo = np.maximum(-w.cfg.velocity_limit, w.last_u - w.cfg.slew)
        hi = np.minimum(w.cfg.velocity_limit, w.last_u + w.cfg.slew)
        init = np.clip(3.0 * (qref - observed_q), lo, hi)
        self.current_qref = qref.copy()
        self.current_target = target.copy()
        cached = {}

        def objective(u):
            key = tuple(u)
            if key in cached:
                return cached[key]
            out = w.rollout(state, [u] * self.horizon, grip)
            value = 0.0
            for k, (q, v, p, R) in enumerate(out):
                tracking = self.tracking(q, p, R, qref, target)
                value += w.cfg.dt * (
                    1.0 + tracking + 0.02 * float(u @ u) + 0.002 * float(v @ v)
                )
            value += 4.0 * tracking + 0.002 * float((u - w.last_u) @ (u - w.last_u))
            # Soft collision penalty supplements the observed-contact abort.
            for c in w.pred.contact[: w.pred.ncon]:
                names = [w.pred_model.geom(int(g)).name for g in (c.geom1, c.geom2)]
                intended = (
                    all(n.startswith("cube") for n in names)
                    or any(n.startswith("platform") or n == "floor" for n in names)
                    and any(n.startswith("cube") for n in names)
                    or held is not None
                    and f"cube{held}" in names
                    and any(n.endswith("_pad") for n in names)
                )
                if not intended:
                    value += 1.0e5 * min(0.0, float(c.dist)) ** 2
            cached[key] = value
            return value

        result = minimize(
            objective,
            init,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options={"maxiter": self.maxiter, "ftol": 1e-7, "gtol": 2e-4, "maxls": 8},
        )
        u = np.clip(result.x, lo, hi)
        if not np.all(np.isfinite(u)):
            raise RuntimeError("nonfinite MPC command")
        # Iteration-limit exits are retained and logged; no false convergence claim.
        self.records.append(
            dict(
                seconds=time.perf_counter() - start,
                objective=float(result.fun),
                converged=bool(result.success),
                status=int(result.status),
                iterations=int(result.nit),
                evaluations=int(result.nfev),
                ik_residual=ikerr,
                u=u.tolist(),
                target=target.tolist(),
                time=float(w.data.time),
                observed_q=observed_q.tolist(),
                true_q=w.data.qpos[:6].tolist(),
                observation_noise=observation_noise.tolist(),
                phase=self.diagnostics.phase if self.diagnostics is not None else None,
            )
        )
        if self.diagnostics is not None and self.diagnostics.mode == 'all':
            truth = w.forecast_samples(true_state, [u] * self.horizon, grip, plant_model=True)
            estimated = w.forecast_samples(state, [u] * self.horizon, grip)
            self.pending_plot = self.diagnostics.solve(
                w, self.records[-1], truth, estimated, observation_noise, held, previous_command,
                self.allowed_block)
            self.records[-1]['plot_directory'] = str(self.pending_plot[0])
        if self.diagnostics is not None and any(mode in self.diagnostics.mode
                for mode in ('joint-prediction', 'cartesian-prediction')):
            estimated = w.forecast_samples(state, [u] * self.horizon, grip)
            self.diagnostics.record_joint_prediction(self.records[-1], estimated)
        log("MPC optimizer finished; iteration-limit output is not a convergence certificate", converged=bool(result.success), objective=float(result.fun), iterations=int(result.nit), command=u)
        return u

    def apply(self, u, grip, render=False):
        """Retain partial physical motion even when a substep guard aborts."""
        error = None
        try:
            return self.w.step(u, grip, render)
        except Exception as exc:
            error = exc
            raise
        finally:
            try:
                if self.diagnostics is not None:
                    self.diagnostics.applied(self.w, self.pending_plot, error, self.allowed_block)
            except Exception as logging_error:
                if error is None:
                    raise
                error.add_note(f"Plot logging also failed: {logging_error}")
            finally:
                self.pending_plot = None

    @traced("MPC feedback loop: repeatedly predict, apply one interval, observe; return accumulated execution cost")
    def reach(
        self, target, grip, held=None, tolerance=0.0025, timeout=12.0, render=False
    ):
        w = self.w
        cost = 0.0
        dwell = 0
        start = len(self.records)
        completion_checks = []
        try:
            for i in range(round(timeout / w.cfg.dt)):
                log("MPC.reach: receding-horizon iteration", iteration=i + 1, prediction_steps=self.horizon, simulation_time=float(w.data.time))
                u = self.solve(target, grip, held)
                increment = self.apply(u, grip, render) + w.cfg.dt * self.current_tracking()
                cost += increment
                self.total_cost += increment
                if self.monitor:
                    self.monitor()
                p, R = w.pose()
                position_error = float(np.linalg.norm(p - np.asarray(target) - self.target_bias))
                speed = float(np.linalg.norm(w.data.qvel[:6]))
                if position_error < tolerance and speed < 0.035:
                    dwell += 1
                else:
                    dwell = 0
                completion = dict(time=float(w.data.time), target=(np.asarray(target)+self.target_bias).tolist(),
                                  position=p.tolist(), position_error=position_error,
                                  joint_speed_norm=speed, position_tolerance=tolerance,
                                  speed_tolerance=0.035, consecutive_passes=dwell, required_passes=3,
                                  position_pass=position_error < tolerance, speed_pass=speed < 0.035)
                completion_checks.append(completion)
                self.records[-1]['completion'] = completion
                log("Applied one control interval; measured completion", position_error=float(np.linalg.norm(p - np.asarray(target) - self.target_bias)), joint_speed_norm=float(np.linalg.norm(w.data.qvel[:6])), accepted_samples=dwell, required_samples=3)
                if dwell >= 3:
                    return cost
            raise RuntimeError(
                f"MPC timeout: target {target}, pose {w.pose()[0]}, solves {len(self.records) - start}"
            )
        finally:
            if self.diagnostics is not None and self.diagnostics.completion_plots:
                self.diagnostics.save_completion(completion_checks)

    @traced("MPC hold: maintain current continuous tool position while grasp/release checks run")
    def hold(self, duration, grip, held=None, render=False, sample_check=None):
        target = self.w.pose()[0]
        cost = 0.0
        for iteration in range(round(duration / self.w.cfg.dt)):
            log("MPC.hold: receding-horizon iteration", iteration=iteration + 1, prediction_steps=self.horizon, simulation_time=float(self.w.data.time))
            increment = (
                self.apply(self.solve(target, grip, held), grip, render)
                + self.w.cfg.dt * self.current_tracking()
            )
            cost += increment
            self.total_cost += increment
            if self.monitor:
                self.monitor()
            if sample_check and sample_check():
                break
        return cost

    @staticmethod
    def tracking(q, p, R, qref, target):
        ## Why does tracking penalize the difference between the current joint angles and the reference joint angles,
        ## as well as the difference between the current position and the target position, and also the orientation error?
        ## We only have taraget i.e. cartisian refs, to compute q_ref we solve the inverse kinematics
        ## this doesn't give us any extra information, the info about joint angles is 
        ## obtained from the cartisian target via inverse kinematics.
        ## This however adds another layer of error, since the inverse kinematics is not perfect, 
        ## and the joint angles obtained from it may not be accurate.
        return float(
            (q - qref) @ (q - qref)
            + 15 * (p - target) @ (p - target)
            + 30
            * Rotation.from_matrix(np.diag([1.0, -1.0, -1.0]) @ R.T).magnitude() ** 2
        )

    def current_tracking(self):
        p, R = self.w.pose()
        return self.tracking(
            self.w.data.qpos[:6], p, R, self.current_qref, self.current_target
        )
