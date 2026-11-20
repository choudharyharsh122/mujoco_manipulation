from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .model import scene

R_DOWN = np.diag([1.0, -1.0, -1.0])


@dataclass
class Config:
    contact_impratio: float = 1.0
    support_solref: float = 0.02
    dt: float = 0.05
    timestep: float = 0.002
    kv: float = 100.0
    velocity_limit: float = 0.7
    slew: float = 0.20
    mass_scale: float = 1.0
    friction: float = 1.0
    observation_noise: float = 0.0
    observation_bias: float = 0.0
    joint_observation_noise: float = 0.0  # rad; independent of block observations
    joint_observation_bias: float = 0.0  # rad, applied to each arm joint
    joint_kick_half_range: float = 0.0  # uniform [-a, a] rad per joint per interval
    joint_kick_bias: float = 0.0  # rad per interval, applied to each arm joint
    kick_timestamps: tuple[float, ...] = ()  # elapsed seconds after initialization
    kick_magnitudes: tuple[tuple[float, ...], ...] = ()  # joint increments [rad]
    seed: int = 0

    def __post_init__(self):
        for name in ("observation_noise", "joint_observation_noise", "joint_kick_half_range"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("observation_bias", "joint_observation_bias", "joint_kick_bias"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

        timestamps = tuple(float(t) for t in self.kick_timestamps)
        if any(not np.isfinite(t) or t < 0 for t in timestamps):
            raise ValueError("kick timestamps must be finite and nonnegative")
        if len(self.kick_magnitudes) > len(timestamps):
            raise ValueError("More kick magnitudes than timestamps")
        magnitudes = []
        for magnitude in self.kick_magnitudes:
            vector = np.asarray(magnitude, dtype=float)
            if vector.ndim != 1 or not 1 <= len(vector) <= 6 or not np.isfinite(vector).all():
                raise ValueError("Each kick magnitude must contain 1 to 6 finite joint increments [rad]")
            magnitudes.append(tuple(np.pad(vector, (0, 6-len(vector)))))
        magnitudes.extend([(0.0,)*6] * (len(timestamps)-len(magnitudes)))
        self.kick_timestamps = timestamps
        self.kick_magnitudes = tuple(magnitudes)


class World:
    def __init__(self, supports=None, config=None):
        self.cfg = config or Config()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.joint_rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, 1]))
        self.kick_rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, 2]))
        self.last_joint_kick = np.zeros(6)
        self.last_kick_events = []
        self.xml = scene(
            supports,
            self.cfg.mass_scale,
            self.cfg.friction,
            self.cfg.timestep,
            self.cfg.contact_impratio,
            self.cfg.support_solref,
        )
        self.model = mj.MjModel.from_xml_string(self.xml)
        self.data = mj.MjData(self.model)
        self.pred_model = mj.MjModel.from_xml_string(
            scene(
                supports,
                1.0,
                1.0,
                self.cfg.timestep,
                self.cfg.contact_impratio,
                self.cfg.support_solref,
            )
        )
        self.pred = mj.MjData(self.pred_model)
        self.ikdata = mj.MjData(self.pred_model)
        self.tcp = self.model.site("tcp").id
        self.nblock = len(supports) if supports is not None else 5
        self.blocks = [self.model.body(f"block{i}").id for i in range(self.nblock)]
        self.blockgeoms = [self.model.geom(f"cube{i}").id for i in range(self.nblock)]
        self.pads = [self.model.geom(s + "_pad").id for s in ["left", "right"]]
        self.state_spec = mj.mjtState.mjSTATE_INTEGRATION
        self.nstep = round(self.cfg.dt / self.cfg.timestep)
        self.substep_guard = None
        self.trace = []
        self.last_u = np.zeros(6)
        self.grip = 0.0
        self.steps = 0
        self.frames = []
        self.interval_samples = []
        self._record_interval = False
        self._next_frame_time = 0.0
        self.render = None
        self.initial_q = np.array([-4.28, -1.55, 1.45, -1.46, -1.57, -2.71])
        self.data.qpos[:6] = self.initial_q
        mj.mj_forward(self.model, self.data)
        q, err = self.ik(np.array([0.45, -0.56, 0.65]), self.initial_q)
        if err > 0.005:
            raise RuntimeError(f"initial IK failed {err}")
        self.data.qpos[:6] = q
        mj.mj_forward(self.model, self.data)
        self.initialized = True
        for _ in range(150):
            self.step_raw(np.zeros(6), 0.0)
        self.kick_time_origin = float(self.data.time)
        self.pending_kicks = sorted(zip(self.cfg.kick_timestamps, self.cfg.kick_magnitudes), key=lambda event: event[0])
        self.next_kick_index = 0

    def snapshot(self, data=None):
        data = self.data if data is None else data
        state = np.empty(mj.mj_stateSize(self.model, self.state_spec))
        mj.mj_getState(self.model, data, state, self.state_spec)
        return state

    def set_prediction(self, state):
        mj.mj_setState(self.pred_model, self.pred, state, self.state_spec)
        mj.mj_forward(self.pred_model, self.pred)

    def pose(self, data=None, model=None):
        d = self.data if data is None else data
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def blockpos(self, b, data=None):
        return (self.data if data is None else data).xpos[self.blocks[b]].copy()

    def ik(self, target, q0=None, R=R_DOWN):
        d = self.ikdata
        m = self.pred_model
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0
        d.qpos[:6] = self.data.qpos[:6] if q0 is None else q0

        def residual(q):
            d.qpos[:6] = q
            mj.mj_forward(m, d)
            p, rot = self.pose(d)
            return np.r_[
                p - np.asarray(target),
                0.25 * Rotation.from_matrix(R @ rot.T).as_rotvec(),
            ]

        q = d.qpos[:6].copy()
        result = least_squares(
            residual,
            q,
            bounds=(
                [-6.28, -6.28, -3.13, -6.28, -6.28, -6.28],
                [6.28, 6.28, 3.13, 6.28, 6.28, 6.28],
            ),
            max_nfev=150,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        return result.x, float(np.linalg.norm(result.fun))

    def command(self, m, d, u, grip):
        # Identified simulation servo: velocity error plus model gravity/Coriolis feedforward.
        # No direct body forces, object position writes or weld constraints.
        d.ctrl[:6] = np.clip(
            self.cfg.kv * (u - d.qvel[:6]) + d.qfrc_bias[:6],
            m.actuator_ctrlrange[:6, 0],
            m.actuator_ctrlrange[:6, 1],
        )
        d.ctrl[6:8] = grip

    def motion_sample(self, data=None, model=None):
        """State and instantaneous TCP velocity, including all joint DOFs."""
        d = self.data if data is None else data
        m = self.model if model is None else model
        # Query a copy: refreshing kinematic caches must not change servo
        # feedforward, contact warm starts, or the next physical integration.
        query = mj.MjData(m)
        mj.mj_setState(m, query, self.snapshot(d), self.state_spec)
        mj.mj_forward(m, query)
        jac = np.zeros((3, m.nv))
        mj.mj_jacSite(m, query, jac, None, self.tcp)
        contacts = []
        for c in d.contact[:d.ncon]:
            contacts.append(dict(geoms=[m.geom(int(g)).name for g in (c.geom1, c.geom2)],
                                 distance=float(c.dist)))
        return dict(time=float(d.time), q=d.qpos[:6].copy(),
                    velocity=d.qvel[:6].copy(), position=self.pose(query)[0],
                    joint_positions=query.xanchor[:6].copy(),
                    cartesian_velocity=jac @ d.qvel,
                    torque=d.ctrl[:6].copy(), contacts=contacts,
                    blocks=np.array([self.blockpos(b, query) for b in range(self.nblock)]),
                    block_rotations=np.array([query.xmat[b].reshape(3, 3).copy() for b in self.blocks]))

    def measured_joint_state(self):
        """Observe once per solve; never modify the physical plant."""
        noise = self.joint_rng.normal(0, self.cfg.joint_observation_noise, 6)
        q = self.data.qpos[:6].copy()
        observed = q + self.cfg.joint_observation_bias + noise
        state = self.snapshot()
        observation = mj.MjData(self.pred_model)
        mj.mj_setState(self.pred_model, observation, state, self.state_spec)
        observation.qpos[:6] = observed
        measured_state = self.snapshot(observation)
        return measured_state, observed, noise

    def forecast_samples(self, state, commands, grip, *, plant_model=False):
        """Disposable same-command forecast; no optimization or plant writes."""
        m = self.model if plant_model else self.pred_model
        d = mj.MjData(m)
        mj.mj_setState(m, d, state, self.state_spec)
        mj.mj_forward(m, d)
        samples = [self.motion_sample(d, m)]
        for u in commands:
            for _ in range(self.nstep):
                self.command(m, d, u, grip)
                mj.mj_step(m, d)
                samples.append(self.motion_sample(d, m))
        return samples

    def step_raw(self, u, grip):
        self.command(self.model, self.data, u, grip)
        mj.mj_step(self.model, self.data)
        self.steps += 1
        if self._record_interval:
            self.interval_samples.append(self.motion_sample())
        if self.substep_guard:
            self.substep_guard()

    def step(self, u, grip, render=False):
        self.last_joint_kick = np.zeros(6)
        self.last_kick_events = []
        costs = 0.0
        self._record_interval = getattr(self, "diagnostics", None) is not None
        self.interval_samples = [self.motion_sample()] if self._record_interval else []
        try:
            for _ in range(self.nstep):
                self.step_raw(u, grip)
                costs += self.cfg.timestep * (
                    1.0
                    + 0.02 * float(u @ u)
                    + 0.002 * float(self.data.qvel[:6] @ self.data.qvel[:6])
                )
        finally:
            self._record_interval = False
        # Position disturbance after a completed control interval, never during
        # initialization, prediction, or optimizer evaluations. Do not scale by dt:
        # these settings describe a position increment, not a velocity disturbance.
        self.last_joint_kick = np.zeros(6)
        kick_events = []
        elapsed = float(self.data.time) - self.kick_time_origin
        while self.next_kick_index < len(self.pending_kicks):
            timestamp, magnitude = self.pending_kicks[self.next_kick_index]
            if timestamp > elapsed + 1e-9:
                break
            self.last_joint_kick += np.asarray(magnitude)
            kick_events.append(dict(requested_time=timestamp, applied_time=elapsed,
                                    magnitude=list(magnitude)))
            self.next_kick_index += 1
        self.last_kick_events = kick_events
        # Scheduled experiments exclude ongoing random/bias perturbations.
        random_kicks = (not self.cfg.kick_timestamps and
                        (self.cfg.joint_kick_half_range != 0 or self.cfg.joint_kick_bias != 0))
        if random_kicks:
            self.last_joint_kick = self.cfg.joint_kick_bias + self.kick_rng.uniform(
                -self.cfg.joint_kick_half_range, self.cfg.joint_kick_half_range, 6
            )
        if kick_events or random_kicks:
            self.data.qpos[:6] += self.last_joint_kick
            # Refresh pose/contact caches without integrating time or moving objects.
            mj.mj_forward(self.model, self.data)
            if getattr(self, "diagnostics", None) is not None:
                self.interval_samples.append(self.motion_sample())
            if self.substep_guard:
                self.substep_guard()
        self.last_u = np.array(u)
        self.grip = grip
        # Sample at the video playback rate; initialization steps and changes
        # to the physics timestep must not prevent frame capture.
        if render and self.data.time + 1e-9 >= self._next_frame_time:
            self.capture()
            self._next_frame_time = (np.floor(self.data.time * 5 + 1e-9) + 1) / 5
        self.trace.append(
            dict(
                time=float(self.data.time),
                q=self.data.qpos[:8].tolist(),
                velocity=self.data.qvel[:8].tolist(),
                u=np.asarray(u).tolist(),
                grip=float(grip),
                joint_kick=self.last_joint_kick.tolist(),
                kick_events=kick_events,
                kick_elapsed_time=elapsed,
                actuator_force=self.data.actuator_force.tolist(),
                blocks=[self.blockpos(b).tolist() for b in range(self.nblock)],
            )
        )
        return costs

    def rollout(self, state, commands, grip):
        self.set_prediction(state)
        out = []
        for u in commands:
            for _ in range(self.nstep):
                self.command(self.pred_model, self.pred, u, grip)
                mj.mj_step(self.pred_model, self.pred)
            out.append(
                (
                    self.pred.qpos[:6].copy(),
                    self.pred.qvel[:6].copy(),
                    *self.pose(self.pred),
                )
            )
        return out

    def contacts(self, b, data=None):
        d = self.data if data is None else data
        ans = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.blockgeoms[b] not in (g1, g2):
                continue
            f = np.zeros(6)
            mj.mj_contactForce(self.model, d, i, f)
            other = g2 if g1 == self.blockgeoms[b] else g1
            normal = c.frame.reshape(3, 3)[0].copy()
            # MuJoCo contact normal is geom1 -> geom2.
            force = c.frame.reshape(3, 3).T @ f[:3]
            if g1 == self.blockgeoms[b]:
                normal = -normal
                force = -force
            ans.append(
                dict(
                    other=other,
                    pad=other in self.pads,
                    side=self.pads.index(other) if other in self.pads else -1,
                    normal=normal,
                    position=c.pos.copy(),
                    force=force,
                    normal_force=float(f[0]),
                    friction_margin=float(
                        c.friction[0] * f[0] - np.linalg.norm(f[1:3])
                    ),
                    gap=float(c.dist),
                )
            )
        return ans

    def observe(self):
        p = np.array([self.blockpos(b) for b in range(self.nblock)])
        p += self.rng.normal(0, self.cfg.observation_noise, p.shape)
        p[:, 0] += self.cfg.observation_bias
        return p

    def capture(self):
        if self.render is None:
            self.render = mj.Renderer(self.model, height=480, width=640)
        camera = mj.MjvCamera()
        camera.lookat[:] = [0.5, 0, 0.4]
        camera.distance = 2.0
        camera.azimuth = 130
        camera.elevation = -30
        option = mj.MjvOption()
        option.geomgroup[3] = 1
        self.render.update_scene(self.data, camera=camera, scene_option=option)
        self.frames.append(self.render.render().copy())

    def close(self):
        if self.render is not None:
            self.render.close()
