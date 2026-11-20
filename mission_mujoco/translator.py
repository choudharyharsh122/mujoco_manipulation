"""Optional finite-difference fuzzy-sensitivity experiment in free space.

Repeated feedback solves on disposable prediction copies differentiate a fixed
number of closed-loop steps. This is a numerical sensitivity ablation, not an
implementation of the manuscript's KKT derivative or a contact-mode theorem.
"""

import copy, json, argparse
from pathlib import Path
import numpy as np
import mujoco as mj
from .physics import World
from .controller import MPC
from .grounding import anchor_membership


def calibrate_anchor(world, observed_offset=(0.006, 0.0, 0.0), steps=6, epsilon=0.0003):
    target = world.pose()[0].copy()
    state = world.snapshot()
    cache = {}

    def loss(eta):
        key = tuple(eta)
        if key in cache:
            return cache[key]
        shadow = copy.copy(world)
        shadow.model = world.pred_model
        shadow.pred_model = world.pred_model
        shadow.data = mj.MjData(shadow.model)
        shadow.pred = mj.MjData(shadow.model)
        shadow.ikdata = mj.MjData(shadow.model)
        mj.mj_setState(shadow.model, shadow.data, state, shadow.state_spec)
        mj.mj_forward(shadow.model, shadow.data)
        shadow.last_u = world.last_u.copy()
        shadow.trace = []
        shadow.frames = []
        shadow.render = None
        shadow.substep_guard = None
        controller = MPC(shadow)
        controller.target_bias = np.asarray(eta).copy()
        for _ in range(steps):
            shadow.step(controller.solve(target, 0.0), 0.0)
        observation = shadow.pose()[0] + np.asarray(observed_offset)
        degree = anchor_membership(observation, target)
        value = 0.5 * (degree - 1) ** 2
        cache[key] = float(value)
        return float(value)

    def gradient(h):
        return np.array(
            [
                (loss(np.eye(3)[i] * h) - loss(-np.eye(3)[i] * h)) / (2 * h)
                for i in range(3)
            ]
        )

    g = gradient(epsilon)
    g_half = gradient(epsilon / 2)
    relative = float(np.linalg.norm(g - g_half) / max(np.linalg.norm(g_half), 1e-8))
    before = loss(np.zeros(3))
    candidate = np.zeros(3)
    after = before
    valid = bool(np.isfinite(g).all() and relative < 0.25 and np.linalg.norm(g) > 1e-8)
    if valid:
        direction = -g / np.linalg.norm(g)
        for trust in [0.003, 0.0015, 0.00075, 0.000375]:
            proposed = np.clip(trust * direction, -0.005, 0.005)
            value = loss(proposed)
            if value < before:
                candidate = proposed
                after = value
                break
    return dict(
        method="central differences of repeated MPC feedback solves",
        KKT_derivative=False,
        fixed_steps=steps,
        epsilon=epsilon,
        gradient=g.tolist(),
        half_step_gradient=g_half.tolist(),
        relative_gradient_difference=relative,
        sensitivity_valid=valid,
        loss_before=before,
        loss_after=after,
        target_bias_update=candidate.tolist(),
        accepted=bool(after < before),
        bound=0.005,
        observation_offset=list(observed_offset),
        scope="synthetic free-space anchor observation; no contact differentiation",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/translator.json")
    args = p.parse_args()
    w = World()
    before = w.snapshot().copy()
    report = calibrate_anchor(w)
    report["plant_unchanged"] = bool(np.array_equal(before, w.snapshot()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    w.close()
    return 0 if report["accepted"] and report["plant_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
