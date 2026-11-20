"""Finite-pad polyhedral closure/capacity diagnostics and measured grasp evidence.

The LP certifies the declared idealized patch and actuator bounds. It is NOT
a certificate that MuJoCo's instantaneous contact manifold realizes all rays.
"""

import numpy as np
from scipy.optimize import linprog


def rays(mu=0.6, finite_pad=True):
    G = []
    forces = []
    points = []
    sides = []
    corners = (
        [(-0.015, -0.015), (-0.015, 0.015), (0.015, -0.015), (0.015, 0.015)]
        if finite_pad
        else [(0, 0)]
    )
    for side, sign in enumerate([1, -1]):
        for y, z in corners:
            p = np.array([sign * 0.025, y, z])
            for angle in np.arange(8) * np.pi / 4:
                f = np.array([-sign, mu * np.cos(angle), mu * np.sin(angle)])
                G.append(np.r_[f, np.cross(p, f) / 0.05])
                forces.append(f)
                points.append(p)
                sides.append(side)
    return np.array(G).T, np.array(forces), np.array(points), np.array(sides)


def closure(mu=0.6, finite_pad=True):
    G, _, _, _ = rays(mu, finite_pad)
    n = G.shape[1]
    # alpha_i >= epsilon, sum alpha = 1, G alpha = 0.
    result = linprog(
        np.r_[np.zeros(n), -1.0],
        A_ub=np.c_[-np.eye(n), np.ones(n)],
        b_ub=np.zeros(n),
        A_eq=np.c_[np.vstack([G, np.ones(n)]), np.zeros(7)],
        b_eq=np.r_[np.zeros(6), 1],
        bounds=[(0, None)] * (n + 1),
        method="highs",
    )
    rank = int(np.linalg.matrix_rank(G))
    margin = float(result.x[-1]) if result.success else 0.0
    return dict(rank=rank, margin=margin, accepted=bool(rank == 6 and margin > 1e-6))


def capacity(
    mass=0.08,
    mu=0.6,
    jaw_limit=25.0,
    preload=2.0,
    disturbance=0.15,
    arm_matrix=None,
    arm_bias=None,
    arm_limits=None,
):
    G, _, _, sides = rays(mu)
    n = G.shape[1]
    B = np.vstack([sides == 0, sides == 1]).astype(float)
    A = np.vstack([B, -B])
    b = np.r_[np.full(2, jaw_limit), np.full(2, -preload)]
    if arm_matrix is not None:
        A = np.vstack([A, arm_matrix, -arm_matrix])
        b = np.r_[b, arm_limits - arm_bias, arm_limits + arm_bias]
    nominal = np.array([0, 0, mass * 9.81, 0, 0, 0])
    solutions = []
    residual = 0.0
    for d in [np.zeros(6)] + [s * np.eye(6)[i] for i in range(6) for s in [-1, 1]]:
        required = nominal + disturbance * d
        result = linprog(
            np.ones(n) * 1e-6,
            A_ub=A,
            b_ub=b,
            A_eq=G,
            b_eq=required,
            bounds=(0, None),
            method="highs",
        )
        if not result.success:
            return dict(accepted=False, reason=result.message, disturbance=disturbance)
        residual = max(residual, float(np.max(np.abs(G @ result.x - required))))
        solutions.append(result.x.tolist())
    return dict(
        accepted=True,
        residual=residual,
        disturbance=disturbance,
        jaw_limit=jaw_limit,
        preload=preload,
        mu=mu,
        convention="[Fx,Fy,Fz,Mx/0.05,My/0.05,Mz/0.05]; 12 vertices of cross-polytope",
    )


def measured(w, block, reference=None):
    contacts = [c for c in w.contacts(block) if c["pad"]]
    force = [sum(c["normal_force"] for c in contacts if c["side"] == i) for i in (0, 1)]
    p, R = w.pose()
    relative = R.T @ (w.blockpos(block) - p)
    slip = float(np.linalg.norm(relative - reference)) if reference is not None else 0.0
    membership = float(min(1.0, min(force) / 2.0) * max(0.0, 1.0 - slip / 0.008))
    return dict(
        accepted=bool(min(force) > 2 and max(force) < 25.5 and slip < 0.008),
        per_jaw=force,
        relative=relative.tolist(),
        slip=slip,
        membership=membership,
        friction_margin=min([c["friction_margin"] for c in contacts], default=-1.0),
    )


def prospective_capacity(w, block):
    """Static unsupported guard with arm torque and jaw-force bounds.

    This guard is not the paper's acceleration-consistent horizon certificate.
    Its static assumption is explicit; live contact/slip guards run in motion.
    """
    import mujoco as mj

    _, forces, points, sides = rays()
    R = w.data.xmat[w.blocks[block]].reshape(3, 3)
    center = w.blockpos(block)
    B = np.empty((6, len(forces)))
    jac = np.zeros((3, w.model.nv))
    for i, (force, point, side) in enumerate(zip(forces, points, sides)):
        body = w.model.body(("left", "right")[side] + "_finger").id
        mj.mj_jac(w.model, w.data, jac, None, center + R @ point, body)
        B[:, i] = jac[:, :6].T @ (R @ force)
    result = capacity(
        arm_matrix=B,
        arm_bias=w.data.qfrc_bias[:6],
        arm_limits=w.model.actuator_ctrlrange[:6, 1],
    )
    result["arm_torque_bounds_included"] = True
    result["scope"] = (
        "static prospective unsupported load; not horizon-wide servo attainability"
    )
    return result
