"""Finite candidate refinement for the declared single IK branch.

Checks the low grasp/release configuration against supported obstacles. This
rejects geometrically blocked candidates without changing symbolic task rules.
It does not certify the entire path; runtime checks remain active.
"""

import numpy as np
import mujoco as mj
from .model import cells, SIDE


class CandidateGeometry:
    def __init__(self, world):
        self.w = world
        self.data = mj.MjData(world.pred_model)
        self.cache = {}
        self.poses = {}
        self.calls = 0
        self.robot = [
            i
            for i in range(world.model.ngeom)
            if world.model.geom(i).name.startswith("arm_")
            or world.model.geom(i).name
            in ["palm", "left_stem", "right_stem", "left_pad", "right_pad"]
        ]

    def __call__(self, s, held, destination):
        return all(
            self.check_height(s, held, destination, offset)
            for offset in [0.0, 0.035, 0.07, 0.105, 0.14, 0.21, 0.28, 0.35]
        )

    def check_height(self, s, held, destination, offset):
        positions = {}

        def center(b):
            if b not in positions:
                sup = s[b]
                if sup is None:
                    raise ValueError("held block used as support")
                positions[b] = (
                    cells()[sup].copy()
                    if isinstance(sup, str)
                    else center(sup) + [0, 0, SIDE]
                )
            return positions[b]

        target = (
            cells()[destination].copy()
            if isinstance(destination, str)
            else center(destination) + [0, 0, SIDE]
        )
        target = target.copy()
        target[2] = min(0.70, target[2] + offset)
        posekey = tuple(np.round(target, 6))
        m = self.w.pred_model
        if posekey not in self.poses:
            q, error = self.w.ik(target + [0, 0, 0.003], self.w.initial_q)
            self.poses[posekey] = (q, error)
        q, error = self.poses[posekey]
        if error > 0.004:
            return False
        for b in range(len(s)):
            if b == held:
                continue
            p = center(b)
            key = (posekey, tuple(np.round(p, 6)))
            if key not in self.cache:
                self.data.qpos[:] = self.w.data.qpos
                self.data.qpos[:6] = q
                self.data.qpos[6:8] = 0.028
                # This is a disposable geometric-query state, never the plant.
                address = m.jnt_qposadr[m.body(self.w.blocks[0]).jntadr[0]]
                self.data.qpos[address : address + 3] = p
                self.data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
                mj.mj_forward(m, self.data)
                self.cache[key] = all(
                    mj.mj_geomDistance(
                        m, self.data, g, self.w.blockgeoms[0], 0.004, None
                    )
                    > 0.002
                    for g in self.robot
                )
                self.calls += 1
            if not self.cache[key]:
                return False
        return True
