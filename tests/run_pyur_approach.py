#!/usr/bin/env python3
"""Preview the existing mission planner's first approach for a single block."""

from pathlib import Path
import sys


def plan_approach(world):
    from mission_mujoco.execution import Executor
    from mission_mujoco.geometry import CandidateGeometry
    from mission_mujoco.learning import Models
    from mission_mujoco.planning import Planner

    source, destination = "p0c0", "p1c0"
    models = Models()
    executor = Executor(world, models, plot_dir=None, plot_mode="none")
    planner = Planner(models, geometry=CandidateGeometry(world))
    plan, _ = planner.solve(
        executor.supports, [((destination,), 0.0)], [source, destination],
        anchor=executor.anchor, held=executor.held,
    )
    if not plan or plan[0][0] != "MoveEmpty":
        raise RuntimeError(f"Expected the first action to be MoveEmpty, got {plan}")
    print(f"Plan: {plan}", flush=True)
    return executor, plan


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mission_mujoco.physics import World

    world = World({0: "p0c0"})
    try:
        executor, plan = plan_approach(world)
        _, block, _ = plan[0]
        print(f"Selected block {block} center [m]: {world.blockpos(block)}")
        print("Approach waypoints in the MuJoCo world frame [m]")
        q = world.data.qpos[:6].copy()
        for phase, target in executor.move_empty_targets(block):
            q, residual = world.ik(target, q)
            if residual > 0.004:
                raise RuntimeError(f"{phase} target fails IK: {residual:.6f}")
            x, y, z = target
            print(f"  {phase:8} [{x:.4f}, {y:.4f}, {z:.4f}]")
    finally:
        world.close()


if __name__ == "__main__":
    main()
