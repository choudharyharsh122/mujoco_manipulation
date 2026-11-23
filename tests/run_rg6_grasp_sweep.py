#!/usr/bin/env python3
"""Compare closing heights or wrist tilts in an isolated PyBullet grasp test."""

import argparse
import json
from pathlib import Path
import sys


OFFSETS_MM = (44, 47, 50, 53)
WRIST_TILTS_DEG = (-12, -6, 0, 2, 4, 6, 12)
GRID_HEIGHTS_MM = (44, 47, 50, 53, 56)
GRID_TILTS_DEG = (-2, 0, 2, 4, 6, 8, 10)
LATERAL_X_MM = (-3, 0, 3)
LATERAL_Y_MM = (-4, 0, 4)
LIFT_METRES = 0.06
PHYSICS_HZ = 240
JAW_MOTOR_LIMIT = 5.0  # Matches the mission fixture's grip setting of 100 / 20.


def prepare(path, wrist_sweep=False, grid=False, lateral=False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from scipy.spatial.transform import Rotation

    from mission_mujoco.physics import R_DOWN, World

    world = World({0: "p0c0"})
    trials = []
    try:
        if lateral:
            settings = [(47, -2, dx, dy) for dx in LATERAL_X_MM for dy in LATERAL_Y_MM]
        elif grid:
            settings = [(height, tilt, 0, 0) for height in GRID_HEIGHTS_MM for tilt in GRID_TILTS_DEG]
        elif wrist_sweep:
            settings = [(50, tilt, 0, 0) for tilt in WRIST_TILTS_DEG]
        else:
            settings = [(offset, 0, 0, 0) for offset in OFFSETS_MM]
        for offset, tilt, dx, dy in settings:
            pose = world.blockpos(0) + [dx / 1000, dy / 1000, 0.003 + offset / 1000 + 0.119]
            # The final term is the measured open-jaw difference between the
            # MuJoCo TCP and the RG6 pad reference in this fixture.
            joints = []
            seed = world.data.qpos[:6].copy()
            orientation = Rotation.from_euler("y", tilt, degrees=True).as_matrix() @ R_DOWN
            # Keep the calibrated open-jaw point fixed as the tool tilts.
            pose += 0.119 * (np.array([0.0, 0.0, -1.0]) - orientation[:, 2])
            for step in range(25):
                seed, residual = world.ik(pose + [0, 0, LIFT_METRES * step / 24], seed, R=orientation)
                if residual > 0.004:
                    raise RuntimeError(f"IK failed at closing offset {offset} mm")
                joints.append(seed.tolist())
            trials.append({"offset_mm": offset, "tilt_deg": tilt, "xy_mm": [dx, dy], "joints": joints})
        path.write_text(json.dumps(trials, indent=2) + "\n")
    finally:
        world.close()


def sample(bullet, block, finger_ids):
    import numpy as np

    center, quaternion = bullet.getBasePositionAndOrientation(block)
    cube_rotation = np.asarray(bullet.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    left_quaternion = bullet.getLinkState(bullet.robot, finger_ids[0])[5]
    left_rotation = np.asarray(bullet.getMatrixFromQuaternion(left_quaternion)).reshape(3, 3)
    moment = np.zeros(3)
    normal_moment = np.zeros(3)
    face_offsets = []
    fields = bullet.contactPoints
    for link in finger_ids:
        hits = bullet.getContactPoints(bodyA=bullet.robot, bodyB=block, linkIndexA=link)
        weighted_position = np.zeros(3)
        total = 0.0
        for hit in hits:
            force = hit[fields["FORCE"]]
            if force <= 0:
                continue
            lever = np.asarray(hit[fields["POSITIONB"]]) - center
            normal = -force * np.asarray(hit[fields["NORMAL"]])
            friction = sum(hit[fields[f"FRICTION{i}"]] * np.asarray(hit[fields[f"FRICTIONDIR{i}"]])
                           for i in (1, 2))
            normal_moment += np.cross(lever, normal)
            moment += np.cross(lever, normal + friction)
            weighted_position += force * (cube_rotation.T @ lever)
            total += force
        if total < 0.1:
            return None
        face_offsets.append(weighted_position[1:] / total)
    return (np.asarray(face_offsets), np.linalg.norm(moment), np.linalg.norm(normal_moment),
            left_rotation.T @ cube_rotation, center[2],
            np.degrees(np.arccos(np.clip(cube_rotation[2, 2], -1, 1))))


def simulate(path, hold_seconds):
    import numpy as np
    from pyUR.pyur import pyUR

    class TrialClient(pyUR):
        def __del__(self):
            if self.isConnected():
                super().__del__()

    hold_steps = hold_seconds * PHYSICS_HZ
    for trial in json.loads(path.read_text()):
        label = f"{trial['offset_mm']} mm, {trial.get('tilt_deg', 0):+d} deg"
        if "xy_mm" in trial:
            dx, dy = trial["xy_mm"]
            label += f", XY [{dx:+d}, {dy:+d}] mm"
        bullet = TrialClient("ros_mission_approach.yaml")
        try:
            names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
            joints = [bullet.find_joint_id(name)[0] for name in names]
            fingers = [bullet.get_link_id(name) for name in ("left_inner_finger", "right_inner_finger")]
            block = next(obj_id for obj_id, name, *_ in bullet.other_objects if name == "mission_block")
            supports = {obj_id for obj_id, name, *_ in bullet.other_objects if name == "mission_support"}
            peak_support_force = 0.0

            def step_simulation(count):
                nonlocal peak_support_force
                for step in range(count):
                    bullet.stepSimulation()
                    if step % 4 != 0:
                        continue
                    for finger in fingers:
                        for hit in bullet.getContactPoints(bodyA=bullet.robot, linkIndexA=finger):
                            if hit[bullet.contactPoints["IDB"]] in supports:
                                peak_support_force = max(peak_support_force, hit[bullet.contactPoints["FORCE"]])

            finger_joint = bullet.gripper.finger_joint_id
            for joint, position in zip(joints, trial["joints"][0]):
                bullet.resetJointState(bullet.robot, joint, position)
            bullet.setJointMotorControlArray(bullet.robot, joints, bullet.POSITION_CONTROL,
                                             targetPositions=trial["joints"][0])
            bullet.setJointMotorControl2(bullet.robot, finger_joint, bullet.POSITION_CONTROL,
                                         targetPosition=-0.6, force=JAW_MOTOR_LIMIT, maxVelocity=0.5)
            step_simulation(500)
            initial_z = bullet.getBasePositionAndOrientation(block)[0][2]
            bullet.setJointMotorControl2(bullet.robot, finger_joint, bullet.POSITION_CONTROL,
                                         targetPosition=0.6, force=JAW_MOTOR_LIMIT, maxVelocity=0.5)
            step_simulation(900)
            if sample(bullet, block, fingers) is None:
                print(f"{label}: no bilateral grasp after closing", flush=True)
                continue
            for positions in trial["joints"][1:]:
                bullet.setJointMotorControlArray(bullet.robot, joints, bullet.POSITION_CONTROL,
                                                 targetPositions=positions)
                step_simulation(40)
            measurements = []
            for step in range(hold_steps):
                bullet.stepSimulation()
                if step % 4 == 0:
                    measured = sample(bullet, block, fingers)
                    if measured is not None:
                        measurements.append(measured)
            if not measurements:
                print(f"{label}: lost contact after lifting", flush=True)
                continue
            offsets, torques, normal_torques, rotations, heights, cube_tilts = zip(*measurements)
            reference = rotations[0]
            angles = [np.degrees(np.arccos(np.clip((np.trace(reference.T @ rotation) - 1) / 2, -1, 1)))
                      for rotation in rotations]
            medians = np.median(offsets, axis=0) * 1000
            # Measure distances before taking the median so opposite sides of
            # a face do not falsely average into a centered contact.
            max_jaw_offset = np.median(np.linalg.norm(offsets, axis=2), axis=0).max() * 1000
            lift_mm = (min(heights) - initial_z) * 1000
            print(f"{label}: bilateral {len(measurements)}/{hold_steps // 4}, "
                  f"minimum lift {lift_mm:.1f} mm, max rotation drift {max(angles):.2f} deg, "
                  f"worse jaw median distance from face center {max_jaw_offset:.1f} mm, "
                  f"median face offsets [y,z] left {medians[0].round(1)} mm, "
                  f"right {medians[1].round(1)} mm, median force-derived moment {np.median(torques):.4f} Nm, "
                  f"median normal-only moment {np.median(normal_torques):.4f} Nm, "
                  f"peak cube tilt {max(cube_tilts):.2f} deg, "
                  f"peak finger-support contact during setup, close, and lift {peak_support_force:.2f} N",
                  flush=True)
        finally:
            bullet.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "simulate"))
    parser.add_argument("trial_file", type=Path)
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--wrist-sweep", action="store_true", help="Prepare wrist tilts at a fixed 50 mm offset")
    sweep.add_argument("--grid", action="store_true", help="Prepare 35 height and tilt combinations")
    sweep.add_argument("--lateral", action="store_true",
                       help="Prepare nine XY offsets at 47 mm and -2 degrees")
    parser.add_argument("--hold-seconds", type=int, default=3, help="Simulated hold time per trial")
    args = parser.parse_args()
    if args.hold_seconds <= 0:
        parser.error("--hold-seconds must be positive")
    if args.mode == "prepare":
        prepare(args.trial_file, args.wrist_sweep, args.grid, args.lateral)
    else:
        simulate(args.trial_file, args.hold_seconds)


if __name__ == "__main__":
    main()
