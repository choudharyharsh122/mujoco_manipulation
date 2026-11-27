#!/usr/bin/env python3
"""Drive a sequence of free-space PyUR reaches using the MuJoCo MPC predictor."""

import argparse
import json
import socket
import sys
import time
from math import degrees
from pathlib import Path


PULSE_PERIOD = 0.05
PULSE_CYCLES = 160
DEFAULT_OFFSETS = [(0.0, 0.0, 0.03), (0.02, 0.0, 0.03), (0.02, 0.0, 0.01)]
# PyBullet RG6 jaws sweep downward as they close. This fixture's measured
# open-jaw offset leaves both fingers room to clear the support.
RG6_OPEN_JAW_Z_OFFSET = 0.047
# The 75 mm service setting puts the knuckle at 0 rad. The URDF geometry
# gives about 38 mm less pad sweep than the 150 mm setting at -0.6 rad.
RG6_HALF_OPEN_JAW_Z_OFFSET = 0.009
GRASP_XY_TOLERANCE = 0.001
TWO_BLOCK_NAMES = ("mission_block", "mission_block_blue")
TWO_BLOCK_SOURCES = {0: "p0c0", 1: "p0c1"}
TWO_BLOCK_GOAL = ("p1c1", "p1c0")


def send(connection, seq, **fields):
    connection.sendall(json.dumps({"seq": seq, **fields}, allow_nan=False).encode() + b"\n")


def make_controller(approach=False, two_block=False):
    # Prefer this checkout over another installation in the Python environment.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import mujoco
    import numpy as np

    from mission_mujoco.controller import MPC
    from mission_mujoco.physics import Config, R_DOWN, World

    class ReachWorld(World):
        # Use the measured starting tool orientation for this free-space reach.
        orientation = R_DOWN

        def ik(self, target, q0=None, R=None):
            return super().ik(target, q0, R=self.orientation if R is None else R)

    class ReachMPC(MPC):
        def tracking(self, q, p, R, qref, target):
            orientation_in_reference_frame = R_DOWN @ self.w.orientation.T @ R
            return super().tracking(q, p, orientation_in_reference_frame, qref, target)

    supports = TWO_BLOCK_SOURCES if two_block else {0: "p0c0"} if approach else {}
    world = ReachWorld(supports=supports, config=Config(velocity_limit=0.05, slew=0.01))
    return mujoco, np, world, ReachMPC(world, horizon=1, maxiter=3)


def plan_two_block_swap(world):
    from mission_mujoco.execution import Executor
    from mission_mujoco.geometry import CandidateGeometry
    from mission_mujoco.learning import Models
    from mission_mujoco.planning import Planner

    models = Models()
    executor = Executor(world, models, plot_dir=None, plot_mode="none")
    if executor.supports != TWO_BLOCK_SOURCES:
        raise RuntimeError(f"Unexpected initial supports in the predictor: {executor.supports}")
    planner = Planner(models, geometry=CandidateGeometry(world))
    sites = [*TWO_BLOCK_SOURCES.values(), *TWO_BLOCK_GOAL]
    plan, _ = planner.solve(executor.supports, [(TWO_BLOCK_GOAL, 0.0)], sites,
                            anchor=executor.anchor, held=executor.held)
    kinds = ("MoveEmpty", "PickLift", "MoveHolding", "PlaceRelease")
    transfers = []
    for index in range(0, len(plan), len(kinds)):
        actions = plan[index:index + len(kinds)]
        if tuple(action[0] for action in actions) != kinds:
            raise RuntimeError(f"Expected a complete transfer, got {actions}")
        _, block, destination = actions[-1]
        if any(action[1] != block for action in actions) or actions[2][2] != destination:
            raise RuntimeError(f"Inconsistent transfer in planned actions: {actions}")
        transfers.append((block, destination))
    if sorted(transfers) != list(enumerate(TWO_BLOCK_GOAL)):
        raise RuntimeError(f"Expected two direct transfers for the color swap, got {transfers}")
    print(f"Two-block planner actions: {plan}", flush=True)
    return executor, plan, iter(transfers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pulse", action="store_true", help="Test TCP with a fixed shoulder pulse")
    mode.add_argument("--approach", action="store_true", help="Run the planner's MoveEmpty approach only")
    mode.add_argument("--pick", action="store_true", help="Descend, grasp, and verify a short lift from pregrasp")
    mode.add_argument("--carry", action="store_true", help="Pick, hold for three seconds, then carry to preplace")
    mode.add_argument("--place", action="store_true", help="Approach, pick, carry, and release in one run")
    mode.add_argument("--two-block", action="store_true", help="Plan and execute the two-block color derangement")
    grasp_mode = parser.add_mutually_exclusive_group()
    grasp_mode.add_argument("--baseline-grasp", action="store_true",
                            help="Use the centered grasp with no wrist tilt (default)")
    grasp_mode.add_argument("--offset-grasp", action="store_true",
                            help="Test the 47 mm, -2 degree, +3/-4 mm grasp from the sweep")
    slip_options = parser.add_mutually_exclusive_group()
    slip_options.add_argument("--slip-limit-mm", type=float,
                              help="Override the 8 mm slip limit after the hold in a PyBullet --place test")
    slip_options.add_argument("--ignore-slip", action="store_true",
                              help="Log positional slip without stopping after the hold in PyBullet --place")
    parser.add_argument("--offset", type=float, nargs=3, action="append",
                        metavar=("DX", "DY", "DZ"),
                        help="Repeat for multiple targets. Offsets are in meters from the starting tool position.")
    args = parser.parse_args()
    args.place = args.place or args.two_block
    open_jaw_z_offset = RG6_HALF_OPEN_JAW_Z_OFFSET if args.two_block else RG6_OPEN_JAW_Z_OFFSET
    carry = args.carry or args.place
    pick = args.pick or carry
    mission = args.approach or pick
    if (args.offset_grasp or args.baseline_grasp) and not pick:
        parser.error("Grasp options require --pick, --carry, or --place")
    if args.slip_limit_mm is not None and (not args.place or not 8 <= args.slip_limit_mm <= 20):
        parser.error("--slip-limit-mm requires --place and a value between 8 and 20")
    if args.ignore_slip and not args.place:
        parser.error("--ignore-slip requires --place")
    if mission and args.offset:
        parser.error("--offset is only for the free-space reach")
    controller = None if args.pulse else make_controller(mission, args.two_block)
    approaching = args.approach or args.place
    grasp_orientation = None
    grasp_xy_offset = [0.003, -0.004] if args.offset_grasp else [0.0, 0.0]
    if mission:
        from run_pyur_approach import plan_approach

        if args.two_block:
            from mission_mujoco.model import cells

            source_centers = cells()
            executor, plan, transfers = plan_two_block_swap(controller[2])
            block, destination = next(transfers)
            approach_orientation = controller[2].orientation.copy()
        else:
            executor, plan = plan_approach(controller[2])
            _, block, _ = plan[0]
        if args.offset_grasp:
            from scipy.spatial.transform import Rotation

            world = controller[2]
            grasp_orientation = Rotation.from_euler("y", -2, degrees=True).as_matrix() @ world.orientation
            if not approaching:
                world.orientation = grasp_orientation
        if pick:
            print("Grasp: shifted (+3/-4 mm, -2 deg)" if args.offset_grasp
                  else "Grasp: baseline (centered target, no wrist tilt)", flush=True)
        if carry and not args.two_block:
            kind, carry_block, destination = plan[2]
            if kind != "MoveHolding" or carry_block != block:
                raise RuntimeError(f"Expected a MoveHolding action for block {block}: {plan}")
            if args.place and plan[3] != ("PlaceRelease", block, destination):
                raise RuntimeError(f"Expected PlaceRelease after MoveHolding: {plan}")
    block_start = None
    closing = False
    contact_deadline = None
    grasp_relative = None
    hold_started = None
    next_grasp_log = 0.0
    reported_baseline_slip = False
    support_samples = 0
    release_started = None
    settled_since = None
    settled_position = None
    phases = None
    step_limit = 5000 if args.place else 2500 if carry else 900 if mission else 300
    placed = {}
    if args.two_block:
        step_limit *= 2
    targets = None
    target_index = 0
    solve_times = []
    settling_cycles = 0
    dwell = 0
    pulse_start = None

    with socket.create_connection((args.host, args.port), timeout=5.0) as connection:
        connection.settimeout(5.0)
        with connection.makefile("rb") as stream:
            for step in range(step_limit):
                line = stream.readline(4096)
                if not line.endswith(b"\n"):
                    raise RuntimeError("Bridge closed the connection or sent an invalid observation")
                state = json.loads(line)
                seq = state["seq"]
                tick = time.monotonic()

                if args.pulse:
                    shoulder = state["names"].index("shoulder_pan_joint")
                    position = state["q"][shoulder]
                    if pulse_start is None:
                        pulse_start = position
                    if step == PULSE_CYCLES:
                        send(connection, seq, stop=True)
                        change = position - pulse_start
                        print(f"Measured shoulder change: {change:+.4f} rad ({degrees(change):+.1f}°)")
                        return
                    command = [0.05 if name == "shoulder_pan_joint" else 0.0
                               for name in state["names"]]
                    period = PULSE_PERIOD
                else:
                    mujoco, np, world, mpc = controller
                    order = [state["names"].index(world.model.joint(j).name) for j in range(6)]
                    world.data.qpos[:6] = np.asarray(state["q"])[order]
                    world.data.qvel[:6] = np.asarray(state["qd"])[order]
                    world.last_u[:] = np.asarray(state["last_command"])[order]
                    if args.two_block:
                        block_states = state.get("blocks")
                        if not isinstance(block_states, dict) or set(block_states) != set(TWO_BLOCK_NAMES):
                            raise RuntimeError("Expected both named blocks from the PyUR --two-block bridge")
                        for index, name in enumerate(TWO_BLOCK_NAMES):
                            block_state = block_states[name]
                            position = np.asarray(block_state["position"]) * [-1, -1, 1]
                            x, y, z, w = block_state["orientation"]
                            # Convert base_link XYZW to MuJoCo WXYZ after a half-turn about Z.
                            world.data.joint(f"free{index}").qpos[:] = [*position, -z, -y, x, w]
                            if index in placed:
                                original = placed[index]
                                if (np.linalg.norm(position - original) > 0.005
                                        or block_state["destination_load"] < 0.2):
                                    raise RuntimeError(f"Previously placed block {name} moved or lost support")
                        if state["active_block"] != TWO_BLOCK_NAMES[block]:
                            send(connection, seq, select_block=TWO_BLOCK_NAMES[block])
                            continue
                    centering_grasp = pick and not approaching and not closing and target_index == 0
                    if mission:
                        # The corresponding PyUR base_link is rotated 180 degrees around Z.
                        actual_pose = np.asarray(state["grasp_center_base_link"]) * [-1, -1, 1]
                        block_pose = np.asarray(state["block_center_base_link"]) * [-1, -1, 1]
                        if not np.isfinite([actual_pose, block_pose]).all():
                            raise ValueError("Invalid RG6 or block feedback")
                        retreating_after_release = args.two_block and placed and target_index == 0
                        if approaching and not retreating_after_release and actual_pose[2] - block_pose[2] < 0.085:
                            raise RuntimeError("Grasp center is too close to the block for pregrasp")
                        if block_start is None:
                            source = (source_centers[TWO_BLOCK_SOURCES[block]]
                                      if args.two_block else world.blockpos(block))
                            if np.linalg.norm(block_pose - source) > 0.015:
                                raise RuntimeError("PyUR block is outside the expected source cell")
                            block_start = block_pose
                        elif not closing and np.linalg.norm(block_pose - block_start) > 0.008:
                            raise RuntimeError("The PyUR block moved before grasping")
                        if pick and not approaching:
                            if targets is None:
                                relative = actual_pose - block_pose
                                if (np.linalg.norm(relative[:2]) > 0.015
                                        or abs(relative[2] - 0.12) > 0.02
                                        or state["finger_separation"] < 0.065):
                                    raise RuntimeError("Run --approach first and leave the RG6 open at pregrasp")
                            actual_pose = np.asarray(state["pad_center_base_link"]) * [-1, -1, 1]
                            if not np.isfinite(actual_pose).all():
                                raise ValueError("Invalid RG6 pad-center feedback")
                            if not closing and actual_pose[2] - block_pose[2] < -0.006:
                                raise RuntimeError("RG6 descended below the expected grasp height")
                            if target_index == 1 and hold_started is None and np.linalg.norm(block_pose - actual_pose - grasp_relative) > 0.025:
                                raise RuntimeError("Block slipped relative to the RG6 during lift")
                        if centering_grasp:
                            grasp_target = block_pose + [
                                *grasp_xy_offset, 0.003 + open_jaw_z_offset
                            ]
                        world.data.joint(f"free{block}").qpos[:3] = block_pose
                    if closing and target_index >= 1:
                        world.data.qpos[6:8] = 0.038
                    elif args.two_block:
                        world.data.qpos[6:8] = 0.0
                    mujoco.mj_forward(world.model, world.data)
                    pose, rotation = world.pose()
                    if release_started is not None:
                        connection.settimeout(5.0)
                        center = executor.support_center(destination)
                        horizontal = np.linalg.norm(block_pose[:2] - center[:2])
                        height = abs(block_pose[2] - center[2])
                        released = (not state["gripper_closed"] and max(state["per_jaw_force"]) < 0.05
                                    and state["finger_separation"] >= 0.065)
                        if state["destination_load"] >= 0.2 and released and horizontal <= 0.015 and height <= 0.012:
                            if settled_position is None or np.linalg.norm(block_pose - settled_position) > 0.002:
                                settled_position = block_pose.copy()
                                settled_since = tick
                            elif tick - settled_since >= 1.0:
                                print(f"PlaceRelease verified at {destination}. Horizontal error "
                                      f"{horizontal * 1000:.1f} mm, vertical error {height * 1000:.1f} mm, "
                                      f"support load {state['destination_load']:.2f} N.", flush=True)
                                if args.two_block:
                                    placed[block] = block_pose.copy()
                                    following = next(transfers, None)
                                    if following is not None:
                                        block, destination = following
                                        world.orientation = approach_orientation
                                        approaching, closing = True, False
                                        targets = phases = block_start = None
                                        release_started = settled_since = settled_position = None
                                        contact_deadline = grasp_relative = hold_started = None
                                        target_index = dwell = support_samples = 0
                                        next_grasp_log = 0.0
                                        reported_baseline_slip = False
                                        send(connection, seq, select_block=TWO_BLOCK_NAMES[block])
                                        continue
                                    print("Two-block derangement verified: blue then red on platform 1.",
                                          flush=True)
                                send(connection, seq, stop=True)
                                return
                        else:
                            settled_position = None
                        if tick - release_started > 15.0:
                            raise RuntimeError(f"Cube did not settle at {destination} after release. "
                                               f"Horizontal error {horizontal * 1000:.1f} mm, "
                                               f"support load {state['destination_load']:.2f} N")
                        send(connection, seq, velocity=[0.0] * 6)
                        time.sleep(world.cfg.dt)
                        continue
                    if targets is None:
                        if approaching:
                            phases, targets = zip(*executor.move_empty_targets(block, start=actual_pose))
                            print(f"Initial MuJoCo TCP minus RG6 midpoint [m]: {pose - actual_pose}", flush=True)
                        elif pick:
                            phases = ("descend", "lift")
                            targets = [grasp_target, None]
                        else:
                            offsets = np.asarray(args.offset or DEFAULT_OFFSETS)
                            distances = np.linalg.norm(offsets, axis=1)
                            if not np.isfinite(offsets).all() or np.any(distances < 0.01) or np.any(distances > 0.05):
                                raise ValueError("Use finite target offsets between 1 and 5 cm")
                            targets = pose + offsets
                            world.orientation = rotation
                        for index, waypoint in enumerate(targets, start=1):
                            if waypoint is None:
                                continue
                            phase = f" {phases[index - 1]}" if phases else ""
                            print(f"Target {index}/{len(targets)}{phase} in MuJoCo base frame [m]: {waypoint}", flush=True)
                    if pick and closing and target_index == 0:
                        connection.settimeout(5.0)
                        if state["gripper_closed"] and state["bilateral_contact"]:
                            grasp_relative = block_pose - actual_pose
                            targets[1] = actual_pose + [0, 0, 0.06]
                            target_index = 1
                            print("Bilateral RG6 contact confirmed. Lifting 60 mm.", flush=True)
                        else:
                            if contact_deadline is None:
                                contact_deadline = time.monotonic() + 2.0
                            if time.monotonic() >= contact_deadline:
                                raise RuntimeError("RG6 closed without bilateral block contact. "
                                                   f"Block contact links: {state['block_contact_links']}; "
                                                   f"finger separation: {state['finger_separation'] * 1000:.1f} mm")
                            send(connection, seq, velocity=[0.0] * 6)
                            time.sleep(world.cfg.dt)
                            continue
                    if args.place and phases and phases[target_index] == "placement descent":
                        center = executor.support_center(destination)
                        on_destination = (state["destination_load"] >= 0.2
                                          and np.linalg.norm(block_pose[:2] - center[:2]) <= 0.015
                                          and abs(block_pose[2] - center[2]) <= 0.015)
                        if on_destination:
                            support_samples += 1
                            if support_samples >= 3:
                                print(f"Destination carries {state['destination_load']:.2f} N. Opening RG6.", flush=True)
                                send(connection, seq, open_gripper=True)
                                connection.settimeout(15.0)
                                release_started = time.monotonic()
                            else:
                                send(connection, seq, velocity=[0.0] * 6)
                                time.sleep(world.cfg.dt)
                            continue
                        support_samples = 0
                    if hold_started is not None:
                        relative = block_pose - actual_pose
                        slip = np.linalg.norm(relative - grasp_relative)
                        limit_mm = (args.slip_limit_mm or 8.0) if target_index > 1 else 8.0
                        log_only = args.ignore_slip and target_index > 1
                        exceeded = not log_only and slip * 1000 > limit_mm
                        if (log_only or limit_mm > 8) and slip > 0.008 and not reported_baseline_slip:
                            policy = "logging slip only" if log_only else f"allowing up to {limit_mm:g} mm of slip"
                            print(f"Slip exceeds the usual 8 mm limit. In this simulation test, {policy} "
                                  f"while both fingers retain contact.", flush=True)
                            reported_baseline_slip = True
                        if tick >= next_grasp_log or not state["bilateral_contact"] or exceeded:
                            stage = "hold" if target_index == 1 else phases[target_index]
                            print(f"Grasp {stage} +{tick - hold_started:.2f}s: "
                                  f"forces [N] {state['per_jaw_force']}, separation "
                                  f"{state['finger_separation'] * 1000:.1f} mm, "
                                  f"block relative to RG6 [mm] {np.round(relative * 1000, 1)}, "
                                  f"slip {slip * 1000:.1f} mm", flush=True)
                            next_grasp_log = tick + (0.25 if target_index == 1 else 2.0)
                        if not state["bilateral_contact"]:
                            send(connection, seq, stop=True)
                            raise RuntimeError("Bilateral grasp contact lost; carry stopped")
                        if exceeded:
                            send(connection, seq, stop=True)
                            raise RuntimeError(f"Block slipped more than {limit_mm:g} mm; carry stopped")
                        if target_index == 1:
                            if tick - hold_started < 3.0:
                                send(connection, seq, velocity=[0.0] * 6)
                                time.sleep(world.cfg.dt)
                                continue
                            grasp_relative = relative
                            waypoints = executor.move_holding_targets(destination, start=actual_pose)
                            for phase, waypoint in waypoints:
                                if phase != "lifted transit height":
                                    waypoint[:2] -= grasp_relative[:2]
                                phases += (phase,)
                                targets.append(waypoint)
                                print(f"MoveHolding {phase} [m]: {waypoint}", flush=True)
                            target_index += 1
                            print(f"Hold passed. Carrying to {destination}.", flush=True)
                    target = grasp_target if centering_grasp else targets[target_index]
                    if args.place and phases[target_index] in ("preplace", "placement descent"):
                        target = target.copy()
                        center_xy = executor.support_center(destination)[:2]
                        target[:2] = actual_pose[:2] + center_xy - block_pose[:2]
                    measured_pose = actual_pose if mission else pose
                    error = np.linalg.norm(target - measured_pose)
                    tolerance = 0.003 if pick and not approaching and target_index == 0 else 0.006
                    aligned = True
                    if grasp_orientation is not None and not approaching and target_index == 0:
                        aligned = Rotation.from_matrix(world.orientation @ rotation.T).magnitude() <= np.deg2rad(1)
                    needs_motion = error > tolerance or not aligned
                    if centering_grasp:
                        xy_error = np.linalg.norm(target[:2] - actual_pose[:2])
                        needs_motion = needs_motion or xy_error > GRASP_XY_TOLERANCE
                    dwell = dwell + 1 if not needs_motion and np.linalg.norm(world.data.qvel[:6]) <= 0.025 else 0
                    if dwell >= 3:
                        phase = f" {phases[target_index]}" if phases else ""
                        print(f"Reached target {target_index + 1}/{len(targets)}{phase}. "
                              f"Position error: {error * 1000:.1f} mm", flush=True)
                        if args.place and approaching:
                            target_index += 1
                            dwell = 0
                            if target_index == len(targets):
                                approaching = False
                                targets = None
                                target_index = 0
                                if grasp_orientation is not None:
                                    world.orientation = grasp_orientation
                                print("Approach verified. Starting PickLift in the same session.", flush=True)
                            send(connection, seq, velocity=[0.0] * 6)
                            time.sleep(world.cfg.dt)
                            continue
                        if centering_grasp:
                            print(f"Closing RG6. XY error to measured grasp target: {xy_error * 1000:.2f} mm.",
                                  flush=True)
                            send(connection, seq, close_gripper=True)
                            connection.settimeout(15.0)
                            closing = True
                            dwell = 0
                            continue
                        if pick and target_index == 1:
                            rise = block_pose[2] - block_start[2]
                            if rise < 0.04 or not state["bilateral_contact"]:
                                raise RuntimeError(f"PickLift not confirmed. Measured block rise: {rise * 1000:.1f} mm")
                            print(f"PickLift verified. Block rose {rise * 1000:.1f} mm with bilateral contact.", flush=True)
                            if carry:
                                grasp_relative = block_pose - actual_pose
                                hold_started = time.monotonic()
                                dwell = 0
                                send(connection, seq, velocity=[0.0] * 6)
                                time.sleep(world.cfg.dt)
                                continue
                        if args.place and phase == "placement descent":
                            raise RuntimeError("Reached descent limit without verified destination support")
                        target_index += 1
                        dwell = 0
                        if target_index == len(targets):
                            if carry:
                                center = executor.support_center(destination)
                                horizontal_error = np.linalg.norm(block_pose[:2] - center[:2])
                                clearance = block_pose[2] - center[2]
                                if horizontal_error > 0.015 or clearance < 0.08:
                                    raise RuntimeError(f"Block missed preplace at {destination}: "
                                                       f"horizontal error {horizontal_error * 1000:.1f} mm, "
                                                       f"clearance {clearance * 1000:.1f} mm")
                                print(f"MoveHolding verified at {destination}. Block horizontal error "
                                      f"{horizontal_error * 1000:.1f} mm; clearance above placement center "
                                      f"{clearance * 1000:.1f} mm.")
                                if args.place:
                                    # Match the executor's lower-to-support logic using the observed grasp offset.
                                    target = actual_pose + center - block_pose - [0, 0, 0.008]
                                    targets.append(target)
                                    phases += ("placement descent",)
                                    print(f"Lowering toward {destination}; stop the descent on verified support.", flush=True)
                                    send(connection, seq, velocity=[0.0] * 6)
                                    time.sleep(world.cfg.dt)
                                    continue
                            send(connection, seq, stop=True)
                            if args.approach:
                                delta = actual_pose - block_pose
                                print(f"RG6 midpoint above block center: {delta[2] * 1000:.1f} mm; "
                                      f"horizontal offset: {np.linalg.norm(delta[:2]) * 1000:.1f} mm")
                            if solve_times:
                                p50, p95 = np.percentile(solve_times, [50, 95])
                                over_budget = sum(ms > world.cfg.dt * 1000 for ms in solve_times)
                                print(f"MPC solves: {len(solve_times)} | median {p50:.1f} ms | "
                                      f"p95 {p95:.1f} ms | max {max(solve_times):.1f} ms | "
                                      f"{over_budget} over {world.cfg.dt * 1000:.0f} ms")
                            print(f"Zero-velocity settling cycles: {settling_cycles}")
                            return
                    if needs_motion:
                        solve_started = time.perf_counter()
                        mpc_target = target + pose - actual_pose if mission else target
                        holding = closing and target_index >= 1
                        speeds = mpc.solve(mpc_target, grip=0.038 if holding else 0.0,
                                           held=block if holding else None)
                        solve_times.append((time.perf_counter() - solve_started) * 1000)
                        if step % 10 == 0:
                            print(f"Cycle {step}: target {target_index + 1}/{len(targets)}, "
                                  f"position error {error * 1000:.1f} mm, solve {solve_times[-1]:.1f} ms", flush=True)
                    else:
                        settling_cycles += 1
                        speeds = np.zeros(6)
                    command = [0.0] * len(order)
                    for model_index, command_index in enumerate(order):
                        command[command_index] = float(speeds[model_index])
                    period = world.cfg.dt

                send(connection, seq, velocity=command)
                time.sleep(max(0.0, period - (time.monotonic() - tick)))

    raise RuntimeError(f"Target sequence did not settle within {step_limit} steps ({target_index} reached)")


if __name__ == "__main__":
    main()
