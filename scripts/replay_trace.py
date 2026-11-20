#!/usr/bin/env python3
"""Reproduce saved applied inputs and independently re-ground action boundaries.

This verifies deterministic plant/grounding replay with the released source.
It does not replace the original logged closed-loop MPC run or its latency data.
"""

import argparse, json, hashlib
from pathlib import Path
import numpy as np
from mission_mujoco.physics import World, Config
from mission_mujoco.grounding import Grounder
from mission_mujoco.grasp import measured
from mission_mujoco.benchmarks import fixture


def replay(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    actions = json.loads((folder / "actions.json").read_text())
    trajectory = json.loads((folder / "trajectory.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    round_ends = set(np.cumsum([r["actions"] for r in summary["rounds"]]) - 1)
    kind = manifest["task"]
    initial = (
        {i: f"p0c{i}" for i in range(5)}
        if kind == "color"
        else {0: "p0c0", 1: 0, 2: 1, 3: 2, 4: 3}
    )
    if kind in ("b0", "b1"):
        initial = fixture(kind)[0]
    w = World(initial, Config(**manifest["config"]))
    g = Grounder()
    assert g.measure(w).accepted
    index = 0
    held = None
    max_error = 0.0
    boundaries = []
    dwell_grounder = None
    dwell_samples = 0
    for sample in trajectory:
        w.step(np.asarray(sample["u"]), sample["grip"])
        difference = np.max(
            np.abs(
                np.array([w.blockpos(i) for i in range(w.nblock)]) - sample["blocks"]
            )
        )
        max_error = max(max_error, float(difference))
        if index < len(actions):
            current = actions[index]
            if (
                current["kind"] == "PlaceRelease"
                and w.data.time > current["end"] - 0.35 + 1e-6
            ):
                if dwell_grounder is None:
                    dwell_grounder = Grounder()
                    dwell_grounder.previous = g.previous.copy()
                released = dwell_grounder.measure(
                    w,
                    require_settled=True,
                    settled_blocks=None
                    if current.get("settle_all") or index in round_ends
                    else [current["block"]],
                )
                intended = {
                    int(b): d for b, d in current["grounding"]["intended"].items()
                }
                assert released.accepted and released.supports == intended, (
                    index,
                    "released dwell",
                    released.reason,
                )
                dwell_samples += 1
        if index < len(actions) and w.data.time >= actions[index]["end"] - 1e-6:
            action = actions[index]
            if action["status"] != "success":
                raise AssertionError("Replay input contains a failed action")
            if action["kind"] == "PickLift":
                held = action["block"]
                assert measured(w, held)["accepted"]
            elif action["kind"] == "PlaceRelease":
                assert max(abs(x) for x in w.data.qpos[6:8]) < 0.003
                held = None
            observed = g.measure(
                w,
                held,
                require_settled=action["kind"] == "PlaceRelease",
                settled_blocks=None if action.get("settle_all") else [action["block"]],
            )
            intended = {int(b): d for b, d in action["grounding"]["intended"].items()}
            assert observed.accepted and observed.supports == intended, (
                index,
                observed.reason,
            )
            boundaries.append(index)
            if index in round_ends:
                terminal = g.measure(w, require_settled=True)
                assert terminal.accepted and terminal.supports == intended, (
                    index,
                    "terminal speed",
                    terminal.reason,
                )
            index += 1
            dwell_grounder = None
    assert index == len(actions) and max_error < 1e-7, (index, len(actions), max_error)
    # Continue free-body dynamics after the recorded goal to expose unstable
    # stacks that happen to satisfy a single terminal observation.
    assert held is None
    terminal_supports = g.previous.copy()
    terminal_positions = np.array([w.blockpos(i) for i in range(w.nblock)])
    post_goal_speeds = []
    for _ in range(60):
        w.step(np.zeros(6), 0.0)
        observed = g.measure(w)
        assert observed.accepted and observed.supports == terminal_supports, (
            "post-goal stability",
            observed.reason,
        )
        velocities = []
        for body in w.blocks:
            dof = w.model.jnt_dofadr[w.model.body(body).jntadr[0]]
            v = w.data.qvel[dof : dof + 6]
            velocities.append(
                (float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:])))
            )
        post_goal_speeds.append(
            dict(
                time=float(w.data.time),
                linear=max(v[0] for v in velocities),
                angular=max(v[1] for v in velocities),
            )
        )
    terminal_drift = float(
        np.max(
            np.linalg.norm(
                np.array([w.blockpos(i) for i in range(w.nblock)]) - terminal_positions,
                axis=1,
            )
        )
    )
    package = Path(__file__).resolve().parents[1] / "mission_mujoco"
    result = dict(
        passed=True,
        scope="action/round acceptance and three-second support stability; later velocity invariance is a separate diagnostic",
        kind=kind,
        action_boundaries=len(boundaries),
        samples=len(trajectory),
        max_position_replay_error=max_error,
        model_hash_matches=hashlib.sha256(w.xml.encode()).hexdigest()
        == manifest["model_sha256"],
        additional_terminal_dwell_seconds=3.0,
        terminal_supports_retained=True,
        terminal_drift_metres=terminal_drift,
        released_dwell_control_samples=dwell_samples,
        linear_speed_limit_mps=0.01,
        angular_speed_limit_radps=0.10,
        post_goal_speed_limits_maintained=all(
            s["linear"] <= 0.01 and s["angular"] <= 0.10 for s in post_goal_speeds
        ),
        post_goal_speed_samples=post_goal_speeds,
        replay_source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.glob("*.py")
        },
    )
    w.close()
    (folder / "replay_verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    args = p.parse_args()
    print(json.dumps([replay(x) for x in args.folders], indent=2))
