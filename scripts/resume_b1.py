#!/usr/bin/env python3
"""Verify the B1 settling correction with a replayed prefix and live final MPC.

The first 43 actions were executed by the earlier closed-loop controller. Their
applied inputs are replayed from physical initialization, with current substep
guards and independent support grounding. No object pose is restored or reset.
The final action and terminal planning decision run live with the released MPC.
This is explicitly checkpoint verification, not a fresh 44-action optimizer run.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mission_mujoco.physics import World, Config
from mission_mujoco.execution import Executor
from mission_mujoco.learning import Models
from mission_mujoco.planning import Planner
from mission_mujoco.geometry import CandidateGeometry
from mission_mujoco.benchmarks import fixture
from mission_mujoco.grasp import measured, prospective_capacity


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    read = lambda n: json.loads((source / (n + ".json")).read_text())
    manifest, old_actions, old_trace = (
        read("manifest"),
        read("actions"),
        read("trajectory"),
    )
    assert manifest["task"] == "b1" and len(old_actions) == 44
    assert all(a["status"] == "success" for a in old_actions[:43])
    assert old_actions[-1]["kind"] == "PlaceRelease"
    initial, goals, sites = fixture("b1")
    w = World(initial, Config(**manifest["config"]))
    assert hashlib.sha256(w.xml.encode()).hexdigest() == manifest["model_sha256"]
    models = Models()
    e = Executor(w, models)
    planner = Planner(models, seed=manifest["seed"], geometry=CandidateGeometry(w))
    start = time.perf_counter()
    position_error = 0.0
    index = 0
    lift_armed = False
    terminal = []
    status = "failed"
    error = None
    try:
        for sample in old_trace:
            if index == 43:
                break
            a = old_actions[index]
            if a["kind"] == "PickLift":
                e.allowed_block = a["block"]
                lift_time = next(p["time"] for p in a["phases"] if p["name"] == "lift")
                if not lift_armed and w.data.time >= lift_time - 1e-6:
                    grasp = measured(w, a["block"])
                    assert (
                        grasp["accepted"]
                        and prospective_capacity(w, a["block"])["accepted"]
                    )
                    e.reference = np.array(grasp["relative"])
                    e.monitored_block = a["block"]
                    lift_armed = True
            if a["kind"] == "PlaceRelease":
                opening = next(
                    p["time"]
                    for p in a["phases"]
                    if p["name"] == "open and released dwell"
                )
                if w.data.time >= opening - 1e-6:
                    e.monitored_block = None
            w.step(np.array(sample["u"]), sample["grip"])
            position_error = max(
                position_error, float(np.max(np.abs(w.observe() - sample["blocks"])))
            )
            assert position_error < 1e-7
            if w.data.time >= a["end"] - 1e-6:
                if a["kind"] == "PickLift":
                    e.held = a["block"]
                elif a["kind"] == "PlaceRelease":
                    e.held = e.reference = e.allowed_block = None
                observed = e.grounder.measure(
                    w,
                    e.held,
                    require_settled=a["kind"] == "PlaceRelease",
                    settled_blocks=[a["block"]],
                )
                expected = {int(b): d for b, d in a["grounding"]["observed"].items()}
                assert observed.accepted and observed.supports == expected
                e.supports = observed.supports
                e.anchor = tuple(a["target_anchor"])
                feedback = models.execution(a["kind"], a["feedback"]["R_o"])
                assert abs(feedback["C_bar"] - a["feedback"]["C_bar"]) < 1e-9
                e.actions.append(a)
                e.phases.extend(a["phases"])
                index += 1
                lift_armed = False
        assert index == 43
        n_prefix = sum(a["solves"] for a in old_actions[:43])
        e.mpc.records = read("mpc")[:n_prefix]
        planner.records = read("planner")[:43]
        plan, _ = planner.solve(e.supports, goals, sites, e.anchor, e.held)
        assert plan[0] == (
            "PlaceRelease",
            old_actions[-1]["block"],
            old_actions[-1]["destination"],
        )
        e.execute(plan[0], settle_all=True)
        plan, actual = planner.solve(e.supports, goals, sites, e.anchor, e.held)
        assert not plan and e.held is None
        observation = e.grounder.measure(w, require_settled=True)
        assert observation.accepted and observation.supports == e.supports
        order = next(g[2] for g in goals if g[0] == actual)
        terminal = [
            dict(
                round=0,
                source=0,
                destination=1,
                previous=list(range(w.nblock)),
                order=order,
                rating=None,
                exploration=False,
                actions=len(e.actions),
                supports=e.supports.copy(),
            )
        ]
        status = "success"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        protocol = dict(
            kind="saved closed-loop prefix replay plus live final MPC action",
            prefix_source=str(source),
            prefix_actions=43,
            live_actions=len(e.actions) - 43,
            max_prefix_position_error=position_error,
            source_manifest_sha256=hashlib.sha256(
                (source / "manifest.json").read_bytes()
            ).hexdigest(),
            source_summary=read("summary"),
            wall_time_scope="replay and live tail; not the wall time of a fresh full closed-loop run",
        )
        manifest.update(
            config=asdict(w.cfg),
            execution_protocol=protocol,
            source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__).resolve().parents[1] / "mission_mujoco").glob("*.py")
            },
        )
        summary = dict(
            status=status,
            error=error,
            completed_rounds=len(terminal),
            rounds=terminal,
            wall_seconds=time.perf_counter() - start,
            simulation_seconds=float(w.data.time),
            actions=len(e.actions),
            successful_actions=sum(a["status"] == "success" for a in e.actions),
            replans=len(planner.records),
            mpc_solves=len(e.mpc.records),
            latency_p50=float(np.median([r["seconds"] for r in e.mpc.records])),
            latency_p95=float(np.quantile([r["seconds"] for r in e.mpc.records], 0.95)),
            solver_converged_fraction=float(
                np.mean([r["converged"] for r in e.mpc.records])
            ),
            max_slip=max([r["slip"] for r in e.monitor], default=0),
            final_positions=w.observe().tolist(),
            memory=terminal[0]["order"] if terminal else list(range(w.nblock)),
            learned_action_costs=models.theta_C,
            color_estimates=models.color.tolist(),
            letter_estimates=models.letter.tolist(),
            execution_protocol=protocol["kind"],
        )
        for name, value in [
            ("manifest", manifest),
            ("summary", summary),
            ("actions", e.actions),
            ("planner", planner.records),
            ("mpc", e.mpc.records),
            ("grasp_observations", e.monitor),
            ("trajectory", w.trace),
        ]:
            (output / (name + ".json")).write_text(
                json.dumps(value, indent=2, allow_nan=False)
            )
        (output / "scene.xml").write_text(w.xml)
        w.close()
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "status",
                    "error",
                    "actions",
                    "successful_actions",
                    "mpc_solves",
                    "wall_seconds",
                ]
            },
            indent=2,
        )
    )
    return 0 if status == "success" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("output")
    a = p.parse_args()
    raise SystemExit(run(a.source, a.output))
