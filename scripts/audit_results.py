#!/usr/bin/env python3
"""Audit executed evidence; do not infer a task pass from exit status alone."""

import json, argparse
from pathlib import Path
import numpy as np


def audit(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    actions = json.loads((folder / "actions.json").read_text())
    trajectory = json.loads((folder / "trajectory.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    task = manifest["task"]
    checks = {}
    checks["complete_run"] = (
        summary["status"] == "success"
        and summary["completed_rounds"] == manifest["requested_rounds"]
    )
    checks["every_action_observed"] = all(
        a["status"] == "success"
        and a["grounding"]["accepted"]
        and a["grounding"]["observed"] == a["grounding"]["intended"]
        for a in actions
    )
    checks["reward_signs"] = all(
        abs(a["feedback"]["R"] + a["feedback"]["R_o"]) < 1e-9
        and abs(a["feedback"]["C_hat"] - a["feedback"]["R_o"]) < 1e-9
        and abs(
            a["feedback"]["E_C"] - (a["feedback"]["C_hat"] - a["feedback"]["C_bar"])
        )
        < 1e-9
        for a in actions
        if a["status"] == "success"
    )
    checks["four_types"] = set(a["kind"] for a in actions) == {
        "MoveEmpty",
        "PickLift",
        "MoveHolding",
        "PlaceRelease",
    }
    checks["replanning_boundaries"] = summary["replans"] >= len(actions) + len(
        summary["rounds"]
    )
    checks["memory_chain"] = all(
        a["order"] == b["previous"]
        for a, b in zip(summary["rounds"], summary["rounds"][1:])
    )
    checks["phase_ownership"] = all(
        not any(p["name"] in ["retreat", "transit", "carry"] for p in a["phases"])
        for a in actions
        if a["kind"] in ["PickLift", "PlaceRelease"]
    )
    checks["bounded_commands"] = all(
        max(abs(v) for v in t["u"]) <= 0.7000001 for t in trajectory
    )
    checks["no_nonfinite_trace"] = all(
        np.isfinite(np.r_[t["q"], t["velocity"], np.asarray(t["blocks"]).ravel()]).all()
        for t in trajectory
    )
    if task == "color":
        checks["spatial_derangement"] = all(
            all(x != y for x, y in zip(r["previous"], r["order"]))
            for r in summary["rounds"]
        )
    if task == "letter":
        checks["new_bottom_rule"] = all(
            r["order"][0] not in (r["previous"][0], r["previous"][-1])
            for r in summary["rounds"]
        )
    terminal_metrics = []
    consumed = 0
    for round_record in summary["rounds"]:
        consumed += round_record["actions"]
        end = actions[consumed - 1]["end"]
        sample = min(trajectory, key=lambda row: abs(row["time"] - end))
        xy = np.asarray(sample["blocks"])[:, :2]
        if task == "color":
            spread = (
                sum(
                    float(np.sum((xy[i] - xy[j]) ** 2))
                    for i in range(5)
                    for j in range(i)
                )
                / 10
                / 0.47**2
            )
            terminal_metrics.append(
                dict(
                    round=round_record["round"],
                    measured_spread=spread,
                    external_utility=round_record["rating"],
                )
            )
        elif task == "letter":
            order = round_record["order"]
            utility = 0.5 * sum(order[i] in [0, 4] for i in [1, 3])
            terminal_metrics.append(
                dict(
                    round=round_record["round"],
                    independent_vowel_utility=utility,
                    external_rating=round_record["rating"],
                )
            )
    if task == "b1" and summary["rounds"]:
        graph = {int(b): s for b, s in summary["rounds"][-1]["supports"].items()}
        colors = "RBGYRBGY"
        roots = {}
        ancestors = {}
        for b in graph:
            chain = []
            s = graph[b]
            while isinstance(s, int):
                chain.append(s)
                s = graph[s]
            roots[b] = s
            ancestors[b] = chain
        checks["B1_red_on_blue"] = all(
            isinstance(graph[b], int) and colors[graph[b]] == "B"
            for b in graph
            if colors[b] == "R"
        )
        checks["B1_green_below_yellow"] = all(
            any(colors[a] == "Y" and b in ancestors[a] for a in graph)
            for b in graph
            if colors[b] == "G"
        )
        checks["B1_identity_precedence"] = 2 in ancestors[0] and 6 in ancestors[4]
        checks["B1_two_height_four_towers"] = set(roots.values()) == {
            "p1c0",
            "p1c2",
        } and all(
            sum(r == site for r in roots.values()) == 4 for site in ["p1c0", "p1c2"]
        )
        checks["B1_red_count"] = all(
            sum(colors[b] == "R" and roots[b] == site for b in graph) <= 1
            for site in roots.values()
        )
    # Check the last completed round at its own boundary, not at a later
    # failed/incomplete round's final physical configuration.
    if summary["rounds"]:
        completed_actions = sum(r["actions"] for r in summary["rounds"])
        end = actions[completed_actions - 1]["end"]
        sample = min(trajectory, key=lambda row: abs(row["time"] - end))
        positions = np.asarray(sample["blocks"])
        supports = summary["rounds"][-1]["supports"]
        z_ok = True
        for b, s in supports.items():
            height = 0.345 if isinstance(s, str) else positions[int(s), 2] + 0.05
            z_ok &= abs(positions[int(b), 2] - height) < 0.009
            if isinstance(s, int):
                z_ok &= np.linalg.norm(positions[int(b), :2] - positions[s, :2]) < 0.016
        checks["last_completed_round_heights"] = bool(z_ok)
    return dict(
        task=task,
        folder=str(folder),
        passed=all(checks.values()),
        checks=checks,
        rounds=summary["completed_rounds"],
        actions=summary["actions"],
        mpc_solves=summary["mpc_solves"],
        simulation_seconds=summary["simulation_seconds"],
        wall_seconds=summary["wall_seconds"],
        max_slip_mm=summary["max_slip"] * 1000,
        latency_p95_ms=summary["latency_p95"] * 1000,
        converged_fraction=summary["solver_converged_fraction"],
        terminal_metrics=terminal_metrics,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("--out", default="results/verification.json")
    args = p.parse_args()
    reports = [audit(x) for x in args.folders]
    result = dict(
        scope="development smoke verification, not independent-run statistical validation",
        runs=reports,
        passed=all(x["passed"] for x in reports),
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
