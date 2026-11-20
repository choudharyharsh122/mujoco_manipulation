#!/usr/bin/env python3
"""Positive mass/friction mismatch transfer and expected low-friction abort."""

import json
from pathlib import Path
from mission_mujoco.physics import World, Config
from mission_mujoco.execution import Executor
from mission_mujoco.learning import Models


def check_case(name, config, expect_success):
    w = World(config=config)
    e = Executor(w, Models())
    error = None
    try:
        for action in [
            ("MoveEmpty", 0, None),
            ("PickLift", 0, "p0c0"),
            ("MoveHolding", 0, "p1c0"),
            ("PlaceRelease", 0, "p1c0"),
        ]:
            e.execute(action)
        success = e.supports.get(0) == "p1c0" and e.held is None
    except RuntimeError as exc:
        success = False
        error = str(exc)
    result = dict(
        name=name,
        expected_success=expect_success,
        physical_success=success,
        expectation_met=success == expect_success,
        error=error,
        actions=e.actions,
        simulation_seconds=float(w.data.time),
        model=w.xml,
        max_slip=max([x["slip"] for x in e.monitor], default=0),
    )
    w.close()
    return result


if __name__ == "__main__":
    reports = [
        check_case(
            "mass_130_percent_friction_070",
            Config(mass_scale=1.3, friction=0.7, seed=1),
            True,
        ),
        check_case("insufficient_friction_001", Config(friction=0.01, seed=2), False),
    ]
    Path("results/stress.json").write_text(json.dumps(reports, indent=2))
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ["actions", "model"]}
                for r in reports
            ],
            indent=2,
        )
    )
    raise SystemExit(0 if all(r["expectation_met"] for r in reports) else 1)
