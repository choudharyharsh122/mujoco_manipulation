#!/usr/bin/env python3
"""Replay the earlier stack failure and vary regularization in disposable states.

This is a numerical diagnostic, not closed-loop mission verification.
"""

import argparse
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from mission_mujoco.physics import World, Config
from mission_mujoco.grounding import Grounder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("contact_sensitivity.json"))
    args = parser.parse_args()
    manifest = json.loads((args.folder / "manifest.json").read_text())
    if manifest["task"] != "letter":
        parser.error("this diagnostic requires the earlier letter-stack trace")
    world = World({0: "p0c0", 1: 0, 2: 1, 3: 2, 4: 3}, Config(**manifest["config"]))
    for row in json.loads((args.folder / "trajectory.json").read_text()):
        world.step(np.array(row["u"]), row["grip"])
    state = world.snapshot()
    reports = []
    for tau in [0.02, 0.01, 0.006, 0.004]:
        mj.mj_setState(world.model, world.data, state, world.state_spec)
        world.model.geom_solref[world.blockgeoms, 0] = tau
        mj.mj_forward(world.model, world.data)
        for _ in range(100):
            world.step(np.zeros(6), 0.0)
        observed = Grounder().measure(world)
        reports.append(
            dict(
                tau=tau,
                accepted=observed.accepted,
                reason=observed.reason,
                positions=[world.blockpos(b).tolist() for b in range(world.nblock)],
                mu={b: max(v.values()) for b, v in observed.memberships.items()},
            )
        )
    args.out.write_text(json.dumps(reports, indent=2))
    print(json.dumps([dict(tau=r["tau"], accepted=r["accepted"]) for r in reports]))
    world.close()


if __name__ == "__main__":
    main()
