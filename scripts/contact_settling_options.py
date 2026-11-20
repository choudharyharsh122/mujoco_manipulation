"""Disposable-state numerical diagnostic; does not validate a mission run."""

import json
from pathlib import Path
import mujoco as mj
import numpy as np
from mission_mujoco.physics import World, Config
from mission_mujoco.grounding import Grounder

p = Path("results/color_settled")
m = json.loads((p / "manifest.json").read_text())
w = World({i: f"p0c{i}" for i in range(5)}, Config(**m["config"]))
for t in json.loads((p / "trajectory.json").read_text()):
    w.step(np.array(t["u"]), t["grip"])
state = w.snapshot()
base = w.model.geom_solref.copy()
reports = []
for label, dt, ratio, tau, noslip, ccd in [
    ("unchanged", 0.002, 1, 0.02, 0, 1e-6),
    ("stack_settings", 0.002, 10, 0.01, 0, 1e-6),
    ("noslip3", 0.002, 1, 0.02, 3, 1e-6),
    ("half_step", 0.001, 1, 0.02, 0, 1e-6),
    ("ccd_tight", 0.002, 1, 0.02, 0, 1e-10),
]:
    mj.mj_setState(w.model, w.data, state, w.state_spec)
    w.model.geom_solref[:] = base
    w.model.geom_solref[w.blockgeoms, 0] = tau
    w.model.opt.timestep = dt
    w.model.opt.impratio = ratio
    w.model.opt.noslip_iterations = noslip
    w.model.opt.ccd_tolerance = ccd
    w.nstep = round(0.05 / dt)
    mj.mj_forward(w.model, w.data)
    g = Grounder()
    streak = longest = accepted = 0
    geom_bad = 0
    for k in range(200):
        w.step(np.zeros(6), 0.0)
        geom = g.measure(w)
        a = g.measure(w, require_settled=True)
        accepted += int(a.accepted)
        geom_bad += int(not geom.accepted)
        streak = streak + 1 if a.accepted else 0
        longest = max(longest, streak)
    reports.append(
        dict(
            label=label,
            dt=dt,
            impratio=ratio,
            tau=tau,
            noslip=noslip,
            ccd_tolerance=ccd,
            accepted=accepted,
            total=200,
            longest_consecutive=longest,
            geometry_failures=geom_bad,
        )
    )
    print(reports[-1], flush=True)
Path("results/contact_settling_options.json").write_text(json.dumps(reports, indent=2))
w.close()
