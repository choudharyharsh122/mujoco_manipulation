"""Replay failure states; report settling without modifying task evidence."""

import json
from pathlib import Path
import numpy as np
from mission_mujoco.physics import World, Config
from mission_mujoco.grounding import Grounder

out = []
for name in ["color_release", "letter_release_final"]:
    p = Path("results") / name
    m = json.loads((p / "manifest.json").read_text())
    a = json.loads((p / "actions.json").read_text())
    initial = (
        {i: f"p0c{i}" for i in range(5)}
        if m["task"] == "color"
        else {0: "p0c0", 1: 0, 2: 1, 3: 2, 4: 3}
    )
    w = World(initial, Config(**m["config"]))
    for t in json.loads((p / "trajectory.json").read_text()):
        w.step(np.array(t["u"]), t["grip"])
    g = Grounder()
    g.previous = {int(k): v for k, v in a[-2]["grounding"]["observed"].items()}
    if m["task"] == "color":
        g.previous = {int(k): v for k, v in a[-1]["grounding"]["observed"].items()}
    state = w.snapshot()
    rows = []
    streak = longest = 0
    for i in range(201):
        if i:
            w.step(np.zeros(6), 0.0)
        geom = g.measure(w)
        settled = g.measure(w, require_settled=True)
        velocities = []
        for body in w.blocks:
            j = w.model.jnt_dofadr[w.model.body(body).jntadr[0]]
            v = w.data.qvel[j : j + 6]
            velocities.append(
                [float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))]
            )
        streak = streak + 1 if settled.accepted else 0
        longest = max(streak, longest)
        rows.append(
            dict(
                t=round(i * 0.05, 2),
                geometry=geom.accepted,
                reason=geom.reason,
                settled=settled.accepted,
                speeds=velocities,
            )
        )
    out.append(dict(run=name, longest_all_settled=longest, initial=rows[0], rows=rows))
    w.close()
Path("results/settling_diagnostic.json").write_text(json.dumps(out, indent=2))
for r in out:
    print(r["run"], r["longest_all_settled"], r["initial"])
