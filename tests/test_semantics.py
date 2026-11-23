from itertools import permutations
import numpy as np
from mission_mujoco.learning import Models, HiddenEvaluator
from mission_mujoco.planning import (
    color_goals,
    letter_goals,
    Planner,
    order_from_supports,
)
from mission_mujoco.model import cells
from mission_mujoco.grounding import valid_forest, Grounder, square, stable_com
from mission_mujoco.grasp import closure, capacity
from mission_mujoco.physics import World


def test_terminal_counts_and_variable_spread():
    m = Models()
    for old in permutations(range(5)):
        color = color_goals(old, 1, m)
        letter = letter_goals(old, "p1c0", m)
        assert len({g[2] for g in color}) == 44
        assert len(letter) == 72
        assert all(g[2][0] not in (old[0], old[-1]) for g in letter)
    assert len({round(g[3], 8) for g in color}) > 1


def test_planner_four_types_and_observed_held_state():
    m = Models()
    planner = Planner(m)
    s = {i: f"p0c{i}" for i in range(5)}
    goals = color_goals(order_from_supports(s, "color"), 1, m)
    plan, _ = planner.solve(s, goals, list(cells()))
    assert len(plan) == 20
    assert {a[0] for a in plan} == set(m.theta_C)
    b = plan[1][1]
    s.pop(b)
    remaining, _ = planner.solve(s, goals, list(cells()), ("lift", b), b)
    assert remaining[0][0] == "MoveHolding"
    assert remaining[1][0] == "PlaceRelease"


def test_no_cycles_or_shared_support():
    assert not valid_forest({0: 1, 1: 0})
    assert not valid_forest({0: "p0c0", 1: "p0c0"})
    assert valid_forest({0: "p0c0", 1: 0, 2: 1})


def test_force_closure_needs_finite_pad_and_capacity_is_bounded():
    assert closure()["accepted"]
    assert not closure(finite_pad=False)["accepted"]
    assert capacity()["accepted"]
    assert not capacity(mass=100)["accepted"]
    assert not capacity(mu=0.0)["accepted"]


def test_reward_sign_missing_rating_and_no_duplicate_terminal_cost():
    m = Models()
    x = m.execution("MoveEmpty", 7.0)
    assert x["E_C"] == 4 and x["R"] == -7 and m.theta_C["MoveEmpty"] == 7
    m.rate_color(1, 3, None)
    assert m.color_n.sum() == 0
    m.rate_color(1, 3, 1.0)
    assert m.color[1, 3] == 1
    assert m.theta_C["MoveEmpty"] == 7


def test_ridge_gauge_and_identity_features():
    m = Models()
    hidden = HiddenEvaluator()
    for p in permutations(range(5)):
        m.rate_letter(p, hidden.letter(p))
    assert np.max(np.abs(m.letter.sum(axis=0))) < 1e-12
    assert np.max(np.abs(m.letter.sum(axis=1))) < 1e-12
    assert m.letter_utility((1, 0, 2, 4, 3)) > m.letter_utility((0, 1, 4, 2, 3)) + 0.8


def test_predictor_does_not_mutate_plant_or_teleport_objects():
    w = World()
    before = w.snapshot().copy()
    a = w.rollout(before, [np.zeros(6)] * 3, 0.0)
    assert np.array_equal(before, w.snapshot())
    b = w.rollout(before, [np.zeros(6)] * 3, 0.0)
    assert np.allclose(a[-1][0], b[-1][0])
    assert w.model.neq == 0
    assert all(w.model.jnt_type[w.model.body(i).jntadr[0]] == 0 for i in w.blocks)


def test_grounding_rejects_floating_object():
    import mujoco as mj

    w = World()
    g = Grounder()
    assert g.measure(w).accepted
    # Deliberate test-fixture corruption, never used by the executor.
    qadr = w.model.jnt_qposadr[w.model.body(w.blocks[0]).jntadr[0]]
    w.data.qpos[qadr + 2] += 0.025
    mj.mj_forward(w.model, w.data)
    assert not g.measure(w).accepted


def test_entire_substack_com_not_individual_pair_only():
    p = np.array(
        [
            cells()["p0c0"] + [0, 0, 0],
            cells()["p0c0"] + [0.019, 0, 0.05],
            cells()["p0c0"] + [0.038, 0, 0.10],
        ]
    )
    footprints = [square(x, 0.025) for x in p]
    # Upper two individually sit within lower footprint, but their combined
    # COM is outside the bottom support's right edge.
    assert not stable_com({0: "p0c0", 1: 0, 2: 1}, p, footprints)


def test_b1_rule_directions_and_terminal_search():
    from mission_mujoco.benchmarks import fixture, Milestone

    s, goals, sites = fixture("b1")
    assert len(goals) == 32
    planner = Planner(Models())
    plan, target = planner.solve(s, goals, sites)
    assert len(plan) > 0 and target in {g[0] for g in goals}
    monitor = Milestone(False)
    assert [monitor.observe(x) for x in [False, True, False]] == [False, True, True]
    assert Milestone(True).value


def test_hysteresis_survives_pick_and_unrelated_release():
    import mujoco as mj

    w = World({0: "p0c0", 1: 0, 2: "p1c0", 3: "p1c1", 4: "p1c2"})
    grounder = Grounder()
    assert grounder.measure(w).accepted
    assert grounder.measure(w, held=4).accepted
    # Perturb a previously accepted, untouched support within its retention band.
    adr = w.model.jnt_qposadr[w.model.body(w.blocks[1]).jntadr[0]]
    w.data.qpos[adr] += 0.0115
    mj.mj_forward(w.model, w.data)
    observed = grounder.measure(w)
    assert 0.70 < observed.memberships[1][0] < 0.80
    assert observed.accepted
    assert observed.supports[4] == "p1c2"


def test_ur10e_tcp_jacobian_and_cube_inertia():
    import mujoco as mj

    w = World()
    data = w.ikdata
    data.qpos[:] = w.data.qpos
    mj.mj_forward(w.pred_model, data)
    J = np.zeros((3, w.pred_model.nv))
    mj.mj_jacSite(w.pred_model, data, J, None, w.tcp)
    baseline = data.site_xpos[w.tcp].copy()
    finite = np.empty((3, 6))
    h = 1e-6
    for j in range(6):
        data.qpos[:] = w.data.qpos
        data.qpos[j] += h
        mj.mj_forward(w.pred_model, data)
        finite[:, j] = (data.site_xpos[w.tcp] - baseline) / h
    assert np.max(np.abs(finite - J[:, :6])) < 1e-6
    assert np.allclose(w.model.body_inertia[w.blocks[0]], 0.08 * 0.05**2 / 6)


def test_tall_stack_contact_regularization_dwell():
    from mission_mujoco.physics import Config

    supports = {0: "p0c0", 1: 0, 2: 1, 3: 2, 4: 3}
    w = World(supports, Config(contact_impratio=10, support_solref=0.01))
    grounder = Grounder()
    for _ in range(200):  # Ten seconds of unsupported, free-body stack dynamics.
        w.step(np.zeros(6), 0.0)
    observed = grounder.measure(w)
    assert observed.accepted and observed.supports == supports
    assert all(abs(w.blockpos(b)[2] - (0.345 + b * 0.05)) < 0.002 for b in supports)


def test_supported_geometry_does_not_override_released_velocity_guard():
    w = World()
    assert Grounder().measure(w).accepted
    dof = w.model.jnt_dofadr[w.model.body(w.blocks[0]).jntadr[0]]
    w.data.qvel[dof] = 0.02
    observed = Grounder().measure(w, require_settled=True)
    assert not observed.accepted and "velocity" in observed.reason
    assert Grounder().measure(w, require_settled=True, settled_blocks=[1]).accepted
