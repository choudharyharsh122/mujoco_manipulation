"""Physical kicks occur once per control interval and never inside prediction."""
import numpy as np
import pytest
from mission_mujoco.physics import Config, World


@pytest.mark.parametrize('name,value', [
    ('joint_kick_half_range', -1), ('joint_kick_half_range', float('nan')),
    ('joint_kick_bias', float('inf')),
])
def test_invalid_kick_config(name, value):
    with pytest.raises(ValueError):
        Config(**{name: value})


def test_physical_kick_and_exact_feedback():
    baseline = World(config=Config(joint_kick_half_range=0))
    kicked = World(config=Config(joint_kick_half_range=.001, joint_kick_bias=.0002, seed=3))
    try:
        # Initialization receives no kicks; predictor calls consume no kick samples.
        np.testing.assert_array_equal(baseline.snapshot(), kicked.snapshot())
        before = kicked.snapshot().copy()
        u = np.zeros(6)
        predicted = kicked.rollout(before, [u], 0)[0][0]
        kicked.forecast_samples(before, [u], 0)
        np.testing.assert_array_equal(kicked.snapshot(), before)
        kicked.diagnostics = object()  # enable interval samples
        baseline.step(u, 0)
        kicked.step(u, 0)
        expected = .0002 + np.random.default_rng(
            np.random.SeedSequence([3, 2])).uniform(-.001, .001, 6)
        np.testing.assert_allclose(kicked.data.qpos[:6] - baseline.data.qpos[:6], expected, atol=1e-14)
        np.testing.assert_allclose(kicked.data.qpos[:6] - predicted, expected, atol=1e-14)
        np.testing.assert_array_equal(kicked.data.qpos[6:], baseline.data.qpos[6:])
        np.testing.assert_array_equal(kicked.data.qvel, baseline.data.qvel)
        assert kicked.data.time == baseline.data.time
        _, observed, noise = kicked.measured_joint_state()
        np.testing.assert_array_equal(observed, kicked.data.qpos[:6])
        np.testing.assert_array_equal(noise, np.zeros(6))
        np.testing.assert_allclose(kicked.trace[-1]['joint_kick'], expected)
        assert np.all(np.abs(expected - .0002) <= .001)
        assert len(kicked.interval_samples) == kicked.nstep + 2
        np.testing.assert_array_equal(kicked.interval_samples[-1]['q'], observed)
        assert np.linalg.norm(kicked.pose()[0] - baseline.pose()[0]) > 0
    finally:
        baseline.close()
        kicked.close()


def test_scheduled_kicks_once_and_missing_vectors_are_zero():
    config = Config(kick_timestamps=(.05, .125, .2),
                    kick_magnitudes=((.001, 0, 0), (.001, .0001, 0)))
    assert config.kick_magnitudes[-1] == (0.,)*6
    w = World(config=config)
    w.diagnostics = object()
    try:
        assert w.next_kick_index == 0  # initialization did not consume schedule
        np.testing.assert_array_equal(w.last_joint_kick, np.zeros(6))
        expected = [(.001, 0, 0, 0, 0, 0), (0,)*6,
                    (.001, .0001, 0, 0, 0, 0), (0,)*6, (0,)*6]
        for step, delta in enumerate(expected):
            state = w.snapshot()
            prediction = w.rollout(state, [np.zeros(6)], 0)[0][0]
            w.forecast_samples(state, [np.zeros(6)], 0)
            before_index = w.next_kick_index
            w.rollout(state, [np.zeros(6)], 0)
            assert w.next_kick_index == before_index
            w.step(np.zeros(6), 0)
            np.testing.assert_allclose(w.last_joint_kick, delta)
            np.testing.assert_allclose(w.data.qpos[:6]-prediction, delta, atol=1e-12)
        assert w.next_kick_index == 3
        assert w.trace[0]['kick_events'][0]['applied_time'] == pytest.approx(.05)
        assert w.trace[2]['kick_events'][0]['requested_time'] == .125
        assert w.trace[2]['kick_events'][0]['applied_time'] == pytest.approx(.15)
        assert not w.trace[4]['kick_events']
    finally:
        w.close()


def test_schedule_configuration_and_cli_parsing():
    from mission_mujoco.cli import parse_kick_timestamps, parse_kick_magnitudes
    times = parse_kick_timestamps('2,5,7')
    vectors = parse_kick_magnitudes(['(1e-3,0,0), (1e-3,1e-4,0)'])
    c = Config(kick_timestamps=times, kick_magnitudes=vectors)
    assert c.kick_magnitudes == ((.001, 0, 0, 0, 0, 0), (.001, .0001, 0, 0, 0, 0), (0,)*6)
    assert Config().joint_kick_half_range == Config().joint_kick_bias == 0
    for kwargs in [dict(kick_timestamps=(-1,)), dict(kick_timestamps=(float('nan'),)),
                   dict(kick_magnitudes=((1, 2, 3),)),
                   dict(kick_timestamps=(1,), kick_magnitudes=((1,)*7,)),
                   dict(kick_timestamps=(1,), kick_magnitudes=((float('inf'),),))]:
        with pytest.raises(ValueError):
            Config(**kwargs)
