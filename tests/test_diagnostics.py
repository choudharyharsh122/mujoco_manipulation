"""Diagnostics preserve physical execution and the exact sampled observations."""
import json
import numpy as np
import pytest
from mission_mujoco.controller import MPC
from mission_mujoco.diagnostics import PlotLogger
from mission_mujoco.execution import Executor
from mission_mujoco.grounding import Grounder, support_components
from mission_mujoco.learning import Models
from mission_mujoco.physics import Config, World


def test_forecast_and_logging_do_not_change_physical_state(tmp_path):
    w = World()
    try:
        before = w.snapshot().copy()
        command = np.array([.01, -.02, 0, 0, 0, 0])
        forecast = w.forecast_samples(before, [command], 0, plant_model=True)
        assert np.array_equal(w.snapshot(), before)
        assert len(forecast) == w.nstep + 1
        # Independent identical worlds avoid changing integration caches while
        # restoring a state; only one world records diagnostics.
        other = World()
        try:
            other.diagnostics = PlotLogger(tmp_path)
            w.step(command, 0)
            other.step(command, 0)
            assert np.array_equal(w.snapshot(), other.snapshot())
            assert len(other.interval_samples) == w.nstep + 1
            assert np.array_equal(other.interval_samples[-1]['q'], other.data.qpos[:6])
        finally:
            other.close()
    finally:
        w.close()


def test_solve_records_same_command_forecasts_and_applied_error(tmp_path):
    w = World(config=Config(joint_observation_bias=.004, joint_observation_noise=.0001))
    logger = PlotLogger(tmp_path)
    w.diagnostics = logger
    record = dict(id=0, kind='MoveEmpty', block=0, destination=None,
                  start=float(w.data.time), status='running')
    logger.begin_action(w, record)
    controller = MPC(w, horizon=1, maxiter=1, diagnostics=logger)
    try:
        before = w.snapshot().copy()
        u = controller.solve(w.pose()[0], 0)
        assert np.array_equal(before, w.snapshot())
        directory, pending = controller.pending_plot
        assert np.allclose(pending['perturbation'], .004 + pending['joint_noise'])
        assert np.allclose(pending['estimated_forecast'][0]['q'], controller.records[-1]['observed_q'])
        controller.apply(u, 0)
        saved = json.loads((directory / 'solve.json').read_text())
        assert saved['application_status'] == 'applied'
        assert saved['full_control_interval']
        assert saved['comparison_dt'] == pytest.approx(w.cfg.dt)
        assert len(saved['executed']) == w.nstep + 1
        assert np.allclose(saved['one_step_error']['q'],
                           np.asarray(saved['executed'][-1]['q']) - saved['estimated_forecast'][-1]['q'])
        assert len(list(directory.glob('*.png'))) == 12
        assert directory.parts[-3] == 'moveEmpty'
        assert 'red_block0' in directory.parent.name
        assert saved['cartesian_velocity_limits'] is None
        assert any(x['name'] == 'joint command change' for x in saved['executed_constraints'])
        Grounder(diagnostics=logger).measure(w)
        grounding_folder = next(directory.parent.glob('grounding_*'))
        grounding = json.loads((grounding_folder / 'grounding.json').read_text())
        assert grounding['predicted_components'] is not None
        assert logger.latest_prediction['time'] == pytest.approx(float(w.data.time))
        record.update(status='success', end=float(w.data.time))
        logger.end_action(w, record)
        assert list(directory.parent.glob('*one_step_position_error_xyz.png'))
        assert list(directory.parent.glob('*executed_commands*.png'))
    finally:
        w.close()


def test_grounding_logs_used_observation_once_and_all_membership_factors(tmp_path):
    w = World(config=Config(observation_bias=.001, observation_noise=.0001))
    logger = PlotLogger(tmp_path)
    grounder = Grounder(diagnostics=logger)
    samples = []
    observe = w.observe
    def capture():
        result = observe()
        samples.append(result.copy())
        return result
    w.observe = capture
    try:
        result = grounder.measure(w)
        assert len(samples) == 1
        folder = next((logger.root / 'initial').glob('grounding_*'))
        saved = json.loads((folder / 'grounding.json').read_text())
        assert np.array_equal(saved['observed_positions'], samples[0])
        for b, options in result.memberships.items():
            for support, combined in options.items():
                factors = saved['observed_components'][str(b)][str(support)]
                assert combined == pytest.approx(factors['area'] * factors['height'] * factors['orientation'])
        assert len(list(folder.glob('*area_height_orientation.png'))) == w.nblock
        # Early velocity rejection also retains geometry and rejection reason.
        dof = w.model.jnt_dofadr[w.model.body(w.blocks[0]).jntadr[0]]
        w.data.qvel[dof] = .02
        result = grounder.measure(w, require_settled=True)
        assert not result.accepted
        assert logger.ground_count == 2
        saved = json.loads(sorted((logger.root / 'initial').glob('grounding_*'))[-1].joinpath('grounding.json').read_text())
        assert 'velocity' in saved['reason'] and saved['observed_components']
    finally:
        w.close()


def test_abort_retains_partial_interval_and_action_failure(tmp_path, monkeypatch):
    w = World()
    executor = Executor(w, Models(), plot_dir=tmp_path)
    # Trigger a real command/step, then a guard error inside the physical interval.
    def fail_guard():
        raise RuntimeError('test substep abort')
    w.substep_guard = fail_guard
    def one_interval(*args, **kwargs):
        executor.mpc.apply(executor.mpc.solve(w.pose()[0], 0), 0)
    monkeypatch.setattr(executor, 'reach', one_interval)
    try:
        with pytest.raises(RuntimeError, match='test substep abort'):
            executor.execute(('MoveEmpty', 0, None))
        folder = next(executor.diagnostics.root.glob('moveEmpty/action_*'))
        saved = json.loads(next(folder.glob('mpc_*/solve.json')).read_text())
        assert saved['application_status'] == 'failed'
        assert not saved['full_control_interval']
        assert saved['comparison_dt'] == pytest.approx(w.cfg.timestep)
        assert len(saved['executed']) == 2
        assert saved['executed'][-1]['time'] < saved['estimated_forecast'][w.nstep]['time']
        action = json.loads((folder / 'action.json').read_text())
        assert action['status'] == 'failed'
        assert 'test substep abort' in action['error']
        assert list(folder.glob('*executed_position_xyz.png'))
    finally:
        w.close()


def test_repeated_actions_and_sessions_never_overwrite(tmp_path):
    w = World()
    try:
        logger = PlotLogger(tmp_path)
        other = PlotLogger(tmp_path)
        assert logger.root != other.root
        record = dict(kind='MoveHolding', block=2, destination='p1c0', start=0, status='running')
        logger.begin_action(w, dict(record, id=0))
        first = logger.directory
        logger.begin_action(w, dict(record, id=1))
        assert first != logger.directory
        assert 'blue_block2' in logger.directory.name
        assert first.joinpath('action.json').exists()
    finally:
        w.close()


@pytest.mark.parametrize('name,value', [('joint_observation_noise', -.1),
                                       ('joint_observation_bias', np.nan)])
def test_invalid_observation_settings_rejected(name, value):
    with pytest.raises(ValueError):
        Config(**{name: value})


def test_final_grounding_mode_skips_mpc_and_intermediate_plots(tmp_path, monkeypatch):
    w = World()
    e = Executor(w, Models(), plot_dir=tmp_path, plot_mode='final-grounding')
    e.mpc.horizon = 1
    e.mpc.maxiter = 1
    # Execute real MPC hold intervals to exercise the motion hooks cheaply.
    def short_reach(target, grip, phase, held=None):
        e.phases.append(dict(name=phase, time=float(w.data.time), target=np.asarray(target).tolist()))
        e.diagnostics.phase = phase
        return e.mpc.hold(w.cfg.dt, 0)
    monkeypatch.setattr(e, 'reach', short_reach)
    try:
        assert e.diagnostics.ground_count == 0  # initial observation is not plotted
        assert e.mpc.diagnostics is None
        result = e.execute(('MoveEmpty', 0, None))
        assert result['status'] == 'success'
        folder = next(e.diagnostics.root.glob('moveEmpty/action_*'))
        assert (folder / 'grounding.json').exists()
        assert not list(folder.glob('mpc_*'))
        assert not list(folder.glob('grounding_*'))
        assert not list(folder.glob('*executed*.png'))
        assert len(list(folder.glob('*.png'))) == 2*w.nblock
        assert len(e.mpc.records) == 3
        assert w.interval_samples == []
        e.grounder.measure(w)
        assert e.diagnostics.ground_count == 1  # extra intermediate checks suppressed
        e.grounder.measure(w, plot_final=True)
        e.grounder.measure(w, plot_final=True)
        assert len(list((e.diagnostics.root / 'terminal').glob('grounding_*'))) == 2
    finally:
        w.close()


@pytest.mark.parametrize('abort', [False, True])
@pytest.mark.parametrize('with_grounding', [False, True])
def test_executed_cartesian_mode_saves_only_two_action_plots(tmp_path, monkeypatch, abort, with_grounding):
    w = World()
    e = Executor(w, Models(), plot_dir=tmp_path,
                 plot_mode=['executed-cartesian', 'final-grounding'] if with_grounding else 'executed-cartesian')
    e.mpc.horizon = 1
    e.mpc.maxiter = 1
    def no_forecasts(*args, **kwargs):
        raise AssertionError('Summary mode must not run extra shadow forecasts')
    monkeypatch.setattr(w, 'forecast_samples', no_forecasts)
    def short_reach(target, grip, phase, held=None):
        e.phases.append(dict(name=phase, time=float(w.data.time), target=np.asarray(target).tolist()))
        e.diagnostics.phase = phase
        return e.mpc.hold(w.cfg.dt, 0)
    monkeypatch.setattr(e, 'reach', short_reach)
    if abort:
        def fail_guard():
            raise RuntimeError('test summary abort')
        w.substep_guard = fail_guard
    try:
        if abort:
            with pytest.raises(RuntimeError, match='test summary abort'):
                e.execute(('MoveEmpty', 0, None))
        else:
            assert e.execute(('MoveEmpty', 0, None))['status'] == 'success'
        folder = next(e.diagnostics.root.glob('moveEmpty/action_*'))
        names = {p.name for p in e.diagnostics.root.rglob('*executed*.png')}
        assert names == {'red_block0_executed_position_xyz.png',
                         'red_block0_executed_cartesian_velocity_xyz.png'}
        assert not [p for p in folder.iterdir() if p.is_dir()]
        assert len(list(e.diagnostics.root.rglob('grounding.json'))) == int(with_grounding)
        assert len(list(e.diagnostics.root.rglob('*.png'))) == 2 + (2*w.nblock if with_grounding else 0)
        samples = e.diagnostics.action_samples
        assert len(samples) == (2 if abort else 3*(w.nstep+1))
        assert samples[-1]['time'] == pytest.approx(float(w.data.time))
        assert len(samples[-1]['position']) == len(samples[-1]['cartesian_velocity']) == 3
        assert e.diagnostics.ground_count == int(with_grounding)
        assert e.diagnostics.solve_count == 0
        assert json.loads((folder/'action.json').read_text())['status'] == ('failed' if abort else 'success')
    finally:
        w.close()


def test_multiple_plot_selections_normalize():
    from mission_mujoco.diagnostics import normalize_plot_mode
    assert normalize_plot_mode(['final-grounding', 'executed-cartesian']) == 'executed-cartesian+final-grounding'
    assert normalize_plot_mode('executed-cartesian,final-grounding') == 'executed-cartesian+final-grounding'
    assert normalize_plot_mode(['all', 'executed-cartesian']) == 'all'
    with pytest.raises(ValueError, match='cannot be combined'):
        normalize_plot_mode(['none', 'final-grounding'])


def test_completion_thresholds_logged_on_timeout(tmp_path):
    w = World()
    e = Executor(w, Models(), plot_dir=tmp_path, plot_mode='executed-cartesian', completion_plots=True)
    e.mpc.horizon = 1
    e.mpc.maxiter = 1
    try:
        with pytest.raises(RuntimeError, match='MPC timeout'):
            e.mpc.reach(w.pose()[0], 0, tolerance=1e-12, timeout=2*w.cfg.dt)
        saved = json.loads(next(e.diagnostics.root.rglob('*completion*.json')).read_text())
        assert len(saved['checks']) == 2
        for check in saved['checks']:
            assert check['position_pass'] == (check['position_error'] < check['position_tolerance'])
            assert check['speed_pass'] == (check['joint_speed_norm'] < .035)
        assert e.mpc.records[-1]['completion'] == saved['checks'][-1]
        assert list(e.diagnostics.root.rglob('*completion*.png'))
    finally:
        w.close()


def test_joint_prediction_figures_per_complete_action(tmp_path, monkeypatch):
    w = World(config=Config(joint_kick_bias=.0001, joint_kick_half_range=0))
    e = Executor(w, Models(), plot_dir=tmp_path, plot_mode='joint-prediction')
    e.mpc.horizon = 2
    e.mpc.maxiter = 1
    captured = []
    original_save = e.diagnostics.save
    def save(fig, path):
        captured.append(fig)
        assert len(fig.axes) == 6
        for ax in fig.axes:
            if path.name.endswith('_joint_prediction.png'):
                assert sum(line.get_color() == 'tab:orange' for line in ax.lines) == 3
            assert any(line.get_color() == 'tab:blue' for line in ax.lines)
        original_save(fig, path)
    monkeypatch.setattr(e.diagnostics, 'save', save)
    def short_reach(target, grip, phase, held=None):
        e.phases.append(dict(name=phase, time=float(w.data.time), target=np.asarray(target).tolist()))
        e.diagnostics.phase = phase
        return e.mpc.hold(w.cfg.dt, grip)
    monkeypatch.setattr(e, 'reach', short_reach)
    try:
        assert e.execute(('MoveEmpty', 0, None))['status'] == 'success'
        assert len(captured) == 2
        assert len(list(e.diagnostics.root.rglob('*.png'))) == 2
        assert not list(e.diagnostics.root.rglob('solve.json'))
        assert not list(e.diagnostics.root.rglob('grounding.json'))
        payload = json.loads(next(e.diagnostics.root.rglob('joint_prediction.json')).read_text())
        assert len(payload['predictions']) == 3
        assert list(e.diagnostics.root.rglob('*_one_step_joint_error.png'))
        for solve in payload['predictions']:
            error = solve['one_step_error']
            assert error['time'] == pytest.approx(solve['time'] + w.cfg.dt)
            np.testing.assert_allclose(error['q'],
                np.asarray(error['actual_q']) - error['predicted_q'])
            np.testing.assert_allclose(error['q'], .0001, atol=1e-10)
        assert len(payload['executed']) == 3*(w.nstep+2)
        assert all(len(p['predicted']) == 2*w.nstep+1 for p in payload['predictions'])
    finally:
        w.close()


def test_cartesian_prediction_action_figure(tmp_path, monkeypatch):
    w = World()
    e = Executor(w, Models(), plot_dir=tmp_path, plot_mode='cartesian-prediction')
    e.mpc.horizon = 2
    e.mpc.maxiter = 1
    original_save = e.diagnostics.save
    def save(fig, path):
        assert len(fig.axes) == 18
        for ax in fig.axes:
            assert sum(line.get_color() == 'tab:orange' for line in ax.lines) == 3
            assert any(line.get_color() == 'tab:blue' for line in ax.lines)
        original_save(fig, path)
    monkeypatch.setattr(e.diagnostics, 'save', save)
    def short_reach(target, grip, phase, held=None):
        e.phases.append(dict(name=phase, time=float(w.data.time), target=np.asarray(target).tolist()))
        e.diagnostics.phase = phase
        return e.mpc.hold(w.cfg.dt, grip)
    monkeypatch.setattr(e, 'reach', short_reach)
    try:
        assert e.execute(('MoveEmpty', 0, None))['status'] == 'success'
        assert len(list(e.diagnostics.root.rglob('*.png'))) == 1
        assert not list(e.diagnostics.root.rglob('solve.json'))
        assert not list(e.diagnostics.root.rglob('grounding.json'))
        payload = json.loads(next(e.diagnostics.root.rglob('cartesian_prediction.json')).read_text())
        actual = payload['executed'][-1]
        np.testing.assert_allclose(actual['joint_positions'], w.data.xanchor[:6])
        assert np.asarray(actual['joint_positions']).shape == (6, 3)
        assert len(payload['predictions']) == 3
        # Predicted anchor positions come from forward kinematics of the forecast state.
        sample = w.motion_sample()
        np.testing.assert_allclose(sample['joint_positions'], w.data.xanchor[:6])
    finally:
        w.close()


@pytest.mark.parametrize('mode', ['joint-prediction', 'cartesian-prediction', 'executed-cartesian'])
def test_action_plots_mark_applied_nonzero_kicks(tmp_path, monkeypatch, mode):
    w = World(config=Config(kick_timestamps=(.025, .1, .125),
                           kick_magnitudes=((.0001, 0, 0), (0, 0, 0), (0, .0001, 0))))
    e = Executor(w, Models(), plot_dir=tmp_path, plot_mode=mode)
    e.mpc.horizon = 1
    e.mpc.maxiter = 1
    images = []
    original_save = e.diagnostics.save
    def save(fig, path):
        images.append(path)
        for ax in fig.axes:
            markers = [line for line in ax.lines
                       if line.get_color() == 'tab:red' and line.get_linestyle() == '--']
            np.testing.assert_allclose([line.get_xdata()[0] for line in markers], [.05, .15], atol=1e-9)
            assert sum(line.get_label() == 'Kick applied' for line in markers) == 1
        original_save(fig, path)
    monkeypatch.setattr(e.diagnostics, 'save', save)
    def short_reach(target, grip, phase, held=None):
        e.phases.append(dict(name=phase, time=float(w.data.time), target=np.asarray(target).tolist()))
        e.diagnostics.phase = phase
        return e.mpc.hold(w.cfg.dt, grip)
    monkeypatch.setattr(e, 'reach', short_reach)
    try:
        e.execute(('MoveEmpty', 0, None))
        assert images
        assert len(e.diagnostics.action_kicks) == 2
        assert e.diagnostics.action_kicks[0]['requested_time'] == .025
        if mode != 'executed-cartesian':
            payload = json.loads(next(e.diagnostics.root.rglob('*prediction.json')).read_text())
            assert len(payload['kick_events']) == 2
    finally:
        w.close()
