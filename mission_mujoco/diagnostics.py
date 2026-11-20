"""Incremental, headless MPC and fuzzy-grounding artifacts.

Forecasts compare the same command from true/observed initial states; they are
not independent baseline-policy runs. Logging never resamples observations.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .model import COLORS
from .grounding import support_components

COLOR_NAMES = ('red', 'yellow', 'blue', 'green', 'purple')
ACTION_NAMES = dict(MoveEmpty='moveEmpty', PickLift='pickLift',
                    MoveHolding='moveHolding', PlaceRelease='placeRelease')


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def slug(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(value))[:100]


def normalize_plot_mode(mode):
    """Accept one mode or multiple selections; retain a canonical string."""
    values = [mode] if isinstance(mode, str) else list(mode)
    selected = {part for value in values for part in value.replace(',', '+').split('+')}
    allowed = {'all', 'none', 'executed-cartesian', 'final-grounding', 'joint-prediction', 'cartesian-prediction'}
    if not selected or not selected <= allowed:
        raise ValueError('Unknown or empty plot mode selection')
    if 'none' in selected and len(selected) > 1:
        raise ValueError('none cannot be combined with other plot modes')
    if 'all' in selected:
        return 'all'
    return '+'.join(value for value in ('none', 'executed-cartesian', 'final-grounding', 'joint-prediction', 'cartesian-prediction') if value in selected)


class PlotLogger:
    def __init__(self, root, mode='all', completion_plots=False):
        self.mode = normalize_plot_mode(mode)
        if self.mode == 'none':
            raise ValueError('Do not create a plot logger for mode none')
        self.completion_plots = completion_plots
        self.completion_count = 0
        self.final_grounding_saved = False
        # A new session prevents reruns/repeated block colors overwriting evidence.
        session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.root = Path(root) / session
        self.root.mkdir(parents=True, exist_ok=False)
        self.directory = self.root / 'initial'
        self.phase = 'initial'
        self.prefix = 'scene'
        self.solve_count = 0
        self.ground_count = 0
        self.action_samples = []
        self.action_kicks = []
        self.action_solves = []
        self.action = None
        self.latest_prediction = None
        self.write(self.root / 'README.json', dict(
            curves={'executed true': 'physical plant under the applied command',
                    'true-start forecast': 'plant-model shadow under the SAME command',
                    'MPC estimated forecast': 'nominal predictor from noisy joint observation'},
            independent_baseline_run=False,
            constraints='Cartesian targets are references, not hard position bounds. No Cartesian speed limits are configured. Joint commands/slew are optimizer bounds; actual speed/contact limits are runtime guards.'))

    @staticmethod
    def write(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically so interrupted runs retain the previous complete record.
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(serial(payload), indent=2, allow_nan=False))
        temporary.replace(path)

    @staticmethod
    def color(w, block):
        rgb = w.model.geom_rgba[w.blockgeoms[block], :3]
        index = int(np.argmin(np.linalg.norm(np.asarray(COLORS)[:, :3] - rgb, axis=1)))
        return COLOR_NAMES[index]

    def begin_action(self, w, record):
        self.action = record
        self.final_grounding_saved = False
        self.prefix = f'{self.color(w, record["block"])}_block{record["block"]}'
        self.directory = self.root / ACTION_NAMES.get(record['kind'], slug(record['kind'])) / f'action_{record["id"]:04d}_{self.prefix}_to_{slug(record["destination"])}'
        self.phase = 'start'
        self.action_samples = []
        self.action_kicks = []
        self.action_solves = []
        self.write(self.directory / 'action.json', record)

    @staticmethod
    def figure(count):
        # Fixed spacing avoids repeatedly solving Matplotlib's layout system
        # for thousands of incremental figures.
        fig = Figure(figsize=(9, 3.6 if count == 1 else 2.8 * count))
        FigureCanvasAgg(fig)
        axes = np.atleast_1d(fig.subplots(count, 1))
        fig.subplots_adjust(left=.13, right=.98, bottom=.24 if count == 1 else .08,
                            top=.84 if count == 1 else .90, hspace=.65)
        return fig, axes

    @staticmethod
    def save(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100)
        fig.clear()

    def mark_kicks(self, ax, origin):
        # Use actual application times, not requested times rounded to a boundary.
        times = sorted({float(event['time']) for event in self.action_kicks})
        for index, timestamp in enumerate(times):
            ax.axvline(timestamp-origin, color='tab:red', linestyle='--',
                       linewidth=1, alpha=.8, label='Kick applied' if index == 0 else None)

    def curves(self, folder, name, sources, field, labels, unit, *, bounds=None,
               target=None, origin=0.0, note=None):
        # Never more than three panels per image, including six-joint quantities.
        for first in range(0, len(labels), 3):
            selection = list(range(first, min(first + 3, len(labels))))
            fig, axes = self.figure(len(selection))
            for ax, j in zip(axes, selection):
                for label, samples in sources.items():
                    if not samples:
                        continue
                    t = np.array([s['time'] for s in samples]) - origin
                    y = np.array([s[field][j] for s in samples])
                    if field == 'command' and len(t) > 1:
                        ax.step(t, y, where='post', label=label, linewidth=1.2)
                    else:
                        ax.plot(t, y, '.-' if len(t) == 1 else '-', label=label,
                                linewidth=1.2, markersize=5)
                if target is not None:
                    ax.axhline(target[j], color='gray', linestyle=':', label='target reference')
                if bounds is not None:
                    lo, hi, description = bounds
                    if np.isfinite(lo[j]):
                        ax.axhline(lo[j], color='red', linestyle=':', label=f'{description} lower')
                    if np.isfinite(hi[j]):
                        ax.axhline(hi[j], color='red', linestyle='--', label=f'{description} upper')
                if folder == self.directory:
                    previous_phase = None
                    events = self.action_solves or [dict(phase=p['name'], time=p['time'])
                                                   for p in (self.action or {}).get('phases', [])]
                    for solve in events:
                        if solve['phase'] != previous_phase:
                            ax.axvline(solve['time']-origin, color='gray', linestyle='--', alpha=.35)
                            ax.text(solve['time']-origin, .98, solve['phase'], rotation=90,
                                    transform=ax.get_xaxis_transform(), va='top', fontsize=6)
                            previous_phase = solve['phase']
                if folder == self.directory:
                    self.mark_kicks(ax, origin)
                ax.set(xlabel='time from solve/action start [s]', ylabel=f'{labels[j]} [{unit}]')
                ax.grid(alpha=.2)
                ax.legend(fontsize=7, loc='best')
            fig.suptitle(f'{self.prefix}: {self.phase} — {name}' + (f'\n{note}' if note else ''), fontsize=10)
            suffix = '' if len(labels) <= 3 else f'_{first+1}-{selection[-1]+1}'
            self.save(fig, folder / f'{self.prefix}_{name}{suffix}.png')

    @staticmethod
    def constraint_report(w, samples, u, previous, held, allowed_block=None):
        report = []
        def add(name, residuals, unit, enforcement):
            a = np.asarray(residuals)
            index = np.unravel_index(np.argmin(a), a.shape)
            value = float(a[index])
            report.append(dict(name=name, minimum_residual=value, unit=unit,
                               index=index, status='violated' if value < -1e-7 else 'active' if value <= 1e-5 else 'inactive',
                               enforcement=enforcement))
        add('joint command magnitude', w.cfg.velocity_limit - np.abs(u), 'rad/s', 'optimizer bound')
        add('joint command change', w.cfg.slew - np.abs(u - previous), 'rad/s per interval', 'optimizer bound')
        if samples:
            v = np.array([s['velocity'] for s in samples])
            add('actual joint speed guard', 1.0 - np.abs(v), 'rad/s', 'runtime guard')
            q = np.array([s['q'] for s in samples])
            for j in range(6):
                if w.model.jnt_limited[j]:
                    lo, hi = w.model.jnt_range[j]
                    add(f'joint {j+1} lower range', q[:, j] - lo, 'rad', 'MuJoCo joint limit')
                    add(f'joint {j+1} upper range', hi - q[:, j], 'rad', 'MuJoCo joint limit')
            unintended = []
            allowed = held if allowed_block is None else allowed_block
            for i, s in enumerate(samples):
                for c in s['contacts']:
                    names = c['geoms']
                    intended = (all(n.startswith('cube') for n in names)
                                or (any(n.startswith('platform') or n == 'floor' for n in names)
                                    and any(n.startswith('cube') for n in names))
                                or (allowed is not None and f'cube{allowed}' in names
                                    and any(n.endswith('_pad') for n in names)))
                    if not intended:
                        unintended.append((c['distance'] + 0.00005, i, names))
            if unintended:
                value, sample, names = min(unintended, key=lambda v: v[0])
                report.append(dict(name='unintended contact penetration guard', minimum_residual=value,
                                   unit='m', sample=sample, geoms=names,
                                   status='violated' if value < 0 else 'active' if value <= 1e-5 else 'inactive',
                                   enforcement='runtime guard; optimizer has a soft penalty'))
        return report

    def solve(self, w, record, truth, estimated, noise, held, previous, allowed_block=None):
        index = self.solve_count
        self.solve_count += 1
        folder = self.directory / f'mpc_{index:06d}_{slug(self.phase)}'
        payload = dict(record=record, phase=self.phase, time=truth[0]['time'],
                       held=held, previous_command=previous, joint_noise=noise,
                       joint_bias=w.cfg.joint_observation_bias,
                       perturbation=np.asarray(estimated[0]['q']) - truth[0]['q'],
                       true_start_forecast=truth, estimated_forecast=estimated,
                       executed=[], application_status='not applied',
                       estimated_constraints=self.constraint_report(w, estimated, np.asarray(record['u']), previous, held, allowed_block),
                       true_start_forecast_constraints=self.constraint_report(w, truth, np.asarray(record['u']), previous, held, allowed_block),
                       position_limits=None, cartesian_velocity_limits=None)
        self.write(folder / 'solve.json', payload)
        self.plot_solve(w, folder, payload)
        self.action_solves.append(payload)
        return folder, payload

    def plot_solve(self, w, folder, payload):
        record = payload['record']
        sources = {'true-start forecast': payload['true_start_forecast'],
                   'MPC estimated forecast': payload['estimated_forecast'],
                   'executed true': payload['executed']}
        origin = payload['time']
        self.curves(folder, 'position_xyz', sources, 'position', list('xyz'), 'm',
                    target=record['target'], origin=origin, note='Target is a reference; no Cartesian position bounds configured')
        self.curves(folder, 'cartesian_velocity_xyz', sources, 'cartesian_velocity', list('xyz'), 'm/s',
                    origin=origin, note='No Cartesian velocity limits configured')
        limits = w.model.jnt_range[:6].copy()
        limits[~w.model.jnt_limited[:6].astype(bool)] = [-np.inf, np.inf]
        self.curves(folder, 'joint_angles', sources, 'q', [f'q{i+1}' for i in range(6)], 'rad',
                    bounds=(limits[:, 0], limits[:, 1], 'model joint range'), origin=origin)
        self.curves(folder, 'joint_velocity', sources, 'velocity', [f'qdot{i+1}' for i in range(6)], 'rad/s',
                    bounds=(-np.ones(6), np.ones(6), 'actual speed guard'), origin=origin)
        u = np.asarray(record['u'])
        command_sources = {'previous command': [dict(time=origin, command=payload['previous_command'])],
                           'optimized held command': [dict(time=s['time'], command=u) for s in payload['estimated_forecast']]}
        lo = np.maximum(-w.cfg.velocity_limit, np.asarray(payload['previous_command'])-w.cfg.slew)
        hi = np.minimum(w.cfg.velocity_limit, np.asarray(payload['previous_command'])+w.cfg.slew)
        self.curves(folder, 'joint_commands', command_sources, 'command', [f'u{i+1}' for i in range(6)], 'rad/s',
                    bounds=(lo, hi, 'command + slew bound'), origin=origin)
        reports = [('MPC estimated forecast', payload['estimated_constraints']),
                   ('true-start forecast', payload['true_start_forecast_constraints'])]
        if payload.get('executed_constraints'):
            reports.append(('executed true', payload['executed_constraints']))
        fig, axes = self.figure(len(reports))
        fig.set_size_inches(11, 3.5 * len(reports))
        fig.subplots_adjust(hspace=.35)
        for ax, (name, rows) in zip(axes, reports):
            ax.axis('off')
            ax.set_title(name, fontsize=9)
            table = ax.table(cellText=[[r['name'], f"{r['minimum_residual']:+.3e} {r['unit']}", r['status']]
                                       for r in rows],
                             colLabels=['Constraint / guard', 'Worst residual (>= 0 feasible)', 'Status'],
                             colWidths=[.48,.34,.18], bbox=[0,0,1,.90])
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            for index, row in enumerate(rows, 1):
                table[index,2].set_facecolor(dict(violated='#ffb3b3', active='#ffe9a6', inactive='#ccedcc')[row['status']])
        fig.suptitle(f'{self.prefix}: solver status={record["status"]}, converged={record["converged"]}; '
                     f'application={payload["application_status"]}', fontsize=10)
        self.save(fig, folder / f'{self.prefix}_constraint_status.png')
        fig, axes = self.figure(1)
        ax = axes[0]
        ax.scatter(np.arange(1,7), payload['perturbation'], label='measured minus true')
        ax.axhline(payload['joint_bias'], color='gray', linestyle=':', label='fixed sensor bias')
        ax.set(xlabel='joint index', ylabel='observation perturbation [rad]', xticks=np.arange(1,7))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        self.save(fig, folder / f'{self.prefix}_observation_perturbation.png')
        if payload.get('one_step_error'):
            fig, axes = self.figure(1)
            fig.suptitle('True minus prediction at equal physical time' +
                         (' (partial aborted interval)' if not payload.get('full_control_interval', True) else ''), fontsize=10)
            axes[0].bar(np.arange(1,7), payload['one_step_error']['q'])
            axes[0].set(xlabel='joint index', ylabel='true minus predicted angle [rad]', xticks=np.arange(1,7))
            axes[0].axhline(0, color='gray', linewidth=.8)
            self.save(fig, folder / f'{self.prefix}_one_step_joint_error.png')
            fig, axes = self.figure(2)
            fig.suptitle('True minus prediction at equal physical time' +
                         (' (partial aborted interval)' if not payload.get('full_control_interval', True) else ''), fontsize=10)
            for ax, field, unit in zip(axes, ['position','cartesian_velocity'], ['m','m/s']):
                ax.bar(list('xyz'), payload['one_step_error'][field])
                ax.set(xlabel='Cartesian component', ylabel=f'true minus predicted {field} [{unit}]')
                ax.axhline(0, color='gray', linewidth=.8)
            self.save(fig, folder / f'{self.prefix}_one_step_cartesian_error.png')

    def record_joint_prediction(self, record, samples):
        # Keep separate segments: each replan starts from a fresh observation.
        self.action_solves.append(dict(time=record['time'], phase=self.phase,
            predicted=[dict(time=s['time'], q=s['q'].copy(),
                            joint_positions=s['joint_positions'].copy()) for s in samples]))

    def plot_joint_prediction(self, record):
        if not self.action_samples:
            return
        fig = Figure(figsize=(14, 10))
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 2, sharex=True).ravel()
        origin = record['start']
        actual_t = np.array([s['time'] for s in self.action_samples]) - origin
        actual_q = np.array([s['q'] for s in self.action_samples])
        for j, ax in enumerate(axes):
            for index, solve in enumerate(self.action_solves):
                segment = solve['predicted']
                ax.plot([s['time']-origin for s in segment], [s['q'][j] for s in segment],
                        color='tab:orange', alpha=.65, linewidth=1,
                        label='MPC predicted horizons' if index == 0 else None)
            ax.plot(actual_t, actual_q[:, j], color='tab:blue', linewidth=1.4,
                    label='Actual robot (including kicks)')
            self.mark_kicks(ax, origin)
            ax.set(xlabel='time from action start [s]', ylabel=f'q{j+1} [rad]')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f'{self.prefix}: {record["kind"]} — {record["status"]}\n'
                     'Orange segments: predictions from each observation; blue: actual execution')
        fig.subplots_adjust(hspace=.35, wspace=.2, top=.9, bottom=.07)
        self.save(fig, self.directory / f'{self.prefix}_joint_prediction.png')
        errors = [solve['one_step_error'] for solve in self.action_solves
                  if solve.get('one_step_error') is not None]
        if errors:
            fig = Figure(figsize=(14, 10))
            FigureCanvasAgg(fig)
            axes = fig.subplots(3, 2, sharex=True).ravel()
            t = np.array([sample['time'] for sample in errors]) - origin
            values = np.array([sample['q'] for sample in errors])
            for j, ax in enumerate(axes):
                ax.plot(t, values[:, j], '.-', color='tab:blue', markersize=3,
                        label='Actual minus one-step prediction')
                ax.axhline(0, color='gray', linestyle=':', linewidth=1)
                self.mark_kicks(ax, origin)
                ax.set(xlabel='time from action start [s]', ylabel=f'q{j+1} error [rad]')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle(f'{self.prefix}: {record["kind"]} — next-step joint prediction error\n'
                         'Post-kick actual minus prediction at the same time; completed intervals only')
            fig.subplots_adjust(hspace=.35, wspace=.2, top=.9, bottom=.07)
            self.save(fig, self.directory / f'{self.prefix}_one_step_joint_error.png')
        self.write(self.directory / 'joint_prediction.json', dict(
            action=record, executed=self.action_samples, predictions=self.action_solves,
            kick_events=self.action_kicks,
            note='Short horizons are separate replans, not a single open-loop action plan.'))

    def plot_cartesian_prediction(self, record):
        if not self.action_samples:
            return
        fig = Figure(figsize=(16, 20))
        FigureCanvasAgg(fig)
        axes = fig.subplots(6, 3, sharex=True)
        origin = record['start']
        times = np.array([sample['time'] for sample in self.action_samples]) - origin
        positions = np.array([sample['joint_positions'] for sample in self.action_samples])
        for joint in range(6):
            for coordinate, name in enumerate('xyz'):
                ax = axes[joint, coordinate]
                for index, solve in enumerate(self.action_solves):
                    segment = solve['predicted']
                    ax.plot([sample['time']-origin for sample in segment],
                            [sample['joint_positions'][joint][coordinate] for sample in segment],
                            color='tab:orange', alpha=.65, linewidth=1,
                            label='MPC predicted horizons' if index == 0 else None)
                ax.plot(times, positions[:, joint, coordinate], color='tab:blue', linewidth=1.3,
                        label='Actual robot (including kicks)')
                self.mark_kicks(ax, origin)
                ax.set(ylabel=f'joint {joint+1} {name} [m]', xlabel='action time [s]')
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .96), ncol=2)
        fig.suptitle(f'{self.prefix}: {record["kind"]} — {record["status"]}\n'
                     'World-frame joint anchors: rows 1–6, columns X/Y/Z; separate prediction horizons')
        fig.subplots_adjust(hspace=.45, wspace=.3, top=.92, bottom=.04)
        self.save(fig, self.directory / f'{self.prefix}_cartesian_prediction.png')
        self.write(self.directory / 'cartesian_prediction.json', dict(
            action=record, executed=self.action_samples, predictions=self.action_solves,
            kick_events=self.action_kicks,
            coordinates='World-frame hinge joint anchors, not joint angles or tool position'))

    def applied(self, w, pending, error=None, allowed_block=None):
        for event in getattr(w, 'last_kick_events', []):
            if np.any(np.asarray(event['magnitude']) != 0):
                self.action_kicks.append(dict(event, time=w.kick_time_origin+event['applied_time']))
        if 'joint-prediction' in self.mode or 'cartesian-prediction' in self.mode:
            if self.action_solves and w.interval_samples:
                solve = self.action_solves[-1]
                actual = w.interval_samples[-1]  # includes the post-kick endpoint
                endpoint = solve['time'] + w.cfg.dt
                predicted = min(solve['predicted'], key=lambda sample: abs(sample['time']-endpoint))
                if (np.isclose(actual['time'], endpoint, atol=1e-9, rtol=0)
                        and np.isclose(predicted['time'], endpoint, atol=1e-9, rtol=0)):
                    solve['one_step_error'] = dict(time=float(actual['time']),
                        q=np.asarray(actual['q'])-predicted['q'],
                        actual_q=actual['q'], predicted_q=predicted['q'],
                        joint_kick=w.last_joint_kick.copy())
            self.action_samples.extend({key: sample[key] for key in
                ('time', 'q', 'position', 'cartesian_velocity', 'joint_positions')} for sample in w.interval_samples)
            return
        if 'executed-cartesian' in self.mode:
            self.action_samples.extend({key: sample[key] for key in
                                        ('time', 'position', 'cartesian_velocity')}
                                       for sample in w.interval_samples)
            return
        if pending is None:
            return
        folder, payload = pending
        payload['executed'] = list(w.interval_samples)
        payload['application_status'] = 'failed' if error else 'applied'
        payload['application_error'] = None if error is None else f'{type(error).__name__}: {error}'
        payload['executed_constraints'] = self.constraint_report(
            w, payload['executed'], np.asarray(payload['record']['u']),
            np.asarray(payload['previous_command']), payload['held'], allowed_block)
        if payload['executed']:
            actual = payload['executed'][-1]
            payload['comparison_dt'] = actual['time'] - payload['time']
            payload['full_control_interval'] = bool(np.isclose(payload['comparison_dt'], w.cfg.dt, atol=1e-9, rtol=0))
            # Partial failed intervals compare equal physical times, not horizon ends.
            predicted = min(payload['estimated_forecast'], key=lambda s: abs(s['time']-actual['time']))
            payload['one_step_error'] = {field: actual[field] - predicted[field]
                                         for field in ('q', 'velocity', 'position', 'cartesian_velocity')}
            self.action_samples.extend(payload['executed'])
            self.latest_prediction = predicted
        self.write(folder / 'solve.json', payload)
        self.plot_solve(w, folder, payload)

    def failed_solve(self, w, error):
        if self.mode != 'all':
            return
        folder = self.directory / f'mpc_{self.solve_count:06d}_{slug(self.phase)}_failed'
        self.solve_count += 1
        sample = w.motion_sample()
        self.write(folder / 'failure.json', dict(error=f'{type(error).__name__}: {error}',
                                                phase=self.phase, actual=sample))
        self.curves(folder, 'failed_position_xyz', {'executed true': [sample]}, 'position', list('xyz'), 'm', origin=sample['time'])

    def save_completion(self, checks, note=None):
        if not checks:
            return
        stem = f'{self.prefix}_completion_{self.completion_count:04d}_{slug(self.phase)}'
        self.completion_count += 1
        self.write(self.directory / (stem+'.json'), dict(checks=checks, note=note,
                   criterion='position error < tolerance AND joint speed norm < 0.035 rad/s for 3 consecutive checks'))
        fig, axes = self.figure(3)
        times = np.array([c['time'] for c in checks])-checks[0]['time']
        for ax, key, threshold, scale, ylabel in zip(axes,
                ['position_error','joint_speed_norm','consecutive_passes'],
                ['position_tolerance','speed_tolerance','required_passes'],
                [1000,1,1], ['position error [mm]','joint speed norm [rad/s]','consecutive accepted checks']):
            ax.plot(times, [c[key]*scale for c in checks], label='actual completion check')
            ax.plot(times, [c[threshold]*scale for c in checks], 'r--', label='threshold')
            ax.set(xlabel='time from first check [s]', ylabel=ylabel)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle(f'{self.prefix}: {self.phase} — waypoint completion'+ (' (reconstructed)' if note else ''),fontsize=10)
        self.save(fig,self.directory/(stem+'.png'))

    def end_action(self, w, record):
        self.write(self.directory / 'action.json', record)
        if 'joint-prediction' in self.mode:
            self.plot_joint_prediction(record)
        if 'cartesian-prediction' in self.mode:
            self.plot_cartesian_prediction(record)
        if self.action_samples and (self.mode == 'all' or 'executed-cartesian' in self.mode):
            sources = {'executed true': self.action_samples}
            origin = record['start']
            self.curves(self.directory, 'executed_position_xyz', sources, 'position', list('xyz'), 'm', origin=origin)
            self.curves(self.directory, 'executed_cartesian_velocity_xyz', sources, 'cartesian_velocity', list('xyz'), 'm/s', origin=origin)
            if self.mode != 'all':
                self.action = None
                self.directory = self.root / 'terminal'
                self.prefix = 'scene'
                self.phase = 'terminal'
                return
            limits = w.model.jnt_range[:6].copy()
            limits[~w.model.jnt_limited[:6].astype(bool)] = [-np.inf, np.inf]
            self.curves(self.directory, 'executed_joint_angles', sources, 'q', [f'q{i+1}' for i in range(6)], 'rad',
                        bounds=(limits[:,0], limits[:,1], 'model joint range'), origin=origin)
            self.curves(self.directory, 'executed_joint_velocity', sources, 'velocity', [f'qdot{i+1}' for i in range(6)], 'rad/s',
                        bounds=(-np.ones(6),np.ones(6),'actual speed guard'), origin=origin)
        if self.action_solves and self.mode == 'all':
            origin = record['start']
            observations = [dict(time=p['time'], perturbation=p['perturbation']) for p in self.action_solves]
            commands = [dict(time=p['time'], command=p['record']['u']) for p in self.action_solves]
            if self.action_samples:
                commands.append(dict(time=self.action_samples[-1]['time'], command=commands[-1]['command']))
            self.curves(self.directory, 'executed_commands', {'applied command': commands}, 'command',
                        [f'u{i+1}' for i in range(6)], 'rad/s',
                        bounds=(-np.full(6,w.cfg.velocity_limit),np.full(6,w.cfg.velocity_limit),'command bound'), origin=origin)
            errors = [dict(time=p['executed'][-1]['time'], error=p['one_step_error']['q'])
                      for p in self.action_solves if p.get('one_step_error')]
            bias = [dict(time=p['time'], perturbation=np.full(6,w.cfg.joint_observation_bias)) for p in self.action_solves]
            self.curves(self.directory, 'joint_observation_perturbation', {'bias + noise': observations, 'fixed bias': bias},
                        'perturbation', [f'q{i+1}' for i in range(6)], 'rad', origin=origin)
            if errors:
                self.curves(self.directory, 'one_step_joint_prediction_error', {'true minus prediction': errors},
                            'error', [f'q{i+1}' for i in range(6)], 'rad', origin=origin)
                position_errors = [dict(time=p['executed'][-1]['time'], error=p['one_step_error']['position'])
                                   for p in self.action_solves if p.get('one_step_error')]
                velocity_errors = [dict(time=p['executed'][-1]['time'], error=p['one_step_error']['cartesian_velocity'])
                                   for p in self.action_solves if p.get('one_step_error')]
                self.curves(self.directory, 'one_step_position_error_xyz', {'true minus prediction': position_errors},
                            'error', list('xyz'), 'm', origin=origin)
                self.curves(self.directory, 'one_step_velocity_error_xyz', {'true minus prediction': velocity_errors},
                            'error', list('xyz'), 'm/s', origin=origin)
        self.action = None
        self.directory = self.root / 'terminal'
        self.prefix = 'scene'
        self.phase = 'terminal'

    def grounding(self, w, result, observed, truth, positions, true_positions, grounder, require_settled, plot_final=False):
        if self.mode != 'all' and 'final-grounding' not in self.mode:
            return
        if 'final-grounding' in self.mode and not plot_final:
            return
        folder = self.directory if 'final-grounding' in self.mode else self.directory / f'grounding_{self.ground_count:06d}_{slug(self.phase)}'
        if 'final-grounding' in self.mode and self.action is None:
            folder = self.directory / f'grounding_{self.ground_count:06d}'
        self.final_grounding_saved = self.final_grounding_saved or plot_final
        self.ground_count += 1
        predicted = None
        if self.latest_prediction is not None and abs(self.latest_prediction['time'] - float(w.data.time)) < 1e-7:
            predicted = support_components(self.latest_prediction['blocks'],
                                           self.latest_prediction['block_rotations'], result.held)
        self.write(folder / 'grounding.json', dict(time=float(w.data.time), observed_positions=positions,
                    true_positions=true_positions, observed_components=observed, true_components=truth,
                    predicted_components=predicted,
                    accepted=result.accepted, reason=result.reason, supports=result.supports,
                    gap=result.gap, held=result.held, require_settled=require_settled,
                    eta_on=grounder.eta_on, eta_off=grounder.eta_off,
                    scope='Truth and noisy block observation of the same scene; no independent baseline run.'))
        for b, options in observed.items():
            prefix = f'{self.color(w,b)}_block{b}'
            fig, axes = self.figure(3)
            specifications = [('area_ratio', 'area', np.linspace(0,1,150), lambda x: x, 'overlap / footprint area'),
                              ('height_error', 'height', np.linspace(-.015,.015,200), lambda x: np.clip((.009-np.abs(x))/.007,0,1), 'signed bottom-to-support height error [m]'),
                              ('orientation_error', 'orientation', np.linspace(0,.22,150), lambda x: np.clip((.17-x)/.135,0,1), 'tilt angle [rad]')]
            for ax, (input_key, degree, xs, function, xlabel) in zip(axes, specifications):
                comparison = [('true state', truth[b], 'o'), ('noisy observed state', options, 'x')]
                if predicted is not None:
                    comparison.append(('MPC one-step predicted state', predicted[b], '^'))
                inputs = [v[input_key] for _, source, _ in comparison for v in source.values()]
                xs = np.linspace(min(xs[0], min(inputs)), max(xs[-1], max(inputs)), 400)
                ax.plot(xs, function(xs), color='black', label=f'{degree} membership')
                for label, source, marker in comparison:
                    ax.scatter([v[input_key] for v in source.values()], [v[degree] for v in source.values()],
                               marker=marker, s=24, alpha=.6, label=label)
                ax.set(xlabel=xlabel, ylabel='membership', ylim=(-.05,1.05))
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
            fig.suptitle(f'{prefix}: all candidate support membership inputs — {result.reason}', fontsize=10)
            self.save(fig, folder / f'{prefix}_area_height_orientation.png')
            fig, axes = self.figure(1)
            ax = axes[0]
            supports = list(options)
            x = np.arange(len(supports))
            ax.scatter(x, [truth[b][s]['combined'] for s in supports], label='true state', marker='o')
            ax.scatter(x, [options[s]['combined'] for s in supports], label='noisy observed state', marker='x')
            if predicted is not None:
                ax.scatter(x, [predicted[b][s]['combined'] for s in supports], label='MPC one-step predicted state', marker='^')
            ax.axhline(grounder.eta_on, color='red', linestyle='--', label='entry threshold')
            ax.axhline(grounder.eta_off, color='red', linestyle=':', label='retention threshold')
            ax.set_xticks(x, [str(s) for s in supports], rotation=60, fontsize=6)
            ax.set(ylabel='area × height × orientation', xlabel='candidate support', ylim=(-.05,1.05))
            ax.legend(fontsize=7)
            fig.suptitle(f'{prefix}: combined memberships; accepted={result.accepted}; gap={result.gap:.3g}', fontsize=10)
            self.save(fig, folder / f'{prefix}_combined_memberships.png')
