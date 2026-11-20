import argparse, json, time, platform, sys, hashlib, ast
from pathlib import Path
from dataclasses import asdict
import numpy as np
import mujoco
from .debug import traced, log
from .physics import World, Config
from .execution import Executor
from .diagnostics import normalize_plot_mode
from .learning import Models, HiddenEvaluator
from .planning import Planner, color_goals, letter_goals, order_from_supports
from .model import cells
from .benchmarks import fixture
from .geometry import CandidateGeometry


def parse_kick_timestamps(value):
    try:
        times = tuple(float(t.strip()) for t in value.split(','))
        if any(not np.isfinite(t) or t < 0 for t in times):
            raise ValueError()
        return times
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use nonnegative seconds separated by commas, e.g. 2,5,7") from exc


def parse_kick_magnitudes(values):
    try:
        parsed = ast.literal_eval(' '.join(values))
        if isinstance(parsed, (tuple, list)) and parsed and all(isinstance(x, (int, float)) for x in parsed):
            parsed = [parsed]
        if not isinstance(parsed, (tuple, list)):
            raise ValueError()
        return tuple(tuple(float(x) for x in vector) for vector in parsed)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError('Use quoted vectors, e.g. --kick-magnitude "(1e-3,0,0), (1e-3,1e-4,0)"') from exc


@traced("Outer symbolic planning / physical control / learning loop", run_scope=True)
def run(
    kind="color",
    rounds=1,
    seed=0,
    out="results/run",
    video=False,
    mass_scale=1.0,
    friction=1.0,
    epsilon=0.1,
    impratio=1.0,
    support_solref=0.02,
    verbose=False,
    plots=True,
    plot_mode="all",
    completion_plots=False,
    joint_observation_noise=0.0,
    joint_observation_bias=0.0,
    observation_noise=0.0,
    observation_bias=0.0,
    joint_kick_half_range=Config.joint_kick_half_range,
    joint_kick_bias=Config.joint_kick_bias,
    kick_timestamps=(),
    kick_magnitudes=(),
):
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    config = Config(
        seed=seed,
        mass_scale=mass_scale,
        friction=friction,
        contact_impratio=impratio,
        support_solref=support_solref,
        joint_observation_noise=joint_observation_noise,
        joint_observation_bias=joint_observation_bias,
        observation_noise=observation_noise,
        observation_bias=observation_bias,
        joint_kick_half_range=joint_kick_half_range,
        joint_kick_bias=joint_kick_bias,
        kick_timestamps=kick_timestamps,
        kick_magnitudes=kick_magnitudes,
    )
    initial = (
        {i: f"p0c{i}" for i in range(5)}
        if kind == "color"
        else {0: "p0c0", 1: 0, 2: 1, 3: 2, 4: 3}
    )
    if kind in ("b0", "b1"):
        initial, fixed_goals, fixed_sites = fixture(kind)
        if rounds != 1:
            raise ValueError(
                "B0/B1 are single-instance tasks; use independent seeds for repetition"
            )
    w = World(initial, config)
    models = Models()
    plot_mode = normalize_plot_mode(plot_mode)
    e = Executor(w, models, video, plot_dir=directory / "plots" if plots and plot_mode != "none" else None,
                 plot_mode="all" if plot_mode == "none" else plot_mode, completion_plots=completion_plots)
    planner = Planner(models, seed=seed, geometry=CandidateGeometry(w))
    evaluator = HiddenEvaluator()
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    completed = []
    status = "failed"
    error = None
    source = 0
    previous = (
        order_from_supports(e.supports, kind)
        if kind in ("color", "letter")
        else tuple(range(w.nblock))
    )
    manifest = dict(
        task=kind,
        requested_rounds=rounds,
        seed=seed,
        config=asdict(config),
        epsilon=epsilon,
        observation_condition="joint-angle and block-position observation bias/noise configurable; velocities, rotations and contacts remain oracle simulator channels",
        plot_directory=str(e.diagnostics.root) if e.diagnostics is not None else None,
        plot_mode=plot_mode if e.diagnostics is not None else "none",
        platform=platform.platform(),
        python=sys.version,
        mujoco=mujoco.__version__,
        numpy=np.__version__,
        model_sha256=hashlib.sha256(w.xml.encode()).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        perturbation_condition="scheduled arm joint-position kicks at control endpoints (elapsed from initialization); optional legacy random kicks only without a schedule; nominal predictions exclude future kicks",
        controller="MuJoCo move-blocked shooting; soft collision penalty; external grasp guards",
        ideal_contact_OCP_equivalent=False,
    )
    (directory / "scene.xml").write_text(w.xml)
    try:
        for r in range(rounds):
            destination = (source + 1) % 3 if kind == "color" else 1 - source
            goals = (
                color_goals(previous, destination, models)
                if kind == "color"
                else letter_goals(previous, f"p{destination}c0", models)
            )
            if kind in ("b0", "b1"):
                goals = fixed_goals
            explore = bool(rng.random() < epsilon)
            if explore:
                goals = [goals[int(rng.integers(len(goals)))]]
            sites = (
                list(cells())
                if kind == "color"
                else ["p0c0", "p1c0", "p0c2", "p0c5", "p1c2", "p1c5"]
            )
            if kind in ("b0", "b1"):
                sites = fixed_sites
            round_start = len(e.actions)
            for decision in range(100):
                # Freeze models for this search; dispatch only its first action.
                plan, terminal = planner.solve(
                    e.supports, goals, sites, e.anchor, e.held
                )
                if not plan:
                    break
                action = plan[0]
                log("Dispatch first symbolic action; replan after completion", action=action, supports=e.supports, anchor=e.anchor, held=e.held)
                print(
                    f"round={r} action={len(e.actions)} {action} expansions={planner.records[-1]['expanded']}",
                    flush=True,
                )
                # A goal-completing release must settle the entire scene before
                # its action cost is finalized and any terminal feedback is sent.
                effect = {**e.supports, action[1]: action[2]}
                settle_all = action[0] == "PlaceRelease" and tuple(
                    effect.get(i) for i in range(w.nblock)
                ) in {g[0] for g in goals}
                e.execute(action, settle_all=settle_all)
                # Updated execution costs are already available to the next solve.
            else:
                raise RuntimeError("round action budget exhausted")
            terminal_observation = e.grounder.measure(w, require_settled=True, plot_final=True)
            if (
                not terminal_observation.accepted
                or terminal_observation.supports != e.supports
            ):
                raise RuntimeError("terminal support or velocity acceptance failed")
            actual = tuple(e.supports[i] for i in range(w.nblock))
            if e.held is not None or actual not in {g[0] for g in goals}:
                raise RuntimeError("terminal goal unverified")
            order = (
                order_from_supports(e.supports, kind)
                if kind in ("color", "letter")
                else tuple(next(g[2] for g in goals if g[0] == actual))
            )
            rating = (
                evaluator.color(destination, order[0])
                if kind == "color"
                else (evaluator.letter(order) if kind == "letter" else None)
            )
            if kind == "color":
                models.rate_color(destination, order[0], rating)
            elif kind == "letter":
                models.rate_letter(order, rating)
            else:
                rating = None
            # Memory changes only here, after the observed complete-round goal.
            completed.append(
                dict(
                    round=r,
                    source=source,
                    destination=destination,
                    previous=previous,
                    order=order,
                    rating=rating,
                    exploration=explore,
                    actions=len(e.actions) - round_start,
                    supports=e.supports.copy(),
                )
            )
            previous = order
            source = destination
        status = "success"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, flush=True)
    finally:
        summary = dict(
            status=status,
            error=error,
            plot_directory=str(e.diagnostics.root) if e.diagnostics is not None else None,
            plotted_mpc_attempts=e.diagnostics.solve_count if e.diagnostics is not None else 0,
            plotted_grounding_evaluations=e.diagnostics.ground_count if e.diagnostics is not None else 0,
            completed_rounds=len(completed),
            rounds=completed,
            wall_seconds=time.perf_counter() - started,
            simulation_seconds=float(w.data.time),
            actions=len(e.actions),
            successful_actions=sum(a["status"] == "success" for a in e.actions),
            replans=len(planner.records),
            mpc_solves=len(e.mpc.records),
            latency_p50=float(np.median([x["seconds"] for x in e.mpc.records]))
            if e.mpc.records
            else None,
            latency_p95=float(np.quantile([x["seconds"] for x in e.mpc.records], 0.95))
            if e.mpc.records
            else None,
            solver_converged_fraction=float(
                np.mean([x["converged"] for x in e.mpc.records])
            )
            if e.mpc.records
            else None,
            max_slip=max([x["slip"] for x in e.monitor], default=0),
            final_positions=[w.blockpos(b).tolist() for b in range(w.nblock)],
            memory=previous,
            learned_action_costs=models.theta_C,
            color_estimates=models.color.tolist(),
            letter_estimates=models.letter.tolist(),
        )
        for name, obj in [
            ("manifest", manifest),
            ("summary", summary),
            ("actions", e.actions),
            ("planner", planner.records),
            ("mpc", e.mpc.records),
            ("grasp_observations", e.monitor),
            ("trajectory", w.trace),
        ]:
            (directory / (name + ".json")).write_text(
                json.dumps(obj, indent=2, allow_nan=False)
            )
        if video:
            import imageio.v2 as imageio

            w.capture()
            imageio.mimsave(directory / "rollout.mp4", w.frames, fps=5)
            imageio.imwrite(directory / "final.png", w.frames[-1])
        w.close()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Physical mission-aligned manipulation simulation"
    )
    p.add_argument("--task", choices=["color", "letter", "b0", "b1"], default="color")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/run")
    p.add_argument("--video", action="store_true")
    p.add_argument("--verbose", action="store_true", help="Write labeled function, MPC, grounding and feedback traces to <out>/debug_trace.log")
    p.add_argument("--completion-plots", action="store_true", help="Add waypoint completion error, speed and consecutive-pass threshold plots")
    p.add_argument("--no-plots", action="store_true", help="Disable incremental MPC/grounding PNG and JSON diagnostics")
    p.add_argument("--plot-mode", nargs="+", choices=["all", "final-grounding", "executed-cartesian", "joint-prediction", "cartesian-prediction", "none"], default="all",
                   help="Save all MPC/grounding plots, only final grounding per action/round, only executed Cartesian action summaries, joint prediction action summaries, or no plots; multiple values can be combined")
    p.add_argument("--cartisian-prediction", "--cartesian-prediction", dest="cartesian_prediction", action="store_true", help="Save only action-level 18-panel XYZ joint-anchor prediction comparisons")
    p.add_argument("--kick-timestamps", type=parse_kick_timestamps, default=(), help="Kick times in elapsed simulation seconds after initialization, e.g. 2,5,7; applied at first control endpoint at/after each time")
    p.add_argument("--kick-magnitude", nargs="+", default=[], help="Quoted comma-separated joint-angle vectors [rad]; missing joints/events default to zero")
    p.add_argument("--joint-kick-half-range", type=float, default=Config.joint_kick_half_range, help="Uniform position-kick half-range per arm joint per control interval [rad]; 0 disables random kicks")
    p.add_argument("--joint-kick-bias", type=float, default=Config.joint_kick_bias, help="Physical position-kick mean per arm joint per control interval [rad]")
    p.add_argument("--joint-observation-noise", type=float, default=0.0, help="Joint-angle observation standard deviation [rad]")
    p.add_argument("--joint-observation-bias", type=float, default=0.0, help="Fixed observation bias for each arm joint [rad]")
    p.add_argument("--observation-noise", type=float, default=0.0, help="Block-position observation standard deviation [m]")
    p.add_argument("--observation-bias", type=float, default=0.0, help="Block observation x bias [m]")
    p.add_argument("--mass-scale", type=float, default=1.0)
    p.add_argument("--friction", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument(
        "--impratio",
        type=float,
        default=1.0,
        help="MuJoCo tangential/normal constraint impedance ratio; recorded in manifest",
    )
    p.add_argument(
        "--support-solref",
        type=float,
        default=0.02,
        help="Cube contact time constant in seconds; keep >= twice integration timestep",
    )
    args = p.parse_args()
    if args.support_solref < 0.004:
        p.error("--support-solref must be at least 0.004 seconds")
    try:
        kick_magnitudes = parse_kick_magnitudes(args.kick_magnitude) if args.kick_magnitude else ()
        Config(kick_timestamps=args.kick_timestamps, kick_magnitudes=kick_magnitudes)
    except ValueError as exc:
        p.error(str(exc))
    result = run(
        args.task,
        args.rounds,
        args.seed,
        args.out,
        args.video,
        args.mass_scale,
        args.friction,
        args.epsilon,
        args.impratio,
        args.support_solref,
        verbose=args.verbose,
        plots=not args.no_plots,
        plot_mode="cartesian-prediction" if args.cartesian_prediction else args.plot_mode,
        completion_plots=args.completion_plots,
        joint_observation_noise=args.joint_observation_noise,
        joint_observation_bias=args.joint_observation_bias,
        observation_noise=args.observation_noise,
        observation_bias=args.observation_bias,
        joint_kick_half_range=args.joint_kick_half_range,
        joint_kick_bias=args.joint_kick_bias,
        kick_timestamps=args.kick_timestamps,
        kick_magnitudes=kick_magnitudes,
    )
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
