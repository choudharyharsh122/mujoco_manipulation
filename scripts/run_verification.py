#!/usr/bin/env python3
"""Reproduce the declared four-task pilot protocol; preserve every failed run."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/local/verification")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be new or empty to preserve prior evidence")
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(root)
    if args.video:
        env.setdefault("MUJOCO_GL", "egl")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=root, env=env, check=True
    )
    passed = []
    failures = []
    for task, rounds in [("b0", 1), ("color", 3), ("letter", 2), ("b1", 1)]:
        folder = out / task
        command = [
            sys.executable,
            "-m",
            "mission_mujoco.cli",
            "--task",
            task,
            "--rounds",
            str(rounds),
            "--epsilon",
            "0",
            "--out",
            str(folder),
        ]
        if task in ("b0", "letter", "b1"):
            command += ["--impratio", "10", "--support-solref", ".01"]
        if args.video:
            command += ["--video"]
        with (out / f"{task}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        (passed if result.returncode == 0 else failures).append(str(folder))
        print(task, "passed" if result.returncode == 0 else "FAILED", flush=True)
    audit = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/audit_results.py"),
            *[str(out / t) for t in ("b0", "color", "letter", "b1")],
            "--out",
            str(out / "verification.json"),
        ],
        cwd=root,
        env=env,
    )
    if passed:
        subprocess.run(
            [sys.executable, str(root / "scripts/replay_trace.py"), *passed],
            cwd=root,
            env=env,
            check=True,
        )
    return 1 if failures or audit.returncode else 0


if __name__ == "__main__":
    raise SystemExit(main())
