"""Evaluate policies trained on the base controller against the pass conditions of
docs/design/2026-11-27-learning-setup.md (Issue #82), next to the base controller alone.

    uv run python scripts/evaluate_learning.py runs/split-0 runs/split-1 ... --out results.json

Each run directory holds agent.pt and run.json (scripts/train.py). Every policy and the base controller alone walk
20 s from standing for each command, with the servo model randomised from each evaluation seed, and are measured over
5-20 s (erg.base_controller.run).
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import base_controller as bc  # noqa: E402
from erg import ppo  # noqa: E402
from erg.servo import ServoParams  # noqa: E402

SEEDS = [900001, 900002, 900003, 900004, 900005]  # not used by any training episode
FORWARD = [(0.05, 0.0, 0.0), (0.1, 0.0, 0.0), (0.15, 0.0, 0.0)]
TURN = [(0.1, 0.0, 0.3), (0.1, 0.0, -0.3)]
COMMANDS = FORWARD + TURN
SECONDS, WINDOW = 20.0, 5.0
FORWARD_TOL, DRIFT_SPEED, DRIFT_YAW = 0.2, 0.03, 0.1  # L1
TURN_TOL, TURN_FORWARD_TOL = 0.3, 0.3  # L2
SMOOTH_FACTOR = 1.2  # L4
TORQUE_P95 = 0.5  # L5
SMOOTH_COMMAND = TORQUE_COMMAND = (0.1, 0.0, 0.0)


def load(run_dir: Path) -> ppo.Agent:
    form = json.loads((run_dir / "run.json").read_text())["config"]["form"]
    agent = ppo.Agent(form, "base")
    agent.load_state_dict(torch.load(run_dir / "agent.pt"))
    return agent


def measure(job) -> dict:
    """One 20 s run: job = (run directory or None for the base controller alone, command, seed)."""
    torch.set_num_threads(1)
    run_dir, command, seed = job
    adjust = ppo.PolicyAdjust(load(Path(run_dir))) if run_dir else None
    r = bc.run(command, seed=seed, seconds=SECONDS, window=WINDOW, servo=ServoParams(), randomize_servo=True,
               adjust=adjust)
    start = round(WINDOW / bc.DT)
    return {"policy": run_dir or "base", "command": list(command), "seed": seed, "forward": r.forward,
            "sideways": r.sideways, "yaw_rate": r.yaw_rate, "fell": r.fell, "max_tilt_deg": r.max_tilt_deg,
            "torque_p95": r.torque_p95, "torque_max": r.torque_max,
            "second_difference": float(np.abs(np.diff(r.targets[start:], 2, axis=0)).mean())}


def conditions(rows: list, base_rows: list) -> dict:
    """Pass (True) or fail of L1-L5 for one policy's rows, with the failures listed."""
    base_d2 = {r["seed"]: r["second_difference"] for r in base_rows if tuple(r["command"]) == SMOOTH_COMMAND}
    fails = {k: [] for k in ("L1", "L2", "L3", "L4", "L5")}
    for r in rows:
        c = tuple(r["command"])
        tag = f"{c} seed {r['seed']}"
        if c in FORWARD:
            if abs(r["forward"] - c[0]) > FORWARD_TOL * c[0]:
                fails["L1"].append(f"{tag}: forward {r['forward']:.3f}")
            if abs(r["sideways"]) > DRIFT_SPEED:
                fails["L1"].append(f"{tag}: sideways {r['sideways']:.3f}")
            if abs(r["yaw_rate"]) > DRIFT_YAW:
                fails["L1"].append(f"{tag}: yaw {r['yaw_rate']:.3f}")
        if c in TURN:
            if abs(r["yaw_rate"] - c[2]) > TURN_TOL * abs(c[2]):
                fails["L2"].append(f"{tag}: yaw {r['yaw_rate']:.3f}")
            if abs(r["forward"] - c[0]) > TURN_FORWARD_TOL * c[0]:
                fails["L2"].append(f"{tag}: forward {r['forward']:.3f}")
        if r["fell"]:
            fails["L3"].append(f"{tag}: fell")
        if c == SMOOTH_COMMAND:
            if r["second_difference"] > SMOOTH_FACTOR * base_d2[r["seed"]]:
                fails["L4"].append(f"{tag}: {r['second_difference']:.5f} against {base_d2[r['seed']]:.5f}")
        if c == TORQUE_COMMAND and r["torque_p95"] > TORQUE_P95:
            fails["L5"].append(f"{tag}: {r['torque_p95']:.3f}")
    return fails


def summary(rows: list) -> dict:
    """Ranges over seeds per command, and the mean absolute relative tracking error (reported, not a condition)."""
    out = {}
    for c in COMMANDS:
        sel = [r for r in rows if tuple(r["command"]) == c]
        out[str(c)] = {k: [min(r[k] for r in sel), max(r[k] for r in sel)]
                       for k in ("forward", "sideways", "yaw_rate", "torque_p95", "second_difference", "max_tilt_deg")}
    err = []
    for r in rows:
        c = r["command"]
        err.append(abs(r["forward"] - c[0]) / c[0])
        if c[2]:
            err.append(abs(r["yaw_rate"] - c[2]) / abs(c[2]))
    out["mean_relative_tracking_error"] = float(np.mean(err))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    jobs = [(None, c, s) for c in COMMANDS for s in SEEDS]
    jobs += [(str(d), c, s) for d in a.runs for c in COMMANDS for s in SEEDS]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(measure, jobs))
    base_rows = [r for r in rows if r["policy"] == "base"]
    report = {"rows": rows, "policies": {}}
    for name in ["base"] + [str(d) for d in a.runs]:
        sel = [r for r in rows if r["policy"] == name]
        fails = conditions(sel, base_rows)
        report["policies"][name] = {"pass": {k: not v for k, v in fails.items()}, "failures": fails,
                                    "summary": summary(sel)}
        print(name, " ".join(f"{k}:{'pass' if not v else 'FAIL'}" for k, v in fails.items()),
              f"tracking error {report['policies'][name]['summary']['mean_relative_tracking_error']:.3f}")
        for c in COMMANDS:
            s = report["policies"][name]["summary"][str(c)]
            print("   ", c, " ".join(f"{k} {v[0]:.4f}..{v[1]:.4f}" for k, v in s.items()))
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
