"""Evaluate the base controller (Issue #71) against the pass conditions of docs/design/2026-11-27-base-controller.md.

For each command and seed, walks 20 s in MuJoCo (per-leg mode, messages between neighbours only, through Dyne's wire
format) and prints the mean forward and sideways speed, the yaw rate, the largest tilt, the lowest torso height,
falls, and the worst joint's 95th percentile and maximum of torque / stall torque; then the torque at the speeds
reported only, the pushes and determinism. Exit status 1 if a condition fails.

    uv run python scripts/base_controller.py [--seeds 0,1,2,3,4] [--seconds 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from erg import base_controller as bc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--push-forces", default="",
                    help="report only: push forces (N, for 0.1 s, comma-separated) to count falls with")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.push_forces:
        for f in (float(x) for x in args.push_forces.split(",")):
            falls = sum(bc.run((0.2, 0.0, 0.0), seed, seconds=args.seconds, push=d, push_force=f).fell
                        for seed in seeds for d in (1, -1))
            print(f"push {f:5.1f} N for {bc.PUSH_SECONDS} s ({f * bc.PUSH_SECONDS:.2f} N s): "
                  f"{falls} falls in {2 * len(seeds)} runs", flush=True)
        return 0
    failed = []
    print(f"{'command (vx, vy, wz)':>22} seed  forward  ratio  sideways  yaw rate  tilt deg  min z   "
          "torque p95  max   result")
    for cmd in bc.FORWARD_COMMANDS + bc.TURN_COMMANDS + bc.SIDEWAYS_COMMANDS:
        for seed in seeds:
            r = bc.run(cmd, seed, seconds=args.seconds)
            bad = bc.check(cmd, r)
            # ratio of the measured to the commanded value that the command's condition is about
            if cmd in bc.TURN_COMMANDS:
                ratio = r.yaw_rate / cmd[2]
            elif cmd in bc.SIDEWAYS_COMMANDS:
                ratio = r.sideways / cmd[1]
            else:
                ratio = r.forward / cmd[0]
            print(f"{str(cmd):>22} {seed:4d}  {r.forward:7.3f}  {ratio:5.2f}  {r.sideways:8.3f}  {r.yaw_rate:8.3f}  "
                  f"{r.max_tilt_deg:8.1f}  {r.min_height:.3f}  {r.torque_p95:9.2f}  {r.torque_max:4.2f}   "
                  f"{'pass' if not bad else 'FAIL: ' + '; '.join(bad)}", flush=True)
            if bad:
                failed.append((cmd, seed, bad))
            if cmd == bc.TORQUE_COMMAND and r.torque_p95 > bc.TORQUE_P95:
                failed.append((cmd, seed, [f"torque p95 {r.torque_p95:.2f} > {bc.TORQUE_P95}"]))
    print(f"\ntorque (worst joint's 95th percentile / stall torque; condition 6: <= {bc.TORQUE_P95} at "
          f"{bc.TORQUE_COMMAND[0]} m/s; reported only at the speeds below):")
    for cmd in bc.TORQUE_REPORTED:
        values = [bc.run(cmd, seed, seconds=args.seconds).torque_p95 for seed in seeds]
        print(f"  {cmd}: " + "  ".join(f"{v:.2f}" for v in values), flush=True)
    print(f"\npushes: {bc.PUSH_FORCE} N sideways for {bc.PUSH_SECONDS} s at 10 s, walking at 0.2 m/s")
    for direction in (1, -1):
        for seed in seeds:
            r = bc.run((0.2, 0.0, 0.0), seed, seconds=args.seconds, push=direction)
            side = "left" if direction > 0 else "right"
            print(f"  {side:5s} seed {seed}: forward {r.forward:.3f}  tilt {r.max_tilt_deg:5.1f} deg  "
                  f"min z {r.min_height:.3f}  {'FELL' if r.fell else 'no fall'}", flush=True)
            if r.fell:
                failed.append(("push " + side, seed, ["fell"]))
    a = bc.run((0.2, 0.0, 0.0), 0, seconds=args.seconds)
    b = bc.run((0.2, 0.0, 0.0), 0, seconds=args.seconds)
    same = np.array_equal(a.targets, b.targets)
    print(f"\ndeterminism (two runs, seed 0): {'same bits' if same else 'DIFFERENT'}")
    if not same:
        failed.append(("determinism", 0, ["different"]))
    print("\nfailed conditions:" if failed else "\nall conditions pass")
    for cmd, seed, bad in failed:
        print(f"  {cmd} seed {seed}: {'; '.join(bad)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
