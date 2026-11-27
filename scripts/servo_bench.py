"""Servo bench (Issue #88): run the step and chirp tests on one servo, log them to CSV, and fit the servo model.

The plan and the operator's guide are in docs/design/2026-11-27-servo-model.md. Commands:

  run      drive a real servo through a serial bus adapter (needs pyserial: `uv sync --extra bench`)
             uv run python scripts/servo_bench.py run --port PORT --out log.csv [--schedule full|short]
                 [--load-mass 0.3 --arm-length 0.15 --arm-mass 0.03] [--id 1] [--baud 1000000] [--echo]
  fit      fit the servo model to one or more logs (each with its rig in its header)
             uv run python scripts/servo_bench.py fit log1.csv [log2.csv ...] [--out fit.json]
  dry-run  the same run and fit against a simulated servo with known parameters (no hardware; run in CI);
           exits with 1 if the fit misses the known parameters by more than erg.bench.FIT_TOLERANCE
             uv run python scripts/servo_bench.py dry-run [--out-dir DIR]
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import bench  # noqa: E402


def _rig(args) -> bench.Rig:
    return bench.Rig(arm_mass=args.arm_mass, arm_length=args.arm_length, load_mass=args.load_mass)


def _report(result: bench.FitResult) -> dict:
    p = dataclasses.asdict(result.params)
    print("fitted servo model:")
    for k in ("delay", *bench.FITTED):
        print(f"  {k:14s} {p[k]:.5g}")
    print(f"  rms error: position {result.rms_position:.3g} rad, speed {result.rms_speed:.3g} rad/s; "
          f"load reading / simulated drive torque: {result.load_scale:.3f}")
    return {"params": p, "cost": result.cost, "rms_position": result.rms_position, "rms_speed": result.rms_speed,
            "load_scale": result.load_scale}


def cmd_run(args) -> int:
    try:
        import serial  # pyserial
    except ImportError:
        print("pyserial is missing: uv sync --extra bench", file=sys.stderr)
        return 2
    port = serial.Serial(args.port, args.baud, timeout=0.02)
    limits = bench.Limits(angle=args.max_angle, max_temperature=args.max_temperature, min_voltage=args.min_voltage,
                          max_current=args.max_current)
    try:
        rows = bench.Bench(port, args.id, args.zero, args.echo).run(bench.schedule(args.schedule), limits=limits)
    finally:
        port.close()
    bench.write_csv(args.out, rows, _rig(args), {"servo_id": args.id, "schedule": args.schedule})
    print(f"{len(rows)} rows written to {args.out}")
    return 0


def cmd_fit(args) -> int:
    logs = []
    for path in args.logs:
        cols, rig, _ = bench.read_csv(path)
        logs.append((cols, rig))
    out = _report(bench.fit(logs))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_dry_run(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = bench.Rig()
    port = bench.SimServoPort(bench.DRY_RUN_TRUTH, rig, seed=0)
    rows = bench.Bench(port).run(bench.schedule("short"))
    log_path = out_dir / "servo_bench_dry_run.csv"
    bench.write_csv(log_path, rows, rig, {"servo_id": 1, "schedule": "short", "simulated": True})
    print(f"{len(rows)} rows written to {log_path}")
    cols, rig2, _ = bench.read_csv(log_path)
    result = bench.fit([(cols, rig2)], log=lambda s: None)
    out = _report(result)
    check = bench.within_tolerance(result.params, bench.DRY_RUN_TRUTH)
    out["check"] = {k: {"fitted": a, "truth": b, "ok": ok} for k, (a, b, ok) in check.items()}
    (out_dir / "servo_bench_dry_run.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for k, (a, b, ok) in check.items():
        print(f"  {k:14s} fitted {a:.5g}, truth {b:.5g}: {'ok' if ok else 'OUTSIDE THE TOLERANCE'}")
    return 0 if all(ok for _, _, ok in check.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="drive a real servo")
    r.add_argument("--port", required=True, help="serial port of the bus adapter (e.g. COM5 or /dev/ttyUSB0)")
    r.add_argument("--baud", type=int, default=1_000_000)
    r.add_argument("--id", type=int, default=1)
    r.add_argument("--zero", type=int, default=2048, help="servo steps at the arm's hanging position")
    r.add_argument("--echo", action="store_true", help="the adapter hears its own transmission")
    r.add_argument("--schedule", choices=("full", "short"), default="full")
    r.add_argument("--out", required=True)
    defaults = bench.Rig()
    r.add_argument("--arm-mass", type=float, default=defaults.arm_mass, help="kg")
    r.add_argument("--arm-length", type=float, default=defaults.arm_length, help="m, shaft to the load")
    r.add_argument("--load-mass", type=float, default=defaults.load_mass, help="kg")
    lim = bench.Limits()
    r.add_argument("--max-angle", type=float, default=lim.angle, help="rad from the hanging position")
    r.add_argument("--max-temperature", type=int, default=lim.max_temperature, help="C")
    r.add_argument("--min-voltage", type=float, default=lim.min_voltage, help="V")
    r.add_argument("--max-current", type=float, default=lim.max_current, help="A")
    f = sub.add_parser("fit", help="fit the servo model to logs")
    f.add_argument("logs", nargs="+")
    f.add_argument("--out")
    d = sub.add_parser("dry-run", help="run and fit against a simulated servo")
    d.add_argument("--out-dir", default="target/servo_bench")
    args = ap.parse_args()
    return {"run": cmd_run, "fit": cmd_fit, "dry-run": cmd_dry_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
