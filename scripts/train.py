"""Train one policy with PPO (Issue #42) and save its weights and learning log.

    uv run python scripts/train.py --form split --steps 50000000 --seed 0 --out runs/split-0

One run uses one CPU thread; run several seeds or settings at once as separate processes to use a many-core PC.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from erg import ppo  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--form", choices=["split", "central"], default="split")
    p.add_argument("--links", choices=["none", "neighbours", "all"], default="neighbours")
    p.add_argument("--loss", type=float, default=0.0)
    p.add_argument("--lat-us", type=int, default=0)
    p.add_argument("--jitter-us", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cfg = ppo.Config(form=a.form, links=a.links, loss=a.loss, lat_us=a.lat_us, jitter_us=a.jitter_us,
                     n_envs=a.n_envs, seed=a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    agent, timing = ppo.train(cfg, a.steps, progress=lambda e: print(json.dumps(e), flush=True))
    torch.save(agent.state_dict(), a.out / "agent.pt")
    with open(a.out / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cfg.log[0]))
        w.writeheader()
        w.writerows(cfg.log)
    info = {"config": {k: v for k, v in vars(cfg).items() if k != "log"}, "timing": timing,
            "wall_seconds": time.perf_counter() - t, "actor_parameters": ppo.n_params(agent.actor)}
    (a.out / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info["timing"]))


if __name__ == "__main__":
    main()
