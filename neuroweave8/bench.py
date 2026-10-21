"""Бенчмарк такта: ticks/sec, размеры, экономия событийного пути.

    python -m neuroweave8.bench [--preset poc] [--ticks 200]
"""
from __future__ import annotations

import argparse
import time
from typing import Dict

import torch

from .config import default_cfg
from .organism.organism import Organism


def bench(preset: str = "poc", ticks: int = 200, batch: int = 4) -> Dict[str, float]:
    cfg = default_cfg(preset)
    org = Organism(cfg, batch=batch)
    org.reset_state(batch)
    x = {n: torch.zeros(batch, cfg.col(n).n_in) for n in org.names}
    x[org.names[0]][:, : min(4, cfg.col(org.names[0]).n_in)] = 1.0  # разреженный вход

    # warmup
    for _ in range(10):
        org.tick(x, learn=True)
    t0 = time.perf_counter()
    for _ in range(ticks):
        org.tick(x, learn=True)
    t_tick = (time.perf_counter() - t0) / ticks

    t0 = time.perf_counter()
    y = torch.randint(0, cfg.col(org.names[0]).n_out, (batch,))
    for _ in range(ticks):
        org.behave(x, y, learn=True)
    t_behave = (time.perf_counter() - t0) / ticks

    n_syn = sum(s.edge_count() for c in org.columns.values() for s in c.syn.values())
    res = {
        "preset": preset,
        "neurons": float(cfg.total_neurons()),
        "synapses": float(n_syn),
        "ticks_per_sec": 1.0 / t_tick,
        "behave_per_sec": 1.0 / t_behave,
        "ms_per_tick": t_tick * 1e3,
    }
    for k, v in res.items():
        print(f"  {k:>16}: {v}")
    return res


def bench_scaling(presets=("small", "poc", "wide"),
                  batches=(1, 4, 16), ticks: int = 60) -> list:
    """Свип масштабирования: ticks/sec по размерам организма и батчу.

    Линейность по нейронам и подлинейность по батчу — экономика
    событийного контура: стоимость растёт по активным ребрам, а не по
    плотности матриц.
    """
    rows = []
    for p in presets:
        for b in batches:
            try:
                r = bench(preset=p, ticks=ticks, batch=b)
            except Exception as e:  # noqa: BLE001
                r = {"preset": p, "neurons": float("nan"),
                     "synapses": float("nan"), "ticks_per_sec": 0.0,
                     "behave_per_sec": 0.0, "ms_per_tick": float("inf"),
                     "error": str(e)}
            r["batch"] = float(b)
            rows.append(r)
    print("\n=== scaling sweep ===")
    print(f"  {'preset':>6} {'batch':>5} {'neurons':>8} {'synapses':>9} "
          f"{'ticks/s':>9} {'ms/tick':>8}")
    for r in rows:
        print(f"  {r['preset']:>6} {int(r['batch']):>5} "
              f"{r['neurons']:>8.0f} {r['synapses']:>9.0f} "
              f"{r['ticks_per_sec']:>9.1f} {r['ms_per_tick']:>8.2f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="poc", choices=["small", "poc", "wide"])
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sweep", action="store_true",
                    help="свип пресетов x батчей (масштабирование)")
    args = ap.parse_args()
    if args.sweep:
        bench_scaling(ticks=args.ticks)
    else:
        bench(args.preset, args.ticks, args.batch)
