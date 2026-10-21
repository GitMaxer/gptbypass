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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="poc", choices=["small", "poc", "wide"])
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    bench(args.preset, args.ticks, args.batch)
