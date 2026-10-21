"""NEUROWEAVE 8 — демонстрация жизненного цикла организма.

    python -m neuroweave8.main [--preset small|poc|wide] [--steps N]

Фазы: изоляция (local heads) -> сборка (межобластные пути + freeze) ->
бодрствование с марковским потоком и инверсией -> сон (реплей
гиппокампа, консолидация, прунинг) -> пробуждение -> контроль точности.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch

from .config import default_cfg
from .organism.organism import Organism
from .tasks.environments import RuleShiftTask


def build(cfg, batch: int = 4) -> Organism:
    org = Organism(cfg, batch=batch)
    org.reset_state(batch)
    return org


def run_demo(preset: str = "poc", steps: int = 120, batch: int = 4,
             verbose: bool = True) -> Dict[str, float]:
    t0 = time.time()
    cfg = default_cfg(preset)
    org = build(cfg, batch)
    primary = org.names[0]
    n_out = cfg.col(primary).n_out  # размерность fused-выхода = число классов
    env = RuleShiftTask(vocab=n_out, switch_at=steps // 2)
    xs, ys = env.run_episode(steps, batch)

    def feed(t):
        x = xs[t]
        inp = {n: torch.zeros(batch, cfg.col(n).n_in) for n in org.names}
        inp[primary][:, : n_out] = x
        return inp

    # ---- Фаза 1: изолированная локальная настройка ----
    pre_loss = 0.0
    for t in range(30):
        outs = org.tick(feed(t), learn=True)
        pre_loss += float(outs[primary].pred_err.pow(2).mean())
    pre_loss /= 30

    # ---- Фаза 2: сборка + бодрствование ----
    org.assemble()
    rewards: List[float] = []
    ces: List[float] = []
    accs: List[float] = []
    for t in range(30, steps):
        r = org.behave(feed(t), ys[t], learn=True)
        rewards.append(r["reward"])
        ces.append(r["ce"])
        accs.append(float(r["correct"].float().mean()))
    sw = env.switch_at - 30
    acc_pre_inv = sum(accs[max(0, sw - 20):sw]) / max(1, len(accs[max(0, sw - 20):sw]))
    acc_post_inv = sum(accs[sw + 5:]) / max(1, len(accs[sw + 5:]))
    acc_pre_sleep = acc_post_inv

    # ---- Сон ----
    rep = org.sleep_cycle()

    # ---- Пробуждение: контроль точности (свежий поток той же среды) ----
    xs2, ys2 = env.run_episode(60, batch)

    def feed2(t):
        x = xs2[t]
        inp = {n: torch.zeros(batch, cfg.col(n).n_in) for n in org.names}
        inp[primary][:, : n_out] = x
        return inp

    post_accs: List[float] = []
    for t in range(60):
        r = org.behave(feed2(t), ys2[t], learn=False)
        post_accs.append(float(r["correct"].float().mean()))
    acc_post_sleep = sum(post_accs[-20:]) / max(1, len(post_accs[-20:]))

    tel = org.telemetry
    res = {
        "preset": preset,
        "pre_loss": pre_loss,
        "acc_pre_inversion": acc_pre_inv,
        "acc_post_inversion": acc_post_inv,
        "acc_post_sleep": acc_post_sleep,
        "consolidated": rep.get("consolidated", 0.0),
        "pruned": rep.get("pruned", 0.0),
        "episodes_replayed": rep.get("episodes", 0.0),
        "hippo_stored": tel["hippo_stored"],
        "reward_baseline": tel["reward_baseline"],
        "seconds": time.time() - t0,
    }
    if verbose:
        print("=== NEUROWEAVE 8 lifecycle ===")
        for k, v in res.items():
            print(f"  {k:>20}: {v}")
        print("  surprises:", {n: round(tel[f'surprise.{n}'], 4) for n in org.names})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="poc", choices=["small", "poc", "wide"])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    run_demo(args.preset, args.steps, args.batch)
