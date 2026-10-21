"""Системная обёртка организма: NeuroSystem — развёртываемый блок канона.

Концепт v4.0 описывает организм как живую систему с жизненным циклом
(бодрствование -> давление сна -> сон -> пробуждение -> EWC), а не как
модель с fit/predict. ``NeuroSystem`` — инфраструктурная граница между
организмом и внешним миром:

    boot()          — фаза 1: изоляция + локальная настройка + сборка
    step(inputs)    — один такт жизни (сенсорные паттерны -> состояние)
    act(inputs, y)  — такт поведения (выход + награда -> кредит)
    run_episode     — эпизод на среде RuleShiftTask/совместимой
    lifecycle()     — медленный контур: опрос давления сна, сетпоинты
    quantize()      — int8-режим развёртывания (после обучения)
    report()        — сводная телеметрия JSON-совместимым dict'ом
    save()/load()   — чекпоинт организма (state_dict + метаданные)

Граница ответственности: внутри — только тензорные паттерны; снаружи —
любые источники (среду подключает вызывающий код).
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import torch

from ..config import OrganismCfg, default_cfg
from ..core.quantize import quantize_organism
from .organism import Organism


class NeuroSystem:
    """Живой контейнер организма: цикл, память, телеметрия, развёртывание."""

    def __init__(self, preset: str = "poc", batch: int = 4,
                 cfg: Optional[OrganismCfg] = None, seed: Optional[int] = None):
        self.cfg = cfg or default_cfg(preset)
        if seed is not None:
            self.cfg = self.cfg.replace(seed=seed)
        self.preset = preset
        self.batch = int(batch)
        self.org = Organism(self.cfg, batch=self.batch)
        self.org.reset_state(self.batch)
        self.booted = False
        self.quantized: Optional[Dict[str, float]] = None
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def boot(self, warmup_ticks: int = 30, arm_ewc: bool = False) -> "NeuroSystem":
        """Фаза 1 -> сборка: изоляционный прогрев, затем межобластные пути."""
        org = self.org
        org.reset_state(self.batch)
        for _ in range(warmup_ticks):
            inputs = {n: torch.zeros(self.batch, self.cfg.col(n).n_in)
                      for n in org.names}
            inputs[org.names[0]].uniform_(0.0, 1.0)
            org.tick(inputs, learn=True)
        org.assemble()
        if arm_ewc:
            org.arm_ewc(loss_fn=lambda: sum(
                s.w_slow.pow(2).sum() for c in org.columns.values()
                for s in c.syn.values() if hasattr(s, "w_slow")), n_samples=4)
        self.booted = True
        return self

    # ------------------------------------------------------------------
    def step(self, inputs: Dict[str, torch.Tensor],
             learn: bool = True) -> Dict[str, object]:
        """Один такт: сенсорный вход -> внутреннее состояние."""
        return self.org.tick(inputs, learn=learn)

    def act(self, inputs: Dict[str, torch.Tensor], target: torch.Tensor,
            learn: bool = True) -> Dict[str, object]:
        """Такт поведения: fused-выход + награда -> событийный DA-кредит."""
        return self.org.behave(inputs, target, learn=learn)

    def run_episode(self, env, n_steps: int, learn: bool = True,
                    auto_sleep: bool = True) -> Dict[str, float]:
        """Эпизод на среде с run_episode(n, batch) -> (xs[T,B,V], ys[T,B]).

        auto_sleep=True: медленный контур опрашивается каждый такт —
        организм сам решает, когда спать (гистерезис по sleep pressure).
        """
        if not self.booted:
            self.boot()
        org, cfg = self.org, self.cfg
        primary = org.names[0]
        xs, ys = env.run_episode(n_steps, self.batch)
        accs, rewards = [], []
        t0 = time.time()
        for t in range(n_steps):
            inputs = {n: torch.zeros(self.batch, cfg.col(n).n_in)
                      for n in org.names}
            inputs[primary][:, : xs.shape[-1]] = xs[t]
            r = org.behave(inputs, ys[t], learn=learn)
            accs.append(float(r["correct"].float().mean()))
            rewards.append(float(r["reward"]))
            if auto_sleep:
                self.lifecycle()
        res = {"steps": float(n_steps),
               "acc_mean": sum(accs) / max(1, len(accs)),
               "acc_tail": sum(accs[-40:]) / max(1, min(40, len(accs))),
               "reward_mean": sum(rewards) / max(1, len(rewards)),
               "seconds": time.time() - t0,
               "sleeps": float(self.org.slow.sleeps)}
        self.history.append(res)
        return res

    def lifecycle(self) -> Optional[Dict[str, float]]:
        """Медленный контур: сон по гистерезису + применение сетпоинтов."""
        rep = self.org.maybe_sleep()
        if rep is not None:
            self.org.apply_setpoints()
        return rep

    def sleep(self) -> Dict[str, float]:
        return self.org.sleep_cycle()

    # ------------------------------------------------------------------
    def quantize(self, bits: int = 8) -> Dict[str, float]:
        """int8-режим развёртывания: ~4x памяти синапсов, инференс-only."""
        self.quantized = quantize_organism(self.org, bits=bits)
        return self.quantized

    # ------------------------------------------------------------------
    def report(self) -> Dict[str, object]:
        """Сводная телеметрия: химия, память, структура, режим."""
        tel = dict(self.org.telemetry)
        edges = sum(s.edge_count() for c in self.org.columns.values()
                    for s in c.syn.values())
        neurons = sum(c.cfg.n_neurons() for c in self.org.columns.values())
        syn_bytes = sum(
            (s.memory_bytes() if hasattr(s, "memory_bytes")
             else s.w_eff.numel() * 4)
            for c in self.org.columns.values() for s in c.syn.values())
        return {"preset": self.preset, "steps": tel.get("steps", 0.0),
                "neurons": neurons, "synapses": edges,
                "synapse_bytes": syn_bytes,
                "quantized": self.quantized is not None,
                "sleep_pressure": tel.get("sleep_pressure", 0.0),
                "reward_baseline": tel.get("reward_baseline", 0.0),
                "hippo_stored": tel.get("hippo_stored", 0.0),
                "episodes": len(self.history), "telemetry": tel}

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.org.state_dict(),
                    "preset": self.preset, "batch": self.batch,
                    "seed": self.cfg.seed}, path)

    @classmethod
    def load(cls, path: str, preset: Optional[str] = None) -> "NeuroSystem":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sys = cls(preset=preset or ckpt["preset"], batch=ckpt["batch"])
        sys.org.load_state_dict(ckpt["state_dict"])
        sys.org.reset_state(sys.batch)
        sys.booted = True
        return sys

    def report_json(self) -> str:
        return json.dumps(self.report(), ensure_ascii=False, indent=1)
