"""Тензорная нейрохимия: DA / 5-HT / ACh / NE как поля, а не скаляры.

Наследие канона (разд. 4.2): модулятор — не глобальное число, а
пространственно распределённый тензор, привязанный к топологии. «Настроение»
организма неоднородно по его объёму.

Что добавлено в v8:
  * ДИФФУЗИЯ между областями (соседство = маршруты) — модулятор растекается,
    а не появляется мгновенно везде;
  * ФРОНТ, а не уровень: NE сбрасывает усталость только в момент РОСТА
    (дефект S3 канона v4.0.1: удерживаемый уровень стирал адаптацию каждый
    такт и провоцировал гипервозбуждение);
  * SETPOINT-интерфейс для обучаемого мета-контроллера: химия принимает
    целевые уровни, а не только импульсы;
  * Давление сна как интеграл нагрузки (ур. 20), а не по часам.

Все обновления — in-place по [Z]-тензорам: горячий такт не порождает
мусора для аллокатора и не требует обмена с хостом.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..config import ModCfg

AGENTS = ("da", "ht", "ach", "ne")


class NeuroChemistry(nn.Module):
    def __init__(
        self,
        zone_names: Sequence[str],
        cfg: ModCfg,
        ht_baseline: Optional[Dict[str, float]] = None,
        adjacency: Optional[Sequence[tuple]] = None,
        device="cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.names = list(zone_names)
        self.index = {n: i for i, n in enumerate(self.names)}
        z = len(self.names)

        da0, ht0, ach0, ne0 = cfg.basal
        self.register_buffer("da", torch.full((z,), da0, device=device))
        self.register_buffer("ne", torch.full((z,), ne0, device=device))
        self.register_buffer("ach", torch.full((z,), ach0, device=device))
        ht = torch.full((z,), ht0, device=device)
        if ht_baseline:
            for n, v in ht_baseline.items():
                if n in self.index:
                    ht[self.index[n]] = v
        self.register_buffer("ht", ht)
        self.register_buffer("prev_ne", self.ne.clone())
        self.register_buffer("ht_base", ht.clone())

        self.register_buffer("sp_user", torch.zeros((z,), device=device))
        self.register_buffer("set_da", torch.zeros((z,), device=device))
        self.register_buffer("set_ach", self.ach.clone())
        self.register_buffer("set_ht", ht.clone())
        self.sleep_pressure = 0.0
        self._buffer = {a: torch.zeros((z,), device=device) for a in AGENTS}

        # Диффузия: размытие по рёбрам (маршруты = анатомическая близость)
        adj = torch.eye(z, device=device)
        for a, b in (adjacency or []):
            if a in self.index and b in self.index:
                adj[self.index[a], self.index[b]] = 1.0
                adj[self.index[b], self.index[a]] = 1.0
        adj = adj / adj.sum(1, keepdim=True)
        self.register_buffer("adj", adj)

    # ------------------------------------------------------------------
    def reset_pressure(self) -> None:
        self.sleep_pressure = 0.0,
        self.sp_user.zero_()

    def pulse(self, agent: str, zone: str, magnitude, cap: float = 1.0) -> None:
        """Фазический выброс в одну область (тензорная амплитуда)."""
        i = self.index[zone]
        buf = self._buffer[agent]
        if isinstance(magnitude, torch.Tensor):
            buf[i] = buf[i] + magnitude.reshape(-1)[0] if magnitude.numel() else buf[i]
        else:
            buf[i] = buf[i] + float(magnitude)
        buf.clamp_(max=cap)

    def broadcast(self, agent: str, magnitude, cap: float = 1.0) -> None:
        self._buffer[agent].add_(magnitude).clamp_(max=cap)

    def set_setpoints(self, da: torch.Tensor, ach: torch.Tensor,
                      ht: torch.Tensor) -> None:
        with torch.no_grad():
            self.set_da.copy_(da.clamp(0.0, 1.0))
            self.set_ach.copy_(ach.clamp(0.0, 1.0))
            self.set_ht.copy_(ht.clamp(0.05, 1.0))

    # ------------------------------------------------------------------
    def level(self, agent: str, zone: str) -> torch.Tensor:
        """Уровень модулятора в области как 0-мерный тензор (для нейрона)."""
        return getattr(self, agent)[self.index[zone]]

    def zone_vec(self, agent: str, zones: Sequence[str]) -> torch.Tensor:
        idx = [self.index[z] for z in zones]
        return getattr(self, agent)[idx]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, spike_sums: Optional[Dict[str, torch.Tensor]] = None,
             rpe: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        # 1. Импульсы, накопленные за такт
        for a in AGENTS:
            getattr(self, a).add_(self._buffer[a])
            self._buffer[a].zero_()

        # 2. Распад и релаксация к сетпоинту (метаботропная динамика)
        self.da.mul_(cfg.da_decay)
        self.ne.mul_(cfg.ne_decay)
        self.ach.mul_(cfg.ach_decay).add_(self.set_ach * (1.0 - cfg.ach_decay))
        self.ht.mul_(cfg.ht_decay).add_(self.set_ht * (1.0 - cfg.ht_decay))

        # 3. Диффузия между областями
        if cfg.diffusion > 0.0:
            for a in AGENTS:
                t = getattr(self, a)
                mixed = self.adj @ t
                t.mul_(1.0 - cfg.diffusion).add_(cfg.diffusion * mixed)

        # 4. RPE -> дофамин (глобальный импульс при отсутствии адресации)
        if rpe is not None:
            self.da.add_(torch.clamp(rpe, min=0.0)).clamp_(max=1.0)
        for a in AGENTS:
            getattr(self, a).clamp_(0.0, 1.0)

        # 5. Фронт NE: сброс усталости — событие, а не состояние
        rise = torch.clamp(self.ne - self.prev_ne, min=0.0)
        self.prev_ne.copy_(self.ne)

        # 6. Давление сна (ур. 20): интеграл нагрузки минус истощение 5-HT
        if spike_sums is not None and spike_sums:
            load = sum(float(v.sum()) for v in spike_sums.values())
            self.sp_user.add_(1.0).clamp_(max=1e9)
            self.sleep_pressure += (cfg.pressure_gain if hasattr(cfg, "pressure_gain")
                                    else 2e-4) * load
            self.sleep_pressure -= 0.00005 * float(self.ht.mean())
            self.sleep_pressure = max(0.0, self.sleep_pressure)
        return {"ne_rise": rise, "da": self.da.clone()}

    # ------------------------------------------------------------------
    def telemetry(self) -> Dict[str, float]:
        out = {}
        for a in AGENTS:
            t = getattr(self, a)
            out[f"{a}_mean"] = float(t.mean())
            out[f"{a}_sd"] = float(t.std()) if t.numel() > 1 else 0.0
        out["sleep_pressure"] = float(self.sleep_pressure)
        return out

    def state_vec(self) -> torch.Tensor:
        """[4*Z] — для мета-контроллера и логирования."""
        return torch.cat([getattr(self, a) for a in AGENTS])
