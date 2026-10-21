"""S7 — гомеостатический стабилизатор (Этап 5: «S7 спасла проект»).

Назначение: удерживать живую сеть в рабочем окне без вмешательства хоста.

  * всплеск активности -> GABA-всплеск (ШУНТИРУЮЩИЙ: E_Cl = V_rest =>
    безопасен по построению — новая архитектура GABA из Этапа 6 делает
    аварийное торможение физически неспособным «провалить» мембрану);
  * молчание зоны -> ACh-подкачка (сомethическая возбудимость);
  * NaN/Inf в V -> мгновенная очистка до V_rest;
  * журнал событий — кольцевой буфер на хосте (вне CUDA-графа).

Все решения ВНУТРИ контура — чистые тензорные операции (сигмоиды),
поэтому S7 работает и внутри захваченного графа без синхронизаций.
"""
from collections import deque

import torch

from .config import S7Cfg


class S7:
    def __init__(self, cfg: S7Cfg, v_rest: float, device):
        self.cfg = cfg
        self.v_rest = v_rest
        self.device = device
        self.events: deque = deque(maxlen=256)  # журнал на хосте (вне графа)

    # ---------- тензорные эффекты (граф-безопасные) ----------
    def gaba_surge(self, rate_ema: torch.Tensor) -> torch.Tensor:
        """Добавка к g_GABA по зонам при превышении r_hi (мягкая сигмоида)."""
        c = self.cfg
        return c.surge_max * torch.sigmoid((rate_ema - c.r_hi) / c.slope)

    def ach_boost(self, rate_ema: torch.Tensor) -> torch.Tensor:
        """Добавка ACh при молчании зоны (r < r_lo)."""
        c = self.cfg
        return c.ach_boost * torch.sigmoid((c.r_lo - rate_ema) / c.slope)

    def scrub(self, V: torch.Tensor) -> torch.Tensor:
        """NaN/Inf -> V_rest (тензорная операция, работает в графе)."""
        bad = ~torch.isfinite(V)
        return torch.where(bad, torch.full_like(V, self.v_rest), V)

    # ---------- хост-мониторинг (ВНЕ графа, по желанию раз в K реплеев) ----------
    def log_stats(self, step: int, rates: torch.Tensor, V: torch.Tensor) -> None:
        r = rates.detach()
        v = V.detach()
        if not torch.isfinite(r).all() or not torch.isfinite(v).all():
            self.events.append((step, "ALARM: не-finite состояние очищено S7"))
        if (r > self.cfg.r_hi).any():
            self.events.append((step, f"SURGE: rate={r.max().item():.3f} -> GABA-шунт"))
        if (r < self.cfg.r_lo).any():
            self.events.append((step, f"BOOST: rate={r.min().item():.3f} -> ACh-подкачка"))

    def drain_events(self):
        ev, self.events = list(self.events), deque(maxlen=256)
        return ev
