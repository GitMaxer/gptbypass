"""Обучаемый мета-контроллер нейромодуляции.

Новое в v8. В каноне уровни DA/5-HT/ACh/NE либо реактивны (событие ->
импульс), либо заданы расписанием. Здесь режим химии выбирает ОБУЧАЕМЫЙ
контроллер — маленькая линейная политика над телеметрией:

    y_t  = [surprise_ema, reward_ema, rate_mean, wfast_mass, drift_err]
    a_t  = W_c y_t + b_c  ->  сетпоинты {ach, ht, da_gain} + gate sleep

Отображение Yu & Dayan (2005) (science_grounding §9):
    ACh <- ОЖИДАЕМАЯ неопределённость: EMA ошибки предсказания — высокий
           ACh = «сенсорика важнее прогноза» (усиление базального пути).
    NE  <- НЕОЖИДАЕМАЯ неопределённость: транзиент Surprise выше фона —
           фазический аларм, сброс усталости, рост исследования.

Обучение — дельта-правило на RPE (тот же дофаминовый сигнал, что у
синапсов): контроллер сам нейромодулируем — мета-петля организма.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MetaController(nn.Module):
    N_IN = 5  # surprise_ema, reward_ema, rate_mean, wfast_mass, pred_err_ema

    def __init__(self, n_zones: int, cfg_lr: float = 0.02, cfg_sigma: float = 0.35):
        super().__init__()
        self.n_zones = int(n_zones)
        self.lr = float(cfg_lr)
        self.sigma = float(cfg_sigma)
        # Выходы на зону: [ach_target, ht_target, da_gain] — диагональная
        # пер-зонная политика (W [Z*3, N_IN], b [Z*3])
        self.register_buffer("w", torch.zeros(self.n_zones * 3, self.N_IN))
        self.register_buffer("b", torch.zeros(self.n_zones * 3))
        self.register_buffer("surprise_ema", torch.zeros(()))
        self.register_buffer("reward_ema", torch.zeros(()))
        self.register_buffer("pred_err_ema", torch.zeros(()))
        self.register_buffer("ne_gain", torch.tensor(1.0))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def observe(self, surprise: float, reward: Optional[float],
                rate: float, wfast_mass: float, pred_err: float) -> None:
        """Обновить EMA-агрегаты такта (медленный сигнал контроллера)."""
        self.surprise_ema.mul_(0.95).add_(0.05 * float(surprise))
        if reward is not None:
            self.reward_ema.mul_(0.95).add_(0.05 * float(reward))
        self.pred_err_ema.mul_(0.95).add_(0.05 * float(pred_err))
        self._last_rate = float(rate)
        self._last_wfast = float(wfast_mass)

    @torch.no_grad()
    def setpoints(self) -> Dict[str, torch.Tensor]:
        """Целевые уровни [Z] для ach/ht и скаляр da_gain."""
        y = torch.tensor([
            float(self.surprise_ema), float(self.reward_ema),
            getattr(self, "_last_rate", 0.0), getattr(self, "_last_wfast", 0.0),
            float(self.pred_err_ema)])
        a = (self.w @ y + self.b).reshape(self.n_zones, 3)
        ach = torch.sigmoid(a[:, 0])                    # [Z] в (0,1)
        ht = torch.sigmoid(a[:, 1]).clamp(0.05, 0.95)   # [Z]
        da_gain = torch.sigmoid(a[:, 2].mean()) * 2.0   # 0..2
        # Yu&Dayan: ожидаемая неопределённость (pred_err) сдвигает ACh вверх
        ach = torch.clamp(ach + 0.5 * self.pred_err_ema, 0.0, 1.0)
        return {"ach": ach, "ht": ht, "da_gain": da_gain}

    @torch.no_grad()
    def ne_alarm(self, surprise: float, threshold_ratio: float = 3.0) -> float:
        """Неожиданная неопределённость: транзиент Surprise vs фон.
        Возвращает амплитуду NE-события (0 = нет аларма)."""
        bg = float(self.surprise_ema)
        if bg <= 1e-9:
            return 0.0
        excess = surprise - threshold_ratio * bg
        return float(torch.clamp(torch.tensor(excess / (bg + 1e-9)),
                                 0.0, float(self.ne_gain)))

    @torch.no_grad()
    def learn(self, rpe: float) -> None:
        """Мета-обучение: RPE-модулированный Хебб над последним входом."""
        y = torch.tensor([
            float(self.surprise_ema), float(self.reward_ema),
            getattr(self, "_last_rate", 0.0), getattr(self, "_last_wfast", 0.0),
            float(self.pred_err_ema)])
        # исследование: шум пропорционален (1 - reward_ema)
        eps = torch.randn_like(self.w) * self.sigma * (1.0 - float(self.reward_ema).real
                                                       if hasattr(self.reward_ema, 'real') else 0.0)
        self.w.add_(self.lr * float(rpe) * (eps + torch.outer(
            torch.tanh(self.w @ y), y)))
