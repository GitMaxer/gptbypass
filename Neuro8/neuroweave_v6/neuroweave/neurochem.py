"""Нейрохимия NEUROWEAVE: ACh, 5-HT, DA, NE — тензорные расписания.

ВСЕ модуляторы живут в тензорах [T, Z] на устройстве и индексируются
ТЕNSOR-счётчиком шага t (in-place t.add_(1) внутри графа) — поэтому
быстрый контур с пульсами DA/NE полностью граф-безопасен
(torch.cuda.CUDAGraph), без единого sync host<->device.

Эффекты:
  ACh  — соматическая возбудимость: v_th_eff = v_th - k_ach*ACh (уравнение
         сомы, Этап 3);
  5-HT — модуляция шума: sigma_eff = sigma_ht*(0.35 + 5-HT) (в глубоком сне
         тонус 5-HT -> минимум: мембрана не «дёргается» стохастически);
  DA   — трёхфакторная пластичность: eta_eff = eta*DA (гейт w_slow) +
         фазические пульсы вознаграждения на претрейне (тензорные);
  NE   — заострение WTA сразу после переключения контекста (пульс
         детектируется ВНУТРИ контура: |ctx - ctx_ema| -> NE).
"""
from dataclasses import dataclass

import torch

from .config import NWConfig


@dataclass
class ModulatorSchedules:
    """Тензорные расписания [T, Z]: ACh / 5-HT / DA / NE / GABA-сон."""
    ach: torch.Tensor
    ht: torch.Tensor
    da: torch.Tensor
    ne: torch.Tensor
    g_sleep: torch.Tensor

    def slice_(self, T: int):
        return ModulatorSchedules(self.ach[:T], self.ht[:T], self.da[:T],
                                  self.ne[:T], self.g_sleep[:T])


def zeros_schedules(T: int, Z: int, device) -> ModulatorSchedules:
    return ModulatorSchedules(
        ach=torch.zeros(T, Z, device=device),
        ht=torch.zeros(T, Z, device=device),
        da=torch.zeros(T, Z, device=device),
        ne=torch.zeros(T, Z, device=device),
        g_sleep=torch.zeros(T, Z, device=device))


def tonic_schedules(T: int, Z: int, device, ach=0.5, ht=0.35, da=0.0, ne=0.0,
                    g_sleep=0.0) -> ModulatorSchedules:
    """Тонические (постоянные) уровни по всем зонам."""
    s = zeros_schedules(T, Z, device)
    s.ach.fill_(ach)
    s.ht.fill_(ht)
    s.da.fill_(da)
    s.ne.fill_(ne)
    s.g_sleep.fill_(g_sleep)
    return s


def da_pulse_train(sched: ModulatorSchedules, start: int, period: int,
                   length: int, amp: float, zones=None):
    """Фазические пульсы DA (тензорные, граф-безопасные): прямоугольники
    амплитуды amp на зонах zones (по умолчанию все)."""
    Z = sched.da.shape[1]
    zs = range(Z) if zones is None else zones
    for t0 in range(start, sched.da.shape[0], period):
        sched.da[t0:t0 + length, list(zs)] = amp
    return sched


def ne_pulse(sched: ModulatorSchedules, t0: int, length: int, amp: float,
             zones=None):
    Z = sched.ne.shape[1]
    zs = range(Z) if zones is None else zones
    sched.ne[t0:t0 + length, list(zs)] = amp
    return sched


def build_pretrain_schedule(cfg: NWConfig, T: int, device) -> ModulatorSchedules:
    """Претрейн: тонический ACh/5-HT + фазические пульсы DA (каждые 40 шагов)."""
    s = tonic_schedules(T, len(cfg.zones), device, ach=0.55, ht=0.35)
    da_pulse_train(s, start=16, period=40, length=6, amp=0.9)
    return s


def build_exam_schedule(cfg: NWConfig, T: int, device) -> ModulatorSchedules:
    """Экзамен: DA = 0 (пластичность w_slow полностью выключена),
    лёгкий тонус NE. Пульс NE при переключении рождается ВНУТРИ контура
    (детектор |ctx - ctx_ema|), а не извне."""
    return tonic_schedules(T, len(cfg.zones), device, ach=0.5, ht=0.3,
                           da=0.0, ne=0.05)
