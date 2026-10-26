"""Ламинарная колонка — канонический микроконтур NEUROWEAVE 8.

ЭТО ГЛАВНАЯ АРХИТЕКТУРНАЯ ЗАМЕНА v8. В v6 «зона» была плоским рекуррентным
пулом, в v7 — пулом с общим апикальным входом. Обе модели не различают
НИСХОДЯЩИЙ сенсорный поток и ВОСХОДЯЩИЙ контекстный прогноз на уровне
топологии: у них один и тот же вход мог прийти куда угодно. Здесь вводится
ламинарная структура (Douglas & Martin, Bastos et al.):

    L4    — вход из таламуса/предыдущей области (feedforward, БАЗАЛЬНЫЙ)
    L2/3  — локальная рекуррентная репрезентация; исток feedforward-выхода
    L5    — выход области; КОМПАРТМЕНТНО ГЕЙТИРОВАН: чтобы выстрелить,
            нужен базальный драйв от L2/3 И апикальный контекст на L1
    L6    — контролирующая петля: возвращает прогноз (feedback) и в свою
            очередь адресует апикальные дендриты L5 других областей
    PV    — перисоматическое торможение: делительный шунт (gain control)
    SST   — дендритно-адресное торможение: гасит апикальный путь (селекция
            того, какому контексту вообще разрешено влиять)

Следствия, которых не было раньше:

 1. Выход области не является функцией только её входа: он КОНТЕКСТНО-
    ГЕЙТИРОВАН. Это строгое обобщение NMDA-коинциденции с уровня нейрона
    на уровень области, и это проверяется тестом (без апикального входа
    L5 молчит даже при сильном L4/L2-3 драйве).
 2. PV и SST выполняют РАЗНЫЕ функции (шунт против дендритного гейта), а не
    одну «тормозную маску»: усиление PV снижает частоту, усиление SST
    избирательно отключает контекст, не трогая сенсорный путь.
 3. Появляется ЛОКАЛЬНОЕ ПРЕДСКАЗАНИЕ (PredictiveHead): область учится
    предсказывать свой следующий вход по своей же активности. Ошибка
    предсказания — плотный локальный учитель, не требующий глобального
    backprop, и одновременно источник сигнала новизны (Surprise).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ColumnCfg, NeuronCfg, PlasticityCfg, PhaseCfg
from ..core.neuron import LaminarALIF
from ..core.synapse import StatefulSynapse
from ..core.phase import ThetaGammaBinder
from ..core.sparse import sparse_drive, active_indices

# Дивергенция полушарий (табл. 2 канона): одни размеры — разная физика.
HEMISPHERE_PRESETS = {
    "left": dict(tau_m=8.0, sigma=0.001, surrogate="fast_sigmoid", k=10.0,
                 ht_baseline=0.75, lr_scale=1.0),
    "right": dict(tau_m=35.0, sigma=0.05, surrogate="arctan", k=2.0,
                  ht_baseline=0.30, lr_scale=0.35),
}


class PredictiveHead(nn.Module):
    """Локальное предсказание следующего входа по активности L2/3.

    Обучение — дельта-правило Уидроу-Хоффа на ОШИБКЕ, без autograd:
        e = target - W h ;   dW = eta * e h^T / B
    Это и есть «локальный predictive coding»: сигнал ошибки живёт в самой
    области, не проходя через глобальный лосс. Память O(n_in * n_l23).
    """

    def __init__(self, n_from: int, n_to: int, lr: float = 0.02):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_to, n_from))
        self.b = nn.Parameter(torch.zeros(n_to))
        self.lr = lr
        self.register_buffer("err_ema", torch.zeros(()))
        self.register_buffer("prec_ema", torch.ones(()))

    @property
    def precision(self) -> torch.Tensor:
        """Π = 1/σ² ошибки — вес предиктивного кодинга (Feldman&Friston)."""
        return 1.0 / (self.err_ema + 1e-3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(h, self.w, self.b)

    @torch.no_grad()
    def local_update(self, h: torch.Tensor, target: torch.Tensor,
                     learn: bool = True) -> torch.Tensor:
        pred = self.forward(h)
        err = target - pred
        if learn:
            # Точность-взвешенное дельта-правило: надёжные области учатся
            # быстрее, шумные — медленнее (precision-weighted prediction error)
            pi = self.precision.clamp(max=10.0)
            b = float(h.shape[0])
            self.w.add_(self.lr * pi * (err.t() @ h) / b)
            self.b.add_(self.lr * pi * err.mean(0))
        self.err_ema.mul_(0.98).add_(0.02 * err.pow(2).mean())
        return err

    @property
    def surprise(self) -> torch.Tensor:
        """Точность-взвешенная ошибка — сигнал новизны для NE-аларма."""
        return self.err_ema * self.precision.clamp(max=10.0)


@dataclass
class ColumnOut:
    """Выходы такта колонки (ссылки на живые тензоры, копируйте при нужде)."""

    spikes: Dict[str, torch.Tensor]
    v: Dict[str, torch.Tensor]
    theta: Dict[str, torch.Tensor]
    rates: Dict[str, torch.Tensor]
    pred_err: torch.Tensor
    surprise: torch.Tensor
    apical_gate: torch.Tensor
    gaba: Dict[str, torch.Tensor]


class LaminarColumn(nn.Module):
    """Одна область коры = один канонический ламинарный микроконтур."""

    def __init__(self, cfg: ColumnCfg, nc: NeuronCfg, pc: PlasticityCfg,
                 seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.name
        self.side = cfg.side
        preset = HEMISPHERE_PRESETS[cfg.side]
        self.ht_baseline = preset["ht_baseline"]
        self.lr_scale = preset["lr_scale"]

        gen = torch.Generator().manual_seed(
            seed * 100003 + sum((i + 1) * ord(ch) for i, ch in enumerate(cfg.name)))

        base = replace(
            nc,
            tau_m=preset["tau_m"],
            sigma=preset["sigma"],
            surrogate=preset["surrogate"],
            k_surrogate=preset["k"],
        )
        n = cfg.pools()
        self.pools = nn.ModuleDict({
            "l4": LaminarALIF(n["l4"], replace(base, tau_m=max(3.0, base.tau_m * 0.6)), "l4"),
            "l23": LaminarALIF(n["l23"], base, "l23"),
            "l5": LaminarALIF(n["l5"], replace(base, tau_m=base.tau_m * 1.6), "l5"),
            "l6": LaminarALIF(n["l6"], replace(base, tau_m=base.tau_m * 2.0), "l6"),
            "pv": LaminarALIF(n["pv"], replace(base, tau_m=4.0, sigma=base.sigma * 0.5,
                                               beta_adapt=0.35), "pv"),
            "sst": LaminarALIF(n["sst"], replace(base, tau_m=10.0, sigma=base.sigma * 0.5,
                                                 beta_adapt=0.3), "sst"),
        })

        d = cfg.density
        S = StatefulSynapse
        self.syn = nn.ModuleDict({
            "in_4": S(cfg.n_in, n["l4"], pc, density=min(1.0, d + 0.3), name=f"{cfg.name}.in_4", generator=gen),
            "4_23": S(n["l4"], n["l23"], pc, density=d, name=f"{cfg.name}.4_23", generator=gen),
            "rec": S(n["l23"], n["l23"], pc, density=0.20, name=f"{cfg.name}.rec", generator=gen),
            "23_5": S(n["l23"], n["l5"], pc, density=d, name=f"{cfg.name}.23_5", generator=gen),
            "5_6": S(n["l5"], n["l6"], pc, density=d, name=f"{cfg.name}.5_6", generator=gen),
            "6_5": S(n["l6"], n["l5"], pc, density=d, name=f"{cfg.name}.6_5", generator=gen),
            "23_pv": S(n["l23"], n["pv"], pc, density=0.5, name=f"{cfg.name}.23_pv", generator=gen),
            "5_pv": S(n["l5"], n["pv"], pc, density=0.4, name=f"{cfg.name}.5_pv", generator=gen),
            "23_sst": S(n["l23"], n["sst"], pc, density=0.5, name=f"{cfg.name}.23_sst", generator=gen),
        })

        # Тормозные маски: W[out, in] (ориентация F.linear), фиксированные,
        # с сохранением среднего драйва при росте пула-источника.
        self.register_buffer("w_pv_4", self._mask(n["l4"], n["pv"], 0.5,
                                                  cfg.inh_strength))
        self.register_buffer("w_pv_23", self._mask(n["l23"], n["pv"], 0.5,
                                                   cfg.inh_strength))
        self.register_buffer("w_pv_5", self._mask(n["l5"], n["pv"], 0.5,
                                                  cfg.inh_strength))
        self.register_buffer("w_sst_5", self._mask(n["l5"], n["sst"], 0.5, 1.0))
        self.sst_gain = 0.70
        # Калибровка эффективности путей: при k-WTA-разреженности (15% пула)
        # активных пре-нейронов мало, и «сырая» fan-in нормировка даёт
        # субпороговый драйв. Коэффициенты подобраны так, чтобы каждый слой
        # работал в физиологическом диапазоне 3-15 Гц, а L5 оставался
        # субпороговым БЕЗ контекста (проверяется тестом).
        syn_gain = {"4_23": 2.6, "rec": 1.2, "23_5": 0.65, "5_6": 3.2,
                    "6_5": 3.0, "23_pv": 1.0, "5_pv": 1.0, "23_sst": 1.0}
        with torch.no_grad():
            for k, g in syn_gain.items():
                self.syn[k].w_slow.mul_(g)
                self.syn[k].w_snapshot.mul_(g)
        # Параметры дендритного плато L5: апикальная деполяризация снижает
        # порог сомы до k_plateau мВ — физический AND-гейт «вход И контекст».
        self.k_plateau = 8.0
        self.theta_apical = 0.10
        self.plateau_scale = 0.25

        # Голова чтения локальной задачи: L2/3 (репрезентация) + L5 (выход)
        self.readout: Optional[nn.Linear] = None

        # Голова чтения локальной задачи: L2/3 (репрезентация) + L5 (выход)
        # θ-γ рабочая память области (7±2 слота поверх L2/3)
        self.binder: Optional[ThetaGammaBinder] = (
            ThetaGammaBinder(n["l23"], PhaseCfg(n_slots=cfg.wm_slots))
            if cfg.wm_slots > 0 else None)
        in_read = (n["l23"] + n["l5"]
                   + (n["l23"] if self.binder is not None else 0)
                   + cfg.n_in)  # + прямой таламо-кортикальный shortcut входа
        if cfg.n_out is not None:
            self.readout = nn.Linear(in_read, cfg.n_out)
            # w_fast_readout: быстрый буфер выходной проекции (канон).
            # При инверсии правила переобучается features->tokens без
            # трогания замороженного readout.weight (rule inversion).
            self.register_buffer(
                "w_fast_readout", torch.zeros(cfg.n_out, in_read))
        self._wfr_lr = pc.lr_fast
        self._wfr_max = pc.w_fast_max
        self._wfr_decay = pc.w_fast_decay

        # Локальное предсказание входа
        self.head = PredictiveHead(n["l23"], cfg.n_in, lr=pc.pred_lr)

        self.register_buffer("isolation", torch.zeros(()))
        self.decay = cfg.readout_decay
        self._init_state = False
        self.trace: Dict[str, Optional[torch.Tensor]] = {"l23": None, "l5": None}
        self.prev: Dict[str, Optional[torch.Tensor]] = {k: None for k in n}
        self.prev_in: Optional[torch.Tensor] = None
        self._k = {p: max(3, int(0.15 * n[p])) for p in ("l4", "l23", "l5", "l6")}

    # ------------------------------------------------------------------
    @staticmethod
    def _mask(rows: int, cols: int, density: float, strength: float) -> torch.Tensor:
        gen = torch.Generator().manual_seed(rows * 7919 + cols * 104729 + 17)
        m = (torch.rand(rows, cols, generator=gen) < density).float()
        # нормировка: ожидаемая сумма строки = strength (масштаб-инвариантность)
        return m * (strength / max(density * cols, 1e-6))

    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        if self.binder is not None:
            self.binder.reset_state(batch, device, dtype)
        n = self.cfg.pools()
        for p in self.pools.values():
            p.reset_state(batch, device, dtype)
        for s in self.syn.values():
            s.reset_state(batch, device, dtype)
        self.trace["l23"] = torch.zeros(batch, n["l23"], device=device, dtype=dtype)
        self.trace["l5"] = torch.zeros(batch, n["l5"], device=device, dtype=dtype)
        for k in self.prev:
            self.prev[k] = torch.zeros(batch, n[k], device=device, dtype=dtype)
        self.prev_in = torch.zeros(batch, self.cfg.n_in, device=device, dtype=dtype)
        self._init_state = True

    def _ensure(self, ref: torch.Tensor) -> None:
        if not self._init_state or self.prev_in is None \
                or self.prev_in.shape[0] != ref.shape[0] \
                or self.prev_in.device != ref.device:
            self.reset_state(ref.shape[0], ref.device, ref.dtype)

    def isolate(self) -> None:
        self.isolation.fill_(0.0)

    def connect_apical(self) -> None:
        self.isolation.fill_(1.0)

    # ------------------------------------------------------------------
    def step(
        self,
        s_in: torch.Tensor,
        extra_basal: Optional[torch.Tensor] = None,
        extra_apical: Optional[torch.Tensor] = None,
        arousal: Optional[torch.Tensor] = None,
        ach: Optional[torch.Tensor] = None,
        ht: Optional[torch.Tensor] = None,
        ne_rise: Optional[torch.Tensor] = None,
        sltt: bool = False,
        learn: bool = False,
        generator: Optional[torch.Generator] = None,
        astro_gain: Optional[torch.Tensor] = None,
    ) -> ColumnOut:
        """Один такт колонки.

        s_in         [B, n_in]  сенсорный/feedforward вход (базальный, в L4)
        extra_basal  [B, n_l4]  дополнительный базальный драйв (межзонные ff)
        extra_apical [B, n_l5]  апикальный контекст (feedback, L1)
        arousal      [B, n_l5]  неспецифическая активация L1 (NE/ACh): НЕ
                                маскируется изоляцией — она внутренняя, а не
                                межобластная; именно она разрешает выход.
        Порядок: L4 -> L2/3 -> L5 (гейт) -> L6 -> интернейроны -> трассы.
        """
        self._ensure(s_in)
        n = self.cfg.pools()
        pv_prev = self.prev["pv"]
        sst_prev = self.prev["sst"]

        # ---- L4: вход + перисоматический шунт PV -----------------------
        # Событийный путь: при разреженном входе платим за |active|, не за n_in
        if float((s_in.abs() > 0).float().mean()) < 0.4:
            drive4 = self.cfg.input_gain * sparse_drive(
                s_in, self.syn["in_4"].w_eff)
        else:
            drive4 = self.cfg.input_gain * self.syn["in_4"](s_in)
        if extra_basal is not None:
            drive4 = drive4 + extra_basal
        gaba4 = F.linear(pv_prev, self.w_pv_4) if pv_prev is not None else None
        o4 = self.pools["l4"].step(drive4, None, gaba4, ach=ach, ht=ht,
                                   ne_rise=ne_rise, sltt=sltt, k_spike=self._k["l4"],
                                   generator=generator)

        # ---- L2/3: локальная рекуррентность ----------------------------
        drive23 = self.syn["4_23"](self.prev["l4"]) + self.syn["rec"](self.prev["l23"])
        if astro_gain is not None:
            # Астроцитарная STP-фасилитация (глиотрансмиттер): удержание
            # активного семантического ансамбля на секундном горизонте.
            drive23 = drive23 * astro_gain
        gaba23 = F.linear(pv_prev, self.w_pv_23) if pv_prev is not None else None
        o23 = self.pools["l23"].step(drive23, None, gaba23, ach=ach, ht=ht,
                                     ne_rise=ne_rise, sltt=sltt, k_spike=self._k["l23"],
                                     generator=generator)

        # ---- L5: КОМПАРТМЕНТНЫЙ ГЕЙТ (базальный + апикальный) -----------
        apical = self.syn["6_5"](self.prev["l6"])
        if extra_apical is not None:
            apical = apical + extra_apical * self.isolation
        if arousal is not None:
            apical = apical + arousal
        if sst_prev is not None:
            # SST гасит апикальный путь: избирательное отключение контекста
            shunt = torch.clamp(1.0 - self.sst_gain * F.linear(sst_prev, self.w_sst_5),
                                min=0.0, max=1.0)
            apical = apical * shunt
            gate = shunt.mean(dim=-1, keepdim=True)
        else:
            gate = torch.ones(s_in.shape[0], 1, device=s_in.device)
        drive5 = self.syn["23_5"](self.prev["l23"])
        gaba5 = F.linear(pv_prev, self.w_pv_5) if pv_prev is not None else None
        # Дендритное плато: апикальная деполяризация выше порога снижает
        # порог сомы (до k_plateau мВ) — без контекста L5 остаётся молчащей.
        plateau = self.k_plateau * torch.tanh(
            torch.clamp(apical - self.theta_apical, min=0.0) / self.plateau_scale)
        o5 = self.pools["l5"].step(drive5, apical, gaba5, ach=ach, ht=ht,
                                   ne_rise=ne_rise, theta_shift=plateau,
                                   sltt=sltt, k_spike=self._k["l5"],
                                   generator=generator)

        # ---- L6: контролирующая петля ----------------------------------
        drive6 = self.syn["5_6"](self.prev["l5"])
        o6 = self.pools["l6"].step(drive6, None, None, ach=ach, ht=ht,
                                   ne_rise=ne_rise, sltt=sltt, k_spike=self._k["l6"],
                                   generator=generator)

        # ---- Интернейроны ----------------------------------------------
        drive_pv = self.syn["23_pv"](self.prev["l23"]) + self.syn["5_pv"](self.prev["l5"])
        opv = self.pools["pv"].step(drive_pv, None, None, ach=ach, ht=ht,
                                    sltt=sltt, k_spike=max(2, n["pv"] // 3),
                                    generator=generator)
        drive_sst = self.syn["23_sst"](self.prev["l23"])
        osst = self.pools["sst"].step(drive_sst, None, None, ach=ach, ht=ht,
                                      sltt=sltt, generator=generator)

        # ---- Локальное предсказание входа ------------------------------
        s23 = o23["spikes"]
        if sltt:
            h = s23  # градиент течёт через текущую активность
            self.trace["l23"] = self.trace["l23"].detach() * self.decay + s23
        else:
            h = s23.detach()
            self.trace["l23"] = self.trace["l23"] * self.decay + h
        err = self.head.local_update(h, s_in, learn=learn)
        if (learn and self.readout is not None and err is not None
                and err.shape[-1] == self.w_fast_readout.shape[0]):
            with torch.no_grad():
                feat = self.features().detach()
                self.w_fast_readout.mul_(self._wfr_decay).add_(
                    self._wfr_lr * torch.einsum(
                        "bo,bi->oi", err.detach(), feat)
                    / max(1, feat.shape[0])).clamp_(
                        -self._wfr_max, self._wfr_max)
        # head учится всегда, когда разрешено (это не autograd-параметр)

        spikes = {"l4": o4["spikes"], "l23": s23, "l5": o5["spikes"],
                  "l6": o6["spikes"], "pv": opv["spikes"], "sst": osst["spikes"]}
        if sltt:
            self.trace["l5"] = self.trace["l5"].detach() * self.decay + spikes["l5"]
        else:
            self.trace["l5"] = self.trace["l5"] * self.decay + spikes["l5"].detach()

        # ---- Обновление отложенных состояний (дендритная задержка 1 такт) --
        # ВСЕГДА detach: это и есть контракт SLTT — история состояний не входит
        # в граф, суррогатный градиент идёт только через ТЕКУЩИЕ синапсы.
        for k, sp in spikes.items():
            self.prev[k] = sp.detach()
        self.prev_in = s_in.detach()
        if self.binder is not None:
            self.binder.write(s23.detach())

        rates = {k: spikes[k].mean() for k in spikes}
        return ColumnOut(
            spikes=spikes,
            v={"l4": o4["v"], "l23": o23["v"], "l5": o5["v"], "l6": o6["v"]},
            theta={"l4": o4["theta"], "l23": o23["theta"], "l5": o5["theta"]},
            rates=rates,
            pred_err=err,
            surprise=self.head.surprise,
            apical_gate=gate,
            gaba={"l4": gaba4, "l23": gaba23, "l5": gaba5} if gaba5 is not None else {},
        )

    # ------------------------------------------------------------------
    def features(self) -> torch.Tensor:
        """Репрезентация области для чтения задачи: [B, n_l23+n_l5(+n_l23 WM)]."""
        f = torch.cat([self.trace["l23"], self.trace["l5"]], dim=-1)
        if self.binder is not None and self.binder.buf is not None:
            # свёртка рабочей памяти: среднее содержимое слотов
            f = torch.cat([f, self.binder.buf.mean(dim=1)], dim=-1)
        if self.prev_in is not None:
            # прямой доступ к сенсорному паттерну (таламо-кортикальный
            # shortcut) — развязывает идентичность входа и внутреннее состояние
            f = torch.cat([f, self.prev_in], dim=-1)
        return f

    def logits(self) -> torch.Tensor:
        if self.readout is None:
            raise RuntimeError(f"{self.name}: голова чтения не сконфигурирована")
        return self.readout(self.features()) + F.linear(
            self.features(), self.w_fast_readout)

    def output_spikes(self) -> torch.Tensor:
        """Выход области = спайки L5 (а не L2/3: выход контекстно-гейтирован)."""
        return self.prev["l5"]

    def prediction_spikes(self) -> torch.Tensor:
        """Прогноз области = спайки L6 (контекст для других областей)."""
        return self.prev["l6"]

    # ------------------------------------------------------------------
    def freeze_basis(self) -> None:
        for s in self.syn.values():
            s.freeze_for_phase2()
        if self.readout is not None:
            for p in self.readout.parameters():
                p.requires_grad_(False)

    def unfreeze_basis(self) -> None:
        for s in self.syn.values():
            s.unfreeze()
        if self.readout is not None:
            for p in self.readout.parameters():
                p.requires_grad_(True)

    def parameter_groups(self, lr: float) -> list:
        g = [{"params": [p for s in self.syn.values() for p in [s.w_slow]],
              "lr": lr}]
        if self.readout is not None:
            g.append({"params": list(self.readout.parameters()), "lr": lr})
        return g

    def report(self) -> Dict[str, float]:
        out = {"n_neurons": float(self.cfg.n_neurons())}
        for k, s in self.syn.items():
            out[f"edges.{k}"] = float(s.edge_count())
            out[f"wfast.{k}"] = float(s.w_fast.abs().sum())
        out["surprise"] = float(self.head.surprise)
        return out
