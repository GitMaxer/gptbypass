"""Сборка организма NEUROWEAVE 8: единый такт жизни.

Быстрый контур (устройство): колонки -> шина -> workspace -> пластичность
-> химия -> гиппокамп. Медленный контур (хост): slow_loop — редкие
макро-решения (сон, сетпоинты контроллера, структурный рост).

Жизненный цикл:
    Phase 1 (вне): изолированное обучение колонок (local heads + SLTT);
    assemble(): connect_apical + freeze w_slow — шина включена;
    wake: behave()/tick() — Tag-and-Capture под событийным DA;
    sleep_cycle(): реплей гиппокампа -> консолидация -> прунинг -> wake;
    EWC ребазируется после каждого сна.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import OrganismCfg
from ..microcircuit.column import LaminarColumn
from ..modulation.chemistry import NeuroChemistry
from ..modulation.controller import MetaController
from ..topology.callosum import CorpusCallosum
from ..topology.workspace import GlobalWorkspace
from ..learning.credit import CreditPipeline
from ..learning.sleep import SleepProtocol
from ..learning.ewc import EWCRegularizer
from ..learning.structural import StructuralPlasticity
from ..memory.hippocampus import Hippocampus
from .slow_loop import SlowLoopController


class Organism(nn.Module):
    def __init__(self, cfg: OrganismCfg, batch: int = 1):
        super().__init__()
        self.cfg = cfg
        self.batch = int(batch)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)

        self.columns = nn.ModuleDict({
            c.name: LaminarColumn(c, cfg.neuron, cfg.plasticity, seed=cfg.seed + i)
            for i, c in enumerate(cfg.columns)})
        self.names = list(self.columns.keys())
        self.left = [n for n in self.names if cfg.col(n).side == "left"]
        self.right = [n for n in self.names if cfg.col(n).side == "right"]

        # Маршруты: (dst, src, src_layer, kind). 'ff' -> базальный вход L4
        # области-адресата; 'fb' -> апикальный контекст L5 через Callosum.
        self.routes = list(cfg.routes)
        fb_routes = [r for r in self.routes if r[3] == "fb"]
        self.fb_routes = fb_routes
        self._callosum: Dict[str, CorpusCallosum] = {}
        for dst, src, layer, _ in fb_routes:
            sc, dc = cfg.col(src), cfg.col(dst)
            self._callosum[f"{src}->{dst}"] = CorpusCallosum(
                sc.n_l6, dc.n_l5, delay=cfg.callosum.delay,
                density=cfg.callosum.density, seed=cfg.seed + 17)

        readout_cols = [n for n in self.names if cfg.col(n).n_out is not None]
        self.readout_names = readout_cols
        self.workspace = GlobalWorkspace(
            max(2, len(readout_cols)),
            gamma_lat=cfg.workspace.lateral_gain / 2.0,
            beta_focus=1.0 / max(cfg.workspace.temperature, 1e-3),
            storm_rate=cfg.workspace.storm_rate,
            storm_gain=cfg.workspace.storm_gain,
            competence_rho=cfg.workspace.competence_rho,
            competence_floor=cfg.workspace.competence_floor)

        adjacency = [(r[0], r[1]) for r in self.routes]
        self.neuro = NeuroChemistry(
            self.names, cfg.mod,
            ht_baseline={n: self.columns[n].ht_baseline for n in self.names},
            adjacency=adjacency, device=self.device)
        self.controller = MetaController(len(self.names), cfg_lr=cfg.mod.ctrl_lr,
                                         cfg_sigma=cfg.mod.ctrl_sigma)
        self.credit = CreditPipeline(self.names, device=self.device,
                                     lr_fast=cfg.plasticity.lr_fast)
        n_in0 = cfg.col(self.names[0]).n_in
        self.hippo = Hippocampus(n_in0, cfg.hippo, seed=cfg.seed + 31)
        self.sleep_proto = SleepProtocol(cfg_sleep=cfg.sleep)
        self.structural = StructuralPlasticity(cfg.structural)
        self.slow = SlowLoopController(theta_sleep=cfg.sleep.threshold,
                                       poll_interval=200)
        self._ewc: Optional[EWCRegularizer] = None
        self._steps = 0
        self._apical_in: Dict[str, torch.Tensor] = {}
        self._pending_ff: Dict[str, List[torch.Tensor]] = {}
        self._ne_rise_prev = torch.zeros(len(self.names), device=self.device)

    # ------------------------------------------------------------------
    def reset_state(self, batch: Optional[int] = None) -> None:
        b = batch or self.batch
        for c in self.columns.values():
            c.reset_state(b, self.device)
        for bus in self._callosum.values():
            bus.reset_state(b, self.device)
        self.hippo.reset_state(self.device)
        self.credit.reset_episode()
        self._apical_in = {}
        self._pending_ff = {}

    def assemble(self) -> None:
        """Фаза 2: включить межобластные пути, заморозить базис."""
        for c in self.columns.values():
            c.connect_apical()
            c.freeze_basis()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def tick(self, inputs: Dict[str, torch.Tensor],
             learn: bool = True) -> Dict[str, object]:
        """Один такт жизни. inputs: имя области -> [B, n_in]."""
        for c in self.columns.values():
            c._ensure(next(iter(inputs.values())))
        B = next(iter(inputs.values())).shape[0]
        dev = self.device

        # 1. Доставка контекстов шины (отложенный апикальный путь)
        apical: Dict[str, torch.Tensor] = {n: torch.zeros(
            B, self.cfg.col(n).n_l5, device=dev) for n in self.names}
        for dst, src, layer, kind in self.fb_routes:
            bus = self._callosum[f"{src}->{dst}"]
            fwd_msg = self.columns[src].prediction_spikes()
            to_dst, _ = bus.step(fwd_msg, torch.zeros(
                B, bus.dim_out, device=dev))
            apical[dst] = apical[dst] + to_dst

        # 2. Прямые маршруты ff -> базальный драйв L4 адресата
        ff: Dict[str, torch.Tensor] = {n: torch.zeros(
            B, self.cfg.col(n).n_l4, device=dev) for n in self.names}
        for dst, src, layer, kind in self.routes:
            if kind != "ff":
                continue
            s = self.columns[src].prev.get(layer)
            if s is not None:
                ff[dst] = ff[dst] + F.pad(
                    s, (0, max(0, ff[dst].shape[-1] - s.shape[-1])))[:, : ff[dst].shape[-1]]

        # 3. Такт колонок с родной нейрохимией
        outs: Dict[str, object] = {}
        for n, col in self.columns.items():
            z = self.neuro.index[n]
            outs[n] = col.step(
                inputs[n], extra_basal=ff[n], extra_apical=apical[n],
                ach=self.neuro.ach[z],
                ht=self.neuro.ht[z],
                ne_rise=self._ne_rise_prev[z],
                sltt=False, learn=learn)

        # 4. NE-алармы: транзиент Surprise -> фазический выброс (Yu&Dayan)
        for n, col in self.columns.items():
            mag = self.controller.ne_alarm(float(col.head.surprise))
            if mag > 0.0:
                self.neuro.pulse("ne", n, mag, cap=self.cfg.mod.ne_pulse_max)

        # 5. Пластичность: раздача событийных DA-импульсов
        if learn:
            for n, col in self.columns.items():
                post = {k: outs[n].spikes[k] for k in ("l4", "l23", "l5", "l6")}
                self.credit.step_column(col, inputs[n].detach(), post,
                                        lr_scale=col.lr_scale)

        # 6. Гиппокамп: эпизодическая запись сенсорного паттерна первой области
        hippo_in = inputs[self.names[0]].detach()
        if hippo_in.shape[-1] != self.hippo.n_in:
            hippo_in = F.pad(hippo_in, (0, max(0, self.hippo.n_in - hippo_in.shape[-1])))[
                :, : self.hippo.n_in]
        self.hippo.process(hippo_in)

        # 7. Химия: распады, диффузия, давление сна; фронт NE для след. такта
        info = self.neuro.step(spike_sums={
            n: outs[n].spikes["l23"].detach().sum() for n in self.names})
        self._ne_rise_prev = info["ne_rise"]
        self._steps += 1
        self._outs = outs
        return outs

    # ------------------------------------------------------------------
    @torch.no_grad()
    def behave(self, inputs: Dict[str, torch.Tensor],
               target: torch.Tensor, learn: bool = True) -> Dict[str, object]:
        """Такт поведения: fused-выход workspace + награда -> DA-кредит."""
        outs = self.tick(inputs, learn=learn)
        logits = [self.columns[n].logits() for n in self.readout_names]
        bids = self.workspace.calibrated_bids(logits)
        alpha, winner, eff = self.workspace.compete(bids)
        fused = self.workspace.fuse(alpha, logits)
        self.workspace.update_track(logits, target)
        correct = fused.argmax(-1) == target
        reward = float(correct.float().mean())
        da = self.credit.observe_reward(reward)
        ce = float(F.cross_entropy(fused, target))
        # Локальный обучаемый выход: дельта-правило softmax на признаках
        # области (L2/3+L5). Не autograd: спайковый тракт вне графа, голова
        # чтения — единственный надзорный сигнал задачи.
        if learn:
            lr_r = 1.0
            for n in self.readout_names:
                col = self.columns[n]
                if col.readout is None:
                    continue
                h = col.features()
                h = torch.tanh(4.0 * h)  # частоты -> ограниченный диапазон
                lg = col.readout(h)
                e = F.one_hot(target, lg.shape[-1]).float() - torch.softmax(lg, -1)
                col.readout.weight.add_(lr_r * e.t() @ h / max(1, h.shape[0]))
                col.readout.bias.add_(lr_r * e.mean(0))
        # мета-контур: контроллер наблюдает и учится на том же RPE
        self.controller.observe(
            surprise=float(sum(float(c.head.surprise) for c in self.columns.values())),
            reward=reward,
            rate=float(sum(float(outs[n].rates["l23"]) for n in self.names) / len(self.names)),
            wfast_mass=sum(float(s.w_fast.abs().sum())
                           for c in self.columns.values() for s in c.syn.values()),
            pred_err=float(sum(float(o.pred_err.pow(2).mean()) for o in outs.values())))
        self.controller.learn(self.critic_rpe())
        return {"fused": fused, "alpha": alpha, "winner": winner,
                "correct": correct, "reward": reward, "da": da, "ce": ce,
                "outs": outs}

    def critic_rpe(self) -> float:
        return float(self.credit.stats["rpe_last"])

    # ------------------------------------------------------------------
    def maybe_sleep(self) -> Optional[Dict[str, float]]:
        """Медленный контур: опрос давления сна с гистерезисом."""
        cmd = self.slow.poll(float(self.neuro.sleep_pressure))
        if cmd == SlowLoopController.SLEEP:
            return self.sleep_cycle()
        return None

    @torch.no_grad()
    def sleep_cycle(self) -> Dict[str, float]:
        rep = self.sleep_proto.run_cycle(
            self.columns.values(), self.neuro, self.hippo,
            prune_threshold=self.cfg.structural.prune_threshold)
        if self._ewc is not None:
            self._ewc.rebaseline()
            rep["ewc_rebaselined"] = 1.0
        self.slow.register_sleep()
        return rep

    def arm_ewc(self, loss_fn, lam: float = 50.0, n_samples: int = 32) -> float:
        params = {}
        for n, col in self.columns.items():
            for k, s in col.syn.items():
                params[f"{n}.{k}"] = s.w_slow
        self._ewc = EWCRegularizer(params, lam=lam)
        return self._ewc.estimate_fisher(loss_fn, n_samples=n_samples)

    def apply_setpoints(self) -> None:
        sp = self.controller.setpoints()
        self.neuro.set_setpoints(sp["da_gain"].expand(len(self.names)) * 0.0,
                                 sp["ach"], sp["ht"])

    # ------------------------------------------------------------------
    @property
    def telemetry(self) -> Dict[str, float]:
        tel = dict(self.neuro.telemetry())
        tel.update(steps=float(self._steps),
                   reward_baseline=self.credit.critic.baseline,
                   hippo_stored=self.hippo.report()["stored"])
        for n, c in self.columns.items():
            tel[f"surprise.{n}"] = float(c.head.surprise)
        return tel
