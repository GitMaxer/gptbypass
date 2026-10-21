"""CortexOrganism — организм NEUROWEAVE v9.0 «Cortex» (The Great Merge).

Единица топологии — ЛАМИНАРНАЯ КОЛОНКА (NEUROWEAVE 8), а не плоская зона
(Spiker v6.0). Три области: A (левое, синтаксис/токены), B (левое,
локальная структура), C (правое, контекст; токенов не видит — получает
код Зоны A).

Из Spiker сохранены МАТЕМАТИЧЕСКИ ПРОВЕРЕННЫЕ компоненты:
  * LaplaceTimingBank — мультимасштабный спектр контекста декодера;
  * BasalGangliaGate — дрифт-диффузионный темп эмиссии токенов;
  * ThetaGammaOscillator — PAC-гейт окна Corpus Callosum;
  * метод обучения «одна большая партия» (train_cortex.py).

Из NEUROWEAVE 8 сохранены ИННОВАЦИОННЫЕ компоненты:
  * LaminarColumn (L4->L2/3->L5->L6 + PV/SST, дендритный AND-гейт);
  * CorpusCallosum с настоящей задержкой и GABA-фильтром;
  * phase.py: ThetaGammaBinder + PhaseVSA — структурная синхронизация
    полушарий (sync.py);
  * NeuroChemistry + MetaController (сетпоинты ACh/5-HT, NE-аларм);
  * CreditPipeline (событийный DA, Tag-and-Capture, нормировка следов);
  * Hippocampus DG->CA3->CA1 с приоритизированным реплеем;
  * SleepProtocol (реплей -> консолидация -> прунинг -> пробуждение);
  * ActiveInference (EFE-внимание -> arousal-гейт L5), StructuralPlasticity,
    EWCRegularizer, int8-квантизация.

Выход step() совместим с пайплайном Spiker (out["left"]["zone_a"]["spikes"]
и т.д.) — это позволяет переиспользовать измеренный обучающий контур без
изменений его семантики.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroweave8.config import (
    CallosumCfg, ColumnCfg, HippoCfg, OrganismCfg, PhaseCfg,
    PlasticityCfg, SleepCfg, StructuralCfg, WorkspaceCfg, NeuronCfg, ModCfg,
)
from neuroweave8.microcircuit.column import LaminarColumn
from neuroweave8.modulation.chemistry import NeuroChemistry
from neuroweave8.modulation.controller import MetaController
from neuroweave8.topology.callosum import CorpusCallosum
from neuroweave8.topology.workspace import GlobalWorkspace
from neuroweave8.learning.credit import CreditPipeline
from neuroweave8.learning.sleep import SleepProtocol
from neuroweave8.learning.ewc import EWCRegularizer
from neuroweave8.learning.structural import StructuralPlasticity
from neuroweave8.memory.hippocampus import Hippocampus
from neuroweave8.organism.slow_loop import SlowLoopController
from neuroweave8.organism.active_inference import ActiveInference

from ..memory.laplace import LaplaceTimingBank
from ..action.basal_ganglia import BasalGangliaGate
from ..modulation.pacemaker import ThetaGammaOscillator
from ..modulation.orexin import OrexinInteroceptiveLoop
from ..modulation.no_field import NitricOxideField
from .sync import HemisphereSync


def cortex_cfg(
    vocab: int,
    input_dim: Optional[int] = None,
    scale: float = 1.0,
    seed: int = 42,
    wm_slots: int = 7,
) -> OrganismCfg:
    """Конфигурация коры v9: A,B (левое) + C (правое) из ламинарных колонок.

    scale=1.0 ~ пресет Spiker 64/48/64 по числу нейронов коры (152/152/152).
    """
    n_in = int(input_dim or vocab)
    s = float(scale)
    l23 = max(8, int(round(48 * s)))

    A = ColumnCfg(name="A", side="left", n_in=n_in, n_out=vocab,
                  n_l4=max(6, int(24 * s)), n_l23=l23, n_l5=max(8, int(32 * s)),
                  n_l6=max(6, int(24 * s)), n_pv=max(4, int(16 * s)),
                  n_sst=max(2, int(8 * s)), input_gain=12.0, wm_slots=wm_slots)
    B = ColumnCfg(name="B", side="left", n_in=n_in, n_out=vocab,
                  n_l4=max(6, int(24 * s)), n_l23=l23, n_l5=max(8, int(32 * s)),
                  n_l6=max(6, int(24 * s)), n_pv=max(4, int(16 * s)),
                  n_sst=max(2, int(8 * s)), input_gain=12.0, wm_slots=wm_slots)
    # C видит код A (n_l23_A), а не токены — ур. (1) канона.
    C = ColumnCfg(name="C", side="right", n_in=l23, n_out=l23,
                  n_l4=max(6, int(24 * s)), n_l23=l23, n_l5=max(8, int(32 * s)),
                  n_l6=max(6, int(24 * s)), n_pv=max(4, int(16 * s)),
                  n_sst=max(2, int(8 * s)), input_gain=4.0, wm_slots=wm_slots)
    return OrganismCfg(
        columns=[A, B, C], vocab=vocab, blank=vocab, seed=seed,
        neuron=NeuronCfg(tau_m=8.0, sigma=0.001, surrogate="fast_sigmoid"),
        callosum=CallosumCfg(delay=6, density=0.35),
        workspace=WorkspaceCfg(temperature=0.25, lateral_gain=2.5),
        plasticity=PlasticityCfg(w_fast_max=0.02),
        hippo=HippoCfg(n_slots=64),
        structural=StructuralCfg(enabled=True),
        sleep=SleepCfg(),
    )


class CortexOrganism(nn.Module):
    """Цифровой организм-кортекс: колонки + шина + память + гейт решения."""

    def __init__(
        self,
        vocab_size: int,
        input_dim: Optional[int] = None,
        scale: float = 1.0,
        dt: float = 1.0,
        num_laplace_scales: int = 8,
        vsa_dim: int = 512,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
        lr_left: float = 0.020,
        lr_right: float = 0.006,
        horizon_delta_t: int = 3,
        lambda_ewc: float = 100.0,
    ):
        super().__init__()
        self.device = device
        self.dt = float(dt)
        self.vocab_size = int(vocab_size)
        self.input_dim = int(input_dim or vocab_size)
        self.cfg = cortex_cfg(vocab_size, input_dim=self.input_dim,
                              scale=scale, seed=seed)
        self.lr_left, self.lr_right = lr_left, lr_right
        self.horizon_delta_t = int(horizon_delta_t)
        self.seed = seed

        torch.manual_seed(seed)
        self.columns = nn.ModuleDict({
            c.name: LaminarColumn(c, self.cfg.neuron, self.cfg.plasticity,
                                  seed=seed + i)
            for i, c in enumerate(self.cfg.columns)})
        self.names = ["A", "B", "C"]
        self.left_names, self.right_names = ["A", "B"], ["C"]

        # Межполушарная шина C -> A (апикальный контекст L6 -> L5)
        ca, cc = self.cfg.col("A"), self.cfg.col("C")
        self.callosum = CorpusCallosum(
            cc.n_l6, ca.n_l5, delay=self.cfg.callosum.delay,
            density=self.cfg.callosum.density, seed=seed + 17)

        # Спайковые модули Spiker: осциллятор, Лапласова память, БГ
        self.oscillator = ThetaGammaOscillator(dt=dt)
        self.laplace_memory = LaplaceTimingBank(
            feature_dim=self.input_dim, num_scales=num_laplace_scales, dt=dt)
        self.basal_ganglia = BasalGangliaGate(dt=dt)
        # Порты Spiker v7.0: орексиновый интероцептивный контур и NO-поле
        self.orexin = OrexinInteroceptiveLoop(dt=dt)
        self.no_field = NitricOxideField(num_neurons=self.cfg.col("C").n_l23, dt=dt)
        # STP Цодыкса-Маркрама на рекуррентности L2/3 — рабочая память
        for col in self.columns.values():
            if "rec" in col.syn:
                col.syn["rec"].enable_stp("depressing", dt=dt)

        # Точка слияния: структурная θ-γ синхронизация полушарий
        n_l23 = self.cfg.col("A").n_l23
        self.sync = HemisphereSync(code_dim=n_l23, vsa_dim=vsa_dim,
                                   n_slots=self.cfg.phase.n_slots, seed=seed)

        # Модули NEUROWEAVE 8
        self.workspace = GlobalWorkspace(
            3, gamma_lat=self.cfg.workspace.lateral_gain / 2.0,
            beta_focus=1.0 / max(self.cfg.workspace.temperature, 1e-3),
            storm_rate=self.cfg.workspace.storm_rate,
            storm_gain=self.cfg.workspace.storm_gain,
            competence_rho=self.cfg.workspace.competence_rho,
            competence_floor=self.cfg.workspace.competence_floor)
        self.neuro = NeuroChemistry(
            self.names, self.cfg.mod,
            ht_baseline={n: self.columns[n].ht_baseline for n in self.names},
            adjacency=[("A", "B"), ("B", "C"), ("A", "C")], device=device)
        self.controller = MetaController(
            3, cfg_lr=self.cfg.mod.ctrl_lr, cfg_sigma=self.cfg.mod.ctrl_sigma)
        self.credit = CreditPipeline(self.names, device=device,
                                     lr_fast=self.cfg.plasticity.lr_fast)
        self.hippo = Hippocampus(self.input_dim, self.cfg.hippo, seed=seed + 31)
        self.sleep_proto = SleepProtocol(cfg_sleep=self.cfg.sleep)
        self.structural = StructuralPlasticity(self.cfg.structural)
        self.slow = SlowLoopController(theta_sleep=self.cfg.sleep.threshold,
                                     poll_interval=200)
        self.infer = ActiveInference(3, gamma=4.0, gain=0.6)

        # Токен-голова правого полушария для конкуренции workspace
        c_feat = (cc.n_l23 + cc.n_l5
                  + (cc.n_l23 if self.columns["C"].binder is not None else 0)
                  + cc.n_in)
        self.token_head_c = nn.Linear(c_feat, vocab_size)

        self._ewc: Optional[EWCRegularizer] = None
        self.phase: int = 1
        self._steps = 0
        self._ne_rise_prev = torch.zeros(3, device=device)
        self._prev_c_in: Optional[torch.Tensor] = None
        self.last_metrics: Dict[str, float] = {}
        self.to(device)

    # ------------------------------------------------------------------
    def set_phase(self, phase: int) -> None:
        """1 — изоляция (шина разорвана); 2 — сборка (апикальный путь открыт,
        w_slow заморожен); 3 — сон."""
        assert phase in (1, 2, 3)
        self.phase = phase
        if phase >= 2:
            for c in self.columns.values():
                c.connect_apical()
                c.freeze_basis()
        else:
            for c in self.columns.values():
                c.isolate()

    def reset_state(self, batch_size: int = 1,
                    device: Optional[torch.device] = None) -> None:
        dev = device or self.device
        for c in self.columns.values():
            c.reset_state(batch_size, dev)
        self.callosum.reset_state(batch_size, dev)
        self.sync.reset_state(batch_size, dev)
        self.laplace_memory.reset_state(batch_size, dev)
        self.basal_ganglia.reset_state(batch_size, dev)
        self.orexin.reset_state(batch_size, dev)
        self.no_field.reset_state(batch_size, dev)
        self._last_activity = torch.zeros(batch_size, 1, device=dev)
        self.hippo.reset_state(dev)
        self.oscillator.reset_state(dev)
        self.credit.reset_episode()
        self._ne_rise_prev = torch.zeros(3, device=dev)
        self._apical = {n: torch.zeros(batch_size, self.cfg.col(n).n_l5,
                                       device=dev) for n in self.names}
        self._ff = {n: torch.zeros(batch_size, self.cfg.col(n).n_l4,
                                   device=dev) for n in self.names}
        self._steps = 0

    @property
    def ewc(self) -> Optional[EWCRegularizer]:
        return self._ewc

    def arm_ewc(self, loss_fn, lam: float = 50.0, n_samples: int = 32) -> float:
        params = {}
        for n, col in self.columns.items():
            for k, s in col.syn.items():
                params[f"{n}.{k}"] = s.w_slow
            if col.readout is not None:
                params[f"{n}.readout"] = col.readout.weight
        self._ewc = EWCRegularizer(params, lam=lam)
        return self._ewc.estimate_fisher(loss_fn, n_samples=n_samples)

    # ------------------------------------------------------------------
    def step_isolated(
        self,
        name: str,
        x_in: torch.Tensor,
        learn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Изолированный такт одной колонки (Фаза 1): шина разорвана,
        апикальный контекст = 0, sltt=True — суррогатный градиент живёт."""
        col = self.columns[name]
        col._ensure(x_in)
        B = x_in.shape[0]
        out = col.step(
            x_in, extra_apical=None,
            arousal=torch.zeros(B, self.cfg.col(name).n_l5, device=x_in.device),
            sltt=True, learn=learn)
        return {"spikes": out.spikes["l23"], "logits": col.logits(),
                "l5": out.spikes["l5"], "surprise": col.head.surprise}

    def _column_inputs(
        self, token_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """A,B <- токен; C <- выход L5 зоны A (в Фазе 1 — ноль, изоляция)."""
        B = token_input.shape[0]
        cin = self._prev_c_in
        if cin is None or cin.shape[0] != B:
            cin = torch.zeros(B, self.cfg.col("C").n_in, device=token_input.device)
        return {"A": token_input, "B": token_input, "C": cin}

    def step(
        self,
        token_input: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        sync_metrics: bool = True,
        learn: bool = True,
    ) -> Dict[str, object]:
        """Один такт быстрого контура (1 мс)."""
        B = token_input.shape[0]
        dev = token_input.device
        if not hasattr(self, "_apical") or self._apical["A"].shape[0] != B:
            self.reset_state(B, dev)

        # 1. θ-γ пейсмейкер: PAC + фазовое окно Callosum
        osc = self.oscillator.step()
        gate = osc["callosum_phase_gate"]

        # 2. Шина C->A: апикальный контекст через задержку Callosum,
        #    открытая только на пике θ (в Фазе 1 шина разорвана).
        apical = self._apical
        for t in apical.values():
            t.zero_()
        if self.phase >= 2:
            msg = self.columns["C"].prediction_spikes() * gate
            to_a, _ = self.callosum.step(
                msg, torch.zeros(B, self.callosum.dim_out, device=dev))
            apical["A"] = apical["A"] + to_a

        # 3. EFE-внимание: компетентность x новизна -> arousal-гейт L5
        comp = self.workspace.track[:3].to(dev)
        nov = torch.tensor([float(self.columns[n].head.surprise)
                            for n in self.names], device=dev)
        att = self.infer.gains(comp, nov)

        # 4. Такт колонок (в Фазе 1 — sltt для суррогатного градиента)
        inputs = self._column_inputs(token_input)
        outs: Dict[str, object] = {}
        sltt = (self.phase == 1)
        for i, n in enumerate(self.names):
            col = self.columns[n]
            z = self.neuro.index[n]
            outs[n] = col.step(
                inputs[n], extra_apical=apical[n],
                arousal=att[i].view(1, 1).expand(B, self.cfg.col(n).n_l5),
                ach=self.neuro.ach[z], ht=self.neuro.ht[z],
                ne_rise=self._ne_rise_prev[z],
                sltt=sltt, learn=learn)

        # 5. NE-алармы: транзиент surprise -> фазический выброс
        for n in self.names:
            mag = self.controller.ne_alarm(float(self.columns[n].head.surprise))
            if mag > 0.0:
                self.neuro.pulse("ne", n, mag, cap=self.cfg.mod.ne_pulse_max)

        # 6. Пластичность Фазы 2: событийные DA-импульсы -> w_fast
        if learn:
            for n in self.names:
                col = self.columns[n]
                post = {k: outs[n].spikes[k] for k in ("l4", "l23", "l5", "l6")}
                self.credit.step_column(col, inputs[n].detach(), post,
                                        lr_scale=col.lr_scale)
        if reward is not None:
            self.credit.observe_reward(float(
                reward.mean().item() if torch.is_tensor(reward) else reward))

        # 7. Вход C на следующий такт — выход L5 зоны A (ур. 1)
        a_out = self.columns["A"].output_spikes()
        n_in_c = self.cfg.col("C").n_in
        self._prev_c_in = F.pad(
            a_out.detach(), (0, max(0, n_in_c - a_out.shape[-1])))[:, :n_in_c]

        # 8. Гиппокамп + Лапласова память + синхронизация полушарий + NO-поле
        self.hippo.process(token_input.detach())
        laplace_out = self.laplace_memory.step(token_input)
        sync_out = self.sync.step(outs["A"].spikes["l23"].detach(),
                                  outs["C"].spikes["l23"].detach())
        no_out = self.no_field.step(outs["C"].spikes["l23"].detach())

        # 9. Workspace: конкуренция зон -> fused логиты
        logits_a = self.columns["A"].logits()
        logits_b = self.columns["B"].logits()
        logits_c = self.token_head_c(self.columns["C"].features())
        logits_list = [logits_a, logits_b, logits_c]
        bids = self.workspace.calibrated_bids(logits_list)
        alpha, winner, eff = self.workspace.compete(bids)
        fused = self.workspace.fuse(alpha, logits_list, winner=winner)

        # 10. Интероцептивный контур (Spiker §7.0): активность такта =
        #     метаболическая стоимость; энтропия выхода = давление цели.
        probs = F.softmax(fused, dim=-1)
        entropy_t = (-(probs * probs.clamp_min(1e-9).log()).sum(-1)).mean()
        firing_t = torch.cat([outs[n].spikes["l23"] for n in self.names],
                             dim=-1).mean()
        self._last_activity = firing_t.detach().reshape(-1, 1).clone()
        entropy_mean = float(entropy_t.detach())
        if entropy_mean > 0.9:
            self.orexin.register_goal("reduce_output_entropy")
        elif "reduce_output_entropy" in self.orexin.unfinished_goals:
            self.orexin.satisfy_goal("reduce_output_entropy")
        orexin_out = self.orexin.step(activity=self._last_activity)

        # 11. Базальные ганглии: темп решения (порог снижает орексин)
        z_a = self.neuro.index["A"]
        bg_out = self.basal_ganglia(
            fused,
            serotonin_5ht=self.neuro.ht[z_a],
            ach_level=self.neuro.ach[z_a],
            orexin_level=orexin_out["orexin"])

        # 12. Химия + медленный контур
        info = self.neuro.step(spike_sums={
            n: outs[n].spikes["l23"].detach().sum() for n in self.names})
        self._ne_rise_prev = info["ne_rise"]
        self.slow.poll(float(self.neuro.sleep_pressure))
        self._steps += 1

        metrics = {}
        if sync_metrics:
            metrics = {
                "entropy": entropy_mean,
                "orexin": float(orexin_out["orexin"].mean()),
                "energy": float(orexin_out["energy"].mean()),
                "firing_rate": float(firing_t.detach()),
                "sleep_pressure": float(self.neuro.sleep_pressure),
                "winner_zone": float(winner.float().mean()),
                "sync_coherence": float(sync_out["coherence"]),
                "wm_occupancy": float(sync_out["occupancy"]),
                "callosum_gate": float(gate),
            }
            self.last_metrics = metrics
        else:
            metrics = self.last_metrics

        spikes_a = outs["A"].spikes["l23"]
        spikes_b = outs["B"].spikes["l23"]
        spikes_c = outs["C"].spikes["l23"]
        return {
            "output_logits": fused,
            "left": {
                "zone_a": {"spikes": spikes_a, "logits": logits_a,
                           "surprise": self.columns["A"].head.surprise,
                           "modulators": {
                               "serotonin_5ht": self.neuro.ht[z_a],
                               "ach": self.neuro.ach[z_a],
                               "ne": self.neuro.ne[z_a]}},
                "zone_b": {"spikes": spikes_b, "logits": logits_b}},
            "right": {
                "zone_c": {"spikes": spikes_c},
                "token_logits": logits_c,
                "proj_state_a_future": logits_c},
            "workspace": {"wta_weights": alpha, "winner_zone_idx": winner},
            "basal_ganglia": bg_out,
            "laplace_memory": laplace_out,
            "sync": sync_out,
            "orexin": orexin_out,
            "no_field": no_out,
            "oscillator": osc,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sleep_cycle(self, n_steps: Optional[int] = None) -> Dict[str, float]:
        rep = self.sleep_proto.run_cycle(
            self.columns.values(), self.neuro, self.hippo,
            prune_threshold=self.cfg.structural.prune_threshold)
        if self._ewc is not None:
            self._ewc.rebaseline()
            rep["ewc_rebaselined"] = 1.0
        self.slow.register_sleep()
        # Ключи совместимости с отчётностью Spiker
        rep.setdefault("swr_replay_steps", rep.get("replay_steps", 0.0))
        rep.setdefault("rem_steps", 0.0)
        rep.setdefault("rem_chimera_insights", 0.0)
        rep["sleep_steps"] = rep.get("replay_steps", 0.0)
        return rep

    @torch.no_grad()
    def run_episode(
        self,
        token_seq: torch.Tensor,
        reward_seq: Optional[torch.Tensor] = None,
        target_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        seq_len, batch, _ = token_seq.shape
        self.reset_state(batch, token_seq.device)
        correct, total = 0, 0
        for t in range(seq_len):
            reward = reward_seq[t] if reward_seq is not None else None
            out = self.step(token_seq[t], reward=reward)
            if target_seq is not None:
                pred = out["output_logits"].argmax(dim=-1)
                correct += int((pred == target_seq[t]).sum())
                total += int(pred.numel())
        return {"steps": float(seq_len),
                "accuracy": correct / max(1, total),
                "sleep_pressure": float(self.neuro.sleep_pressure)}

    def telemetry(self) -> Dict[str, float]:
        tel = dict(self.neuro.telemetry())
        tel.update(steps=float(self._steps),
                   hippo_stored=self.hippo.report()["stored"],
                   sync_coherence=float(self.sync.coherence),
                   wm_occupancy=float(self.sync.binder.occupancy())
                   if self.sync.binder.buf is not None else 0.0,
                   bg_emissions=float(self.basal_ganglia.total_emissions),
                   **{f"orexin_{k}": v for k, v in self.orexin.state_summary().items()})
        return tel

    def cortical_dim(self) -> int:
        return sum(self.cfg.col(n).n_l23 for n in self.names)

    def feature_dim_c(self) -> int:
        cc = self.cfg.col("C")
        return (cc.n_l23 + cc.n_l5
                + (cc.n_l23 if self.columns["C"].binder is not None else 0)
                + cc.n_in)
