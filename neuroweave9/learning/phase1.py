"""Фаза 1 v9 «Cortex»: изолированное обучение ЛАМИНАРНЫХ колонок.

Метод — «одна большая партия» из Spiker (train_scale_original.py,
ARCHITECTURE_REVIEW §8.1): весь корпус — один батч, SLTT с detach() на
каждом такте даёт память O(1) по времени, суррогатный градиент течёт через
спайки ТЕКУЩЕГО такта.

Локальные цели (табл. 3 канона, ур. 1-2):
  * A (левое): каузальное предсказание следующего токена — CE на readout;
  * B (левое): реконструкция короткого спайкового окна — MSE(sigmoid
    logits, EMA one-hot);
  * C (правое): предсказание траектории состояния A на горизонте delta_t
    через каузальное кольцо задержки (C получает s_A(t-dt), предсказывает
    s_A(t)) — MSE(sigmoid, след A) с энергетической нормировкой.

После эпох — EWC: accumulate_fisher_step() по градиентам w_slow/readout,
затем finalize (нормировка Фишера + снапшот + жёсткая заморозка базиса).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from ..cortex.organism import CortexOrganism


class CortexEWC:
    """EWC над w_slow и readout всех колонок (интерфейс Spiker)."""

    def __init__(self, organism: CortexOrganism, lambda_ewc: float = 100.0):
        self.org = organism
        self.lam = float(lambda_ewc)
        self.params: Dict[str, torch.nn.Parameter] = {}
        for n, col in organism.columns.items():
            for k, s in col.syn.items():
                self.params[f"{n}.{k}.w_slow"] = s.w_slow
            if col.readout is not None:
                self.params[f"{n}.readout.weight"] = col.readout.weight
        self.fisher = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.snapshot = {k: p.detach().clone() for k, p in self.params.items()}
        self.is_consolidated = False

    def accumulate_fisher_step(self) -> None:
        for k, p in self.params.items():
            if p.grad is not None:
                self.fisher[k] += p.grad.detach() ** 2

    def finalize_phase1_and_freeze(
        self, num_samples: int = 32, hard_freeze_synapses: bool = True
    ) -> int:
        for k in self.fisher:
            self.fisher[k] = self.fisher[k] / max(1, num_samples)
            m = self.fisher[k].max()
            if float(m) > 0:
                self.fisher[k] = self.fisher[k] / m
        for k, p in self.params.items():
            self.snapshot[k] = p.detach().clone()
        self.is_consolidated = True
        if hard_freeze_synapses:
            self.org.set_phase(2)
        return len(self.params)

    def rebaseline_after_sleep(self) -> int:
        for k, p in self.params.items():
            self.snapshot[k] = p.detach().clone()
        return len(self.params)

    def compute_ewc_loss(self) -> torch.Tensor:
        if not self.is_consolidated:
            return torch.zeros(())
        return sum(
            (self.fisher[k] * (p - self.snapshot[k]) ** 2).sum()
            for k, p in self.params.items()) * self.lam


class Phase1ColumnTrainer:
    """Тренер изолированных колонок — прямой аналог Phase1IsolatedTrainer."""

    def __init__(
        self,
        organism: CortexOrganism,
        lr_left: float = 0.020,
        lr_right: float = 0.006,
        horizon_delta_t: int = 3,
        lambda_ewc: float = 100.0,
    ):
        self.org = organism
        self.delta_t = max(1, int(horizon_delta_t))
        self.ewc = CortexEWC(organism, lambda_ewc=lambda_ewc)
        self.opt_a = torch.optim.Adam(organism.columns["A"].parameters(), lr=lr_left)
        self.opt_b = torch.optim.Adam(organism.columns["B"].parameters(), lr=lr_left)
        self.opt_c = torch.optim.Adam(
            list(organism.columns["C"].parameters())
            + list(organism.token_head_c.parameters()), lr=lr_right)

    def train_sequence_epoch(
        self,
        token_seq_one_hot: torch.Tensor,
        target_token_ids: torch.Tensor,
        accumulate_fisher: bool = False,
    ) -> Dict[str, float]:
        org = self.org
        seq_len, batch_size, _ = token_seq_one_hot.shape
        dev = token_seq_one_hot.device
        org.set_phase(1)
        org.reset_state(batch_size, dev)

        n_l23_a = org.cfg.col("A").n_l23
        za_trace = torch.zeros((batch_size, n_l23_a), device=dev)
        history: List[torch.Tensor] = [
            torch.zeros((batch_size, n_l23_a), device=dev)
            for _ in range(self.delta_t)]
        window_target = torch.zeros(
            (batch_size, org.vocab_size), device=dev)
        w_decay = 0.8
        trace_decay = 0.75

        sa = sb = sc = 0.0
        for t in range(seq_len):
            self.opt_a.zero_grad()
            self.opt_b.zero_grad()
            self.opt_c.zero_grad()

            x_t, y_t = token_seq_one_hot[t], target_token_ids[t]

            # Левое полушарие: A (синтаксис), B (структура)
            out_a = org.step_isolated("A", x_t, learn=True)
            loss_a = F.cross_entropy(out_a["logits"], y_t)
            out_b = org.step_isolated("B", x_t, learn=True)
            window_target = (w_decay * window_target.detach()
                             + (1.0 - w_decay) * x_t[:, : org.vocab_size].detach())
            loss_b = F.mse_loss(torch.sigmoid(out_b["logits"]), window_target)

            # Правое: C получает ОТСЕЧЁННЫЙ след A из прошлого, предсказывает текущий
            delayed = history.pop(0)
            raw_a = out_a["spikes"].detach()
            za_trace = trace_decay * za_trace + (1.0 - trace_decay) * raw_a
            current_a = za_trace.detach()
            history.append(current_a)
            out_c = org.step_isolated("C", delayed, learn=True)
            pred = torch.sigmoid(out_c["logits"][:, :n_l23_a])
            mse = F.mse_loss(pred, current_a)
            scale = current_a.pow(2).mean().detach().clamp_min(0.05)
            loss_c = mse / scale

            loss_a.backward()
            loss_b.backward()
            loss_c.backward()
            if accumulate_fisher:
                self.ewc.accumulate_fisher_step()
            self.opt_a.step()
            self.opt_b.step()
            self.opt_c.step()

            sa += float(loss_a.item())
            sb += float(loss_b.item())
            sc += float(loss_c.item())

        return {"loss_zone_a": sa / seq_len, "loss_zone_b": sb / seq_len,
                "loss_zone_c": sc / seq_len}
