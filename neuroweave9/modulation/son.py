"""Ансамбль нейронов само-наблюдения (Self-Observation Neurons) — порт из Spiker v6.0.

«Это ново?» — сигнал переключения режима:
  1. Компрессия кортикального состояния в латентный спайковый вектор z_SON(t);
  2. Авторегрессионное само-предсказание z_hat и детекция Surprise(t) —
     ошибка предсказания собственного состояния;
  3. Энергоэффективный гейтинг Zero-Compute: при Surprise < theta шина
     молчит (на предсказуемом потоке SON активен <30% тактов);
  4. Декодирование контекста в апикальные дендриты (ур. 16 канона).

В v9 SON стоит на объединённом L2/3 состоянии коры (A+B+C) — самонаблюдение
над всей корой, а не над одной зоной.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from neuroweave8.config import NeuronCfg, PlasticityCfg
from neuroweave8.core.neuron import LaminarALIF
from neuroweave8.core.synapse import StatefulSynapse


class SonBus(nn.Module):
    """Ансамбль само-наблюдения: компрессия + само-предсказание + surprise-гейт."""

    def __init__(
        self,
        zone_size: int,
        son_size: int = 16,
        dt: float = 1.0,
        tau_son_trace: float = 15.0,
        surprise_threshold: float = 0.15,
        encoder_gain: float = 2.0,
        recurrent_gain: float = 0.5,
        bus_apical_gain: float = 1.0,
        seed: int = 777,
    ):
        super().__init__()
        self.zone_size = int(zone_size)
        self.son_size = int(son_size)
        self.dt = float(dt)
        self.surprise_threshold = float(surprise_threshold)
        self.bus_apical_gain = float(bus_apical_gain)
        self.encoder_gain = float(encoder_gain)
        self.recurrent_gain = float(recurrent_gain)

        pc = PlasticityCfg()
        gen = torch.Generator().manual_seed(seed)
        self.encoder_synapse = StatefulSynapse(
            zone_size, son_size, cfg=pc, dt=dt, generator=gen)
        self.recurrent_synapse = StatefulSynapse(
            son_size, son_size, cfg=pc, dt=dt, generator=gen)
        self.decoder_synapse = StatefulSynapse(
            son_size, zone_size, cfg=pc, dt=dt, generator=gen)

        nc = NeuronCfg(v_th=-50.0, beta_adapt=1.2, sigma=0.001, dt=dt)
        self.son_neurons = LaminarALIF(son_size, nc, name="son")

        # Метакогнитивный предиктор следующего латентного состояния (ур. 17)
        self.predictor = nn.Linear(son_size, son_size)

        self.register_buffer(
            "decay_trace",
            torch.exp(torch.tensor(-dt / tau_son_trace)))

        self.z_trace: Optional[torch.Tensor] = None
        self.prev_z_trace: Optional[torch.Tensor] = None
        self.prev_son_spikes: Optional[torch.Tensor] = None
        self.zero_gaba: Optional[torch.Tensor] = None

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (batch_size, self.son_size)
        self.z_trace = torch.zeros(shape, device=device, dtype=dtype)
        self.prev_z_trace = torch.zeros(shape, device=device, dtype=dtype)
        self.prev_son_spikes = torch.zeros(shape, device=device, dtype=dtype)
        self.zero_gaba = torch.zeros(shape, device=device, dtype=dtype)
        self.son_neurons.reset_state(batch_size, device, dtype)
        self.encoder_synapse.reset_state(batch_size, device, dtype)
        self.recurrent_synapse.reset_state(batch_size, device, dtype)
        self.decoder_synapse.reset_state(batch_size, device, dtype)

    def detach_state(self) -> None:
        if self.z_trace is not None:
            self.z_trace = self.z_trace.detach()
            self.prev_z_trace = self.prev_z_trace.detach()
            self.prev_son_spikes = self.prev_son_spikes.detach()
            self.son_neurons.detach_state()

    def forward(
        self,
        zone_spikes: torch.Tensor,
        incoming_callosum_msg: Optional[torch.Tensor] = None,
        sltt_mode: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Один такт SON.

        Возвращает (z_export, apical_context, surprise, meta_loss,
        bus_active_mask): z_export — гейтированный пакет на экспорт;
        surprise — «это ново?»; meta_loss — лосс предиктора для обучения.
        """
        if (self.z_trace is None
                or self.z_trace.shape[0] != zone_spikes.shape[0]
                or self.z_trace.device != zone_spikes.device):
            self.reset_state(zone_spikes.shape[0], zone_spikes.device,
                             zone_spikes.dtype)

        if sltt_mode:
            self.detach_state()

        # Вход от коры отсекается в Фазе 1 — метакогниция не искажает лосс зон
        s_zone_in = zone_spikes.detach() if sltt_mode else zone_spikes

        # 1. Компрессия состояния в спайки SON (нормировка по плотности
        #    спайков — чувствительность шины не должна зависеть от rate)
        density = s_zone_in.mean(dim=-1, keepdim=True)
        density_scale = 1.0 / torch.sqrt(density.clamp_min(1e-3))
        raw_basal = (self.encoder_gain * self.encoder_synapse(s_zone_in)
                     * density_scale)
        raw_apical = (self.recurrent_gain
                      * self.recurrent_synapse(self.prev_son_spikes))
        o = self.son_neurons.step(raw_basal, raw_apical, self.zero_gaba,
                                  sltt=sltt_mode)
        z_spikes = o["spikes"]

        self.z_trace = (self.decay_trace * self.z_trace
                        + (1.0 - self.decay_trace) * z_spikes)

        # 2. Прогноз z_hat(t) из prev_z_trace внутри текущего такта
        #    (граф создаётся и потребляется в пределах одного шага).
        z_pred_current = torch.sigmoid(self.predictor(self.prev_z_trace))
        pred_error = self.z_trace.detach() - z_pred_current
        surprise = (pred_error * pred_error).mean(dim=-1, keepdim=True)
        meta_loss = surprise.mean()

        self.prev_z_trace = self.z_trace.detach()
        self.prev_son_spikes = (z_spikes.detach() if sltt_mode else z_spikes)

        # 3. Zero-Compute гейтинг: на предсказуемом потоке шина молчит
        bus_active_mask = (surprise.detach()
                           >= self.surprise_threshold).to(z_spikes.dtype)
        z_export = z_spikes * bus_active_mask

        # 4. Апикальный контекст — только внешний межзонный пакет
        #    (никакой самопетли, патч S2 из Spiker)
        if incoming_callosum_msg is not None:
            apical_context_current = (self.bus_apical_gain
                                      * self.decoder_synapse(
                                          incoming_callosum_msg))
        else:
            apical_context_current = torch.zeros_like(zone_spikes)

        return (z_export, apical_context_current, surprise.detach(),
                meta_loss, bus_active_mask)

    def update_phase2_plasticity(
        self,
        zone_spikes: torch.Tensor,
        da_reward: torch.Tensor,
        lr_fast: float = 0.01,
    ) -> None:
        """Трёхфакторное обновление синапсов шины в Фазе 2."""
        self.encoder_synapse.update_traces_and_plasticity(
            s_pre=zone_spikes, s_post=self.prev_son_spikes,
            da=da_reward, lr=lr_fast)
        self.decoder_synapse.update_traces_and_plasticity(
            s_pre=self.prev_son_spikes, s_post=zone_spikes,
            da=da_reward, lr=lr_fast)
