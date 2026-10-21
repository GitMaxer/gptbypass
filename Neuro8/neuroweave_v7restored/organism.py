# src/spiker/organism/organism.py
# Этап 3/4 (канон v4.0/v4.0.1, разд. 8 + табл. 5/6): полная сборка организма.
#
# SpikerOrganism — БЫСТРЫЙ GPU-контур жизни: зоны -> полушария -> Callosum
# -> Global Workspace, пошаговая нейрохимия и Reward-STDP — всё в едином
# такте без синхронизации с хостом. Медленный CPU-контур (SlowLoopController)
# изредка опрашивает телеметрию и выносит макро-решения (сон, режимы ACh).
#
# Жизненный цикл (то, что делает SNN «живой самоулучшающейся»):
#   ФАЗА 1  (вне организма): изолированное обучение зон SLTT;
#   ASSEMBLE: connect_apical + freeze w_slow — шина физически включена;
#   БОДРСТВОВАНИЕ: awake_step/behaviour — Tag-and-Capture под DA-гейтом;
#   СОН: sleep_cycle — 5 шагов разд. 7.3 (реплей, консолидация, прунинг);
#   ПРОБУЖДЕНИЕ: навык в w_slow, P_sleep = 0, EWC перебазирован —
#   организм готов к следующему циклу, не забывая предыдущий.
import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..topology.zone import Zone
from ..topology.hemisphere import Hemisphere, LeftHemisphere, RightHemisphere
from ..topology.callosum import CorpusCallosum
from ..topology.workspace import GlobalWorkspace
from ..modulation.neurochemistry import NeuroChemistry
from ..learning.stdp import RewardSTDPPipeline
from ..learning.sleep import SleepProtocol
from ..learning.ewc import EWCRegularizer


class SpikerOrganism(nn.Module):
    """
    Полная сборка организма: быстрый контур жизни.

    Параметры конструктора:
        left_zones / right_zones: списки Zone соответствующих боков;
        lr_fast: базовый темп Reward-STDP (умножается на пресет lr_local
                 зоны: правое полушарие консервативнее, табл. 2);
        bus_apical_gain: PoC-калибровка доставки SON (устранить при
                 обучении декодеров приёма — открытый вопрос Этапа 4);
        lam_ewc: вес штрафа Фишера (вооружается arm_ewc после Фазы 1);
        wta_track_calibration (Этап 6): калибровка заявок WTA бегущим
                 протоколом точности зон (track record, tau_track тактов) —
                 закрытие факта Этапа 5 «уверенность != компетентность»;
        reward_attribution (Этап 6): пер-зонная атрибуция DA-импульса —
                 амплитуда события масштабируется точностью САМОЙ зоны
                 (закрытие открытого вопроса Этапа 3: глобальный кредит
                 усиливает уверенно-ошибающихся экспертов);
        track_tau: константа времени протокола точности (тактов).
    """

    def __init__(
        self,
        left_zones: Sequence[Zone],
        right_zones: Sequence[Zone],
        lr_fast: float = 2e-3,
        bus_apical_gain: float = 0.2,
        lam_ewc: float = 50.0,
        callosum_delay: int = 7,
        tau_v: float = 150.0,
        sleep_kwargs: Optional[dict] = None,
        callosum_kwargs: Optional[dict] = None,
        wta_track_calibration: bool = False,
        reward_attribution: bool = False,
        track_tau: float = 16.0,
        track_epsilon: float = 0.1,
    ):
        super().__init__()
        left_list, right_list = list(left_zones), list(right_zones)
        if not left_list:
            raise ValueError("Организм требует хотя бы одну левую зону")
        if not right_list:
            raise ValueError("Организм требует хотя бы одну правую зону")

        named: Dict[str, Zone] = {}
        for z in left_list + right_list:
            if z.name in named:
                raise ValueError(f"Дублирующееся имя зоны: {z.name!r}")
            named[z.name] = z
        self.zone_names = list(named.keys())
        self.zones = nn.ModuleDict(named)

        self.left = LeftHemisphere(left_list)
        self.right = RightHemisphere(right_list)

        self.callosum = CorpusCallosum(
            left_dim=self.left.bus_dim, right_dim=self.right.bus_dim,
            delay=callosum_delay,
            **(callosum_kwargs or {}),
        )
        self.workspace = GlobalWorkspace(n_zones=len(self.zone_names))

        # Нейрохимия: 5-HT базовые уровни — из зон (дивергенция полушарий)
        ht_baseline = {z.name: z.ht_baseline for z in named.values()}
        self.neuro = NeuroChemistry(zone_names=self.zone_names,
                                    ht_baseline=ht_baseline)

        self.pipeline = RewardSTDPPipeline(self.neuro, lr_fast=lr_fast,
                                           tau_v=tau_v)
        self.sleep_protocol = SleepProtocol(
            **(sleep_kwargs or {}))
        self.bus_apical_gain = float(bus_apical_gain)
        # Этап 5-d: обучение декодеров приёма SON — устранение ручной
        # калибровки BUS_APICAL_GAIN (гейн рампится к 1.0 по мере кредита)
        self.decoder_training = False
        self._gain_target = 1.0
        self._lam_ewc = float(lam_ewc)
        self._ewc: Optional[EWCRegularizer] = None

        # Отложенные сообщения шины и прошлые спайки зон
        self._to_right: Optional[torch.Tensor] = None
        self._to_left: Optional[torch.Tensor] = None
        # Чанки шины, ДОСТАВЛЕННЫЕ зонам в текущем такте (входы узлов
        # приёма SON — аргументы 3-факторного кредита декодеров, Этап 5-d)
        self._delivered: Dict[str, torch.Tensor] = {}
        self._last_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        self._awake_steps = 0

        # Этап 6 — быстрый контекст доверия:
        # 1) протокол точности зон (track record) для калибровки заявок WTA;
        # 2) пер-зонная атрибуция амплитуды DA-импульса. Оба механизма —
        #    ОПЦИИ (дефолт OFF: PoC-семантика Этапов 1-5 воспроизводится
        #    побитово); сессия ЭКЗАМЕНА включает оба (см. Этап 6).
        # Протокол точности — память ОРГАНИЗМА (как база критика V):
        # НЕ сбрасывается в reset_state — доверие переживает эпизоды.
        self.wta_track_calibration = bool(wta_track_calibration)
        self.reward_attribution = bool(reward_attribution)
        self._track_rho = 1.0 - math.exp(-1.0 / float(track_tau))
        self._track_epsilon = float(track_epsilon)
        self.register_buffer(
            "zone_track", torch.full((len(self.zone_names),), 0.5))

    # ------------------------------------------------------------------
    # Фазы жизненного цикла
    # ------------------------------------------------------------------
    def assemble(self):
        """Фаза 2: физически включить шину, заморозить w_slow экспертов."""
        for zone in self.zones.values():
            zone.connect_apical()
            zone.syn_in.freeze_for_phase2()
            zone.syn_rec.freeze_for_phase2()
            if zone.readout is not None:
                for p in zone.readout.parameters():
                    p.requires_grad_(False)

    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32):
        for zone in self.zones.values():
            zone.reset_state(batch_size, device, dtype)
        self.callosum.reset_state(batch_size, device, dtype)
        self._to_right = torch.zeros(batch_size, self.right.bus_dim,
                                     device=device, dtype=dtype)
        self._to_left = torch.zeros(batch_size, self.left.bus_dim,
                                    device=device, dtype=dtype)
        self._last_outputs = {}
        # Новый эпизод: DA-фронт пайплайна отсчитывается с нуля
        self.pipeline.reset_episode()

    # ------------------------------------------------------------------
    # Такт бодрствования (быстрый контур целиком)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def awake_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Один такт организма: шина -> полушария -> пластичность -> химия.

        inputs: имя зоны -> сенсорный вход зоны [B, n_inputs].
        Возвращает: имя зоны -> выходы Zone.step + сводную активность.
        """
        missing = set(self.zone_names) - set(inputs.keys())
        if missing:
            raise KeyError(f"Организму не подан вход зон: {missing}")
        batch = next(iter(inputs.values())).shape[0]
        device = next(iter(inputs.values())).device
        if self._to_right is None or self._to_right.shape[0] != batch \
                or self._to_right.device != device:
            self.reset_state(batch, device)

        # 1. Спайки прошлого такта (для рекуррентной пластичности)
        prev_spikes = {
            name: zone.s_exc_prev.detach().clone() if zone.s_exc_prev is not None
            else torch.zeros(batch, zone.n_exc, device=device)
            for name, zone in self.zones.items()
        }

        # 2. Доставка сообщений шины на апикальные дендриты (ур. 16).
        #    Чанки сохраняются: они — входы узлов приёма SON и аргументы
        #    кредита декодеров (Этап 5-d)
        chunks_l = self.left.bus_chunks(self._to_left)
        chunks_r = self.right.bus_chunks(self._to_right)
        self._delivered = {**chunks_l, **chunks_r}
        apical_l = self.left.decode_chunks(chunks_l, gain=self.bus_apical_gain)
        apical_r = self.right.decode_chunks(chunks_r, gain=self.bus_apical_gain)

        # 3. Полушария (родная нейрохимия: ACh/NE/5-HT из NeuroChemistry).
        #    Патч Этапа 5 (capture-ready): NE-алармы выпускаются ТЕНЗОРНО из
        #    son.alarm_mag (безветвевая амплитуда) — колбэка в горячем такте
        #    больше нет: каждое событие — чистые операции на устройстве.
        inputs_l = {k: inputs[k] for k in self.left.zone_names}
        inputs_r = {k: inputs[k] for k in self.right.zone_names}
        outs_l = self.left.step(inputs_l, apical=apical_l, neuro=self.neuro)
        outs_r = self.right.step(inputs_r, apical=apical_r, neuro=self.neuro)
        outputs = {**outs_l, **outs_r}
        self._last_outputs = outputs

        # 3а. Фазические NE-алармы SON (разд. 6.3): тензорный выброс
        for name, zone in self.zones.items():
            mag = zone.son.alarm_mag
            if mag is not None:
                self.neuro.release_ne(name, mag)

        # 4. Асинхронный обмен SON-сообщениями (задержка Callosum)
        self._to_right, self._to_left = self.callosum.step(
            self.left.son_export(outs_l), self.right.son_export(outs_r))

        # 5. Reward-STDP: следы + пластичность под гейтом DA зоны.
        #    Импульс события раздаётся ОДИН раз — синапсам зоны и (Этап 5-d)
        #    декодеру приёма SON: оба потребителя видят одно событие
        for name, zone in self.zones.items():
            s_exc = outputs[name]["spikes"]
            impulse = self.pipeline.step_zone(
                zone, inputs[name], s_exc, prev_spikes[name],
                lr_fast=self.pipeline.lr_fast * (zone.lr_local / 0.01),
            )
            if self.decoder_training:
                chunk = self._delivered.get(name)
                if chunk is not None:
                    self.pipeline.step_decoder(
                        zone, chunk, s_exc, impulse)

        # 6. Транзитный кредит проекций шины (3-факторный, у DA получателя).
        #    Патч Этапа 5 (capture-ready): DA-факторы — 0-dim ТЕНЗОРЫ уровней
        #    нейрохимии, БЕЗ приведения к хосту (совместимость с CUDA Graphs)
        if self.pipeline.bus_plasticity and self.callosum._filtered_left is not None:
            right_idx = [self.neuro.zone_index[n] for n in self.right.zone_names]
            left_idx = [self.neuro.zone_index[n] for n in self.left.zone_names]
            self.pipeline.step_bus(
                self.callosum,
                self.callosum._filtered_left, self.callosum._filtered_right,
                self._to_right, self._to_left,
                da_right=self.neuro.da[right_idx].mean(),
                da_left=self.neuro.da[left_idx].mean(),
            )

        # 7. Нейрохимия: распад DA, давление сна (ур. 20, каноническая Σs)
        self.neuro.step(spike_sums={
            name: outputs[name]["spikes"].detach().sum()
            for name in self.zone_names
        })
        self._awake_steps += 1
        return outputs

    # ------------------------------------------------------------------
    # Поведение: конкуренция, слияние мнений, исход, награда
    # ------------------------------------------------------------------
    @torch.no_grad()
    def behave(self, inputs: Dict[str, torch.Tensor], target: torch.Tensor):
        """Такт поведения: fused-решение Global Workspace + DA-кредит.

        target: [B] — истинный следующий токен (критерий организма).
        Возвращает сводку такта: логиты, fused, alpha, winner, исход, DA.
        """
        outputs = self.awake_step(inputs)
        readout_zones = [n for n, z in self.zones.items() if z.readout is not None]
        logits = {n: self.zones[n].readout_logits() for n in readout_zones}
        order = [n for n in self.zone_names if n in logits]
        # Заявки: уверенность чтения (Soft-Attention, разд. 2.4)
        bids = torch.stack(
            [torch.softmax(logits[n], -1).max(-1).values for n in order], dim=-1)

        # Этап 6 — калибровка заявок WTA протоколом точности: уверенность
        # != компетентность (факт Этапа 5: уверенно-ошибающийся эксперт
        # держит бид и топит fused). Бегущий протокол точности зоны
        # (tau_track тактов, память организма — переживает эпизоды)
        # модулирует её заявку: bid' = bid·(eps + track). Каузальный
        # порядок: заявка калибруется ПРОШЛЫМ протоколом, затем протокол
        # обновляется точностью текущего такта.
        if self.wta_track_calibration:
            order_idx = torch.tensor(
                [self.neuro.zone_index[n] for n in order],
                device=self.zone_track.device)
            bids = bids * (self._track_epsilon + self.zone_track[order_idx])
            correct_per_zone = self._update_track(logits, target, order)
        else:
            correct_per_zone = None

        alpha, winner, _ = self.workspace.compete(bids)
        fused = self.workspace.fuse_logits(alpha, [logits[n] for n in order])

        correct = fused.argmax(dim=-1) == target
        reward = float(correct.float().mean())
        if self.reward_attribution and correct_per_zone is not None:
            # Этап 6 — пер-зонная атрибуция: амплитуда DA-события зоны
            # масштабируется ЕЁ точностью такта (тот же сигнал награды,
            # та же семантика RPE/привыкания — меняется только адресация:
            # успех глобален, но кредит получают точные зоны).
            da = self.pipeline.observe_reward_multi(
                list(self.zone_names), reward,
                zone_weights=correct_per_zone)
        else:
            da = self.observe_outcome(reward)

        ce_fused = F.cross_entropy(fused, target)
        ce = {n: F.cross_entropy(logits[n], target).item() for n in order}
        return dict(
            outputs=outputs, logits=logits, fused=fused, alpha=alpha,
            winner=winner, correct=correct, reward=reward, da=da,
            ce_fused=float(ce_fused), ce=ce, alpha_map={
                n: float(alpha[:, i].mean()) for i, n in enumerate(order)},
        )

    # ------------------------------------------------------------------
    # Этап 6: протокол точности зон (быстрый контекст доверия)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_track(self, logits, target, order) -> Dict[str, float]:
        """EMA-протокол точности: track_z <- track_z + rho·(corr_z - track_z).

        Возвращает точность такта по зонам (python-float: хост-граница
        behave, вне CUDA-графа)."""
        correct_per_zone: Dict[str, float] = {}
        for n in order:
            corr = (logits[n].argmax(dim=-1) == target).float().mean()
            correct_per_zone[n] = float(corr)
            idx = self.neuro.zone_index[n]
            self.zone_track[idx].mul_(1.0 - self._track_rho)
            self.zone_track[idx].add_(self._track_rho * corr)
        return correct_per_zone

    # ------------------------------------------------------------------
    # Этап 5-d: рампа гейна доставки (устранение BUS_APICAL_GAIN)
    # ------------------------------------------------------------------
    def ramp_bus_gain(self, delta: float) -> float:
        """Продвинуть гейн доставки к 1.0 (после периода кредита декодеров).

        Вызывается медленным контуром/демо между эпизодами: декодеры
        приёма уже получили 3-факторный кредит — NMDA-путь получателей
        приучен к масштабу доставки, ручная калибровка 0.2 отмирает.
        """
        self.bus_apical_gain = min(
            self._gain_target, self.bus_apical_gain + float(delta))
        return self.bus_apical_gain

    def observe_outcome(self, reward: float) -> float:
        """Исход организма -> RPE -> DA-импульс ВСЕМ зонам (успех глобален).

        Критик наблюдается ОДИН раз на такт (RPE и обновление базы V —
        единственные: двойной вызов разносил бы базу быстрее канонической
        релаксации tau_v); амплитуда импульса ширководится всем зонам.
        Отрицательный RPE не наказывает (ур. 14): только затухание w_fast.
        """
        return self.pipeline.observe_reward_multi(
            list(self.zone_names), reward)

    # ------------------------------------------------------------------
    # EWC: защита задач Фазы 1 в цикле организма
    # ------------------------------------------------------------------
    def arm_ewc(self, loss_fn, n_samples: int = 64) -> float:
        """Вооружить EWC после периода устойчивости (см. EWCRegularizer).

        loss_fn: замыкание, возвращающее скалярный лосс защищаемой задачи.
        """
        params = {}
        for name, zone in self.zones.items():
            params[f"{name}.syn_in"] = zone.syn_in.w_slow
            params[f"{name}.syn_rec"] = zone.syn_rec.w_slow
            if zone.readout is not None:
                params[f"{name}.readout_w"] = zone.readout.weight
        self._ewc = EWCRegularizer(params, lam=self._lam_ewc)
        return self._ewc.estimate_fisher(loss_fn, n_samples=n_samples)

    @property
    def ewc(self) -> Optional[EWCRegularizer]:
        return self._ewc

    # ------------------------------------------------------------------
    # Сон: полный протокол + перебазирование EWC (решение v4.0.1)
    # ------------------------------------------------------------------
    def sleep_cycle(
        self,
        sample_input: Optional[Dict[str, torch.Tensor]] = None,
        replay_steps: int = 64,
    ) -> Dict[str, float]:
        """Полный цикл сна организма (5 шагов разд. 7.3 + re-baseline EWC)."""
        report = self.sleep_protocol.run_cycle(
            self.zones.values(), self.neuro,
            sample_input=sample_input, replay_steps=replay_steps)
        if self._ewc is not None:
            self._ewc.rebaseline()
            report["ewc_rebaselined"] = 1.0
        return report

    # ------------------------------------------------------------------
    # Телеметрия (медленный контур, разд. 8)
    # ------------------------------------------------------------------
    @property
    def telemetry(self) -> Dict[str, float]:
        tel = dict(self.neuro.telemetry)
        tel.update(
            awake_steps=self._awake_steps,
            callosum_traffic=self.callosum.traffic,
            w_fast_mass=sum(
                float(z.syn_in.w_fast.abs().sum() + z.syn_rec.w_fast.abs().sum())
                for z in self.zones.values()),
            reward_baseline=self.pipeline.critic.baseline,
            ewc_armed=float(self._ewc.is_armed) if self._ewc is not None else 0.0,
        )
        return tel

    @property
    def awake_steps(self) -> int:
        return self._awake_steps
