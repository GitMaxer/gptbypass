# src/spiker/learning/stdp.py
# Этап 3 (канон v4.0/v4.0.1, разд. 5.2 + 7.2): пайплайн Reward-STDP Фазы 2.
#
# Ядро 3-факторного правила (Tag-and-Capture) живёт в core/synapse.py
# (update_traces_and_plasticity): следы пригодности O(N) маркируются
# спайковой корреляцией, а дофамин ЗАХВАТЫВАЕТ маркированное. Этот модуль
# собирает над ядром organism-level контур Фазы 2:
#
#   1. RewardCritic — бегущая база V (частота успеха) и RPE = R − V (ур. 14);
#      привыкание: при росте мастерства одинаковый успех даёт всё меньший DA
#      (навык перестаёт удивлять — организм экономит нейромодуляцию);
#   2. событийные DA-импульсы в NeuroChemistry (da_pulse, распад tau_da=100);
#   3. пошаговый опрос уровня DA зоны (гейт синаптической пластичности);
#   4. опциональная пластичность проекций Callosum (W_l2r/W_r2l) —
#      транзитный 3-факторный кредит: усиливаются пути, активные в момент
#      награды получателя.
#
# Контракт метрики табл. 9 (валидируется tests/test_reward_stdp.py):
#   * при DA > 0: |w_fast| на коррелированных путях РАСТЁТ;
#   * при DA = 0: роста нет (гомеостаз Y1 только затухает);
#   * задержанная награда захватывается живыми следами (tau_elig=100 > delay).
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from ..modulation.neurochemistry import NeuroChemistry


class RewardCritic:
    """
    Критик организма: RPE = R − V (ур. 14), V — бегущая частота успеха.

    V обновляется экспоненциальным средним по ТАКТАМ (tau_v >> длины
    эпизода): при редких успехах база мала (успех «удивляет» — сильный DA),
    при высокой частоте база догоняет успех (навык привычен — DA гаснет).
    Отрицательный RPE НЕ наказывает: ур. 14 — DA = max(0, RPE), неудачи не
    разрушают собранные связи (только естественное затухание w_fast).
    """

    def __init__(self, tau_v: float = 150.0, dt: float = 1.0, v_init: float = 0.0):
        self.v = float(v_init)
        self._rate = 1.0 - math.exp(-dt / tau_v)

    def reset(self, v_init: float = 0.0):
        self.v = float(v_init)

    def rpe(self, reward: float) -> float:
        """Наблюдаемая награда такта (0/1 или scalar) -> RPE; обновляет базу V."""
        r = float(reward)
        rpe = r - self.v
        self.v += self._rate * (r - self.v)
        return rpe

    @property
    def baseline(self) -> float:
        return self.v


class RewardSTDPPipeline:
    """
    Пайплайн Фазы 2: Tag-and-Capture поверх StatefulSynapse.

    Поток такта бодрствования (вызывает сборка организма):
        1. зоны делают шаг (спайки s_pre/s_post);
        2. pipeline.step_zone(...) обновляет следы и пластичность syn_in/
           syn_rec под гейтом уровня DA зоны (из NeuroChemistry);
        3. по исходу такта (эпизода) pipeline.observe_reward(...) вычисляет
           RPE и вносит событийный DA-импульс (da_pulse) — с задержкой
           распада tau_da импульс живёт ~100 тактов и захватывает следы,
           помеченные ДО награды (Tag-and-Capture, ур. 11-12).
    """

    def __init__(
        self,
        neuro: NeuroChemistry,
        lr_fast: float = 0.01,
        a_plus: float = 1.0,
        a_minus: float = 0.8,
        tau_v: float = 150.0,
        lr_bus: float = 0.02,
        bus_amp: float = 0.3,
        bus_plasticity: bool = True,
        lr_dec: float = 0.05,
    ):
        self.neuro = neuro
        self.critic = RewardCritic(tau_v=tau_v)
        self.lr_fast = float(lr_fast)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.lr_bus = float(lr_bus)
        self.bus_amp = float(bus_amp)
        self.bus_plasticity = bool(bus_plasticity)
        # Темп кредита декодеров приёма SON (Этап 5-d)
        self.lr_dec = float(lr_dec)
        # Отложенные DA-импульсы по зонам (событийная семантика S4) живут
        # на УСТРОЙСТВЕ — нейрохимия pending_da [Z] (патч Этапа 5,
        # capture-ready): observe_reward накапливает амплитуду события
        # такта, step_zone раздаёт её в пластичность СЛЕДУЮЩЕГО такта
        # (биологическая задержка выброса ~ 1 такт) и обнуляет — тензорными
        # операциями, без хост-словаря. Вне событий импульс 0 — роста нет.
        # Телеметрия (опрос — редкий, медленный контур)
        self.stats: Dict[str, float] = dict(
            reward_events=0.0, da_total=0.0, rpe_last=0.0, baseline=0.0
        )

    # ------------------------------------------------------------------
    # Событие награды (эпизодический уровень)
    # ------------------------------------------------------------------
    def observe_reward(self, zone, reward) -> float:
        """Исход организма -> RPE -> положительный DA-импульс зоне.

        Возвращает величину DA-импульса (0 при RPE <= 0 — ур. 14).
        zone: имя зоны (строка) — куда адресован кредит.
        """
        rpe = self.critic.rpe(reward)
        da = max(0.0, rpe)
        if da > 0.0:
            self.neuro.da_pulse(zone, da)
            self.neuro.da_pending_add(zone, da)
        self.stats["reward_events"] += 1.0
        self.stats["da_total"] += da
        self.stats["rpe_last"] = rpe
        self.stats["baseline"] = self.critic.baseline
        return da

    def observe_reward_multi(self, zones, reward, zone_weights=None) -> float:
        """Глобальная награда организма: критик наблюдает ОДИН раз,
        амплитуда RPE раздаётся всем зонам (успех глобален).

        Единственное наблюдение критика на такт — каноническая
        релаксация базы V (tau_v); вызов observe_reward на каждую зону
        отдельно разносил бы базу в N раз быстрее.

        Этап 6 — пер-зонная атрибуция (zone_weights: dict зона -> вес в
        [0, 1]): амплитуда импульса зоны масштабируется ЕЁ вкладом в
        успех такта (точность чтения зоны). Тот же RPE, та же база V —
        адресация кредита вместо широковещания. None (дефолт) —
        каноническая широковещательная семантика Этапов 1-5.
        Закрытие открытого вопроса Этапа 3: глобальный DA-кредит давал
        уверенно-ошибающимся экспертам усиление (демо: зонд ЭКЗАМЕНА
        Этапа 6, probe_stage6_exam.py).
        """
        rpe = self.critic.rpe(reward)
        da = max(0.0, rpe)
        if da > 0.0:
            for zone in zones:
                amount = da if zone_weights is None \
                    else da * float(zone_weights.get(zone, 1.0))
                if amount > 0.0:
                    self.neuro.da_pulse(zone, amount)
                    self.neuro.da_pending_add(zone, amount)
        self.stats["reward_events"] += 1.0
        self.stats["da_total"] += da
        self.stats["rpe_last"] = rpe
        self.stats["baseline"] = self.critic.baseline
        return da

    # ------------------------------------------------------------------
    # Такт пластичности зон (импульсная семантика DA — патч S4)
    # ------------------------------------------------------------------
    def step_zone(self, zone, s_input, s_exc, s_exc_prev, lr_fast: Optional[float] = None,
                  impulse: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Обновить syn_in/syn_rec зоны под DA-импульсом события.

        Порядок вызова (сборка организма): ПОСЛЕ zone.step, при этом
        s_exc_prev — спайки ПРЕДЫДУЩЕГО такта (сохранённые сборкой ДО
        шага зоны: zone.s_exc_prev внутри Zone.step уже перезаписан).

        Патч S4 (канон v4.0.1): DA-гейтинг — СОБЫТИЙНЫЙ (импульс события,
        ур. 12), а не удерживаемый уровень — см. CHANGELOG. Патч Этапа 5
        (capture-ready): импульс — 0-dim ТЕНЗОР на устройстве
        (neuro.da_pending_consume); нормировка S6 — целиком тензорная,
        без float()-синхронизаций с хостом (совместимость с CUDA Graphs).

        Возвращает потреблённый импульс (сырой 0-dim тензор): его можно
        раздать другим потребителям того же события (обучение
        SON-декодеров приёма, Этап 5).
        """
        lr = self.lr_fast if lr_fast is None else float(lr_fast)
        if impulse is None:
            impulse = self.neuro.da_pending_consume(zone.name)
        batch = int(s_input.shape[0])
        dtype = s_input.dtype

        # Патч S6 (канон v4.0.1): нормировка импульса на масштаб следа —
        # СИНАПСОСПЕЦИФИЧНАЯ, теперь ПОЛНОСТЬЮ на устройстве (Этап 5):
        # следы e_pre — интегралы активности, стационар ~ tau_elig*rate;
        # у syn_in активна малая доля входов (e ~ 5), у syn_rec — весь пул
        # (e ~ 40); общий импульс раздувал рекуррентный синапс. Дофамин
        # взвешивает ОТКЛОНЕНИЯ от фона, не стационар. clamp(max=1) —
        # тензорный аналог прежнего min(..., 1.0): при импульсе 0
        # обновление естественно нулевое (принцип Y5). След может ещё не
        # существовать (до первого такта синапса) — масштаб 0.
        e_in = zone.syn_in.e_pre
        e_rec = zone.syn_rec.e_pre
        mean_in = e_in.detach().abs().mean() if e_in is not None \
            else torch.zeros_like(impulse)
        mean_rec = e_rec.detach().abs().mean() if e_rec is not None \
            else torch.zeros_like(impulse)
        imp_in = (impulse / (1.0 + mean_in)).clamp(max=1.0)
        imp_rec = (impulse / (1.0 + mean_rec)).clamp(max=1.0)
        da_in = imp_in.to(dtype).view(1, 1).expand(batch, 1)
        da_rec = imp_rec.to(dtype).view(1, 1).expand(batch, 1)

        zone.syn_in.update_traces_and_plasticity(
            s_input.detach(), s_exc.detach(), da_reward=da_in,
            lr_fast=lr, a_plus=self.a_plus, a_minus=self.a_minus,
        )
        zone.syn_rec.update_traces_and_plasticity(
            s_exc_prev.detach(), s_exc.detach(), da_reward=da_rec,
            lr_fast=lr, a_plus=self.a_plus, a_minus=self.a_minus,
        )
        return impulse

    # ------------------------------------------------------------------
    # Кредит декодеров приёма SON (Этап 5-d: устранение BUS_APICAL_GAIN)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step_decoder(
        self,
        zone,
        z_chunk: torch.Tensor,
        s_exc: torch.Tensor,
        impulse: torch.Tensor,
        lr_dec: Optional[float] = None,
        amp_dec: float = 0.5,
        decay_dec: float = 1.0e-3,
    ):
        """3-факторный кредит декодера приёма зоны (ур. 12 на W_dec).

        Декодер приёма (SONBus.decoder, ур. 16) переводит латент шины в
        апикальный ток зоны-получателя. До Этапа 5 его масштаб страховал
        BUS_APICAL_GAIN=0.2: необученный декодер в полную силу раскачивает
        NMDA-интегратор (tau=40) получателей (калибровка Этапа 2).

        Кредит (канон 5.2, симметрия ур. 12):
            пре  = доставленный чанк z (сообщение шины для зоны);
            пост = спайки получателя В ЭТОМ такте — апикальный вклад
                   участвует в них через NMDA-гейт (без базального входа
                   гейт не откроется — кредит не терпит спонтанных спайков);
            фактор = DA-импульс события (тот же, что у синапсов зоны).

        Гомеостаз (ур. Y1, симметрия w_fast): каждый такт — медленный
        распад W_dec; рост — только под DA-импульсом, нормирован на фон
        активности (S6) и ограничен клэмпом. Равновесие распада и gated
        роста держит декодер на масштабе ПОЛЕЗНОЙ доставки — путь
        «декодер обучен» = рампа гейна к 1.0 без раскачки получателей
        (устранение BUS_APICAL_GAIN, открытый вопрос Этапа 2).
        """
        lr = self.lr_dec if lr_dec is None else float(lr_dec)
        W = zone.son.decoder.weight                     # [n_zone, n_son]
        batch = int(s_exc.shape[0])
        # гомеостатический распад (каждый такт, вне событий — Y1-симметрия)
        W.mul_(1.0 - float(decay_dec))
        # средние масштабы активности (тензорно, без хост-синхронизаций)
        mean_s = s_exc.detach().abs().mean()
        mean_z = z_chunk.detach().abs().mean()
        # коактивация: einsum пост×пре, нормированная на фон (S6-урок)
        co = torch.einsum(
            "bi,bj->ij", s_exc.detach(), z_chunk.detach()) / max(batch, 1)
        delta = lr * impulse * co / ((1.0 + mean_s) * (1.0 + mean_z))
        W.add_(delta)
        W.clamp_(-float(amp_dec), float(amp_dec))

    # ------------------------------------------------------------------
    # Пластичность проекций Callosum (транзитный 3-факторный кредит)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step_bus(
        self,
        callosum,
        filtered_left: torch.Tensor,
        filtered_right: torch.Tensor,
        to_right: torch.Tensor,
        to_left: torch.Tensor,
        da_right: Union[float, torch.Tensor],
        da_left: Union[float, torch.Tensor],
    ):
        """Транзитный кредит проекций шины (W_l2r/W_r2l).

        PoC-решение Этапа 3 (зарегистрировано в CHANGELOG): пост-активностью
        проекции считается сам транзит (сообщение, доставленное получателю),
        пре-активностью — отфильтрованный пакет; DA получателя — третий
        фактор. Усиливаются пути, доставившие трафик незадолго до награды.
        Амплитудный клэмп сохраняет устойчивость (аналог w_fast_max).

        Патч Этапа 5 (capture-ready): DA-факторы — 0-dim ТЕНЗОРЫ (уровни
        нейрохимии, без приведения к хосту); ветвлений по данным нет —
        при DA = 0 обновление естественно нулевое (принцип Y5).
        """
        if not self.bus_plasticity:
            return
        b = float(filtered_left.shape[0])
        da_r = da_right if isinstance(da_right, torch.Tensor) \
            else torch.tensor(float(da_right))
        da_l = da_left if isinstance(da_left, torch.Tensor) \
            else torch.tensor(float(da_left))
        da_r_pos = torch.clamp(da_r.detach().float(), min=0.0)
        da_l_pos = torch.clamp(da_l.detach().float(), min=0.0)
        # L2R: получатель — правое полушарие (кредит по DA правого борта)
        delta = self.lr_bus * da_r_pos * torch.einsum(
            "bo,bi->oi", to_right.detach(), filtered_left.detach()) / b
        callosum.W_l2r.add_(delta * callosum.mask_l2r)
        callosum.W_l2r.clamp_(-self.bus_amp, self.bus_amp)
        # R2L: получатель — левое полушарие
        delta = self.lr_bus * da_l_pos * torch.einsum(
            "bo,bi->oi", to_left.detach(), filtered_right.detach()) / b
        callosum.W_r2l.add_(delta * callosum.mask_r2l)
        callosum.W_r2l.clamp_(-self.bus_amp, self.bus_amp)

    # ------------------------------------------------------------------
    def reset_stats(self):
        self.stats = dict(
            reward_events=0.0, da_total=0.0, rpe_last=0.0, baseline=0.0
        )

    def reset_episode(self):
        """Новый эпизод: отложенные импульсы гаснут (события не переносятся)."""
        self.neuro.reset_pending()
