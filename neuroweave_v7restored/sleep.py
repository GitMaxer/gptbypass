# src/spiker/learning/sleep.py
# Этап 3/4 (канон v4.0/v4.0.1, разд. 7.3): протокол сна — замыкает цикл
# «живого» организма: бодрствование (Tag-and-Capture накапливает w_fast)
# -> сон (консолидация в w_slow, прунинг, сброс нагрузки) -> пробуждение.
#
# Пять шагов протокола (разд. 7.3 канона):
#   1. ПОДАВЛЕНИЕ ВНЕШНИХ AMPA-ВХОДОВ: сенсорный поток заглушается
#      (sleep_input_gain ~ 0.25); сон — эндогенный режим;
#   2. ACh -> МИНИМУМ (ach_sleep = -0.8): ур. (5) переключает баланс на
#      апикальные эндогенные токи (top-down реплеи вместо сенсорики);
#      перевод быстрый — прямой записью буфера (макро-команда медленного
#      контура; релаксация tau_ach=500 заняла бы сотни тактов);
#   3. ГЕНЕРАТИВНЫЙ РЕПЛЕЙ при повышенной энтропии: базовые 5-HT зон
#      временно снижаются (replay_5ht_drop) — шум мембраны растёт до
#      3x (ур. разд. 3.4), зоны реактивируются на ослабленных записанных
#      паттернах; DA НЕТ — Tag-and-Capture не срабатывает, следы только
#      затухают (гомеостаз Y1): сон НЕ должен захватывать случайности;
#   4. МЕДЛЕННАЯ КОНСОЛИДАЦИЯ w_fast -> w_slow: исправленное ур. (21)
#      v4.0.1 (пороговая селекция, StatefulSynapse.consolidate_to_slow);
#   5. ОБНУЛЕНИЕ БУФЕРОВ w_fast/следов + структурный прунинг: синапсы
#      слабее theta_prune (шумовая пыль) снимаются с графа совсем —
#      разрежённость растёт, энергия падает, тракт остаётся.
#
# Пробуждение: ACh -> +0.3 (приоритет входа), давление сна обнуляется,
# модуляторы чистые. EWC перебазируется СНАРУЖИ протокола (SpikerOrganism
# вызывает ewc.rebaseline() после цикла — решение v4.0.1).
from typing import Dict, Iterable, List, Optional

import torch

from ..topology.zone import Zone
from ..modulation.neurochemistry import NeuroChemistry


class SleepProtocol:
    """Оффлайн-консолидация и реплей (разд. 7.3)."""

    def __init__(
        self,
        lam_consol: float = 0.3,
        theta_prune: float = 0.05,
        theta_c: float = 0.03,
        temperature: float = 0.015,
        ach_sleep: float = -0.8,
        ach_wake: float = 0.3,
        sleep_input_gain: float = 0.25,
        replay_5ht_drop: float = 0.5,
    ):
        self.lam_consol = float(lam_consol)
        self.theta_prune = float(theta_prune)
        self.theta_c = float(theta_c)
        self.temperature = float(temperature)
        self.ach_sleep = float(ach_sleep)
        self.ach_wake = float(ach_wake)
        self.sleep_input_gain = float(sleep_input_gain)
        self.replay_5ht_drop = float(replay_5ht_drop)
        # Снапшот базовых 5-HT для восстановления при пробуждении
        self._ht_baseline_backup: Optional[torch.Tensor] = None
        self.asleep = False

    # ------------------------------------------------------------------
    # Шаги 1-2: погружение
    # ------------------------------------------------------------------
    def enter(self, zones: Iterable[Zone], neuro: NeuroChemistry):
        """Перевести организм в сон: ACh -> минимум, энтропия -> вверх."""
        zone_list = list(zones)
        names = [z.name for z in zone_list]
        with torch.no_grad():
            for name in names:
                neuro.set_ach_target(name, self.ach_sleep)
            # Быстрый переход в режим сна (макро-команда, а не релаксация)
            neuro.ach.fill_(self.ach_sleep)
            # Повышенная энтропия реплея: временно снижаем базовые 5-HT
            if self._ht_baseline_backup is None:
                self._ht_baseline_backup = neuro.ht_baseline.clone()
            neuro.ht_baseline.mul_(self.replay_5ht_drop)
            neuro.ht.copy_(neuro.ht_baseline)
            # Модуляторы бодрствования гаснут
            neuro.da.zero_()
            neuro.ne.zero_()
        self.asleep = True

    # ------------------------------------------------------------------
    # Шаг 3: генеративный реплей (эндогенная реактивация)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def replay(
        self,
        zones: Iterable[Zone],
        neuro: NeuroChemistry,
        sample_input: Optional[Dict[str, torch.Tensor]] = None,
        steps: int = 64,
        apical: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Эндогенная реактивация: зоны гоняются на заглушенном входе.

        sample_input: имя зоны -> записанный паттерн входа (средний снимок
        бодрствования); при отсутствии — лёгкий эндогенный шум того же
        масштаба. Вход ослаблен sleep_input_gain (шаг 1), ACh на минимуме
        (шаг 2), 5-HT снижен (энтропия). Пластичность без DA: следы
        затухают, w_fast релаксирует — реплей НЕ захватывает случайности.
        """
        zone_list = list(zones)
        if not zone_list:
            return dict(replay_steps=0, replay_spikes=0.0)
        batch = next(iter(zone_list)).spike_accum.shape[0] \
            if zone_list[0].spike_accum is not None else 1
        device = neuro.da.device

        # Заглушенный вход: записанный паттерн либо эндогенный шум
        inputs: Dict[str, torch.Tensor] = {}
        for z in zone_list:
            if sample_input is not None and z.name in sample_input:
                base = sample_input[z.name].detach().to(device)
                if base.shape[0] != batch:
                    base = base[:batch] if base.shape[0] > batch \
                        else base.repeat(batch, 1)
                inputs[z.name] = base * self.sleep_input_gain
            else:
                inputs[z.name] = torch.randn(
                    batch, z.syn_in.in_features, device=device
                ) * 0.1 * self.sleep_input_gain

        replay_spikes = 0.0
        for _ in range(int(steps)):
            for z in zone_list:
                out = z.step(
                    inputs[z.name],
                    s_apical=None if apical is None else apical.get(z.name),
                    ach=neuro.get("ach", z.name, batch),
                    ne=neuro.get("ne", z.name, batch),
                    serotonin_5ht=neuro.get("ht", z.name, batch),
                )
                # Пластичность сна: БЕЗ награды — только следы и затухание
                z.syn_in.update_traces_and_plasticity(
                    inputs[z.name].detach(), out["spikes"].detach(),
                    da_reward=None,
                )
                z.syn_rec.update_traces_and_plasticity(
                    z.s_exc_prev.detach(), out["spikes"].detach(),
                    da_reward=None,
                )
                replay_spikes += float(out["spikes"].sum())
            neuro.step(spike_sums={z.name: 0.0 for z in zone_list})  # DA=0, NE распад

        return dict(replay_steps=int(steps), replay_spikes=replay_spikes)

    # ------------------------------------------------------------------
    # Шаг 4: медленная консолидация (исправленное ур. 21)
    # ------------------------------------------------------------------
    def consolidate(self, zones: Iterable[Zone]) -> float:
        """Перенос подтвердившихся быстрых весов в долговременные.

        Калибровка Этапа 3 (зонд живого цикла, lr_fast = 2e-3): масштаб
        Tag-and-Capture с событийными импульсами и синапсо-специфичной
        нормировкой (патч S6) — пики w_fast ~0.03-0.1 за период
        бодрствования. Селекция theta_c = 0.03, T = 0.015, lam = 0.3:
        переносится только МНОГОКРАТНО подтверждённое (порог отсекает
        одноразовый шум и фоновую диффузию), за 5-10 циклов сна
        накапливается значимая консолидированная структура.

        Возвращает массу перенесённого |lam * g * w_fast| (телеметрия).
        """
        transferred = 0.0
        for z in list(zones):
            for syn in (z.syn_in, z.syn_rec):
                before_fast = syn.w_fast.abs().sum().item()
                syn.consolidate_to_slow(
                    lam_consol=self.lam_consol,
                    theta_c=self.theta_c,
                    temperature=self.temperature,
                )
                after_fast = syn.w_fast.abs().sum().item()
                transferred += (before_fast - after_fast)
        return transferred

    # ------------------------------------------------------------------
    # Шаг 5: прунинг + обнуление оперативной памяти
    # ------------------------------------------------------------------
    def prune(self, zones: Iterable[Zone]) -> Dict[str, float]:
        """Структурный прунинг слабых путей и сброс буферов w_fast/следов.

        Слабые синапсы |w_slow| < theta_prune — шумовая пыль необученной
        инициализации и полузатухших флуктуаций — снимаются с графа:
        разрежённость растёт, тракт остаётся. Возвращает метрики.
        """
        pruned = 0
        total = 0
        w_fast_mass = 0.0
        for z in list(zones):
            for syn in (z.syn_in, z.syn_rec):
                with torch.no_grad():
                    mask = syn.w_slow.abs() < self.theta_prune
                    pruned += int(mask.sum())
                    total += mask.numel()
                    syn.w_slow.data[mask] = 0.0
                    # Обнуление оперативных буферов (канон, шаг 5)
                    syn.w_fast.zero_()
                    w_fast_mass += float(syn.w_fast.abs().sum())
                    # Следы пригодности: ленивая ре-инициализация guard-ом
                    syn.z_pre = syn.z_post = syn.e_pre = syn.e_post = None
        return dict(
            pruned=pruned,
            total=total,
            pruned_frac=pruned / max(1, total),
            w_fast_mass=w_fast_mass,
        )

    # ------------------------------------------------------------------
    # Пробуждение
    # ------------------------------------------------------------------
    def wake(self, zones: Iterable[Zone], neuro: NeuroChemistry):
        """Вернуть организм в бодрствование: ACh -> +0.3, P_sleep = 0."""
        with torch.no_grad():
            for z in list(zones):
                neuro.set_ach_target(z.name, self.ach_wake)
            neuro.ach.fill_(self.ach_wake)
            if self._ht_baseline_backup is not None:
                neuro.ht_baseline.copy_(self._ht_baseline_backup)
                neuro.ht.copy_(self.ht_wake_fill(neuro))
            neuro.da.zero_()
            neuro.ne.zero_()
            neuro.sleep_pressure.zero_()
        self._ht_baseline_backup = None
        self.asleep = False

    @staticmethod
    def ht_wake_fill(neuro: NeuroChemistry) -> torch.Tensor:
        return neuro.ht_baseline.clone()

    # ------------------------------------------------------------------
    # Полный цикл (шаги 1-5 + пробуждение)
    # ------------------------------------------------------------------
    def run_cycle(
        self,
        zones: Iterable[Zone],
        neuro: NeuroChemistry,
        sample_input: Optional[Dict[str, torch.Tensor]] = None,
        replay_steps: int = 64,
    ) -> Dict[str, float]:
        """Полный протокол сна. Возвращает отчёт для телеметрии организма."""
        zone_list = list(zones)
        report: Dict[str, float] = dict(
            p_sleep_before=float(neuro.sleep_pressure),
            w_fast_mass_before=sum(
                float(z.syn_in.w_fast.abs().sum() + z.syn_rec.w_fast.abs().sum())
                for z in zone_list
            ),
        )

        self.enter(zone_list, neuro)                                   # шаги 1-2
        rep = self.replay(zone_list, neuro, sample_input, replay_steps)  # шаг 3
        report["consolidated_mass"] = self.consolidate(zone_list)       # шаг 4
        pr = self.prune(zone_list)                                     # шаг 5
        report.update(pr)
        self.wake(zone_list, neuro)                                    # пробуждение

        report.update(
            replay_steps=rep["replay_steps"],
            replay_spikes=rep["replay_spikes"],
            p_sleep_after=float(neuro.sleep_pressure),
            w_fast_mass_after=sum(
                float(z.syn_in.w_fast.abs().sum() + z.syn_rec.w_fast.abs().sum())
                for z in zone_list
            ),
            ach=float(neuro.ach.mean()),
        )
        return report
