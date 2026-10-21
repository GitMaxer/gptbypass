"""Конфигурация NEUROWEAVE 8.

Все величины — в мВ и мс. Проводимости безразмерны и нормированы на утечку
g_L = 1. Конфигурация — чистые dataclass'ы: никакого импорта torch, чтобы
её можно было сериализовать в YAML/JSON и использовать в отчётах.

Ключевое отличие от v6/v7: единица топологии — не «зона», а ЛАМИНАРНАЯ
КОЛОНКА (канонический микроконтур: L4 -> L2/3 -> L5 -> L6, плюс PV/SST
интернейроны). Область (area) = одна колонка; полушарие = набор областей.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Биофизика одного нейрона
# ---------------------------------------------------------------------------
@dataclass
class NeuronCfg:
    """Параметры компартментного ALIF-нейрона.

    Проводимостная схема (не токовая!): мембрана интегрирует ПРОВОДИМОСТИ,
    поэтому торможение с E_Cl >= V_rest физически не может увести мембрану
    ниже покоя — это инвариант, а не подгонка (см. MATH в core/transmitters.py).
    """

    v_rest: float = -70.0
    v_th: float = -50.0          # порог E-клеток
    v_reset: float = -58.0
    v_floor: float = -90.0       # предохранитель
    e_exc: float = 0.0           # реверсия AMPA/NMDA
    e_cl: float = -70.0          # реверсия GABA; >= v_rest => чистый шунт
    ecl_mode: str = "homeostat"  # 'homeostat' (KCC2) | 'raw' (патология)

    tau_m: float = 8.0
    tau_ampa: float = 1.5
    tau_nmda: float = 100.0
    tau_gaba: float = 8.0
    tau_a: float = 100.0
    beta_adapt: float = 0.15
    r_m: float = 1.0
    dt: float = 1.0

    mg: float = 1.0              # мМ, блок Mg2+ (Jahr-Stevens)
    theta_mg: float = 0.25       # порог базальной деполяризации для снятия блока
    k_mg: float = 12.0           # крутизна гейта совпадения
    alpha_nmda: float = 1.0      # вес апикального NMDA-пути в соме
    gamma_ach: float = 0.5       # глубина ACh-модуляции проводимостей
    k_ach_th: float = 2.0        # мВ/ед. ACh — сдвиг порога (возбудимость)

    k_surrogate: float = 10.0
    surrogate: str = "fast_sigmoid"   # | "arctan"
    refractory: int = 2
    stochastic: bool = True
    sigma: float = 0.001


@dataclass
class ColumnCfg:
    """Ламинарная колонка: 4 возбуждающих слоя + 2 класса интернейронов."""

    name: str
    side: str = "left"           # 'left' | 'right'
    n_in: int = 32
    n_l4: int = 24
    n_l23: int = 48
    n_l5: int = 32
    n_l6: int = 24
    n_pv: int = 16               # перисоматическое торможение (делительный шунт)
    n_sst: int = 8               # дендритное торможение апикального пути
    n_out: Optional[int] = 16    # голова чтения (None — без головы)

    input_gain: float = 1.0
    wm_slots: int = 7            # слотов θ-γ рабочей памяти (0 = выкл)
    density: float = 0.35        # плотность разреженных масок
    inh_strength: float = 0.45
    readout_decay: float = 0.8
    predict_horizon: int = 1     # горизонт локального предсказания (такты)

    def pools(self) -> Dict[str, int]:
        return {
            "l4": self.n_l4, "l23": self.n_l23, "l5": self.n_l5, "l6": self.n_l6,
            "pv": self.n_pv, "sst": self.n_sst,
        }

    def n_neurons(self) -> int:
        return sum(self.pools().values())


@dataclass
class PhaseCfg:
    """Тета-гамма связывание (рабочая память) и фазовая VSA (структура)."""

    n_slots: int = 7             # ёмкость рабочей памяти (7 +/- 2)
    gamma_ticks: int = 4         # тактов на один гамма-цикл
    vsa_dim: int = 512           # размерность фазовых векторов
    vsa_codebook: int = 256      # символов в кодовой книге
    cleanup_sharpen: float = 8.0


@dataclass
class HippoCfg:
    """Гиппокампальный эпизодический буфер (DG -> CA3 -> CA1)."""

    n_dg: int = 512              # экспансия (паттерн-сепарация)
    dg_sparsity: float = 0.04    # доля активных DG-клеток
    ca3_fan_in: float = 0.06     # плотность рекуррентности CA3
    ca3_steps: int = 4           # итераций аттрактора
    ca3_gain: float = 1.6
    n_slots: int = 64            # эпизодических слотов
    novelty_threshold: float = 0.25
    decay: float = 0.999


@dataclass
class ModCfg:
    """Нейрохимия: тензорные поля модуляторов + обучаемый мета-контроллер."""

    da_decay: float = 0.94
    ht_decay: float = 0.999
    ach_decay: float = 0.995
    ne_decay: float = 0.85
    ne_pulse_max: float = 1.0
    diffusion: float = 0.02      # межобластная диффузия модуляторов
    basal: Tuple[float, float, float, float] = (0.0, 0.55, 0.45, 0.0)
    #                                                       DA  5-HT  ACh  NE
    ctrl_lr: float = 0.02
    ctrl_sigma: float = 0.35
    rate_target: float = 0.08
    rate_band: float = 0.06


@dataclass
class PlasticityCfg:
    lr_fast: float = 3e-3
    lr_slow: float = 2e-3
    a_plus: float = 1.0
    a_minus: float = 0.8
    tau_pre: float = 20.0
    tau_post: float = 20.0
    tau_elig: float = 100.0
    w_fast_max: float = 0.5
    w_fast_decay: float = 0.9995
    w_max: float = 1.2
    theta_consol: float = 0.25   # порог доверия консолидации (в долях w_fast_max)
    temp_consol: float = 0.02
    eta_consol: float = 0.35
    pred_lr: float = 0.02        # локальное предсказание (predictive coding)
    pred_gain: float = 1.0       # вес ошибки предсказания в пластичности


@dataclass
class StructuralCfg:
    """Структурная пластичность: синаптогенез, прунинг, критический период."""

    enabled: bool = True
    max_edges_frac: float = 1.5      # потолок числа рёбер к исходному
    grow_rate: float = 0.02          # доля кандидатов за событие роста
    prune_threshold: float = 0.004
    grace_ticks: int = 400           # молодые синапсы не прунятся
    critical_ticks: int = 6000       # закрытие критического периода
    ne_reopen: float = 0.6           # NE вновь открывает окно (стресс)
    growth_cost: float = 1e-4


@dataclass
class WorkspaceCfg:
    temperature: float = 0.5
    lateral_gain: float = 1.2
    storm_rate: float = 0.35
    storm_gain: float = 2.5
    competence_rho: float = 0.12     # EMA компетентности (метакогниция)
    competence_floor: float = 0.1


@dataclass
class CallosumCfg:
    delay: int = 7
    density: float = 0.25
    gaba_gain: float = 0.8
    quantize: bool = False           # int8/тернарная шина (Этап-5-совместимо)


@dataclass
class SleepCfg:
    pressure_gain: float = 2e-4
    threshold: float = 1.0
    n1_ticks: int = 8
    n2_ticks: int = 16
    n3_ticks: int = 24
    rem_ticks: int = 12
    ripple_batch: int = 8
    pr_frac: float = 0.02


@dataclass
class OrganismCfg:
    """Полная конфигурация организма."""

    neuron: NeuronCfg = field(default_factory=NeuronCfg)
    phase: PhaseCfg = field(default_factory=PhaseCfg)
    hippo: HippoCfg = field(default_factory=HippoCfg)
    mod: ModCfg = field(default_factory=ModCfg)
    plasticity: PlasticityCfg = field(default_factory=PlasticityCfg)
    structural: StructuralCfg = field(default_factory=StructuralCfg)
    workspace: WorkspaceCfg = field(default_factory=WorkspaceCfg)
    callosum: CallosumCfg = field(default_factory=CallosumCfg)
    sleep: SleepCfg = field(default_factory=SleepCfg)

    vocab: int = 16
    blank: int = 16
    columns: List[ColumnCfg] = field(default_factory=list)
    # имя области-адресата -> (источник, слой-источник, слой-адресат, вид)
    # вид: 'ff' (базальный, кормит L4) | 'fb' (апикальный, кормит L1/L5)
    routes: List[Tuple[str, str, str, str]] = field(default_factory=list)
    seed: int = 8
    device: str = "cpu"

    def col(self, name: str) -> ColumnCfg:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def by_side(self, side: str) -> List[ColumnCfg]:
        return [c for c in self.columns if c.side == side]

    def total_neurons(self) -> int:
        return sum(c.n_neurons() for c in self.columns)

    def replace(self, **kw) -> "OrganismCfg":
        return replace(self, **kw)


def _base_columns(vocab: int) -> List[ColumnCfg]:
    """Три области PoC-масштаба: сенсорная, структурная, контекстная.

    Дивергенция полушарий — в пресетах нейрона (tau_m, sigma, суррогат),
    а не в размерах: копии одинакового размера, но физически разные.
    """
    return [
        ColumnCfg(
            name="A", side="left", n_in=vocab * 2, n_out=vocab,
            n_l4=24, n_l23=48, n_l5=32, n_l6=24, n_pv=16, n_sst=8,
            input_gain=2.0,
        ),
        ColumnCfg(
            name="B", side="left", n_in=32, n_out=vocab,
            n_l4=24, n_l23=48, n_l5=32, n_l6=24, n_pv=16, n_sst=8,
            input_gain=1.4,
        ),
        ColumnCfg(
            name="C", side="right", n_in=32, n_out=vocab,
            n_l4=24, n_l23=48, n_l5=32, n_l6=24, n_pv=16, n_sst=8,
            input_gain=1.4,
        ),
    ]


def default_cfg(preset: str = "poc") -> OrganismCfg:
    """Готовые пресеты.

    poc   — рабочий стенд по умолчанию (быстрый на CPU);
    small — для юнит-тестов (секунды);
    wide  — демонстрация масштабирования (постоянный fan-in).
    """
    if preset == "small":
        n = NeuronCfg(tau_m=8.0, sigma=0.0, k_surrogate=10.0, refractory=1)
        cols = [
            ColumnCfg("A", "left", n_in=8, n_out=4, n_l4=6, n_l23=12, n_l5=8,
                      n_l6=6, n_pv=4, n_sst=2, input_gain=2.0),
            ColumnCfg("C", "right", n_in=12, n_out=4, n_l4=6, n_l23=12, n_l5=8,
                      n_l6=6, n_pv=4, n_sst=2, input_gain=1.4),
        ]
        return OrganismCfg(
            neuron=n, columns=cols, vocab=8, blank=8,
            hippo=HippoCfg(n_dg=128, n_slots=16),
            phase=PhaseCfg(vsa_dim=128, vsa_codebook=64, n_slots=5),
            routes=[("C", "A", "l23", "ff")],
            structural=StructuralCfg(enabled=False),
        )

    if preset == "wide":
        base = default_cfg("poc")
        cols = []
        for c in base.columns:
            cols.append(replace(
                c, n_l4=c.n_l4 * 4, n_l23=c.n_l23 * 4, n_l5=c.n_l5 * 4,
                n_l6=c.n_l6 * 4, n_pv=c.n_pv * 4, n_sst=c.n_sst * 4))
        return replace(base, columns=cols)

    cols = _base_columns(16)
    routes = [
        ("B", "A", "l23", "ff"),   # A -> B: прямой путь (базальный, L4)
        ("C", "A", "l5", "ff"),    # A -> C: сырые спайки выхода (правое НЕ видит токены)
        ("A", "C", "l6", "fb"),    # C -> A: контекст на апикальные дендриты L1/L5
    ]
    return OrganismCfg(columns=cols, routes=routes)
