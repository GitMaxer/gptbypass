"""Конфигурация NEUROWEAVE v6.0.

Все биофизические константы в мВ / мс. Проводимости нормированы
на утечку g_L = 1 (безразмерные).
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MembraneConfig:
    """Мембрана + рецепторы. Ключ Этапа 6: E_Cl и v_floor."""
    v_rest: float = -70.0        # мВ, потенциал покоя
    v_th: float = -50.0          # мВ, порог спайка (E-клетки)
    v_th_i_offset: float = 8.0   # мВ, надбавка порога интернейронов
    v_reset: float = -58.0       # мВ, сброс после спайка
    v_floor: float = -90.0       # жёсткий предохранитель; протокол сна ставит = v_rest
    tau_m: float = 4.0           # мс (быстрая кортикальная мембрана)
    dt: float = 1.0              # мс
    e_exc: float = 0.0           # мВ, реверсия AMPA/NMDA
    e_cl: float = -70.0          # мВ, реверсия GABA (= v_rest -> чистый шунт)
    ecl_mode: str = "homeostat"  # 'homeostat' (KCC2-насос) | 'raw' (патология для демо)
    mg: float = 1.0              # mM Mg2+ для NMDA (Jahr–Stevens)
    tau_ampa: float = 1.5        # мс (очень короткая память: окна токенов не «протекают»)
    tau_nmda: float = 100.0      # мс
    tau_gaba: float = 8.0        # мс
    refractory: int = 3          # шагов
    sigma_ht: float = 0.6        # мВ, базовый шум (модулируется 5-HT)
    k_ach_th: float = 6.0        # мВ на ед. ACh — соматическая возбудимость
    nmda_frac: float = 0.3       # доля NMDA в возбуждающих связях


@dataclass
class ZoneSpec:
    """Зона: имя, размер, роль. Изоляция зон — по построению топологии."""
    name: str
    n_e: int                     # возбуждающие клетки
    n_i: int                     # интернейроны (GABA, локальные)
    role: str = ""
    fan_in_rec: int = 16         # рекуррентный fan-in (постоянен при масштабировании)
    fan_in_i: int = 10           # fan-in I->E


@dataclass
class WTACfg:
    """WTA: строб маршрутизации по зонам + NE-заострение."""
    k_ctx: float = 3.0           # вес быстрого контекста в логитах WTA
    rate_ref: float = 0.15       # нормировка энергий зон
    temp: float = 0.5            # температура softmax
    k_ne: float = 2.0            # усиление контраста пульсом NE
    kwt_k_frac: float = 1 / 12.  # доля E-клеток зоны, проходящих k-WTA
    kwt_k_min: int = 8
    rate_alpha: float = 0.3      # EMA частот зон
    ctx_ema_rho: float = 0.35    # скорость следа контекста (быстрый контур)
    ne_k_det: float = 4.0        # детектор переключения: |ctx - ctx_ema| -> пульс NE
    gate_floor: float = 0.02     # минимальный строб входа зоны


@dataclass
class S7Cfg:
    """S7 — гомеостатический стабилизатор (тензорные операции, граф-безопасно)."""
    r_lo: float = 0.02           # ниже -> ACh-подкачка
    r_hi: float = 0.35           # выше -> GABA-всплеск (шунтирующий, reversal-безопасный!)
    surge_max: float = 1.5       # макс. добавка к g_GABA
    ach_boost: float = 0.25      # добавка ACh при молчании
    slope: float = 0.05          # крутизна сигмоид


@dataclass
class NWConfig:
    membrane: MembraneConfig = field(default_factory=MembraneConfig)
    zones: List[ZoneSpec] = field(default_factory=lambda: [
        ZoneSpec("A", 96, 16, role="сенсорная/вход", fan_in_rec=16),
        ZoneSpec("B", 96, 16, role="синтаксис", fan_in_rec=16),
        ZoneSpec("C", 96, 16, role="логика/семантика", fan_in_rec=16),
        ZoneSpec("D", 48, 8, role="ассоциативная/контекст", fan_in_rec=12),
    ])
    # индексы зон задач для WTA-маршрутизации (зона B = синтаксис, C = логика)
    task_zones: List[str] = field(default_factory=lambda: ["B", "C"])
    n_tokens: int = 33           # 0..15 синтаксис, 16..31 логика, 32 blank
    blank_token: int = 32
    in_gain: float = 1.0         # масштаб входных линий
    sched_len: int = 8192        # длина тензоров расписаний нейромодуляторов
    seed: int = 7
    w_max: float = 1.2           # клип w_slow
    eta_readout: float = 0.15    # скорость дельта-правила readout (претрейн)
    eta_hebb: float = 8e-5      # трёхфакторный Хебб w_slow: ОЧЕНЬ мягкий (DA-гейт)
    hebb_decay: float = 3e-3    # доминирующий распад: w_slow живёт у инициализации
    wta: WTACfg = field(default_factory=WTACfg)
    s7cfg: S7Cfg = field(default_factory=S7Cfg)

    def zone(self, name: str) -> ZoneSpec:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(name)

    def scaled(self, factor: int) -> "NWConfig":
        """Масштабирование: размеры зон x factor, fan-in ПОСТОЯНЕН -> O(N) шаг."""
        import copy
        cfg = copy.deepcopy(self)
        for z in cfg.zones:
            z.n_e = int(z.n_e * factor)
            z.n_i = int(z.n_i * factor)
        return cfg

    def total_n(self) -> int:
        return sum(z.n_e + z.n_i for z in self.zones)
