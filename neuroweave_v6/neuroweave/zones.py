"""Топология зон NEUROWEAVE: физическая изоляция + gather-связность.

ФИЗИЧЕСКАЯ ИЗОЛЯЦИЯ — ПО ПОСТРОЕНИЮ:
  * рекуррентные связи существуют только ВНУТРИ зоны (блок-структура);
  * межзонные связи — ТОЛЬКО через явно заданные проекции
    (A->B, A->C, B->D, C->D), каждая со своим стробом маршрутизации;
  * интернейроны (I) проецируются только на E-клетки СВОЕЙ зоны.

Связность с фиксированным fan-in -> память O(N*fan_in), шаг O(N*fan_in):
масштабирование зон НЕ меняет локальную плотность (Этап 6).

w_slow — ЗНАЧЕНИЯ связей (values), тензор [n_dst, fan_in]: обучение
трёхфакторным Хеббом (DA-гейт), на «Экзамене» заморожены.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .config import NWConfig


@dataclass
class ConnGroup:
    """Группа связей: dst-нейроны читают fan_in пресинаптических спайков."""
    name: str
    src_idx: torch.Tensor   # [n_dst, fan_in] long — пресинапсы (flat-индексы)
    dst_idx: torch.Tensor   # [n_dst] long — постсинапсы (flat-индексы)
    w: torch.Tensor         # [n_dst, fan_in] — w_slow (значения)
    kind: str = "exc"       # 'exc' (AMPA+NMDA) | 'inh' (GABA)
    f_ampa: float = 0.7     # доля AMPA в возбуждающей группе
    gate_id: int = -1       # индекс строба маршрутизации (-1 = всегда вкл.)
    trace: Optional[torch.Tensor] = None  # [n_dst, fan_in] элегибельность (претрейн)


@dataclass
class InputLine:
    """Входная линия: токен -> ток в зону (строки таблицы = w-подобная проекция)."""
    name: str
    table: torch.Tensor    # [n_tokens, n_dst] — фиксированная рецепторная геометрия
    dst_idx: torch.Tensor  # [n_dst] long
    gate_id: int = -1      # -1 = сенсорный вход всегда активен
    kind: str = "ampa"


@dataclass
class Topology:
    cfg: NWConfig
    zone_names: List[str]
    zone_of: torch.Tensor      # [N] long — зона каждого нейрона
    e_slices: Dict[str, slice]  # flat-срезы E-клеток зон
    i_slices: Dict[str, slice]
    groups: List[ConnGroup]
    lines: List[InputLine]
    n_gates: int

    def learnable_tensors(self) -> Dict[str, torch.Tensor]:
        """Все w_slow-тензоры (для снапшота/заморозки/нулевого дрейфа)."""
        return {f"w:{g.name}": g.w for g in self.groups}

    def snapshot_wslow(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.learnable_tensors().items()}


def _sparse_src(src_pool_lo: int, src_pool_hi: int, n_dst: int, fan_in: int,
                gen: torch.Generator) -> torch.Tensor:
    """Случайные пресинапсы ИЗ ЗАДАННОГО ПУЛА [lo, hi) — изоляция по построению."""
    return torch.randint(src_pool_lo, src_pool_hi, (n_dst, fan_in),
                         generator=gen)


def build_topology(cfg: NWConfig, device, tables: Optional[Dict[str, torch.Tensor]] = None,
                   seed: int = 0) -> Topology:
    """Строит зоны, связность и входные линии.

    tables: словарь имя -> [n_tokens, n_dst] входных таблиц (строятся задачей:
    exam.py / sleep.py). Ожидаемы ключи: cur_A, cur_B, del_B, cur_C, del_C.
    """
    gen = torch.Generator().manual_seed(cfg.seed + seed)
    zone_names = [z.name for z in cfg.zones]
    Z = len(zone_names)

    # flat-раскладка: для каждой зоны блок E, затем блок I
    flat = 0
    e_slices, i_slices, zone_of = {}, {}, []
    for z in cfg.zones:
        e_lo = flat
        e_slices[z.name] = slice(e_lo, e_lo + z.n_e)
        flat += z.n_e
        i_slices[z.name] = slice(flat, flat + z.n_i)
        flat += z.n_i
        zone_of += [zone_names.index(z.name)] * (z.n_e + z.n_i)
    N = flat
    zone_of = torch.tensor(zone_of, dtype=torch.long, device=device)

    groups: List[ConnGroup] = []

    def exc_group(name, src_lo, src_hi, dst_idx, fan_in, gate_id, w_scale):
        n_dst = dst_idx.numel()
        src_idx = _sparse_src(src_lo, src_hi, n_dst, fan_in, gen).to(device)
        w = (torch.rand(n_dst, fan_in, generator=gen) * w_scale).to(device)
        return ConnGroup(name, src_idx, dst_idx.to(device), w, kind="exc",
                         f_ampa=1.0 - cfg.membrane.nmda_frac, gate_id=gate_id,
                         trace=torch.zeros(n_dst, fan_in, device=device))

    def inh_group(name, src_lo, src_hi, dst_idx, fan_in, w_scale):
        n_dst = dst_idx.numel()
        src_idx = _sparse_src(src_lo, src_hi, n_dst, fan_in, gen).to(device)
        w = (torch.rand(n_dst, fan_in, generator=gen) * w_scale).to(device)
        return ConnGroup(name, src_idx, dst_idx.to(device), w, kind="inh",
                         trace=torch.zeros(n_dst, fan_in, device=device))

    # стробы маршрутизации: 0 = B (синтаксис), 1 = C (логика)
    n_gates = 2

    for z in cfg.zones:
        e_dst = torch.arange(e_slices[z.name].start, e_slices[z.name].stop)
        # рекуррентные E->E внутри зоны (изоляция: пул = своя зона)
        groups.append(exc_group(
            f"rec_{z.name}", e_slices[z.name].start, e_slices[z.name].stop,
            e_dst, z.fan_in_rec, gate_id=-1, w_scale=0.035))
        # I->E внутри зоны (GABA с reversal)
        groups.append(inh_group(
            f"inh_{z.name}", i_slices[z.name].start, i_slices[z.name].stop,
            e_dst, z.fan_in_i, w_scale=0.12))

    # межзонные проекции (явные, стробируемые) — ЭТО единственные мосты между зонами
    proj_specs = [
        ("proj_A_B", "A", "B", 0),   # сенсорика -> синтаксис (строб B)
        ("proj_A_C", "A", "C", 1),   # сенсорика -> логика (строб C)
        ("proj_B_D", "B", "D", 0),   # синтаксис -> ассоциации
        ("proj_C_D", "C", "D", 1),   # логика -> ассоциации
    ]
    fan_proj = 8
    for name, src, dst, gate in proj_specs:
        e_dst = torch.arange(e_slices[dst].start, e_slices[dst].stop)
        groups.append(exc_group(
            name, e_slices[src].start, e_slices[src].stop,
            e_dst, fan_proj, gate_id=gate, w_scale=0.05))

    # входные линии (таблицы задач); если не переданы — пусто
    lines: List[InputLine] = []
    if tables:
        for name in ("cur_A", "cur_B", "del_B", "cur_C", "del_C"):
            if name not in tables:
                continue
            zone = name.split("_")[1]
            gate = -1 if zone == "A" else (0 if zone == "B" else 1)
            e_lo, e_hi = e_slices[zone].start, e_slices[zone].stop
            dst_idx = torch.arange(e_lo, e_hi)
            lines.append(InputLine(name, tables[name].to(device),
                                   dst_idx.to(device), gate_id=gate))

    return Topology(cfg, zone_names, zone_of, e_slices, i_slices, groups,
                    lines, n_gates)
