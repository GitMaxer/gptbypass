# src/spiker/topology/hemisphere.py
# Этап 2 (канон v4.0/v4.0.1, разд. 2.2 + табл. 2/6): полушарие —
# сборка зон одного бока с жёстко зафиксированной дивергенцией.
#
# Слом симметрии — не гиперпараметрическая случайность, а конструктивный
# принцип (разд. 2.2): две одинаковые копии на одних данных выродились бы
# в дубликаты репрезентаций. Поэтому полушария расходятся по ЧЕТЫРЁМ осям:
#     tau_m        — короткая (5-10 мс, левое) vs длинная (25-50 мс, правое);
#     sigma_noise  — 0.001 (почти детерминизм) vs 0.05 (выраженная стохастика);
#     5-HT базис   — высокий 0.8 (эксплуатация) vs пониженный 0.3 (поиск);
#     lr_local     — высокая vs умеренно-консервативная адаптация;
# плюс суррогатный градиент: FastSigmoid (левое) vs Arctan с тяжёлыми
# хвостами (правое, разд. 7.1). Пресеты дублированы YAML-конфигами
# configs/hemisphere_{left,right}.yaml (листинг 6).
#
# Критерий готовности Этапа 2 (табл. 9): статистически значимое расхождение
# распределений спайков L/R на одном входе — tests/test_symmetry_breaking.py.
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .zone import Zone, HEMISPHERE_PRESETS


# Значения по умолчанию совпадают с configs/hemisphere_*.yaml (табл. 2 канона)
HEMISPHERE_DEFAULTS = {
    "left": dict(tau_m=8.0, sigma_noise=0.001, surrogate_mode="fast_sigmoid",
                 surrogate_slope=10.0, surrogate_alpha=2.0,
                 ht_baseline=0.8, lr_local=0.01),
    "right": dict(tau_m=35.0, sigma_noise=0.05, surrogate_mode="arctan",
                  surrogate_slope=10.0, surrogate_alpha=2.0,
                  ht_baseline=0.3, lr_local=0.002),
}

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class Hemisphere(nn.Module):
    """
    Полушарие: именованный набор зон одного бока + зафиксированная
    дивергенция (табл. 2), доступная сборке организма и телеметрии.

    Обязанности:
      * пошаговый прогон своих зон (единый такт dt полушария);
      * выдача конкатенированного SON-экспорта [B, Z*n_son] для шины
        Corpus Callosum (сжатые сообщения, ур. 15);
      * приём сообщения шины и раздача его зонам: чанк [B, n_son] на зону
        декодируется SON-узлом приёма этой зоны на её апикальные дендриты
        (ур. 16) — межзонное влияние контекстно, NMDA-гейт его не пустит
        без собственного базального входа (разд. 6.1);
      * трансляция модуляторов нейрохимии на свои зоны (разд. 4.2).
    """

    def __init__(self, side: str, zones: Sequence[Zone]):
        super().__init__()
        if side not in ("left", "right"):
            raise ValueError(f"side должен быть 'left' или 'right': {side!r}")
        if len(zones) == 0:
            raise ValueError("Полушарие требует хотя бы одну зону")

        self.side = side
        preset = HEMISPHERE_DEFAULTS[side]
        self.tau_m = float(preset["tau_m"])
        self.sigma_noise = float(preset["sigma_noise"])
        self.surrogate_mode = str(preset["surrogate_mode"])
        self.ht_baseline = float(preset["ht_baseline"])
        self.lr_local = float(preset["lr_local"])

        named: Dict[str, Zone] = {}
        for zone in zones:
            if zone.side != side:
                raise ValueError(
                    f"Зона {zone.name!r} ({zone.side}) не принадлежит "
                    f"полушарию {side!r}"
                )
            if zone.name in named:
                raise ValueError(f"Дублирующееся имя зоны: {zone.name!r}")
            named[zone.name] = zone
        self.zone_names = list(named.keys())
        self.zones = nn.ModuleDict(named)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(
        cls,
        side: str,
        zone_specs: Sequence[Dict],
        config_path: Optional[Path] = None,
    ) -> "Hemisphere":
        """Сборка полушария из YAML-конфига (листинг 6: configs/).

        zone_specs: список словарей {"name": ..., **kwargs зоны};
        параметры дивергенции из YAML накладываются на kwargs зон,
        если зона не задала их явно. При отсутствии YAML-файла
        (пакетная установка) используются встроенные пресеты табл. 2.
        """
        path = Path(config_path) if config_path is not None \
            else _DEFAULT_CONFIG_DIR / f"hemisphere_{side}.yaml"
        overrides = {}
        if path.exists():
            import yaml  # зависимость проекта (requirements.txt)
            with open(path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            overrides = {
                k: v for k, v in cfg.items()
                if k in ("tau_m", "sigma_noise", "surrogate_mode",
                         "surrogate_slope", "surrogate_alpha")
            }
            if isinstance(cfg.get("surrogate"), dict):
                overrides.update({
                    f"surrogate_{k}": v for k, v in cfg["surrogate"].items()
                })

        zones = []
        for spec in zone_specs:
            spec = dict(spec)
            name = spec.pop("name")
            for key, val in overrides.items():
                spec.setdefault(key, val)
            zones.append(Zone(name=name, side=side, **spec))
        return cls(side, zones)

    # ------------------------------------------------------------------
    def reset_state(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype = torch.float32):
        for zone in self.zones.values():
            zone.reset_state(batch_size, device, dtype)

    # ------------------------------------------------------------------
    def step(
        self,
        inputs: Dict[str, torch.Tensor],
        apical: Optional[Dict[str, torch.Tensor]] = None,
        neuro=None,
        on_son_alarm=None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Один такт dt полушария.

        Параметры:
            inputs: имя зоны -> сенсорный вход зоны [B, n_inputs];
            apical: имя зоны -> внешний апикальный ток [B, n_exc]
                    (сообщения Callosum, уже декодированные SON-узлом
                    приёма — см. callosum_apical; учитываются только при
                    включённой шине, isolation_mask, Фаза 2);
            neuro: NeuroChemistry — трансляция ACh/NE/5-HT на зоны;
            on_son_alarm: колбэк (zone_name, surprise) — выброс NE
                    при детекции новизны (разд. 6.3).
        Возвращает: имя зоны -> словарь выходов Zone.step.
        """
        missing = set(self.zone_names) - set(inputs.keys())
        if missing:
            raise KeyError(f"Полушарию {self.side!r} не подан вход зон: {missing}")
        if apical is None:
            apical = {}

        batch_size = next(iter(inputs.values())).shape[0]
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, zone in self.zones.items():
            modifiers: Dict[str, torch.Tensor] = {}
            if neuro is not None:
                modifiers = dict(
                    ach=neuro.get("ach", name, batch_size),
                    ne=neuro.get("ne", name, batch_size),
                    serotonin_5ht=neuro.get("ht", name, batch_size),
                )
            outputs[name] = zone.step(
                inputs[name],
                s_apical=apical.get(name),
                on_son_alarm=on_son_alarm,
                **modifiers,
            )
        return outputs

    # ------------------------------------------------------------------
    # Шина: экспорт и приём сжатых сообщений (разд. 2.3/6.1)
    # ------------------------------------------------------------------
    def son_export(self, outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Конкатенация SON-экспортов зон: [B, Z*n_son] — поток в Callosum.

        Экспорт уже гейтирован Surprise (Zero-Compute, разд. 6.3):
        предсказуемые зоны шину не занимают — экономия полосы встроена
        в источник, а не в арбитра."""
        chunks = [outputs[name]["z_export"] for name in self.zone_names]
        return torch.cat(chunks, dim=-1)

    def bus_chunks(self, bus_message: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Разбор сообщения шины на чанки зон-получателей (чистое сечение).

        Чанк [B, n_son] зоны — ЕЁ вход узла приёма SON (ур. 16): то, что
        декодер приёма видит на апикальном пути. Этап 5: чанки нужны
        также 3-факторному кредиту декодеров (stdp.step_decoder) —
        сечение вынесено в отдельный метод, callosum_apical — надстройка.
        """
        if bus_message.shape[-1] != self.bus_dim:
            raise ValueError(
                f"Сообщение шины {bus_message.shape[-1]} не совпадает с "
                f"полосой полушария {self.bus_dim}"
            )
        chunks: Dict[str, torch.Tensor] = {}
        cursor = 0
        for name, zone in self.zones.items():
            n_son = zone.son.n_son
            chunks[name] = bus_message[:, cursor:cursor + n_son]
            cursor += n_son
        return chunks

    def decode_chunks(
        self,
        chunks: Dict[str, torch.Tensor],
        gain: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Чанки -> апикальные токи зон (ур. 16): декодер узла приёма зоны."""
        apical: Dict[str, torch.Tensor] = {}
        for name, zone in self.zones.items():
            chunk = chunks.get(name)
            if chunk is None:
                continue
            apical[name] = zone.son.decode(chunk) * gain
        return apical

    def callosum_apical(
        self,
        bus_message: torch.Tensor,
        gain: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Сообщение шины [B, Z*n_son] -> апикальные токи зон (ур. 16).

        Чанк размером n_son достаётся зоне-получателю и декодируется ЕЁ
        SON-декодером (узел приёма, ур. 16): пересланная информация
        модулирует реакцию зоны, но не запускает её без базального входа
        (NMDA-гейт, ур. 4). Активация пути — Фаза 2 (connect_apical).

        gain: масштаб доставки апикального тока. PoC-калибровка Этапа 2:
        декодеры приёма до обучения не знают масштаба NMDA-интегратора
        (tau=40) — гейн страхует получателей от раскачки. Этап 5 (d):
        декодеры обучаются 3-факторным кредитом — гейн РАМПИТСЯ к 1.0
        (устранить; см. stdp.step_decoder и зонд инверсии правил)."""
        return self.decode_chunks(self.bus_chunks(bus_message), gain=gain)

    @property
    def bus_dim(self) -> int:
        """Полоса полушария: суммарная размерность SON-латентов."""
        return int(sum(zone.son.n_son for zone in self.zones.values()))

    # ------------------------------------------------------------------
    def activity(self, outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Сводная активность зон [B, Z] — заявки на GlobalWorkspace."""
        cols = [outputs[name]["spikes"].detach().sum(dim=-1)
                for name in self.zone_names]
        return torch.stack(cols, dim=-1)

    def divergence_summary(self) -> Dict[str, float]:
        """Дивергенция полушария (телеметрия медленного контура)."""
        return dict(
            tau_m=self.tau_m,
            sigma_noise=self.sigma_noise,
            ht_baseline=self.ht_baseline,
            lr_local=self.lr_local,
            surrogate_mode=self.surrogate_mode,
        )


class LeftHemisphere(Hemisphere):
    """Левое (аналитическое) полушарие — табл. 2 канона."""

    def __init__(self, zones: Sequence[Zone]):
        super().__init__("left", zones)


class RightHemisphere(Hemisphere):
    """Правое (ассоциативное) полушарие — табл. 2 канона."""

    def __init__(self, zones: Sequence[Zone]):
        super().__init__("right", zones)
