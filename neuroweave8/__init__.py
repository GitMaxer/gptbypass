"""NEUROWEAVE 8 — цифровой спайковый организм.

Архитектурное ядро (быстрый контур):

    core/        биофизика: суррогат, проводимости, нейрон, синапс, фаза, спарс
    microcircuit/ ламинарная колонка (канонический микроконтур) = область
    memory/      гиппокамп (DG/CA3/CA1) + системная консолидация
    modulation/  тензорная нейрохимия + ОБУЧАЕМЫЙ мета-контроллер
    topology/    мозолистое тело (задержка, GABA-фильтр) + Global Workspace
    learning/    трёхфакторная пластичность, EWC, структурная пластичность, сон
    organism/    единый такт жизни, активный вывод, медленный контур
    tasks/       среды: марковский поток, инверсия правил, связывание, сдвиг

Публичный вход: ``build_organism`` + ``Organism.tick``.
"""
from .config import (
    CallosumCfg, ColumnCfg, HippoCfg, ModCfg, NeuronCfg, OrganismCfg,
    PhaseCfg, PlasticityCfg, SleepCfg, StructuralCfg, WorkspaceCfg,
    default_cfg,
)

__version__ = "8.0.0"
__all__ = [
    "OrganismCfg", "NeuronCfg", "ColumnCfg", "PhaseCfg", "HippoCfg", "ModCfg",
    "PlasticityCfg", "StructuralCfg", "WorkspaceCfg", "CallosumCfg", "SleepCfg",
    "default_cfg", "build_organism",
]


def build_organism(cfg: OrganismCfg | None = None, preset: str | None = None,
                   **kwargs):
    """Фабрика организма (импорт отложен, чтобы core работал без organism).

    ``preset`` ('small'|'poc'|'wide') — короткий путь к default_cfg.
    """
    from .organism.organism import Organism

    if cfg is None:
        cfg = default_cfg(preset or "poc")
    return Organism(cfg, **kwargs)
