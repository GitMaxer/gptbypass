"""NEUROWEAVE v9.0 «Cortex» — гибрид Spiker x NEUROWEAVE 8 (The Great Merge)."""
from __future__ import annotations


def CortexOrganism(*args, **kwargs):  # noqa: N802
    from .cortex.organism import CortexOrganism as _CO
    return _CO(*args, **kwargs)


def CortexSystem(*args, **kwargs):  # noqa: N802
    from .system import CortexSystem as _CS
    return _CS(*args, **kwargs)


def LaplaceTimingBank(*args, **kwargs):  # noqa: N802
    from .memory.laplace import LaplaceTimingBank as _L
    return _L(*args, **kwargs)


def BasalGangliaGate(*args, **kwargs):  # noqa: N802
    from .action.basal_ganglia import BasalGangliaGate as _BG
    return _BG(*args, **kwargs)


def HemisphereSync(*args, **kwargs):  # noqa: N802
    from .cortex.sync import HemisphereSync as _HS
    return _HS(*args, **kwargs)


def LaplaceContextReadout(*args, **kwargs):  # noqa: N802
    from .readout import LaplaceContextReadout as _R
    return _R(*args, **kwargs)


def Phase1ColumnTrainer(*args, **kwargs):  # noqa: N802
    from .learning.phase1 import Phase1ColumnTrainer as _T
    return _T(*args, **kwargs)


__all__ = ["CortexOrganism", "CortexSystem", "LaplaceTimingBank",
           "BasalGangliaGate", "HemisphereSync", "LaplaceContextReadout",
           "Phase1ColumnTrainer"]
