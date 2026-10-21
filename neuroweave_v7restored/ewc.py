# src/spiker/learning/ewc.py
# Этап 3 (канон v4.0/v4.0.1, разд. 7.2 + ур. 13): Elastic Weight Consolidation.
#
# Задача: цикл организма НЕЛИНЕЕН — после сна (консолидация w_fast -> w_slow)
# организм может входить в новый период обучения (новая Фаза 1 локальной
# донастройки). Прямое дообучение размывает старые навыки — катастрофическое
# забывание. EWC штрафует удаление весов, критичных для подтверждённых задач:
#
#     L_total = L_task + lam_EWC * SUM_k F_k (w_k - w_snap_k)^2      (ур. 13)
#
# F_k — диагональная эмпирическая матрица Фишера: E[(dL_task/dw)^2] по
# выборкам ЗАЩИЩАЕМОЙ задачи. Большие |grad| => вес активно используется
# задачей => сильнее держится у снапшота.
#
# Каноническое решение v4.0.1 (CHANGELOG): штраф активен ТОЛЬКО в Фазе 2
# цикла пластичности; после каждого цикла сна выполняется rebaseline()
# (снапшот и Фишер перебазируются на новый w_slow), иначе EWC боролся бы с
# консолидацией w_fast -> w_slow (ур. 21) и замораживал бы организм в
# прошлом. EWC — защита МЕЖДУ циклами сна, не внутри них.
from typing import Callable, Dict

import torch


class EWCRegularizer:
    """
    Диагональный EWC над именованным набором параметров.

    Жизненный цикл в организме:
        1. consolidate_snapshot() — после периода устойчивости (Фаза 1
           завершена / цикл сна окончен): запомнить w_snap;
        2. estimate_fisher(loss_fn, n_samples) — на выборках защищаемой
           задачи (grad^2 без накопления в самих .grad параметров);
        3. penalty() — слагаемое к лоссу нового обучения (Фаза 2 цикла);
        4. rebaseline() — после следующего сна: вернуться к шагу 1.
    """

    def __init__(self, params: Dict[str, torch.nn.Parameter], lam: float = 50.0):
        for name, p in params.items():
            if not isinstance(p, torch.Tensor) or not p.requires_grad:
                raise ValueError(
                    f"EWC-параметр {name!r} должен быть листом графа с "
                    f"requires_grad=True (получен requires_grad={p.requires_grad})"
                )
        self._params: Dict[str, torch.Tensor] = dict(params)
        self.lam = float(lam)
        self.snapshot: Dict[str, torch.Tensor] = {
            k: p.detach().clone() for k, p in self._params.items()
        }
        self.fisher: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(p) for k, p in self._params.items()
        }
        self._fisher_mass = 0.0

    # ------------------------------------------------------------------
    def consolidate_snapshot(self):
        """Зафиксировать текущие веса как защищаемые (шаг 1)."""
        self.snapshot = {k: p.detach().clone() for k, p in self._params.items()}

    # ------------------------------------------------------------------
    @torch.enable_grad()
    def estimate_fisher(self, loss_fn: Callable[[], torch.Tensor], n_samples: int = 64):
        """Диагональный эмпирический Фишер: F_k = E[(dL/dw_k)^2].

        loss_fn: замыкание БЕЗ аргументов, каждый вызов возвращает скалярный
        лосс защищаемой задачи на новом примере/минибатче (случайность
        внутри замыкания — это и есть интегрирование по распределению).
        Градиенты накапливаются во временных копиях: .grad самих параметров
        не загрязняются (совместимо с внешним оптимизатором).
        """
        for p in self._params.values():
            if p.grad is not None:
                p.grad = None
        acc = {k: torch.zeros_like(p) for k, p in self._params.items()}
        for _ in range(int(n_samples)):
            loss = loss_fn()
            grads = torch.autograd.grad(
                loss, list(self._params.values()), retain_graph=False,
                allow_unused=True,
            )
            for (name, _), g in zip(self._params.items(), grads):
                if g is not None:
                    acc[name] += g.detach() ** 2
        n = max(1, int(n_samples))
        for name in self._params:
            self.fisher[name].copy_(acc[name] / n)
        self._fisher_mass = float(
            sum(f.sum().item() for f in self.fisher.values())
        )
        return self._fisher_mass

    # ------------------------------------------------------------------
    def penalty(self) -> torch.Tensor:
        """L_EWC = SUM_k F_k (w_k - w_snap_k)^2 (ур. 13) — к лоссу Фазы 2."""
        total = None
        for name, p in self._params.items():
            if float(self.fisher[name].sum()) == 0.0:
                continue  # параметр не защищён (Фишер не оценён)
            term = (self.fisher[name] * (p - self.snapshot[name]) ** 2).sum()
            total = term if total is None else total + term
        if total is None:
            return torch.zeros((), device=next(iter(self._params.values())).device)
        return total

    def total_loss(self, task_loss: torch.Tensor) -> torch.Tensor:
        """L_task + lam * L_EWC — каноническая композиция Фазы 2."""
        return task_loss + self.lam * self.penalty()

    # ------------------------------------------------------------------
    def rebaseline(self):
        """После цикла сна: снапшот перебазируется, Фишер сбрасывается.

        Иначе штраф тянул бы веса к ДОконсолидационному состоянию и
        конфликтовал с переносом w_fast -> w_slow (решение v4.0.1).
        Следующий период устойчивости переоценит Фишер заново.
        """
        self.consolidate_snapshot()
        for f in self.fisher.values():
            f.zero_()
        self._fisher_mass = 0.0

    # ------------------------------------------------------------------
    @property
    def fisher_mass(self) -> float:
        """Сумма диагонали Фишера — телеметрия «сколько защищено»."""
        return self._fisher_mass

    @property
    def is_armed(self) -> bool:
        return self._fisher_mass > 0.0
