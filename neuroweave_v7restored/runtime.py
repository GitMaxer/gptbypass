# src/spiker/organism/runtime.py
# Этап 5 (канон v4.0/v4.0.1, разд. 8 + табл. 6/9): рантайм быстрого контура.
#
# Три инструмента Этапа 5:
#   1. audit_host_sync() — АУДИТ СИНХРОННО-БЕЗОПАСНОСТИ горячего такта:
#      запрещает .item()/.tolist()/.cpu()/.numpy()/float()/bool() на
#      тензорах. Хост-ветвление по данным — единственное принципиальное
#      препятствие захвата в CUDA Graphs (Y5, T2): пока такт проходит аудит,
#      его тензорный граф захватываем ПО КОНСТРУКЦИИ.
#   2. FastLoopProfiler — профиль производительности быстрого контура
#      (критерий табл. 9: «профиль производительности быстрого контура
#      снят»): такты/с + поэлементная разбивка (зоны, шина, пластичность,
#      нейрохимия).
#   3. CUDAGraphRunner — захват блока тактов в torch.cuda.CUDAGraph:
#      устранение оверхеда запуска ядер на каждом такте (цель табл. 9:
#      снижение оверхеда такта >= 2x). На CPU захват невозможен
#      (capture — операция CUDA-драйвера); раннер честно сообщает статус
#      и даёт информативную ошибку. Необходимое условие захвата
#      (синхронно-безопасный такт) проверяется аудитом на ЛЮБОМ железе.
#
# Контракт захвата (блок из K тактов, K = delay Callosum):
#   * указатель кольца — тензор (S7-инженерия): за K = delay тактов он
#     совершает полный цикл и возвращается в исходное состояние —
#     блок ПОЛУКРАТЕН периоду кольца (замороженные посередине захваты
#     разорвали бы чередование слотов);
#   * входы и DA-гейт — статические буферы: между replay-блоками хост
#     копирует новые сенсорные данные (copy_) и пишет импульсы награды
#     в neuro.pending_da; сами replay-ы не касаются хоста;
#   *_rng_ шум мембран захватывается (Philox offset — штатный механизм
#     CUDA Graphs);
#   * readout/награда (behave) — ВНЕ графа: это хост-граница организма
#     (медленный контур, табл. 5).
import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional

import torch


# ----------------------------------------------------------------------
# 1. Аудит синхронно-безопасности (необходимое условие CUDA Graphs)
# ----------------------------------------------------------------------
_SYNC_METHODS = ("item", "tolist", "numpy", "cpu", "nonzero_zero_check")
_SYNC_DUNDERS = ("__float__", "__bool__", "__index__")


@contextmanager
def audit_host_sync():
    """Запретить хост-синхронизации тензоров внутри контекста.

    Патчит методы torch.Tensor, приводящие тензор к хосту (синхронизация
    device->host — препятствие захвату в CUDA Graphs): .item(), .tolist(),
    .cpu(), .numpy(), float(t), bool(t), t.__index__. Любой вызов внутри
    контекста — AssertionError с именем метода.

    Использование:
        with audit_host_sync():
            organism.awake_step(inputs)   # такт обязан пройти без синхронизаций
    """
    saved = {}
    banned = {}
    for name in _SYNC_METHODS:
        banned[name] = getattr(torch.Tensor, name, None)
    for name in _SYNC_DUNDERS:
        banned[name] = getattr(torch.Tensor, name, None)

    def _make_raiser(method_name):
        def _raise(self, *args, **kwargs):
            raise AssertionError(
                f"host-синхронизация тензорного пути: {method_name}() "
                f"неприемлем в быстром контуре (CUDA Graphs, разд. 8)")
        return _raise

    try:
        for name, fn in banned.items():
            if fn is None:
                continue
            saved[name] = fn
            setattr(torch.Tensor, name, _make_raiser(name))
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.Tensor, name, fn)


# ----------------------------------------------------------------------
# 2. Профилировщик быстрого контура
# ----------------------------------------------------------------------
class FastLoopProfiler:
    """Профиль производительности горячего такта организма (табл. 9).

    Меряет wall-время awake_step и его составляющих (полушария, шина,
    пластичность, нейрохимия) обёртками методов ДАННОГО экземпляра —
    без изменения кода организма. CPU-мера честна для тактов/с; для CUDA
    асинхронных потоков агрегирует время запуска (не исполнения) —
    ограничение задокументировано.
    """

    COMPONENTS = ("hemispheres", "callosum", "plasticity", "neurochemistry")

    def __init__(self, organism):
        self.organism = organism

    @torch.no_grad()
    def profile(self, make_inputs: Callable[[], Dict[str, torch.Tensor]],
                ticks: int = 64, warmup: int = 8) -> Dict[str, float]:
        """make_inputs: замыкание, возвращающее входы зон одного такта.

        Возвращает: ticks_per_sec, ms_per_tick, seconds по компонентам,
        tick_breakdown_pct.
        """
        org = self.organism
        # прогрев (аллокации, ленивые состояния, кэши ядер)
        for _ in range(warmup):
            org.awake_step(make_inputs())

        acc = {name: 0.0 for name in self.COMPONENTS}
        total = 0.0

        def _timed(fn, key):
            def wrapper(*args, **kwargs):
                nonlocal total
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                acc[key] += time.perf_counter() - t0
                return out
            return wrapper

        # обёртки составляющих такта (только на время замера)
        saved = {
            "left_step": org.left.step, "right_step": org.right.step,
            "callosum_step": org.callosum.step,
            "step_zone": org.pipeline.step_zone,
            "neuro_step": org.neuro.step,
        }
        org.left.step = _timed(saved["left_step"], "hemispheres")
        org.right.step = _timed(saved["right_step"], "hemispheres")
        org.callosum.step = _timed(saved["callosum_step"], "callosum")
        org.pipeline.step_zone = _timed(saved["step_zone"], "plasticity")
        org.neuro.step = _timed(saved["neuro_step"], "neurochemistry")
        try:
            t0 = time.perf_counter()
            for _ in range(ticks):
                org.awake_step(make_inputs())
            total = time.perf_counter() - t0
        finally:
            org.left.step = saved["left_step"]
            org.right.step = saved["right_step"]
            org.callosum.step = saved["callosum_step"]
            org.pipeline.step_zone = saved["step_zone"]
            org.neuro.step = saved["neuro_step"]

        report = {
            "ticks": float(ticks),
            "seconds": float(total),
            "ms_per_tick": 1000.0 * total / max(ticks, 1),
            "ticks_per_sec": float(ticks) / max(total, 1e-9),
        }
        for name in self.COMPONENTS:
            report[f"{name}_s"] = float(acc[name])
            report[f"{name}_pct"] = 100.0 * acc[name] / max(total, 1e-9)
        return report


# ----------------------------------------------------------------------
# 3. Захват быстрого контура в CUDA Graph (аппаратно-гейтед)
# ----------------------------------------------------------------------
class CUDAGraphRunner:
    """Захват блока тактов организма в torch.cuda.CUDAGraph (табл. 9).

    Протокол:
        capture(warmup):  прогрев на side-stream, затем захват K тактов
                          awake_step (K = delay Callosum — полное кольцо:
                          тензорный указатель возвращается в исходный слот);
        step(inputs):     copy_ новых входов в статические буферы + replay()
                          (K тактов без единого касания хоста);
        поведение (behave: readout, награда, DA-импульс) — ВНЕ графа:
        хост-граница между блоками (медленный контур, табл. 5).

    На CUDA-недоступном железе: available=False, capture() даёт
    информативную ошибку. Синхронно-безопасность такта (необходимое
    условие захвата) проверяется audit_host_sync() на любом железе.
    """

    def __init__(self, organism, batch_size: int, device: torch.device):
        self.organism = organism
        self.batch = int(batch_size)
        self.device = device
        # блок = полный цикл кольца Callosum (указатель возвращается)
        self.steps_per_graph = int(organism.callosum.delay)
        self.available = bool(torch.cuda.is_available()
                               and device.type == "cuda")
        self._graph: Optional[object] = None
        self._static_inputs: Dict[str, torch.Tensor] = {}
        self._ticks_done = 0

    # ------------------------------------------------------------------
    def _make_static_inputs(self, sample_inputs: Dict[str, torch.Tensor]):
        self._static_inputs = {
            name: t.detach().clone() for name, t in sample_inputs.items()
        }

    @torch.no_grad()
    def capture(self, sample_inputs: Dict[str, torch.Tensor],
                warmup: int = 3):
        """Прогреть и захватить K-тактовый блок быстрого контура."""
        if not self.available:
            raise RuntimeError(
                "CUDAGraphRunner: захват CUDA Graph невозможен — "
                f"устройство {self.device} без CUDA (capture — операция "
                "драйвера). Профиль быстрого контура снимается "
                "FastLoopProfiler'ом; синхронно-безопасность такта "
                "проверяется audit_host_sync(). Критерий табл. 9 "
                "(оверхед такта >= 2x) — аппаратно-гейтед: переносится "
                "на GPU-конфигурацию."
            )
        org = self.organism
        org.reset_state(self.batch, self.device)
        self._make_static_inputs(sample_inputs)

        # Прогрев на side-stream (штатный протокол захвата CUDA Graphs)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                org.awake_step(self._static_inputs)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(self.steps_per_graph):
                org.awake_step(self._static_inputs)
        self._graph = g
        return g

    @torch.no_grad()
    def step(self, inputs: Dict[str, torch.Tensor]) -> int:
        """Прогнать K тактов: новые входы в статические буферы + replay."""
        if self._graph is None:
            raise RuntimeError("граф не захвачен: сначала capture(...)")
        for name, t in inputs.items():
            self._static_inputs[name].copy_(t.detach())
        self._graph.replay()
        self._ticks_done += self.steps_per_graph
        return self.steps_per_graph

    @property
    def ticks_done(self) -> int:
        return self._ticks_done