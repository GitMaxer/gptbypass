# NEUROWEAVE

Цифровой спайковый организм: **вычисление = память = обучение** в одном
контуре. Этот репозиторий целиком посвящён проекту NEUROWEAVE.

`neuroweave9` — **v9.0 «Cortex» (The Great Merge)**: гибрид измеренного
пайплайна обучения Spiker v6.0 и инноваций NEUROWEAVE 8.
`neuroweave8` — ламинарный организм v8, используемый v9 как библиотека
ядра (нейрон/синапс/сон/callosum/credit); зафиксирован в `V8_V9_LOCK`.

## Быстрый старт

```bash
pip install torch pytest          # CPU-режим поддерживается из коробки

# v9 «Cortex» — обучение на TinyStories методом «одной большой партии»
python -m neuroweave9.train_cortex --max-stories 200 --holdout 40 \
    --phase1-epochs 4 --phase2-epochs 10 --max-vocab-size 128

# v8 — демо жизненного цикла и бенчмарк
python -m neuroweave8.main --preset poc --steps 200 --batch 4
python -m neuroweave8.bench --sweep --ticks 40   # свип масштабирования

pytest tests/                     # 48 тестов: v8 ядро/организм/скейлинг + v9 cortex
```

## Состав репозитория

| Путь | Содержимое |
|---|---|
| `neuroweave9/` | **v9.0 «Cortex»** — ламинарные колонки A/B/C + θ-γ синхронизация полушарий (binder+VSA), Лапласова память и базальные ганглии Spiker, пайплайн «одной большой партии», EWC-контроль памяти, сон с реплеем. `train_cortex.py` — обучение + генерация |
| `neuroweave8/` | **v8** — ламинарные колонки, гиппокамп, мета-контроллер нейромодуляции, Global Workspace, θ-γ рабочая память + фазовый VSA, Corpus Callosum, цикл сон/бодрствование, EWC, структурная пластичность, int8, NeuroSystem. Документация — `neuroweave8/README.md` |
| `tests/` | Тест-пайплайн: v8 ядро (нейрон/синапс/колонка/шина/WS/VSA), организм, пайплайн, скейлинг + **v9 cortex** (контракт Spiker, Фаза 1, EWC, синк, сон, фасад) |
| `data/` | TinyStories corpus (4000 историй) для пайплайна v9 |
| `V8_V9_LOCK` | Коммит-хэш ядра v8, на которое рассчитан v9 |

## v9.0 «Cortex» — The Great Merge

Из **Spiker** взяты математически проверенные компоненты (`ARCHITECTURE_REVIEW`):
- метод **«одной большой партии»**: корпус = один батч, один
  `optimizer.step()` на эпоху, экономный `backward(1/seq_len)` — тот же
  градиент при памяти O(1);
- **LaplaceTimingBank** — геометрический спектр экспоненциальных
  интеграторов (клетки времени, Shankar & Howard), мультимасштабный
  контекст декодера;
- **BasalGangliaGate** — дрифт-диффузионная модель решения (Go/No-Go +
  STN): темп эмиссии токенов с нейрохимическим порогом (5-HT/ACh);
- **ThetaGammaOscillator** — PAC-гейт окна Corpus Callosum;
- асимметрия полушарий: левое быстрое (tau_m=8, σ=0.001), правое медленное
  (tau_m=35, σ=0.05); изолированная Фаза 1 с каузальной историей
  (горизонт delta_t=3), EWC-финализация и контроль `L_A` до/после сна.

Из **NEUROWEAVE 8** взяты инновационные компоненты:
- **LaminarColumn** вместо плоской Zone: L4→L2/3→L5→L6 + PV/SST,
  дендритное плато L5 = AND-гейт «вход И контекст», SST контролирует
  апикальный ток, PV — k-WTA;
- **HemisphereSync** (`sync.py`): обе полушарии пишут в общий
  `ThetaGammaBinder` на соседние γ-слоты, а `PhaseVSA` связывает их коды
  в один межполушарный фрейм — синхронизация по структуре, не по сигналу.
  Реляционный канал `frame.real` идёт декодеру;
- NeuroChemistry + MetaController, CreditPipeline (событийный DA,
  Tag-and-Capture), Hippocampus DG→CA3→CA1 с приоритизированным реплеем,
  SleepProtocol, EWC, StructuralPlasticity, int8-квантизация, CortexSystem.

### Структура `neuroweave9/`

- `memory/laplace.py` — LaplaceTimingBank (порт)
- `action/basal_ganglia.py` — BasalGangliaGate (порт; порог ↓ от орексина,
  ↑ от SON-surprise)
- `modulation/pacemaker.py` — ThetaGammaOscillator (порт)
- `modulation/orexin.py` — OrexinInteroceptiveLoop (энергия/цель → drive)
- `modulation/no_field.py` — NitricOxideField (объёмная передача NO)
- `modulation/son.py` — SonBus: самонаблюдение + авторегрессионный
  surprise «это ново?» (порт)
- `topology/thalamus.py` — Thalamus + TRN: сенсорный гейт и сонные веретёна
- `cortex/astrocyte.py` — TripartiteAstrocyte: 3-я шкала памяти
  (Ca2+ ~ 2500 мс, глиотрансмиттер → STP-фасилитация L2/3)
- `cortex/sync.py` — HemisphereSync: общий θ-γ биндер (со слотовой
  утечкой) + VSA-фрейм; когерентность по VSA-наполнителям, фазовый
  сдвиг полушарий
- `cortex/organism.py` — CortexOrganism (контракт совместим с Spiker)
- `organism/development.py` — DevelopmentalController: curriculum по
  EMA-мастерству
- `learning/phase1.py` — Phase1ColumnTrainer + CortexEWC
- `readout.py` — LaplaceContextReadout (+ канал синхронизации)
- `train_cortex.py` — пайплайн «одной большой партии» + `--save`/`--curriculum`
- `generate.py` — генерация по чекпоинту (БГ-пейсинг эмиссии)
- `runtime.py` — FastLoopProfiler (секционный профайлер такта)
- `system.py` — CortexSystem (boot/step/sleep/quantize/save/load)
- `data.py` — загрузчики TinyStories

В v9 также восстановлены: STP Цодыкса-Маркрама и метапластичность Фузи
(`neuroweave8/core/synapse.py`), 2-AG ретроградная депрессия
(`neuroweave8/core/neuron.py`), химерный REM-инсайт и адаптивная
миелинизация (`neuroweave8/learning/sleep.py`, `topology/callosum.py`),
тест пола мембраны `tests/test_gaba_reversal.py`.

## Ключевые уровни v8

- **Ламинарная колонна** L4→L2/3→L5→L6 + PV/SST: базальный/апикальный
  тракты разведены физически; дендритное плато L5 = AND-гейт
  «вход И контекст».
- **Гиппокамп** DG→CA3→CA1: паттерн-сепарация, аттракторное
  восстановление, приоритизированный реплей во сне.
- **Обучаемый мета-контроллер**: сетпоинты ACh/5-HT от телеметрии;
  NE = фазический аларм неожиданной неопределённости (Yu & Dayan).
- **Global Workspace** с заявками компетентности и winner-broadcast
  fusion (WTA, ур. 9).
- **Активный вывод**: EFE-внимание (компетентность × новизна → arousal-
  гейн L1), precision-взвешенный PredictiveHead.
- **θ-γ рабочая память** (7±2 слота) внутри каждой области + фазовый VSA.
- **Событийный разреженный путь**: при плотности входа <40% драйв L4
  идёт через sparse gather.
- **Сон**: приоритизированный реплей гиппокампа, мягкая консолидация
  w_fast→w_slow, прунинг, EWC rebaseline; медленный контур с
  гистерезисом по sleep pressure.
- **NeuroSystem** (`organism/system.py`): развёртываемая системная
  граница — boot/эпизоды/авто-сон/телеметрия/save-load/quantize.
- **int8-квантизация** синапсов: ~10x памяти, max|Δw|<0.01, инференс
  на снимках.
- **Масштабирование**: small→poc→wide (до ~144k синапсов), сублинейная
  цена такта; `bench.py --sweep`.
