# Аудит NEUROWEAVE v6.0 — реестр возможностей реализации

Объект: `neuroweave_v6/` — пакет `neuroweave/` (12 модулей), `scripts/` (3 CLI),
`verify_cuda_graphs.py`, документы `README.md`, `VERIFY.md`, `MATH_GABA_REVERSAL.md`.
Дата: 2026-10-21. Режим: только чтение, существующие файлы не изменялись.

> ## Статус запусков: НЕ ЗАПУСКАЛОСЬ
> В этой сессии нет инструмента исполнения команд (shell/python/sandbox), поэтому
> `python verify_cuda_graphs.py`, `python scripts/run_exam.py` и
> `python scripts/run_sleep.py` **не запускались**. Ни одно число ниже не является
> выводом программы. Пометки у каждого утверждения:
> **[static]** — прочитано в исходнике или посчитано по константам конфигурации;
> **[analytic]** — выведено вручную из формул кода (проверяемо прогоном, но не проверено);
> **[doc]** — цитата из README/VERIFY/MATH, воспроизведение не выполнялось;
> **н/з** — не запускалось.

## 1. Метод и границы

1. Прочитаны все 12 модулей пакета, 3 скрипта, верификатор, 3 документа, `verify.bat`, `requirements.txt`.
2. Прогоны не выполнялись (см. баннер) — стадии PASS/FAIL/SKIP **не зафиксированы**.
3. Все числа в §3–§4 либо помечены как выведенные из кода, либо помечены «н/з».
4. Границы: код не правился, тесты не добавлялись, `neuroweave_v7restored/` не аудировался (упомянут только как источник дельты в §6).

## 2. Что реально реализовано (по модулям)

**`config.py`** — dataclass-конфигурация: `MembraneConfig`, `ZoneSpec`, `WTACfg`, `S7Cfg`, `NWConfig`.
`NWConfig.scaled(factor)` масштабирует `n_e`/`n_i`, fan-in не трогает (это и есть заявленный O(N));
`total_n()`, `zone(name)`. Константы мембраны: `v_rest=-70`, `v_th=-50`, `v_reset=-58`, `v_floor=-90`,
`tau_m=4.0`, `dt=1.0`, `e_cl=-70`, `ecl_mode="homeostat"|"raw"`, `tau_ampa=1.5`, `tau_nmda=100`, `tau_gaba=8`.

**`neurons.py`** — ядро мембраны: `mg_block` (Jahr–Stevens, `1/(1+exp(-0.062V)·mg/3.57)`),
`fixed_point` (V_inf, λ, denom), `membrane_exact_step` (точный экспоненциальный шаг + OU-шум),
`membrane_euler_step` (только для контраста), `e_cl_effective` (KCC2: `clamp(E_cl, min=v_rest)`).
GABA реализована как `g_gaba·(V−E_Cl)` внутри `fixed_point`.

**`neurochem.py`** — `ModulatorSchedules` (пять тензоров `[T,Z]`: `ach, ht, da, ne, g_sleep`),
`tonic_schedules`, `da_pulse_train`, `ne_pulse`, `build_pretrain_schedule` (DA каждые 40 шагов, amp 0.9),
`build_exam_schedule` (DA=0). Расписания читаются по тензорному индексу `t % sched_len`.

**`zones.py`** — `ConnGroup` (src_idx `[n_dst,fan_in]`, dst_idx, `w`, `kind`, `gate_id`, `trace`),
`InputLine` (таблица `[n_tokens,n_dst]`), `Topology` (+`learnable_tensors`, `snapshot_wslow`),
`_sparse_src` (пресинапсы строго из пула зоны), `build_topology`.
Группы: на зону `rec_*` (E→E внутри) и `inh_*` (I→E), плюс 4 моста `proj_A_B/A_C/B_D/C_D` (fan_in=8).

**`wta.py`** — `kwt_mask` (top-k по V среди спайкнувших), `wta_mix` (softmax по 2 зонам задач,
`logits = rate/rate_ref + k_ctx·ctx_bias`, заострение `·(1+k_ne·NE)`), `ctx_bias_from_ctx`.

**`s7.py`** — класс `S7`: `gaba_surge` (σ((r−r_hi)/slope)·1.5), `ach_boost` (σ((r_lo−r)/slope)·0.25),
`scrub` (NaN/Inf→v_rest), `log_stats` (host-журнал deque≤256), `drain_events`.

**`fastloop.py`** — `FastLoop` (главный шаг) и `FastOutputs`. Состояние: `V, refr, g_ampa, g_nmda, g_gaba,
spikes, buf_tok, buf_prev_tok, buf_prev_win, ctx_ema, rate_ema, tr_unit, t` — всё in-place.
Шаг из 15 блоков: счётчик → расписание → S7 → NE-детектор → ctx EMA → WTA-стробы → буферы токенов →
затухание проводимостей → входные линии → gather-связность → сон-GABA+surge → мембрана → k-WTA/рефрактерность →
частоты зон → readout → (опц.) пластичность.

**`cudagraph.py`** — `FastLoopGraph` (прогрев на side-stream, захват `torch.cuda.graph`, `copy_`+`replay`),
`bench_step` (events на CUDA, `perf_counter` на CPU), `graph_vs_eager` (эквивалентность при sigma=0, замер).

**`sleep.py`** — `SleepStats`, `STAGES` (W150→N1 150→N2 200→N3 300→REM 150→W2 100 = **1050 шагов** [static]),
`build_sleep_tables`, `build_sleep_schedule` (веретёна в N2: sin, period 12, depth 0.35), `run_sleep(pathological=)`.

**`exam.py`** — `ExamRunner` (`pretrain`, `_calibrate`, `run_exam`), `ExamReport.passed`, генераторы
`gen_syntax_block`/`gen_logic_block`, `unfold`, `build_task_tables`. Константы задачи: `K=6`, `U=8`,
`LINE_D_B=0.089`, `LINE_D_C1=0.20`, `CUE_LEN=4`, `WIN_PER_BLOCK=12`, `SYN_BLOCKS=3`.

**`scaling.py`** — `ScalePoint`, `bench_scale`, `run_scaling` (log-log наклон, пиковая память, finite-скан).

**`utils.py`** — `get_device`, `set_seed`, `t`, `fmt_device`. Ничего содержательного сверх этого.

## 3. Инварианты: заявлено vs проверяемо

| Инвариант | Где заявлен | Где в коде | Чем проверяется в верификаторе | Факт этой сессии |
|---|---|---|---|---|
| `V ≥ V_rest` при `E_Cl=V_rest`, любом `g_GABA` | README:85, MATH §4 | `neurons.py:48-75` | T1 (g=50, 1000 шагов), T1b (g=500, 2000 шагов), допуск 1e-6 | н/з |
| `V_inf` — выпуклая комбинация | MATH §2 | `neurons.py:52-54` | T4, 1000 случайных `(G_e,G_gaba,E_cl)` | н/з |
| Явный Эйлер «проваливается» | MATH §5 | `neurons.py:79-88` | T2, 60 шагов, g=50 | н/з |
| KCC2 спасает при `E_Cl=-85` | MATH §6 | `neurons.py:91-102` | T3 (g=20, 500 шагов) | н/з |
| N3 без провала + патология | README:86, VERIFY §SLEEP | `sleep.py:88-160` | `run_sleep(False)` / `run_sleep(True)`, `bound_ok` | н/з |
| eager↔graph: `max|ΔV|<1e-4`, спайки бит-в-бит | VERIFY §GRAPHS | `cudagraph.py:129-160` | `graph_vs_eager`, 60 шагов, sigma=0 | н/з (на CPU стадия SKIP by design, `verify:161-164`) |
| Дрейф `w_slow=0` (`torch.equal`) | README:87 | `fastloop.py:118, 280-300`; `exam.py:256-260` | `wslow_drift`, 16 тензоров | н/з |
| Fan-in постоянен при масштабе | README:88 | `config.py:98-105` | косвенно, через наклон log-log | н/з |

Что можно утверждать без прогона:
- **[static]** Инвариант V≥V_rest следует из формулы `V_inf` при `E_Cl≥V_rest` и `G_e≥0` — это свойство
  выпуклой комбинации, а не эмпирика; `clamp(v_floor)` его дублирует.
- **[static]** T4 повторяет формулу **инлайном** (`verify:137`), а не вызывает `fixed_point` — тест
  проверяет математику, но не код `neurons.fixed_point`.
- **[analytic]** T2 при `tau_m=4` даёт множитель `1 − dt·51/tau_m = −11.75` [analytic]; за 60 шагов
  `20·11.75^n` переполняет float32 (≈3.4e38) к n≈35 → ожидаемый вывод `-inf`, а не «−3.4·10¹²» из MATH §5.
  Число `−3.4·10¹²` соответствует `tau_m=20` (множитель −1.55) — старой конфигурации.
  Причина расхождения видна в комментарии `verify:72` («tau_m=20») против `config.py:18` (`tau_m=4.0`).
- **[static]** В нормальном сне `v_floor` ставится равным `v_rest` (`sleep.py:100`), а `membrane_exact_step`
  клампит снизу (`neurons.py:75`) → `bound_ok` в норме истинно **по построению clamp'а**, независимо от
  GABA-математики. Тест SLEEP в норме тавтологичен; содержательна только «патологическая» ветка.

## 4. Ключевые константы и архитектурные решения

| Решение | Реализация | Проверяемые числа |
|---|---|---|
| Точный экспоненциальный интегратор | `membrane_exact_step`: `V_inf+(V−V_inf)·exp(−λdt)`, `λ=(1+G_e+g)/tau_m` | безусловно устойчив при любом dt [static] |
| GABA с reversal `E_Cl ≥ V_rest` | `I=g·(V−E_Cl)`, `e_cl_effective` = clamp | `E_Cl=V_rest` в homeostat → чистый шунт [static] |
| KCC2-гомеостат | одна строка `clamp(E_cl, min=v_rest)` | в `raw`-режиме отключается (`neurons.py:100`) |
| Фиксированный fan-in gather | `src_idx [n_dst,fan_in]`, `(w*spikes[src]).sum(1)` | fan_in 16/16/16/12 (rec), 10 (inh), 8 (proj) [static] |
| In-place состояние | `add_/mul_/copy_/index_add_`, `t.add_(1)` | нет `t=t+1`; проверено чтением `fastloop.py:165` [static] |
| Тензорные расписания модуляторов | `[T,Z]`, индекс `t % sched_len` | `sched_len=8192` [static] |
| S7 | `sigmoid`-пороги, `surge_max=1.5`, `ach_boost=0.25` | `r_lo=0.02`, `r_hi=0.35`, `slope=0.05` [static] |
| S7 внутри графа | `gaba_surge`/`ach_boost`/`scrub` — чистые тензорные операции | host-журнал только в `log_stats` [static] |
| k-WTA | `topk` по V среди спайков, k=max(8, n_e/12) | k=8 для всех зон (96/12, 48/12→8) [static] |
| WTA-маршрутизация | softmax по 2 зонам, гейты только на `gate_id≥0` | 2 гейта: 0=B, 1=C (`zones.py:112`) [static] |
| S7/сон GABA — шунтирующие | входят как `g_gaba` при `E_Cl=V_rest` | безопасно по построению [static] |
| Readout | `W_rB@tr_B`, `W_rC@tr_C`, смешение весами `mix` | 2×96 + 2 bias [static] |

Счётности (посчитаны по `config.py`/`zones.py`, не измерения):
**[static]** N = 392 (A 96+16, B 96+16, C 96+16, D 48+8); Z = 4; синапсов 8048
(1696·3 + 656 + 2304); обучаемых тензоров 12 групп + 4 readout = **16** (совпадает с VERIFY).
**[static]** Экзамен: 6 блоков × 12 окон × 6 шагов + 5 cue × 4 = 452 шага; окон 36+36, оцениваемых 30+30;
претрейн 60·6 + 120·6 = 1080 шагов.
**[analytic]** `g_sleep` не имеет цели затухания: при `tau_gaba=8`, dt=1 установившееся значение
`g/(1−e^(−1/8)) ≈ 8.51·g`, т.е. в N3 эффективная GABA-проводимость ≈ 29.8, а не «3.5» из README:86.

## 5. Что НЕ реализовано или упрощено

1. **Нет компартментной модели нейрона**: одна точка `V[n]`, нет дендритов/сома-аксона; в `neuroweave_v7restored/neuron.py:28` есть `MultiCompartmentALIF` — в v6 аналога нет.
2. **Нет `w_fast`**: есть только `w_slow` (`ConnGroup.w`) и мгновенные проводимости; двухвременнáя синаптическая память не реализована.
3. **Нет SON-шины** (`son.py`/`SONBus` есть только в `v7restored`).
4. **Нет полушарий и каллозальной связи** (`hemisphere.py`, `callosum.py` — только в `v7restored`).
5. **Нет EWC/STDP-модулей** (`ewc.py`, `stdp.py` — только в `v7restored`); пластичность сведена к дельта-правилу readout + упрощённому Хеббу.
6. **Нет реального языка**: задачи синтетические (33 токена, 2 класса, OR-пары), нет эмбеддингов/токенизатора/корпуса.
7. **Зоны A и D функционально мертвы** [analytic]: при штатных таблицах драйв A `0.06·2.06≈0.12` даёт `V_inf≈−62 мВ` против порога `−53` (ACh 0.5); D вообще не влияет на решение (readout только по B/C).
8. **I-интернейроны не спайкуют** [static]: все `ConnGroup` и `InputLine` адресованы только E-срезам (`zones.py:117,136,150`), входов на I нет → `inh_*` дают ровно 0 проводимости. [analytic] порог I `−42` мВ при шуме ~0.35 мВ/шаг недостижим, т.е. GABA-путь E↔I в рантайме мёртв, и вся GABA приходит из `g_sleep` и S7-surge.
9. **Нет сохранения/загрузки**: `torch.save/load` отсутствуют — чекпоинтов, воспроизводимого resume и экспорта весов нет.
10. **Нет тестов** (pytest/unittest), нет CI; единственный «тест» — сам `verify_cuda_graphs.py`.
11. **Нет планировщика обучения** сверх претрейна: эпох, валидации, early-stop, батчей нет.

## 6. Ограничения и риски (файл:строка)

1. `sleep.py:100` + `neurons.py:75` — инвариант в норме обеспечивается `clamp`, а не динамикой: SLEEP-PASS не доказывает GABA-математику.
2. `neurons.py:75` — `clamp(min=v_floor)` маскирует любую ошибку интегратора: при `v_floor=v_rest` провал физически невозможен.
3. `verify_cuda_graphs.py:72` — комментарий `tau_m=20` противоречит `config.py:18` (`4.0`); из-за этого эталон T2 в MATH §5 (`−3.4e12`) не соответствует текущему коду [analytic].
4. `verify_cuda_graphs.py:137` — T4 проверяет инлайн-формулу, а не `neurons.fixed_point`; регрессия в функции не будет поймана.
5. `verify_cuda_graphs.py:186` — критерий `speedup >= 1.5`, тогда как README:7/89 обещают «>2x»; PASS возможен при 1.5x.
6. `verify_cuda_graphs.py:217` — на CPU факторы (1,2): наклон log-log по двум точкам вырожден (секущая), R²=1 тождественно; заявление README:88 «0.6–0.7» этим не подтверждается статистически.
7. `scaling.py:90-91` — `resource` на Windows недоступен → `peak = nan`; метрика памяти на целевой ОС не работает, а `nan` не участвует в `ok`.
8. `scaling.py:44-72` — фоновая нагрузка берётся из `sleep.build_sleep_tables` (амплитуда 0.004) → спайков нет [analytic], `rate_mean→0`; проверка «гомеостаза частот» вырождается.
9. `exam.py:327-336` — в режиме `--graph` на каждом шаге вызывается `.item()`: это host-sync на каждом реплее, что противоречит README:89 («ноль синхронизаций»); сам захват при этом валиден.
10. `exam.py:42-53` — константы задачи (`K,U,LINE_D_*,CUE_LEN,SYN_BLOCKS`) лежат вне `config.py`, хотя README:50 утверждает «все константы» в конфиге.
11. `exam.py:69-77` — `passed` включает `v_min >= -74.0`; при `sigma_ht=0.6` шум ~0.35 мВ/шаг, порог мягкий и не связан с GABA-инвариантом (что и оговорено в комментарии).
12. `exam.py:44` — запас «И»-детектора ≈2 мВ [analytic]: при `LINE_D_B=0.089` две линии дают `V_inf≈−51.2`, порог `−53`; сдвиг ACh/threshold на ~2 мВ ломает синтаксис без видимой ошибки.
13. `fastloop.py:118, 280-300` — при `learn=True` распад `hebb_decay=3e-3` применяется каждый шаг: за 1080 шагов претрейна множитель `0.997^1080≈0.04` [analytic], т.е. `w_slow` в основном затухает, а не обучается (Хебб `eta=8e-5` на этом фоне слаб).
14. `fastloop.py:192-193` — `torch.zeros` + `index_copy` создают новый тензор в каждом шаге: для CUDA Graph допустимо (статический пул), но это скрытая аллокация, а не in-place (заявлено «строго in-place», README:105).
15. `config.py:83` — расписание циклично (`t % 8192`); при прогонах >8192 шагов импульсы DA/NE повторятся, о чём нигде не сказано.
16. `zones.py:114` — `n_gates=2` жёстко зашито; добавление зоны-получателя молча получит чужой строб.
17. Документы противоречат друг другу в числах: README:86 «−84.98», VERIFY:91 «−85.07», MATH:121 «−84.3» — три значения для близких величин; README:87 «логика 0.93–1.00» против VERIFY:93 «log acc=1.000».
18. `README.md:89` и `VERIFY.md:65` описывают GPU-поведение, которое в этой сборке **никем не подтверждено**: прогонов на CUDA не было.

## 7. Таблица «заявлено в README vs фактически в коде»

| Заявлено (README/VERIFY) | Фактически в коде | Вердикт |
|---|---|---|
| «conductance-based LIF, точный интегратор» (README:35) | `neurons.py:58-75`, формула совпадает | подтверждено [static] |
| «GABA с reversal `I=g(V−E_Cl)`» (README:36) | `neurons.py:52-54` | подтверждено [static] |
| «физическая изоляция зон» (README:37) | `zones.py:60-64,117-140` — пулы src по зоне | подтверждено [static] |
| «S7: GABA-всплеск, ACh-подкачка, NaN-скраб» (README:40) | `s7.py:30-56` | подтверждено [static] |
| «w_slow формируется претрейном» (README:42-44) | `fastloop.py:280-300`; распад доминирует [analytic] | частично: обучение есть, эффект сомнителен, н/з |
| «дрейф w_slow = 0 bit-exact» (README:87) | `learn=False` → нет записей; `torch.equal` по 16 тензорам | подтверждено структурно [static], н/з |
| «NE-пульс 8.0» (README:87) | `ne_k_det=4.0` × `|Δctx|≤2` = 8.0 — арифметика констант, не эмерджентность | подтверждено как константа [analytic] |
| «наклон log-log 0.6–0.7» (README:88) | CPU: 2 точки (`verify:217`) → метрика вырождена | не подтверждено, н/з |
| «ноль синхронизаций» (README:89) | `exam.py:327-336` — `.item()` каждый шаг в graph-режиме | опровергнуто для «Экзамена» [static] |
| «цель >2x» (README:7) | `verify:186` PASS при `>=1.5` | расхождение критерия |
| «патология −84.98» (README:86) | VERIFY:91 «−85.07», MATH:121 «−84.3» | документы не согласованы |
| «логика 0.93–1.00» (README:87) | VERIFY:93 «1.000» | документы не согласованы [doc] |
| «все константы в config.py» (README:50) | часть в `exam.py:42-53`, `sleep.py:49-50` | неточно |
| «без GPU полностью работоспособен» (README:117) | `st_graphs` → SKIP, `required=False` | подтверждено [static] |
| «A/B/C/D — сенсорная/синтаксис/логика/ассоциативная» | только имена+`role`; специализация из таблиц задач; A и D не спайкуют [analytic] | упрощение |

## 8. Что нужно, чтобы закрыть пробел (не выполнялось)

```powershell
cd neuroweave_v6
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python verify_cuda_graphs.py          # ожидается 5 PASS + 1 SKIP на CPU
python scripts/run_exam.py            # ожидается код 0 при rep.passed
python scripts/run_sleep.py           # ожидается «МАТЕМАТИКА РАБОТАЕТ»
```

Фиксировать в отчёте следует: тег каждой стадии, `V_min` T1/T1b/T2/T3/T4, `v_min_deep` в обеих ветках
сна, `syntax_acc/logic_acc/follow/drift_max/ne_pulse_max`, `ms_per_step` и наклон по факторам,
а также `g_sleep`-эффективность (§4). До этого все числа в §3–§7 остаются статическими оценками.

## 9. Итог

1. Ядро Этапа 6 (GABA-reversal, точный интегратор, KCC2, fan-in gather, in-place состояние) реализовано
   аккуратно и читается однозначно; математика инварианта корректна как формула.
2. Верификация проверяет **не всё, что заявляет**: SLEEP в норме тавтологичен из-за `clamp`, T4 не тестирует
   `fixed_point`, SCALE на CPU вырожден по числу точек, GRAPHS на CPU не выполняется вовсе.
3. Сеть функционально уже, чем описано: спайкуют практически только E-клетки зон B и C; I-клетки, зоны A и D
   вклада в поведение не дают.
4. Документация расходится с кодом в числах (`−84.98/−85.07/−84.3`), в критериях (>2x vs 1.5x) и в
   утверждении о нулевых синхронизациях; `tau_m=20` в комментарии верификатора устарел.
5. Прогонов не было: PASS/FAIL/SKIP и точные числа **не подтверждены** — отчёт фиксирует статический аудит
   и список того, что нужно измерить.
