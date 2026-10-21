# tests/test_gaba_reversal.py
# ЭТАП 6 — ХЛОРНЫЙ РЕВЕРС GABA (открытый вопрос Этапа 1, критично для
# протокола сна): I_GABA = g_GABA·(V − E_Cl) как ПРОВОДИМОСТЬ, свёрнутая
# в leak мембраны (exact integration, ур. 6' в core/neuron.py).
#
# Математические контракты (ответ на вопрос «как реализовать, чтобы
# мембрана не уходила ниже V_rest при глубоком торможении»):
#   * E_Cl = V_rest (дефолт) — ЧИСТЫЙ ШУНТ: гиперполяризации нет вовсе,
#     вход возбуждения делится на (1+G); вычитаемая форма v4.0 при той же
#     глубине торможения уводит мембрану скользяще под покой — контраст;
#   * E_Cl < V_rest — гиперполяризация ОГРАНИЧЕНА полом на E_Cl:
#     стационар чистого торможения (V_rest + G·E_Cl)/(1+G) > E_Cl всегда;
#   * exact integration: один шаг = замкнутая форма
#     V(t+dt) = α_eff·V + (1−α_eff)·V_inf, α_eff = α_m^(1+G) —
#     безусловная устойчивость при сколь угодно глубокой G;
#   * G = 0: ур. 6' вырождается в ур. 6 канона побитово.
import math

import torch

from src.spiker.core.neuron import MultiCompartmentALIF
from src.spiker.core.transmitters import IonotropicTransmitters

DEV = torch.device("cpu")


def _neuron(mode, e_cl=0.0, tau_m=10.0, gaba_gain=1.0, n=8):
    torch.manual_seed(0)
    nr = MultiCompartmentALIF(
        num_neurons=n, tau_m=tau_m, theta_0=1.0, sigma_noise=0.0,
        stochastic=False, gaba_mode=mode, e_cl=e_cl, gaba_gain=gaba_gain)
    nr.reset_state(1, DEV)
    return nr


# ----------------------------------------------------------------------
# 1. ГЛАВНАЯ ГАРАНТИЯ: мембрана не уходит ниже V_rest (E_Cl >= V_rest)
# ----------------------------------------------------------------------
def test_deep_inhibition_cannot_cross_v_rest():
    """Торможение ЛЮБОЙ глубины при E_Cl = V_rest не гонит мембрану под
    покой: стационар чистого шунта = V_rest. 200 тактов g=50 (G >> 1)."""
    nr = _neuron("conductance", e_cl=0.0)
    zero = torch.zeros(1, 8)
    deep = torch.full((1, 8), 50.0)

    # старт СВЕРХУ покоя (деполяризованное состояние) — шунт тянет К покою
    vs = []
    for _ in range(200):
        _, v, _ = nr(raw_basal=zero, raw_apical=zero, raw_gaba=deep)
        vs.append(v.clone())
    v_min = torch.stack(vs).min()
    assert float(v_min) >= -1e-6, (
        f"мембрана нырнула под V_rest при глубоком торможении: "
        f"min V = {float(v_min):.6f}")


def test_subtractive_mode_dives_below_v_rest_contrast():
    """Контраст с легаси v4.0 (вычитаемый ток): ТО ЖЕ торможение
    беспредельно уводит мембрану под покой — демонстрация проблемы,
    которую решает реверс (мотивация протокола сна)."""
    nr = _neuron("subtractive")
    zero = torch.zeros(1, 8)
    deep = torch.full((1, 8), 50.0)

    v_min = None
    for _ in range(200):
        _, v, _ = nr(raw_basal=zero, raw_apical=zero, raw_gaba=deep)
        v_min = float(v.min()) if v_min is None else min(v_min, float(v.min()))
    assert v_min < -0.5 * 1.0, (
        f"вычитаемая форма должна глубоко нырять (v_min = "
        f"{float(v_min):.3f}); реверс устраняет именно это")


# ----------------------------------------------------------------------
# 2. Пол гиперполяризации при E_Cl < V_rest: ограничение снизу на E_Cl
# ----------------------------------------------------------------------
def test_hyperpolarization_floor_at_e_cl():
    """E_cl = −0.5 < V_rest: мембрана гиперполяризуется, но НЕ глубже
    E_Cl — стационар (V_rest + G·E_Cl)/(1+G) строго больше E_Cl."""
    e_cl = -0.5
    nr = _neuron("conductance", e_cl=e_cl)
    zero = torch.zeros(1, 8)
    deep = torch.full((1, 8), 50.0)

    vs = []
    for _ in range(300):
        _, v, _ = nr(raw_basal=zero, raw_apical=zero, raw_gaba=deep)
        vs.append(v.clone())
    v_min = torch.stack(vs).min()
    assert float(v_min) >= e_cl - 1e-6, (
        f"мембрана пробила пол E_Cl: min V = {float(v_min):.6f} < {e_cl}")
    # и реально приближается к полу (глубокое торможение работает)
    assert float(v_min) < e_cl + 0.05, (
        f"торможение не гиперполяризует к E_Cl: min V = {float(v_min):.4f}")


# ----------------------------------------------------------------------
# 3. Точность exact integration: замкнутая форма ур. 6'
# ----------------------------------------------------------------------
def test_exact_integration_matches_closed_form():
    """Один шаг (докритический вход, спайков нет) совпадает с аналитикой:
    V(t+dt) = α_eff·V + (1−α_eff)·V_inf, где α_eff = exp(ln α_m·(1+G)),
    V_inf = (V_rest + G·E_Cl + r_m·i_exc)/(1+G); токи — стационары
    кинетик после одного шага из нулевого состояния."""
    tau_m, dt = 10.0, 1.0
    raw_basal = torch.full((1, 4), 0.4)   # докритический (нет спайков)
    raw_gaba = torch.full((1, 4), 2.0)
    e_cl = -0.25
    nr = _neuron("conductance", e_cl=e_cl, tau_m=tau_m, n=4)
    zero = torch.zeros(1, 4)

    _, v, _ = nr(raw_basal=raw_basal, raw_apical=zero, raw_gaba=raw_gaba)

    # ручной пересчёт (независимая арифметика, не код ядра)
    tr = nr.transmitters
    i_ampa = float(tr.decay_ampa) * 0.0 + 0.4          # след первого шага
    i_gaba = float(tr.decay_gaba) * 0.0 + 2.0
    G = 1.0 * i_gaba
    mg = torch.sigmoid(torch.tensor(12.0 * (i_ampa - 0.25)))
    mg_gate = float(torch.clamp((mg - tr.mg_zero_baseline)
                                / (1 - tr.mg_zero_baseline), 0, 1))
    i_exc = i_ampa + 1.5 * mg_gate * 0.0               # апикальный = 0
    alpha_eff = math.exp((-dt / tau_m) * (1.0 + G))
    v_inf = (0.0 + G * e_cl + 1.0 * i_exc) / (1.0 + G)
    expected = alpha_eff * 0.0 + (1 - alpha_eff) * v_inf

    assert torch.allclose(v, torch.full_like(v, expected), atol=1e-6), (
        f"exact integration разошёлся с аналитикой: {v.mean().item():.8f} "
        f"vs {expected:.8f}")


def test_shunting_divides_excitation():
    """E_Cl = V_rest: стационар под возбуждением = V_rest + r_m·i/(1+G) —
    торможение ДЕЛИТ вход (shunting), а не вычитает из напряжения."""
    nr = _neuron("conductance", e_cl=0.0)
    basal = torch.full((1, 8), 0.6)      # докритический суммарный ток
    zero = torch.zeros(1, 8)
    gaba_drive = torch.full((1, 8), 1.0)

    for raw_gaba, tag in ((zero, "free"), (gaba_drive, "shunted")):
        nr.reset_state(1, DEV)
        for _ in range(400):
            _, v, _ = nr(raw_basal=basal, raw_apical=zero, raw_gaba=raw_gaba)
        if tag == "free":
            v_free = v.detach().clone()
            i_ampa_ss = float(nr.transmitters.i_ampa.detach().mean())
        else:
            v_shunt = v.detach().clone()

    G = 1.0 * float(nr.transmitters.i_gaba.detach().mean())
    expected = (1.0 * i_ampa_ss) / (1.0 + G)   # v_rest = 0, r_m = 1
    assert torch.allclose(v_shunt, torch.full_like(v_shunt, expected), atol=1e-5), (
        f"шунт не делит возбуждение: стационар {v_shunt.mean().item():.6f}, "
        f"ожидание {expected:.6f}")
    assert float(v_shunt.max()) < float(v_free.max())


def test_unconditional_stability_at_absurd_conductance():
    """G ~ 1e4 (торможение абсурдной глубины): безусловная устойчивость
    exact integration — нет ни взрыва, ни NaN; мембрана в полосе."""
    nr = _neuron("conductance", e_cl=0.0)
    zero = torch.zeros(1, 8)
    absurd = torch.full((1, 8), 1.0e4)
    basal = torch.full((1, 8), 3.0)      # сильное возбуждение одновременно

    for _ in range(100):
        s, v, _ = nr(raw_basal=basal, raw_apical=zero, raw_gaba=absurd)
    assert torch.isfinite(v).all(), "NaN/inf в мембране при глубоком шунте"
    assert float(v.max()) <= 1e-6 + 1e-3, (
        "глубокий шунт обязан прижать мембрану к покою даже при сильном входе")


# ----------------------------------------------------------------------
# 4. G = 0: ур. 6' вырождается в ур. 6 канона побитово
# ----------------------------------------------------------------------
def test_zero_gaba_bitwise_equivalence():
    """Без торможения conductance- и subtractive-формы идентичны
    побитово (α_eff = α_m, V_inf = V_rest + r_m·i_soma): смена режима
    не трогает динамику вне тормозного пути."""
    torch.manual_seed(7)
    a = MultiCompartmentALIF(num_neurons=8, tau_m=8.0, sigma_noise=0.0,
                             stochastic=False, gaba_mode="conductance")
    b = MultiCompartmentALIF(num_neurons=8, tau_m=8.0, sigma_noise=0.0,
                             stochastic=False, gaba_mode="subtractive")
    a.reset_state(2, DEV); b.reset_state(2, DEV)
    g = torch.Generator().manual_seed(11)
    zero = torch.zeros(2, 8)

    for _ in range(60):
        basal = torch.rand(2, 8, generator=g) * 2.0
        sa, va, _ = a(raw_basal=basal, raw_apical=zero, raw_gaba=zero)
        sb, vb, _ = b(raw_basal=basal, raw_apical=zero, raw_gaba=zero)
        assert torch.equal(va, vb), "мембраны разошлись при G = 0"
        assert torch.equal(sa, sb), "спайки разошлись при G = 0"


# ----------------------------------------------------------------------
# 5. Критично для сна: восстановление после глубокого торможения
# ----------------------------------------------------------------------
def test_recovery_after_deep_inhibition_is_instant():
    """Сценарий протокола сна (tau_m = 50, как правое полушарие):
    длинное окно глубокого торможения -> снятие.
    Conductance: мембрана стояла НА покое — отклик лимитирован ТОЛЬКО
    закрытием GABA-каналов (tau_gaba = 6), ~10 тактов.
    Subtractive: мембрана сидела глубоко под покоем (−G) — подъём
    тянет tau_m ~ 50: реплей задержан на порядок (в этом и была
    проблема сна v4.0: длинные мёртвые окна после торможения)."""
    def _recover_ticks(mode):
        nr = _neuron(mode, e_cl=0.0, tau_m=50.0)
        zero = torch.zeros(1, 8)
        deep = torch.full((1, 8), 5.0)   # G_ss ~ 33 — глубокий шунт
        probe = torch.full((1, 8), 2.0)  # надпороговый тест-вход
        for _ in range(80):              # фаза «глубокий сон»
            nr(raw_basal=zero, raw_apical=zero, raw_gaba=deep)
        for t in range(200):             # пробуждение: вход вернулся
            s, _, _ = nr(raw_basal=probe, raw_apical=zero, raw_gaba=zero)
            if float(s.detach().sum()) > 0:
                return t + 1
        return 999

    t_cond = _recover_ticks("conductance")
    t_sub = _recover_ticks("subtractive")
    assert t_cond <= 15, (
        f"чистый шунт отвечает за время закрытия каналов (tau_gaba): "
        f"{t_cond} тактов")
    assert t_sub >= 5 * t_cond, (
        f"вычитаемая форма обязана восстанавливаться на порядок дольше "
        f"({t_sub} vs {t_cond}) — иначе контраст сна не демонстрируется")


# ----------------------------------------------------------------------
# 6. Уровень зоны: реверс в живой рекуррентной динамике
# ----------------------------------------------------------------------
def test_zone_lateral_gaba_shallower_than_subtractive():
    """Зона с живым латеральным торможением (W_ie): conductance-реверс
    держит мембрану ЗНАЧИТЕЛЬНО мельче, чем вычитаемая форма на том же
    потоке (GABA-вклад больше не гиперполяризует). Глубина ниже
    постспайкового сброса/отрицательных AMPA-токов остаётся возможной —
    floor-гарантия касается именно тормозного члена (см. тесты 1-2)."""
    from src.spiker.topology.zone import Zone

    def _vmin(mode):
        torch.manual_seed(5)
        zone = Zone("G", side="left", n_inputs=16, n_exc=48, n_inh=24,
                    input_gain=3.0, gaba_mode=mode)
        zone.reset_state(2, DEV)
        v_min = None
        for t in range(200):
            gen = torch.Generator().manual_seed(100 + t)
            s_in = (torch.rand(2, 16, generator=gen) < 0.3).float()
            out = zone.step(s_in)
            v = out["v_mem"].detach()
            v_min = v.min().clone() if v_min is None \
                else torch.minimum(v_min, v.min())
        return float(v_min)

    v_cond = _vmin("conductance")
    v_sub = _vmin("subtractive")
    assert v_cond > v_sub + 1.0, (
        f"реверс не обмелчил мембрану зоны: conductance {v_cond:.3f} vs "
        f"subtractive {v_sub:.3f}")


def test_conductance_mode_is_zone_default():
    """Дефолт зоны — conductance-реверс (канон Этапа 6): E_Cl = v_rest."""
    from src.spiker.topology.zone import Zone

    zone = Zone("Z", side="left")
    tr = zone.exc.transmitters
    assert tr.gaba_mode == "conductance"
    assert tr.e_cl == 0.0 == zone.exc.v_rest, \
        "дефолтный E_Cl обязан совпадать с v_rest (чистый шунт)"
    assert zone.inh.transmitters.gaba_mode == "conductance"
