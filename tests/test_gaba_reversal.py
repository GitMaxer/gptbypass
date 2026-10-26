"""Доказательство пола мембраны: проводимостная GABA с E_Cl >= V_rest
физически не может увести мембрану ниже покоя — инвариант, а не подгонка
(порт test_gaba_reversal.py из NW8.1 / Spiker)."""
import torch

from neuroweave8.config import NeuronCfg
from neuroweave8.core.neuron import LaminarALIF


def test_gaba_never_drives_v_below_rest():
    """Чистый GABA-драйв без возбуждения: v остаётся >= v_rest всегда."""
    cfg = NeuronCfg(e_cl=-70.0, ecl_mode="homeostat", sigma=0.0)
    pool = LaminarALIF(24, cfg, name="gaba_test")
    pool.reset_state(2, torch.device("cpu"))
    basal = torch.zeros(2, 24)
    gaba = torch.full((2, 24), 5.0)  # мощный тормозной драйв
    for _ in range(40):
        out = pool.step(basal, None, gaba, sltt=False)
        assert float(out["v"].min()) >= cfg.v_rest - 1e-4, (
            f"мембрана ушла ниже покоя: {float(out['v'].min())}")

    # И в 'патологическом' режиме с ECl < v_rest проводимость всё равно
    # ограничена v_floor — инвариант пола сохраняется.
    cfg2 = NeuronCfg(e_cl=-80.0, ecl_mode="raw", sigma=0.0)
    pool2 = LaminarALIF(24, cfg2, name="gaba_raw")
    pool2.reset_state(2, torch.device("cpu"))
    for _ in range(40):
        out = pool2.step(basal, None, gaba, sltt=False)
        assert float(out["v"].min()) >= cfg2.v_floor - 1e-4


def test_shunting_reduces_excitation():
    """Шунт GABA при ECl=V_rest ослабляет возбуждающий драйв (деление)."""
    cfg = NeuronCfg(e_cl=-70.0, ecl_mode="homeostat", sigma=0.0)
    pool = LaminarALIF(24, cfg, name="shunt")
    pool.reset_state(2, torch.device("cpu"))
    basal = torch.full((2, 24), 3.0)
    for _ in range(10):
        out_clean = pool.step(basal, None, torch.zeros(2, 24), sltt=False)
    v_clean = float(out_clean["v_inf"].mean())
    pool.reset_state(2, torch.device("cpu"))
    for _ in range(10):
        out_shunt = pool.step(basal, None, torch.full((2, 24), 5.0),
                              sltt=False)
    v_shunt = float(out_shunt["v_inf"].mean())
    assert v_shunt < v_clean, "шунт не ослабил возбуждение"
    assert v_shunt >= cfg.v_rest - 1e-4
