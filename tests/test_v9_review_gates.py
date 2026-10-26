"""Гейты ревью: регрессионные тесты на фиксы раунда 3-4.

Проверяют, что «починено» не регрессирует: DDM-эмиссия как выбор
(не argmax), модуляция порога, w_fast_readout, нормализация Фишера,
WM-leak и неразвырождённость coherence.
"""
import torch

from neuroweave9.action.basal_ganglia import BasalGangliaGate
from neuroweave9.cortex.organism import CortexOrganism
from neuroweave8.learning.ewc import EWCRegularizer


def test_bg_emit_is_choice_not_argmax():
    """emit_mask: сильный margin -> эмиссия есть, но не каждый такт
    (acc обнуляется при эмиссии); плоские логиты -> порог держит паузу."""
    bg = BasalGangliaGate(base_threshold=0.55, evidence_gain=8.0)
    bg.reset_state(1)
    # Переменная уверенность: то margin 0.3, то ~0 — фракция в полосе выбора.
    emits = []
    for t in range(200):
        logits = torch.zeros(1, 8)
        if t % 3 != 0:
            logits[0, 0], logits[0, 1] = 0.35, 0.05  # уверенный margin
        else:
            logits[0, 0], logits[0, 1] = 0.010, 0.009  # почти плоско
        emits.append(float(bg(logits)["emit_mask"].item()))
    frac = sum(emits) / len(emits)
    assert 0.10 < frac < 0.70, f"emit fraction {frac:.2%} — не выбор"
    # Плоское: порог не достигается.
    bg.reset_state(1)
    flat = torch.zeros(1, 8); flat[0, 0], flat[0, 1] = 0.001, 0.0009
    assert all(float(bg(flat)["emit_mask"]) == 0.0 for _ in range(10))


def test_bg_threshold_modulated():
    """Порог реагирует на 5-HT (+), ACh (+), orexin (-), surprise (+)."""
    bg = BasalGangliaGate()
    bg.reset_state(1)
    logits = torch.zeros(1, 4); logits[0, 0] = 0.3
    base = float(bg(logits)["threshold"])
    bg.reset_state(1)
    hi5ht = float(bg(logits, serotonin_5ht=torch.ones(1, 1))["threshold"])
    bg.reset_state(1)
    hi_orex = float(bg(logits, orexin_level=torch.ones(1, 1))["threshold"])
    bg.reset_state(1)
    hi_surp = float(bg(logits, surprise_level=torch.ones(1, 1))["threshold"])
    assert hi5ht > base and hi_surp > base and hi_orex < base


def test_w_fast_readout_grows_and_shifts_logits():
    """w_fast_readout: обучается локально, меняет logits — rule inversion
    без retrain замороженной головы."""
    org = CortexOrganism(vocab_size=48, input_dim=48, scale=0.3,
                         vsa_dim=96, seed=0)
    org.set_phase(2)
    org.reset_state(1)
    col = org.columns["A"]
    x = torch.zeros(1, 48); x[0, 3] = 1.0
    l0 = col.logits().clone()
    for t in range(30):
        x = torch.zeros(1, 48); x[0, (t * 5) % 48] = 1.0
        org.step(x, learn=True, sync_metrics=False)
    assert float(col.w_fast_readout.abs().sum()) > 0.0
    assert not torch.allclose(col.logits(), l0)


def test_ewc_fisher_normalized():
    """Фишер нормирован на max — штраф в том же масштабе, что и CE."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    ewc = EWCRegularizer({"w": p}, lam=1.0, normalize_fisher=True)
    def loss_fn():
        return (p ** 2).sum()
    ewc.estimate_fisher(loss_fn, n_samples=4)
    assert abs(ewc.fisher["w"].max().item() - 1.0) < 1e-6
    assert ewc.fisher_mass() > 0.0


def test_coherence_and_wm_occupancy_not_degenerate():
    """coherence меняется по тактам (не залипает), occupancy <= 1 и
    затухает в тишине (leak)."""
    org = CortexOrganism(vocab_size=48, input_dim=48, scale=0.3,
                         vsa_dim=96, seed=7)
    org.set_phase(2)
    org.reset_state(1)
    cohs, occs, amps = [], [], []
    for t in range(24):
        x = torch.zeros(1, 48)
        if t < 12:
            x[0, (t * 3) % 48] = 1.0
        org.step(x, learn=False, sync_metrics=False)
        cohs.append(float(org.sync.coherence))
        occs.append(float(org.sync.binder.occupancy()))
        amps.append(float(org.sync.binder.buf.abs().mean()))
    # coh=1.0 в тишине = оба полушария релаксировали в один аттрактор;
    # в активном окне должна быть вариативность (не залипание).
    assert max(cohs[:12]) - min(cohs[:12]) > 0.01, "coherence залипла"
    # WM-leak: затухание буфера проверяется юнит-тестом ниже (биндер пишется
    # каждый такт, поэтому организм-уровень не изолирует leak).
    assert all(0.0 <= o <= 1.0 for o in occs)


def test_binder_leak_decays_content():
    """ThetaGammaBinder: без новых записей содержимое затухает (leak 0.985)."""
    from neuroweave8.core.phase import ThetaGammaBinder
    from neuroweave8.config import PhaseCfg
    b = ThetaGammaBinder(16, PhaseCfg(n_slots=7))
    b.reset_state(1, torch.device("cpu"))
    b.write(torch.rand(1, 16))
    a0 = float(b.buf.abs().mean())
    for _ in range(20):
        b.write(torch.zeros(1, 16))
    a1 = float(b.buf.abs().mean())
    assert a1 < a0 * 0.9, f"leak не затухает: {a0:.4f} -> {a1:.4f}"
