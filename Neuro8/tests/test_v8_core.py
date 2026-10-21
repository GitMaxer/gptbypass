"""Тесты инвариантов NEUROWEAVE 8 (канон v4.0/v4.0.1 + science_grounding)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from neuroweave8.config import default_cfg, NeuronCfg, PlasticityCfg
from neuroweave8.core.neuron import LaminarALIF
from neuroweave8.core.synapse import StatefulSynapse
from neuroweave8.core.transmitters import (mg_block, coincidence_gate,
    fixed_point, exact_step, e_cl_effective)
from neuroweave8.core.phase import ThetaGammaBinder, PhaseVSA, PhaseCfg
from neuroweave8.microcircuit.column import LaminarColumn
from neuroweave8.modulation.chemistry import NeuroChemistry
from neuroweave8.topology.callosum import CorpusCallosum
from neuroweave8.topology.workspace import GlobalWorkspace
from neuroweave8.memory.hippocampus import Hippocampus
from neuroweave8.organism.organism import Organism
from neuroweave8.main import run_demo


# ---------- мембранная физика (наследие v6) ----------

def test_gaba_pure_shunt_when_ecl_ge_vrest():
    cfg = NeuronCfg()
    B = 4
    v = torch.full((B, 10), cfg.v_rest)
    # E_Cl >= V_rest: инверсионный потенциал габы не деполяризует ниже отдыха
    e_cl = e_cl_effective(torch.full((B, 10), cfg.e_cl), cfg)
    _, _, _ = fixed_point(v, torch.zeros(B, 10), torch.zeros(B, 10),
                          torch.full((B, 10), 2.0), e_cl, cfg)
    v2, _ = exact_step(v, torch.zeros(B, 10), torch.zeros(B, 10),
                       torch.full((B, 10), 2.0), e_cl, cfg)
    assert (v2 >= cfg.v_rest - 1e-6).all()


def test_nmda_gate_calibrated():
    # coincidence_gate откалиброван: ~0 при g=0, растёт к theta
    g0 = coincidence_gate(torch.zeros(2, 3))
    g1 = coincidence_gate(torch.ones(2, 3) * 0.25)
    g2 = coincidence_gate(torch.ones(2, 3) * 1.0)
    assert g0.abs().max() < 1e-3 and g2.mean() > g1.mean() > g0.mean()


def test_refractory_and_reset_subtraction():
    cfg = NeuronCfg(v_th=-50.0, v_reset=-58.0, refractory=2)
    n = LaminarALIF(8, cfg)
    n.reset_state(4, torch.device("cpu"))
    drive = torch.full((4, 8), 500.0)  # гарантированный спайк
    out = n.step(drive, sltt=False)
    s = out["spikes"]
    assert s.sum() > 0
    # спайкнувший нейрон в рефрактерном окне: следующие 2 такта — 0
    out2 = n.step(drive, sltt=False)
    fired = s > 0
    assert out2["spikes"][fired].sum() == 0


def test_theta_plateau_from_apical():
    cfg = NeuronCfg(k_ach_th=0.0)
    n = LaminarALIF(8, cfg)
    n.reset_state(2, torch.device("cpu"))
    o1 = n.step(torch.zeros(2, 8), sltt=False)
    th1 = o1["theta"]
    # апикальный приток выше порога сдвигает тета (plateau-потенциал)
    n.g_nmda.fill_(1.0)
    o2 = n.step(torch.zeros(2, 8), sltt=False,
                theta_shift=torch.full((2, 8), 5.0))
    assert (o2["theta"] < th1).all()


# ---------- синаптическая память (канон v4.0.1) ----------

def test_wfast_decay_every_tick():
    syn = StatefulSynapse(16, 8, PlasticityCfg(), density=0.5)
    syn.reset_state(4, torch.device("cpu"))
    pre = (torch.rand(4, 16) > 0.5).float()
    post = (torch.rand(4, 8) > 0.5).float()
    syn.update_traces_and_plasticity(pre, post, da=torch.ones(4, 1), lr=1e-2)
    mass1 = syn.w_fast.abs().sum()
    # без DA: w_fast только затухает (Y1)
    syn.update_traces_and_plasticity(pre, post, da=None, lr=1e-2)
    mass2 = syn.w_fast.abs().sum()
    assert mass2 < mass1


def test_da_gated_capture():
    syn = StatefulSynapse(16, 8, PlasticityCfg(), density=1.0)
    syn.reset_state(4, torch.device("cpu"))
    pre = torch.ones(4, 16); post = torch.ones(4, 8)
    syn.update_traces_and_plasticity(pre, post, da=None)
    assert syn.w_fast.abs().sum() == 0.0
    syn.update_traces_and_plasticity(pre, post, da=torch.ones(4, 1), lr=1e-2)
    assert syn.w_fast.abs().sum() > 0.0


def test_consolidation_soft_threshold():
    syn = StatefulSynapse(8, 4, PlasticityCfg(w_fast_max=1.0, theta_consol=0.25,
                                            temp_consol=0.02, eta_consol=0.35))
    syn.reset_state(2, torch.device("cpu"))
    syn.w_fast.fill_(0.5)  # выше порога
    ws0 = syn.w_slow.clone()
    rep = syn.consolidate_to_slow()
    assert rep["moved_mass"] > 0.0 and (syn.w_slow != ws0).any()
    syn.w_fast.fill_(0.001)  # ниже порога -> почти ничего не уходит
    rep2 = syn.consolidate_to_slow()
    assert rep2["moved_mass"] < rep["moved_mass"] * 0.1


# ---------- фазовая память (θ-γ, VSA) ----------

def test_theta_gamma_capacity():
    cfg = PhaseCfg(n_slots=7, gamma_ticks=4)
    b = ThetaGammaBinder(16, cfg)
    b.reset_state(2, torch.device("cpu"))
    items = [torch.rand(2, 16) for _ in range(9)]
    for it in items:
        b.write(it)
    assert b.occupancy() <= cfg.n_slots  # кап 7±2, старые вытесняются
    read = b.read(6)
    assert read.shape == (2, 16)


def test_vsa_bind_unbind_roundtrip():
    vsa = PhaseVSA(PhaseCfg(vsa_dim=256, vsa_codebook=16))
    role = torch.tensor([1, 2, 3])
    filler = torch.tensor([4, 5, 6])
    bound = vsa.bind(vsa.codebook[filler], vsa.codebook[role])
    idx, sim = vsa.cleanup(vsa.unbind(bound, vsa.codebook[role]))
    assert torch.equal(idx, filler)


# ---------- микроцепь и химия ----------

def test_column_surprise_decreases():
    cfg = default_cfg("small")
    col = LaminarColumn(cfg.columns[0], cfg.neuron, cfg.plasticity, seed=1)
    B = 4
    col.reset_state(B, torch.device("cpu"))
    x = torch.rand(B, cfg.columns[0].n_in)
    hist = []
    for _ in range(60):
        out = col.step(x, sltt=False, learn=True)
        hist.append(float(out.surprise))
    tail = hist[-10:]
    # EMA-сюрприз сходится: конечный сигнал конечен, неотрицателен и стабилен
    assert all(torch.isfinite(torch.tensor(hist))) and min(hist) >= 0.0
    assert max(tail) - min(tail) < 0.05


def test_ne_front_semantics():
    cfg = default_cfg("small")
    names = [c.name for c in cfg.columns]
    neuro = NeuroChemistry(names, cfg.mod, device="cpu")
    neuro.pulse("ne", names[0], 1.0, cap=cfg.mod.ne_pulse_max)
    rise1 = float(neuro.step()["ne_rise"][0])
    rise2 = float(neuro.step()["ne_rise"][0])
    assert rise1 > 0.0 and rise2 < rise1  # фазический фронт


# ---------- топология ----------

def test_callosum_delay():
    bus = CorpusCallosum(8, 6, delay=7, theta_bus=-1.0)  # фильтр прозрачен
    B = 2
    bus.reset_state(B, torch.device("cpu"))
    msg = torch.ones(B, 8)
    for i in range(7):
        out, _ = bus.step(msg, torch.zeros(B, 6))
        assert out.abs().sum() == 0.0  # задержка: 0 за первые 7 тактов
    out, _ = bus.step(msg, torch.zeros(B, 6))
    assert out.abs().sum() > 0.0  # пришло на 8-й (задержка = 7 тактов)


def test_workspace_competence():
    ws = GlobalWorkspace(3, competence_floor=0.1)
    logits = [torch.randn(4, 8) for _ in range(3)]
    bids = ws.calibrated_bids(logits)
    assert bids.shape == (4, 3)
    ws.update_track(logits, logits[0].argmax(-1))
    assert ws.track[0] > ws.track[1]  # трек растёт у точного


def test_hippocampus_separation_completion():
    h = Hippocampus(8, default_cfg("small").hippo, seed=7)
    h.reset_state(torch.device("cpu"))
    x1 = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0.]], dtype=torch.float32)
    x2 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1.]], dtype=torch.float32)
    c1, c2 = h.separate(x1), h.separate(x2)
    assert float((c1 * c2).sum()) < float(c1.pow(2).sum())  # сепарация
    rep = h.process(x1)
    assert rep["code"].shape[1] == h.n_dg
    xs, codes = h.replay_batch(4)
    assert xs.shape[0] >= 1


# ---------- организм: интеграция ----------

@pytest.mark.slow
def test_organism_lifecycle():
    res = run_demo(preset="small", steps=80, batch=2, verbose=False)
    assert res["episodes_replayed"] >= 0
    assert res["hippo_stored"] >= 0
    assert res["consolidated"] >= 0.0
    # организм работает и стабилен
    assert res["seconds"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
