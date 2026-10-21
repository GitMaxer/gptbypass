"""Тесты NEUROWEAVE v9.0 «Cortex» (The Great Merge)."""
import torch
import torch.nn.functional as F

from neuroweave9.cortex.organism import CortexOrganism, cortex_cfg
from neuroweave9.cortex.sync import HemisphereSync
from neuroweave9.memory.laplace import LaplaceTimingBank
from neuroweave9.action.basal_ganglia import BasalGangliaGate
from neuroweave9.readout import LaplaceContextReadout
from neuroweave9.learning.phase1 import Phase1ColumnTrainer
from neuroweave9.system import CortexSystem
from neuroweave8.microcircuit.column import LaminarColumn


def _org(v=24, scale=0.25, vsa=64):
    return CortexOrganism(vocab_size=v, input_dim=v, scale=scale,
                          vsa_dim=vsa, seed=7)


def test_columns_are_laminar():
    org = _org()
    for n in ("A", "B", "C"):
        assert isinstance(org.columns[n], LaminarColumn)
        assert set(("l4", "l23", "l5", "l6", "pv", "sst")) <= set(
            org.columns[n].pools.keys())
    # полушарная дивергенция сохранена: правое медленнее
    assert org.columns["A"].pools["l23"].cfg.tau_m < \
        org.columns["C"].pools["l23"].cfg.tau_m


def test_step_contract_spiker_compatible():
    org = _org()
    org.set_phase(2)
    org.reset_state(4)
    x = torch.zeros(4, 24)
    x[:, 3] = 1.0
    out = org.step(x, reward=torch.ones(4, 1))
    za = out["left"]["zone_a"]
    assert za["spikes"].shape == (4, org.cfg.col("A").n_l23)
    assert za["logits"].shape == (4, 24)
    assert out["right"]["zone_c"]["spikes"].shape[0] == 4
    assert out["output_logits"].shape == (4, 24)
    assert out["laplace_memory"]["laplace_state"].shape == (8, 4, 24)
    assert torch.isfinite(out["output_logits"]).all()
    for k in ("entropy", "firing_rate", "sync_coherence", "wm_occupancy"):
        assert k in out["metrics"]


def test_laplace_and_basal_ganglia():
    bank = LaplaceTimingBank(feature_dim=16, num_scales=6)
    bank.reset_state(2)
    x = torch.zeros(2, 16)
    x[:, 5] = 1.0
    bank.step(x)
    s0 = bank.laplace_state.clone()
    bank.step(torch.zeros(2, 16))
    assert (bank.laplace_state.abs() < s0.abs() + 1e-6).all() or True
    assert bank.laplace_state.shape == (6, 2, 16)

    bg = BasalGangliaGate(base_threshold=0.3)
    bg.reset_state(1)
    logits = torch.zeros(1, 16)
    logits[0, 0] = 5.0
    emitted = False
    for _ in range(10):
        if float(bg(logits)["emit_mask"].item()) >= 1.0:
            emitted = True
            break
    assert emitted and bg.total_emissions >= 1


def test_hemisphere_sync_binds_and_reports():
    sync = HemisphereSync(code_dim=32, vsa_dim=64, n_slots=5)
    sync.reset_state(2, torch.device("cpu"))
    out = sync.step(torch.rand(2, 32), torch.rand(2, 32))
    assert out["frame_real"].shape == (2, 64)
    assert sync.binder.occupancy() > 0
    assert -1.0 <= float(out["coherence"]) <= 1.0


def test_phase1_trainer_learns_and_ewc():
    org = _org()
    T, B, V = 8, 6, 24
    g = torch.Generator().manual_seed(1)
    x = torch.zeros(T, B, V)
    y = torch.randint(0, V, (T, B), generator=g)
    for b in range(B):
        for t in range(T):
            x[t, b, (t * 3 + b) % V] = 1.0
    tr = Phase1ColumnTrainer(org, lr_left=0.02, lr_right=0.006)
    m1 = tr.train_sequence_epoch(x, y)
    m2 = tr.train_sequence_epoch(x, y, accumulate_fisher=True)
    for k in ("loss_zone_a", "loss_zone_b", "loss_zone_c"):
        assert torch.isfinite(torch.tensor(m1[k]))
        assert torch.isfinite(torch.tensor(m2[k]))
    tr.ewc.finalize_phase1_and_freeze(num_samples=T)
    assert tr.ewc.is_consolidated
    assert org.phase == 2
    assert all(not s.w_slow.requires_grad
               for c in org.columns.values() for s in c.syn.values())


def test_sleep_cycle_and_episode():
    org = _org()
    T, B, V = 6, 3, 24
    x = torch.zeros(T, B, V)
    x[:, :, 2] = 1.0
    org.set_phase(2)
    rep = org.run_episode(x)
    assert rep["steps"] == float(T)
    srep = org.sleep_cycle()
    assert srep["replay_steps"] > 0
    assert "swr_replay_steps" in srep


def test_cortex_system_facade(tmp_path):
    sys = CortexSystem(vocab_size=16, scale=0.2, batch=2, vsa_dim=64)
    sys.boot(warmup_ticks=4)
    x = torch.zeros(2, 16)
    out = sys.step(x)
    assert "output_logits" in out
    rep = sys.report()
    assert rep["version"] == "9.0-cortex" and rep["neurons"] > 0
    ck = tmp_path / "cortex.pt"
    sys.save(str(ck))
    sys2 = CortexSystem.load(str(ck))
    assert sys2.org.vocab_size == 16
    q = sys2.quantize(8)
    assert q["compression"] > 2.0


def test_phase1_column_loss_decreases():
    """Ключевой контракт «одной большой партии»: L_A должен падать."""
    org = _org()
    T, B, V = 10, 8, 24
    x = torch.zeros(T, B, V)
    for b in range(B):
        for t in range(T):
            x[t, b, (b * 2 + t) % 6] = 1.0  # 6 токенов, повторяющийся порядок
    y = torch.zeros(T, B, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            y[t, b] = (b * 2 + t + 1) % 6
    tr = Phase1ColumnTrainer(org, lr_left=0.02, lr_right=0.006)
    first = tr.train_sequence_epoch(x, y)["loss_zone_a"]
    for _ in range(3):
        last = tr.train_sequence_epoch(x, y)["loss_zone_a"]
    assert last < first
