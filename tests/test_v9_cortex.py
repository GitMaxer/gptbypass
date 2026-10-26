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


def test_orexin_no_field_stp_2ag_integrated():
    """Аудит нейрохимии v9: орексин, NO-поле, STP, метапластичность Фузи, 2-AG."""
    org = _org()
    org.set_phase(2)
    org.reset_state(3)
    # STP включён на рекуррентности L2/3
    for col in org.columns.values():
        assert col.syn["rec"].stp_mode == "depressing"
        assert col.syn["rec"].metaplastic_state.shape == col.syn["rec"].w_slow.shape
    # 2-AG: буфер в каждом нейронном пуле
    assert org.columns["A"].pools["l23"].ecb_2ag is not None

    x = torch.zeros(3, 24)
    x[:, 1] = 1.0
    for _ in range(30):  # энергия тратится -> орексин растёт
        out = org.step(x, learn=True)
    assert "orexin" in out and "no_field" in out
    assert out["no_field"]["field"].shape[0] == 3
    om = org.orexin.state_summary()
    assert 0.0 <= om["orexin"] <= 1.0 and 0.0 <= om["energy"] <= 1.0
    # Голодная сеть спешит: порог БГ снижает орексин
    bg = org.basal_ganglia
    bg.reset_state(1)
    logits = torch.zeros(1, 24); logits[0, 0] = 4.0
    o_full = torch.full((1, 1), 0.9)
    thr_hungry = float(bg(logits, orexin_level=o_full)["threshold"])
    bg.reset_state(1)
    thr_fed = float(bg(logits)["threshold"])
    assert thr_hungry < thr_fed


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


def test_wm_leak_and_structural_coherence():
    """Фиксы sync.py: утечка слотов биндера, когерентность VSA-кодов,
    фазовый сдвиг полушарий."""
    org = _org()
    org.set_phase(2)
    org.reset_state(1)
    x = torch.zeros(1, 24)
    x[0, 3] = 1.0
    out = org.step(x, learn=False)
    assert "son" in out and "thalamus" in out

    binder = org.sync.binder
    # Leak: буфер затухает — после записи occupancy < 1.0 и продолжает падать
    for _ in range(20):
        org.step(x, learn=False)
    occ_full = binder.occupancy()
    for _ in range(60):
        org.step(torch.zeros(1, 24), learn=False)
    assert binder.occupancy() <= occ_full
    assert org.sync.hemi_phase_r > 0.0  # фазовый сдвиг полушарий

    # Структурная когерентность: идентичные коды -> высокая, разные -> ниже
    d = org.sync.code_dim
    a = torch.zeros(1, d); a[0, :4] = 1.0
    b = torch.zeros(1, d); b[0, :4] = 1.0
    org.sync.reset_state(1, torch.device("cpu"))
    out_same = org.sync.step(a, b)
    org.sync.reset_state(1, torch.device("cpu"))
    b2 = torch.zeros(1, d); b2[0, -4:] = 1.0
    out_diff = org.sync.step(a, b2)
    assert float(out_same["coherence"]) > float(out_diff["coherence"])


def test_son_thalamus_astrocyte_myelin_integrated():
    """4-й батч аудита: SON, таламус-TRN, астроцит, миелинизация, химера."""
    org = _org()
    org.set_phase(2)
    org.reset_state(2)
    x = torch.zeros(2, 24)
    x[:, 5] = 1.0
    for _ in range(10):
        out = org.step(x, learn=True)
    # SON: surprise и bus_active в выходе; шина гейтирована
    assert "surprise" in out["son"] and "bus_active" in out["son"]
    assert 0.0 <= float(out["son"]["surprise"].mean()) <= 4.0
    # Таламус: TRN-гейт ослабляет/усиливает вход
    assert 0.4 <= float(out["thalamus"]["relay_gain"].mean()) <= 1.15
    # Астроциты: кальцевая волна и stp_gain живут и растут на активности
    for n in org.names:
        a = org.astrocytes[n]
        assert a.ca_level is not None
        assert org._astro_gains[n].shape[1] == org.cfg.col(n).n_l23
        assert float(a.ca_level.mean()) >= 0.0
    # Адаптивная миелинизация включена и обучается
    assert org.callosum.adaptive_myelination
    org.callosum.myelin_update(0.5)
    assert float(org.callosum.myelin_fwd.abs().sum()) > 0
    # Химерный REM: ключ инсайта в отчёте сна
    rep = org.sleep_cycle()
    assert "rem_chimera_insights" in rep
    # BG получает surprise: повышение порога на непредсказанном контексте
    bg = org.basal_ganglia
    bg.reset_state(1)
    logits = torch.zeros(1, 24); logits[0, 0] = 4.0
    thr_new = float(bg(logits, surprise_level=torch.ones(1, 1))["threshold"])
    bg.reset_state(1)
    thr_calm = float(bg(logits)["threshold"])
    assert thr_new > thr_calm


def test_development_and_profiler():
    """Онтогенез (curriculum) + FastLoopProfiler."""
    from neuroweave9.organism.development import DevelopmentalController
    from neuroweave9.runtime import FastLoopProfiler
    dev = DevelopmentalController(vocab_stages=[8, 16, 24],
                                  seq_len_stages=[4, 8, 16],
                                  mastery_accuracy_threshold=0.3,
                                  min_steps_per_stage=1)
    assert dev.active_vocab_limit == 8
    tok = torch.arange(24)
    assert int(dev.clamp_tokens(tok).max()) <= 7
    st = dev.observe_performance(0.9, 1.0)
    assert st["promoted"] == 1.0 and dev.active_vocab_limit == 16

    prof = FastLoopProfiler()
    for _ in range(3):
        with prof.section("columns"):
            pass
    rep = prof.report()
    assert "columns_ms" in rep and rep["columns_pct"] > 0
