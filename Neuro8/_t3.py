import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from neuroweave8.config import PhaseCfg, PlasticityCfg
from neuroweave8.core import PhaseVSA, ThetaGammaBinder, StatefulSynapse, capacity_from_frequencies

print("capacity 8/40 =", capacity_from_frequencies(8,40), "| 6/45 =", capacity_from_frequencies(6,45))
vsa = PhaseVSA(PhaseCfg(vsa_dim=512, vsa_codebook=256))
for n in (4,8,12,16,24,32,48):
    print("pairs", n, vsa.capacity_probe(n, seed=3))
x = torch.tensor([1.,0.,1.,1.,0.]); 
b = vsa.encode_pattern(x, role_idx=2)
print("roundtrip eq:", torch.equal(vsa.decode_pattern(b, 2), x))

tg = ThetaGammaBinder(16, PhaseCfg(n_slots=5))
tg.reset_state(2,"cpu")
for k in range(5):
    s = torch.zeros(2,16); s[:, k*3:(k+1)*3] = 1.0
    tg.write(s)
print("occupancy", tg.occupancy(), "slot2 read sum", float(tg.read(2).sum()))
print("soft read 2 sum", float(tg.read_soft(2).sum()))

syn = StatefulSynapse(16, 8, PlasticityCfg(), density=0.5)
s_pre = torch.ones(2,16); s_post = torch.ones(2,8)
d = syn.update_traces_and_plasticity(s_pre, s_post, da=torch.tensor(1.0))
print("wfast mass", float(syn.w_fast.abs().sum()), "mask", syn.edge_count())
print(syn.consolidate_to_slow())
