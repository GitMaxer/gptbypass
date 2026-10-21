import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from neuroweave8.config import NeuronCfg
from neuroweave8.core import LaminarALIF
cfg = NeuronCfg(sigma=0.0, alpha_nmda=2.0)
z=torch.zeros(1,4)
n = LaminarALIF(4, cfg); n.reset_state(1,"cpu")
for i in range(14):
    o = n.step(torch.full((1,4),0.2), z, z, sltt=False)
    print(i, round(float(o["v"][0,0]),3), round(float(n.g_ampa[0,0]),4), float(o["spikes"].sum()), round(float(o["theta"][0,0]),3))
