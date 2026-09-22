"""
Example: VBF di-Higgs coupling scan with ufoamp.
Computes spin/colour-summed |M|^2 for u d -> u d H H on random phase-space
points as a function of (CV, C2V, C3) -- switching couplings is a function
call, never a recompile.

usage:  PYTHONPATH=. python3 examples/vbf_hh_scan.py /path/to/HHVBF_UFO
"""
import sys, os, numpy as np
from ufoamp.ufo_model import load
from ufoamp.process import Process
from ufoamp.phasespace import rambo

ufo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("UFOAMP_HHVBF_UFO", "HHVBF_UFO")
M = load(ufo)
proc = Process(M, initial=[2, 1], final=[2, 1, 25, 25],
               overrides={"CV": 1, "C2V": 1, "C3": 1, "MU": 0, "MD": 0})
MH = float(np.real(proc.params["MH"]))
rng = np.random.default_rng(1)
rs = 1000.0
pts = []
for _ in range(3):
    fin = rambo(rs, [0, 0, MH, MH], rng)
    pts.append([np.array([rs/2, 0, 0, rs/2]), np.array([rs/2, 0, 0, -rs/2])] + list(fin))

print(f"{'(CV,C2V,C3)':>14} " + " ".join(f"{'|M|^2 pt'+str(i):>13}" for i in range(len(pts))))
for cv, c2v, c3 in [(1,1,1), (1,2,1), (1,0,1), (1,1,5), (1,1,-1), (1,1,0)]:
    proc.set_param(overrides={"CV": cv, "C2V": c2v, "C3": c3, "MU": 0, "MD": 0})
    vals = [proc.m2(p) for p in pts]
    print(f"{str((cv,c2v,c3)):>14} " + " ".join(f"{v:13.5e}" for v in vals))
