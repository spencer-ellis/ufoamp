"""
Test 10: coupling-order restriction and helicity frames.
(a) max_orders={"QCD":0} == dropping every gluon vertex (MadGraph QCD=0).
(b) Helicity-summed |M|^2 is invariant under a boost of the event; polarised
    pieces are frame dependent; pol_frame="partonic_cm" recovers the CM values
    from a lab-frame event.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)
from ufoamp.analysis import MatrixElement
from ufoamp.phasespace import rambo
from ufoamp.recursion import boost_matrix
c = {}
MZ = float(np.real(MatrixElement(SM_UFO, 'u d > u d z z h', max_orders={'QCD': 0}).param('MZ'))); MH = float(np.real(MatrixElement(SM_UFO, 'u d > u d z z h', max_orders={'QCD': 0}).param('MH')))
rng = np.random.default_rng(0); fin = rambo(2000., [0, 0, MZ, MZ, MH], rng)
cm = [np.array([1000., 0, 0, 1000.]), np.array([1000., 0, 0, -1000.])] + list(fin)
q0 = MatrixElement(SM_UFO, "u u > u u z z h", couplings=c, max_orders={"QCD": 0}).m2(cm)
g = MatrixElement(SM_UFO, "u u > u u z z h", couplings=c, drop_vertex=lambda v, p: 21 in p).m2(cm)
ok = abs(q0 / g - 1) < 1e-12
print("(a) max_orders QCD=0 == drop gluon vertices:", "PASS" if ok else "FAIL"); assert ok
Lam = np.linalg.inv(boost_matrix(np.array([2500., 300., -200., 1500.])))
lab = [Lam @ p for p in cm]
mc = MatrixElement(SM_UFO, "u d > u d z z h", couplings=c, max_orders={"QCD": 0})
ml = MatrixElement(SM_UFO, "u d > u d z z h", couplings=c, max_orders={"QCD": 0}, pol_frame="lab")
mp = MatrixElement(SM_UFO, "u d > u d z z h", couplings=c, max_orders={"QCD": 0}, pol_frame="partonic_cm")
ok = abs(ml.m2(lab) / mc.m2(cm) - 1) < 1e-10
rc, rl, rp = mc.m2_polarized(cm, ["z"]), ml.m2_polarized(lab, ["z"]), mp.m2_polarized(lab, ["z"])
ok &= abs(rp["LL"] / rc["LL"] - 1) < 1e-10 and abs(rl["LL"] / rc["LL"] - 1) > 1e-3
print(f"    LL: CM {rc['LL']:.4e}  lab-frame {rl['LL']:.4e}  partonic_cm {rp['LL']:.4e}")
print("(b) boost invariance / pol_frame:", "PASS" if ok else "FAIL"); assert ok
