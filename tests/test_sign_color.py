"""
Test 4: fermion-exchange sign and colour sum.

Bhabha e+e- -> e+e- (photon only, massless):
   (1/4) sum|M|^2 = 2 e^4 [ (s^2+u^2)/t^2 + (u^2+t^2)/s^2 + 2 u^2/(s t) ]
The interference term 2u^2/(st) has its sign fixed by the relative fermion-
exchange sign between the s- and t-channel diagrams.  Wrong sign -> fails.

Colour: e+e- -> u ubar (photon only) must equal N_c * (e+e- -> mu+mu-) with
Q_u = 2/3, i.e. N_c * Q_u^2 * e^4(1+cos^2).
"""

import numpy as np
from ufoamp.ufo_model import load
from ufoamp.process import Process
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)


M = load(SM_UFO)
BY = M.p_by_pyname()


def photon_only(proc, extra_ov=None):
    zero = {}
    for v in M.vertices:
        pd = [BY[n].pdg for n in v.particles]
        if 23 in pd or 25 in pd or 21 in pd:      # Z, H, gluon
            for cn in v.couplings.values():
                zero[cn] = 0j
    ov = {"Me": 0.0, "MM": 0.0, "MU": 0.0}
    if extra_ov: ov.update(extra_ov)
    proc.set_param(overrides=ov, restrict_couplings=zero)


def cm(E, th):
    return [np.array(x, float) for x in ([E, 0, 0, E], [E, 0, 0, -E],
            [E, E*np.sin(th), 0, E*np.cos(th)], [E, -E*np.sin(th), 0, -E*np.cos(th)])]


def test_bhabha():
    proc = Process(M, initial=[11, -11], final=[11, -11]); photon_only(proc)
    e = float(np.real(proc.params["ee"])); E = 50.0; s = (2*E)**2
    ok = True
    print(f"{'theta':>6} {'ufoamp':>13} {'analytic':>13} {'ratio':>10}")
    for deg in (30, 60, 90, 120, 150):
        th = np.radians(deg); c = np.cos(th)
        t = -s*(1-c)/2; u = -s*(1+c)/2
        ana = 2*e**4*((s**2+u**2)/t**2 + (u**2+t**2)/s**2 + 2*u**2/(s*t))
        num = proc.m2(cm(E, th))
        print(f"{deg:6d} {num:13.6e} {ana:13.6e} {num/ana:10.7f}")
        ok &= np.isclose(num, ana, rtol=1e-8)
    print("Bhabha (fermion-exchange sign):", "PASS" if ok else "FAIL")
    assert ok


def test_color_sum():
    pm = Process(M, initial=[11, -11], final=[13, -13]); photon_only(pm)
    pq = Process(M, initial=[11, -11], final=[2, -2]);   photon_only(pq)
    E = 50.0; ok = True
    for deg in (45, 90, 135):
        th = np.radians(deg)
        mu = pm.m2(cm(E, th)); uu = pq.m2(cm(E, th))
        ratio = uu / mu; expect = 3 * (2/3)**2
        print(f"theta={deg:3d}  (ee->uu)/(ee->mumu) = {ratio:.6f}   expect N_c Q_u^2 = {expect:.6f}")
        ok &= np.isclose(ratio, expect, rtol=1e-8)
    print("colour sum:", "PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    test_bhabha()
    print()
    test_color_sum()
