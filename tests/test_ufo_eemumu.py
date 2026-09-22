"""
UFO-driven validation of the full recursion pipeline.

Test 1: e+e- -> mu+mu- via PHOTON ONLY (Z couplings switched off), massless
        leptons, from the real SM UFO.  Must reproduce e^4 (1+cos^2 theta).
        This validates: UFO vertex lookup, outgoing-type leg labels, fermion
        slot assignment, photon propagator, Berends-Giele assembly.
"""

import numpy as np
from ufoamp.ufo_model import load
from ufoamp.process import Process
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)


M = load(SM_UFO)


def cm_2to2(E, theta, m3=0.0, m4=0.0):
    p1 = [E, 0, 0, E]; p2 = [E, 0, 0, -E]
    pf = np.sqrt(E**2 - 0.0)  # massless final
    p3 = [E, pf * np.sin(theta), 0, pf * np.cos(theta)]
    p4 = [E, -pf * np.sin(theta), 0, -pf * np.cos(theta)]
    return [np.array(x, dtype=float) for x in (p1, p2, p3, p4)]


def test_photon_only():
    # e-(11) e+(-11) -> mu-(13) mu+(-13)
    proc = Process(M, initial=[11, -11], final=[13, -13],
                   overrides={"MZ": 91.1876})
    # switch off every coupling except the photon-lepton ones
    # (Z-fermion couplings are the ones with order QED and structures FFV2/FFV3;
    #  simplest robust approach: zero couplings appearing in Z vertices)
    zero = {}
    for v in M.vertices:
        pdgs = [M.p_by_pyname()[n].pdg for n in v.particles]
        if 23 in pdgs or 25 in pdgs:              # Z or Higgs vertices
            for cn in v.couplings.values():
                zero[cn] = 0j
    # also zero lepton masses so it's the massless analytic case
    proc.set_param(overrides={"MZ": 91.1876, "Me": 0.0, "MM": 0.0}, restrict_couplings=zero)

    e = float(np.real(proc.params["ee"]))
    E = 50.0
    ok = True
    print(f"{'theta':>6} {'ufoamp |M|^2':>14} {'e^4(1+cos^2)':>14} {'ratio':>9}")
    for deg in (20, 45, 90, 135, 160):
        th = np.radians(deg)
        num = proc.m2(cm_2to2(E, th))
        ana = e**4 * (1 + np.cos(th) ** 2)
        print(f"{deg:6.0f} {num:14.6e} {ana:14.6e} {num/ana:9.6f}")
        ok &= np.isclose(num, ana, rtol=1e-8)
    print("photon-only:", "PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    test_photon_only()
