"""
Test 3: e+e- -> W+W-  (gamma s-channel + Z s-channel + nu_e t-channel).

The three diagrams cancel the bad high-energy growth of LONGITUDINAL W
production only if every relative sign is right: the VVV triple-gauge vertex
momentum convention (P_SIGN), the t-channel fermion propagator sign, and the
gamma/Z interference.  Test: |M(W_L W_L)|^2 must approach a constant as
sqrt(s) grows, not grow like s or s^2.

This is also the first real testbed for the physics you care about: the
longitudinal amplitude and its relation to the Goldstone amplitude.
"""

import numpy as np
from ufoamp.ufo_model import load
from ufoamp.process import Process
import ufoamp.recursion as R
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)


M = load(SM_UFO)


def cm_2to2_massive(rs, theta, m):
    E = rs / 2
    pf = np.sqrt(E**2 - m**2)
    p1 = [E, 0, 0, E]; p2 = [E, 0, 0, -E]
    p3 = [E, pf * np.sin(theta), 0, pf * np.cos(theta)]
    p4 = [E, -pf * np.sin(theta), 0, -pf * np.cos(theta)]
    return [np.array(x, dtype=float) for x in (p1, p2, p3, p4)]


def run(p_sign):
    R.P_SIGN = p_sign
    proc = Process(M, initial=[11, -11], final=[24, -24], overrides={"Me": 0.0})
    # zero Higgs couplings (massless e has no Yukawa anyway; keep it clean)
    zero = {}
    BY = M.p_by_pyname()
    for v in M.vertices:
        pdgs = [BY[n].pdg for n in v.particles]
        if 25 in pdgs:
            for cn in v.couplings.values():
                zero[cn] = 0j
    proc.set_param(overrides={"Me": 0.0}, restrict_couplings=zero)
    MW = float(np.real(proc.params["MW"]))
    th = np.radians(60)
    rows = []
    for rs in (200.0, 400.0, 800.0, 1600.0, 3200.0):
        mom = cm_2to2_massive(rs, th, MW)
        # longitudinal-longitudinal: hels (e-, e+, W+, W-) = (h1,h2,0,0), sum over e hel
        m2_LL = sum(abs(proc.helicity_amplitude(mom, (h1, h2, 0, 0)))**2
                    for h1 in (1, -1) for h2 in (1, -1)) / 4
        m2_tot = proc.m2(mom)
        rows.append((rs, m2_LL, m2_tot))
    return rows


def test_ww_gauge_cancellation():
    for p_sign in (-1.0, +1.0):
        rows = run(p_sign)
        print(f"\nP_SIGN = {p_sign:+.0f}   (theta=60 deg)")
        print(f"{'sqrt s':>8} {'|M_LL|^2':>14} {'|M|^2 total':>14} {'LL growth ratio':>16}")
        prev = None
        growth = []
        for rs, ll, tot in rows:
            g = (ll / prev) if prev else float('nan')
            growth.append(g)
            print(f"{rs:8.0f} {ll:14.6e} {tot:14.6e} {g:16.4f}")
            prev = ll
        # for a unitary (gauge-cancelled) amplitude the LL ratio between
        # successive doublings of sqrt(s) should approach 1 (constant), not 4 (~s)
        last = growth[-1]
        verdict = "unitary (cancellation OK)" if last < 1.5 else "GROWS -> wrong sign"
        print(f"  => LL |M|^2 ratio per doubling at high s: {last:.3f}  -> {verdict}")


if __name__ == "__main__":
    test_ww_gauge_cancellation()
