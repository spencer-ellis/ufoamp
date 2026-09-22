"""
Test 2: e+e- -> mu+mu- with PHOTON + Z, massless leptons, from the SM UFO.

Analytic helicity amplitudes (massless, chiral couplings):
  M_ab = K_ab * [ c_g^e c_g^mu / s  +  c_a^e c_b^mu / (s - MZ^2 + i MZ GZ) ]
  |K_LL|^2 = |K_RR|^2 = s^2 (1+cos)^2 ,  |K_LR|^2 = |K_RL|^2 = s^2 (1-cos)^2
  (1/4) sum |M|^2  =  (1/4) s^2 [ (1+c)^2 (|F_LL|^2+|F_RR|^2) + (1-c)^2 (|F_LR|^2+|F_RL|^2) ]

where c_g is the photon coupling and c_L, c_R the coefficients of gamma^mu P_L,
gamma^mu P_R in the UFO Z-vertex.  The cos-odd (forward-backward) term is only
reproduced if the chiral projector slots are assigned correctly -- this is the
decisive fermion-flow test.
"""

import numpy as np
from ufoamp.ufo_model import load
from ufoamp.process import Process
from ufoamp.couplings import evaluate
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)


M = load(SM_UFO)
BY = M.p_by_pyname()
LOR = M.lorentz_by_name()


def vertex_couplings(pdgs_wanted):
    """Return {structure: coupling_name} for the vertex with these pdgs."""
    for v in M.vertices:
        pdgs = sorted(BY[n].pdg for n in v.particles)
        if pdgs == sorted(pdgs_wanted):
            return {LOR[v.lorentz[li]].structure: cn for (ci, li), cn in v.couplings.items()}
    raise KeyError(pdgs_wanted)


def chiral_couplings(pdgs, coup):
    """(c_gamma, c_L, c_R): coefficients of gamma^mu, gamma^mu P_L, gamma^mu P_R
    summed over the vertex's structures (UFOs may write e.g. ProjM + 2 ProjP)."""
    from ufoamp.vertex_eval import parse
    cg = cL = cR = 0j
    for st, cn in vertex_couplings(pdgs).items():
        g = coup[cn]
        for coeff, atoms in parse(st):
            names = [a.name for a in atoms]
            if "ProjM" in names: cL += coeff * g
            elif "ProjP" in names: cR += coeff * g
            else: cg += coeff * g
    return cg, cL, cR


def cm_2to2(E, theta):
    p1 = [E, 0, 0, E]; p2 = [E, 0, 0, -E]
    p3 = [E, E * np.sin(theta), 0, E * np.cos(theta)]
    p4 = [E, -E * np.sin(theta), 0, -E * np.cos(theta)]
    return [np.array(x, dtype=float) for x in (p1, p2, p3, p4)]


def test_photon_plus_Z():
    ov = {"Me": 0.0, "MM": 0.0}
    proc = Process(M, initial=[11, -11], final=[13, -13], overrides=ov)
    # zero Higgs (Yukawa) couplings so only gamma/Z exchange contribute
    zero = {}
    for v in M.vertices:
        pdgs = [BY[n].pdg for n in v.particles]
        if 25 in pdgs:
            for cn in v.couplings.values():
                zero[cn] = 0j
    proc.set_param(overrides=ov, restrict_couplings=zero)
    params, coup = proc.params, proc.coup

    # UFO couplings: photon-e, photon-mu, Z-e (L,R), Z-mu (L,R)
    ce_g, _, _ = chiral_couplings([-11, 11, 22], coup); cm_g, _, _ = chiral_couplings([-13, 13, 22], coup)
    _, ce_L, ce_R = chiral_couplings([-11, 11, 23], coup); _, cm_L, cm_R = chiral_couplings([-13, 13, 23], coup)
    MZ = float(np.real(params["MZ"])); WZ = float(np.real(params["WZ"]))
    print(f"Z couplings: e_L={ce_L:.4f} e_R={ce_R:.4f}  MZ={MZ} WZ={WZ}")

    ok = True
    print(f"{'sqrt s':>7} {'theta':>6} {'ufoamp':>13} {'analytic':>13} {'ratio':>9}  {'A_FB(ufoamp)':>11} {'A_FB(ana)':>10}")
    for rs in (60.0, 91.1876, 150.0):
        E = rs / 2; s = rs**2
        DZ = s - MZ**2 + 1j * MZ * WZ
        def F(a, b): return ce_g * cm_g / s + a * b / DZ
        FLL, FRR, FLR, FRL = F(ce_L, cm_L), F(ce_R, cm_R), F(ce_L, cm_R), F(ce_R, cm_L)
        def ana(th):
            c = np.cos(th)
            return 0.25 * s**2 * ((1 + c)**2 * (abs(FLL)**2 + abs(FRR)**2)
                                  + (1 - c)**2 * (abs(FLR)**2 + abs(FRL)**2))
        fwd_p = fwd_a = bwd_p = bwd_a = 0.0
        for deg in (30, 60, 90, 120, 150):
            th = np.radians(deg)
            num = proc.m2(cm_2to2(E, th)); an = ana(th)
            ok &= np.isclose(num, an, rtol=1e-7)
            if deg < 90: fwd_p += num; fwd_a += an
            elif deg > 90: bwd_p += num; bwd_a += an
        afb_p = (fwd_p - bwd_p) / (fwd_p + bwd_p); afb_a = (fwd_a - bwd_a) / (fwd_a + bwd_a)
        num90 = proc.m2(cm_2to2(E, np.pi / 2)); an90 = ana(np.pi / 2)
        print(f"{rs:7.1f} {90:6d} {num90:13.5e} {an90:13.5e} {num90/an90:9.6f}  {afb_p:11.5f} {afb_a:10.5f}")
        ok &= np.isclose(afb_p, afb_a, atol=1e-7)
    print("photon+Z incl. A_FB:", "PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    test_photon_plus_Z()
