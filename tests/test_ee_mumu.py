"""
Phase-1 validation: e+ e- -> mu+ mu- via photon (pure QED, massless).

Analytic target (Peskin & Schroeder 5.10, massless limit):
    (1/4) sum_spins |M|^2 = e^4 (1 + cos^2 theta)          [Q_e = Q_mu = -1]

This exercises: Dirac spinors u/v, the FFV vertex gamma^mu from lorentz_eval,
the Feynman-gauge photon propagator, amplitude assembly, spin sum/average.
"""

import numpy as np

from ufoamp.dirac import GAMMA, METRIC, bar, dot
from ufoamp.wavefunctions import u_spinor, v_spinor
from ufoamp.lorentz_eval import ffv_vertex_matrices
from ufoamp.propagators import vector_prop


def kinematics(E, theta):
    """CM frame, massless. e-(p1) e+(p2) -> mu-(p3) mu+(p4)."""
    s = np.sin(theta); c = np.cos(theta)
    p1 = np.array([E, 0, 0, E], dtype=np.complex128)
    p2 = np.array([E, 0, 0, -E], dtype=np.complex128)
    p3 = np.array([E, E * s, 0, E * c], dtype=np.complex128)
    p4 = np.array([E, -E * s, 0, -E * c], dtype=np.complex128)
    return p1, p2, p3, p4


def amplitude(E, theta, h, e=1.0, Qe=-1.0, Qmu=-1.0):
    """M for helicities h=(h1,h2,h3,h4) via s-channel photon."""
    p1, p2, p3, p4 = kinematics(E, theta)
    q = p1 + p2                     # photon momentum, q^2 = s
    # FFV vertex gamma^mu (QED: vertex factor -i e Q gamma^mu)
    Gmu = ffv_vertex_matrices("Gamma(3,2,1)", vector_leg=3, out_leg=2, in_leg=1)

    # electron current J_e^mu = vbar(p2) (-i e Qe gamma^mu) u(p1)
    ue = u_spinor(p1, h[0], 0.0)
    vebar = bar(v_spinor(p2, h[1], 0.0))
    Je = np.array([(-1j * e * Qe) * (vebar @ Gmu[mu] @ ue) for mu in range(4)])

    # muon current J_mu^nu = ubar(p3) (-i e Qmu gamma^nu) v(p4)
    u3bar = bar(u_spinor(p3, h[2], 0.0))
    v4 = v_spinor(p4, h[3], 0.0)
    Jmu = np.array([(-1j * e * Qmu) * (u3bar @ Gmu[nu] @ v4) for nu in range(4)])

    # contract with photon propagator (-i g_{mu nu}/s): M = Je^mu (-i g/s) Jmu^nu
    prop = vector_prop(q, 0.0)      # -i g^{mu nu}/s (matrix)
    M = 0j
    for mu in range(4):
        for nu in range(4):
            M += Je[mu] * prop[mu, nu] * Jmu[nu]
    return M


def m2_avg(E, theta, **kw):
    tot = 0.0
    for h1 in (+1, -1):
        for h2 in (+1, -1):
            for h3 in (+1, -1):
                for h4 in (+1, -1):
                    M = amplitude(E, theta, (h1, h2, h3, h4), **kw)
                    tot += abs(M) ** 2
    return tot / 4.0   # average over 4 initial spin states


def test_ee_mumu():
    E = 50.0
    e = 0.30282212           # sqrt(4 pi alpha), alpha=1/137.036 -> arbitrary here
    ok = True
    print(f"{'theta':>8} {'numeric |M|^2':>16} {'e^4(1+cos^2)':>16} {'ratio':>10}")
    for deg in (20, 45, 60, 90, 120, 150):
        th = np.radians(deg)
        num = m2_avg(E, th, e=e)
        ana = e**4 * (1 + np.cos(th) ** 2)
        ratio = num / ana
        print(f"{deg:8.0f} {num:16.8e} {ana:16.8e} {ratio:10.6f}")
        if not np.isclose(num, ana, rtol=1e-9):
            ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    test_ee_mumu()
