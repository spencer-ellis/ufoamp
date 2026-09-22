"""
wavefunctions.py
================

External-leg wavefunctions in the conventions fixed by :mod:`dirac`.

Fermions: helicity spinors u(p,h), v(p,h) built directly from the boosted
rest-frame spinors, so they are correct for massive AND massless momenta and
carry a definite helicity h = +-1 (we pass h = +-1 meaning +-1/2).

Vectors: polarization vectors eps(p, lam) for lam in {+1, -1, 0}; the
longitudinal one eps_0 is the physically important mode for the GBET study and
is constructed to satisfy eps_0.p = 0 and eps_0.eps_0 = -1.

Validation is via completeness relations (checked in tests):
    sum_h u u_bar = pslash + m
    sum_h v v_bar = pslash - m
    sum_lam eps^mu eps*^nu = -g^{mu nu} + p^mu p^nu / m^2   (massive vector)
"""

from __future__ import annotations

import numpy as np

from .dirac import GAMMA, gamma_slash, bar, ID4

# incoming/outgoing handled by caller via crossing; here p is the physical
# four-momentum with p^0 > 0 for the given particle.


# --------------------------------------------------------------------------- #
#  Dirac spinors                                                              #
# --------------------------------------------------------------------------- #
def _helicity_2spinor(phat: np.ndarray, h: int) -> np.ndarray:
    """2-component helicity eigenspinor chi_h for unit 3-vector phat."""
    px, py, pz = phat
    p = np.linalg.norm([px, py, pz])
    if p < 1e-14:
        # momentum along +z by convention
        px, py, pz, p = 0.0, 0.0, 1.0, 1.0
    ct = pz / p
    # theta, phi
    theta = np.arccos(np.clip(ct, -1, 1))
    phi = np.arctan2(py, px)
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    if h == +1:
        return np.array([c, s * np.exp(1j * phi)], dtype=np.complex128)
    else:
        return np.array([-s * np.exp(-1j * phi), c], dtype=np.complex128)


def u_spinor(p: np.ndarray, h: int, m: float) -> np.ndarray:
    """Particle spinor u(p,h), normalised to ubar u = 2m.

    Derived from (pslash - m) u = 0 in the Dirac basis:
        u = sqrt(E+m) ( chi_h ,  (h|p|/(E+m)) chi_h )
    with chi_h the helicity 2-spinor (sigma.phat chi_h = h chi_h).  Finite and
    correct in both the massive and massless limits.
    """
    E = p[0].real
    p3 = p[1:4].real
    pmag = np.linalg.norm(p3)
    phat = p3 / pmag if pmag > 1e-14 else np.array([0.0, 0.0, 1.0])
    chi = _helicity_2spinor(phat, h)
    a = np.sqrt(E + m)
    kappa = (h * pmag / (E + m)) if (E + m) > 0 else 0.0
    upper = a * chi
    lower = a * kappa * chi
    return np.concatenate([upper, lower]).astype(np.complex128)


def v_spinor(p: np.ndarray, h: int, m: float) -> np.ndarray:
    """Antiparticle spinor v(p,h), normalised to vbar v = -2m.

    Derived from (pslash + m) v = 0 in the Dirac basis:
        v = sqrt(E+m) ( (h|p|/(E+m)) chi_{-h} ,  chi_{-h} )
    (uses chi_{-h}, the standard antiparticle helicity labelling).  Gives
    sum_h v vbar = pslash - m.
    """
    E = p[0].real
    p3 = p[1:4].real
    pmag = np.linalg.norm(p3)
    phat = p3 / pmag if pmag > 1e-14 else np.array([0.0, 0.0, 1.0])
    chi = _helicity_2spinor(phat, -h)
    a = np.sqrt(E + m)
    # sigma.p chi_{-h} = -h|p| chi_{-h}, so the upper component carries -h|p|/(E+m)
    kappa = (-h * pmag / (E + m)) if (E + m) > 0 else 0.0
    upper = a * kappa * chi
    lower = a * chi
    return np.concatenate([upper, lower]).astype(np.complex128)


# --------------------------------------------------------------------------- #
#  Polarization vectors                                                       #
# --------------------------------------------------------------------------- #
def polarization_vector(p: np.ndarray, lam: int, m: float) -> np.ndarray:
    """Polarization eps^mu(p, lam), lam in {+1,-1,0}.

    Transverse (+/-) use the standard circular basis w.r.t. the 3-momentum
    direction; longitudinal (0) is (|p|, E phat)/m so that eps.p = 0.
    """
    E = p[0].real
    p3 = p[1:4].real.astype(float)
    pmag = np.linalg.norm(p3)
    if pmag < 1e-14:
        phat = np.array([0.0, 0.0, 1.0])
    else:
        phat = p3 / pmag

    # build transverse basis (e1, e2) perpendicular to phat
    ref = np.array([1.0, 0.0, 0.0]) if abs(phat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, phat); e1 /= np.linalg.norm(e1)
    e2 = np.cross(phat, e1)

    if lam == 0:
        if m <= 0:
            raise ValueError("longitudinal polarization undefined for massless")
        eps0 = pmag / m
        eps_space = (E / m) * phat
        return np.array([eps0, *eps_space], dtype=np.complex128)
    # transverse: eps_pm = (0, ∓(e1 ± i e2)/sqrt2)
    sign = -1.0 if lam == +1 else 1.0
    space = sign * (e1 + 1j * lam * e2) / np.sqrt(2)
    return np.array([0.0, *space], dtype=np.complex128)


# --------------------------------------------------------------------------- #
#  self-tests                                                                 #
# --------------------------------------------------------------------------- #
def _check(m=1.3, seed=2):
    rng = np.random.default_rng(seed)
    p3 = rng.normal(size=3)
    E = np.sqrt(m**2 + p3 @ p3)
    p = np.array([E, *p3], dtype=np.complex128)

    # spinor completeness
    su = sum(np.outer(u_spinor(p, h, m), bar(u_spinor(p, h, m))) for h in (+1, -1))
    assert np.allclose(su, gamma_slash(p) + m * ID4), "sum u ubar != pslash+m"
    sv = sum(np.outer(v_spinor(p, h, m), bar(v_spinor(p, h, m))) for h in (+1, -1))
    assert np.allclose(sv, gamma_slash(p) - m * ID4), "sum v vbar != pslash-m"
    # Dirac equations
    for h in (+1, -1):
        assert np.allclose((gamma_slash(p) - m * ID4) @ u_spinor(p, h, m), 0)
        assert np.allclose((gamma_slash(p) + m * ID4) @ v_spinor(p, h, m), 0)

    # vector completeness: sum eps eps* = -g + p p/m^2
    from .dirac import METRIC
    S = np.zeros((4, 4), dtype=np.complex128)
    for lam in (+1, -1, 0):
        e = polarization_vector(p, lam, m)
        S += np.outer(e, e.conj())
    target = -METRIC + np.outer(p, p) / m**2
    assert np.allclose(S, target), "vector completeness failed"
    # eps.p = 0 and normalisation
    from .dirac import dot
    for lam in (+1, -1, 0):
        e = polarization_vector(p, lam, m)
        assert abs(dot(e, p)) < 1e-9
        assert np.allclose(dot(e, e.conj()), -1)
    return True


if __name__ == "__main__":
    _check()
    print("wavefunctions OK: spinor & vector completeness, Dirac eqs, eps.p=0")
