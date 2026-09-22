"""
dirac.py
========

The single source of truth for all conventions.  Every sign, phase and basis
choice in the whole generator is fixed *here* and nowhere else, because
convention mismatches are the number-one source of bugs in helicity-amplitude
code.  If a validation fails, this file is the first place to look.

Conventions (frozen)
--------------------
* Metric signature:      g = diag(+1, -1, -1, -1)          [mostly-minus]
* Four-vectors:          p = (E, px, py, pz), contravariant p^mu
* Dirac algebra:         {gamma^mu, gamma^nu} = 2 g^{mu nu}
* Gamma representation:  Dirac basis (gamma^0 diagonal)
* gamma5 = i gamma^0 gamma^1 gamma^2 gamma^3
* Spinor normalisation:  ubar u = 2m,  vbar v = -2m
* Projectors:            ProjM = (1-gamma5)/2 (left),  ProjP = (1+gamma5)/2 (right)

All arrays are complex128 numpy; spinors are length-4, gamma matrices 4x4,
four-vectors length-4.
"""

from __future__ import annotations

import numpy as np

# ---- metric --------------------------------------------------------------- #
METRIC = np.diag([1.0, -1.0, -1.0, -1.0]).astype(np.complex128)


def lower(p: np.ndarray) -> np.ndarray:
    """p^mu -> p_mu."""
    return METRIC @ p


def dot(a: np.ndarray, b: np.ndarray) -> complex:
    """Minkowski dot a.b = a^mu b_mu = a0 b0 - a.b (3-vector)."""
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


# ---- gamma matrices, Dirac basis ------------------------------------------ #
_I2 = np.eye(2, dtype=np.complex128)
_Z2 = np.zeros((2, 2), dtype=np.complex128)

# Pauli matrices
SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
]


def _block(a, b, c, d):
    return np.block([[a, b], [c, d]])


# gamma^0 = [[I,0],[0,-I]] ; gamma^i = [[0, sigma^i],[-sigma^i,0]]
GAMMA = [
    _block(_I2, _Z2, _Z2, -_I2),
    _block(_Z2, SIGMA[0], -SIGMA[0], _Z2),
    _block(_Z2, SIGMA[1], -SIGMA[1], _Z2),
    _block(_Z2, SIGMA[2], -SIGMA[2], _Z2),
]
GAMMA = [g.astype(np.complex128) for g in GAMMA]

# gamma5 = i g0 g1 g2 g3  (Dirac basis -> off-diagonal identity blocks)
GAMMA5 = 1j * GAMMA[0] @ GAMMA[1] @ GAMMA[2] @ GAMMA[3]

# chiral projectors
PROJM = (np.eye(4, dtype=np.complex128) - GAMMA5) / 2.0   # left  (1-g5)/2
PROJP = (np.eye(4, dtype=np.complex128) + GAMMA5) / 2.0   # right (1+g5)/2

ID4 = np.eye(4, dtype=np.complex128)


def gamma_slash(p: np.ndarray) -> np.ndarray:
    """pslash = p_mu gamma^mu = g^{mu nu} p_nu gamma_mu = p^0 g^0 - p^i g^i."""
    return p[0] * GAMMA[0] - p[1] * GAMMA[1] - p[2] * GAMMA[2] - p[3] * GAMMA[3]


def bar(spinor: np.ndarray) -> np.ndarray:
    """Dirac adjoint: ubar = u^dagger gamma^0 (returns a row vector)."""
    return spinor.conj() @ GAMMA[0]


# ---- sanity self-checks (run on import in debug) -------------------------- #
def _check_algebra():
    for mu in range(4):
        for nu in range(4):
            anti = GAMMA[mu] @ GAMMA[nu] + GAMMA[nu] @ GAMMA[mu]
            assert np.allclose(anti, 2 * METRIC[mu, nu] * ID4), (mu, nu)
    assert np.allclose(GAMMA5 @ GAMMA5, ID4)
    for mu in range(4):
        assert np.allclose(GAMMA5 @ GAMMA[mu] + GAMMA[mu] @ GAMMA5, 0)


if __name__ == "__main__":
    _check_algebra()
    print("Dirac algebra OK: {g^mu,g^nu}=2g^{mu nu}, g5^2=1, {g5,g^mu}=0")
