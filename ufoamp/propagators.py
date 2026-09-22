"""
propagators.py
==============

Numerators/denominators for internal lines, in the conventions of :mod:`dirac`
(mostly-minus metric, Feynman gauge for vectors).

We return the full propagator matrix/scalar including the i and the denominator
1/(p^2 - m^2 + i eps).  For tree level the +i eps is irrelevant off resonance;
we keep a tiny width option for numerical safety near poles.

  scalar :  i / (p^2 - m^2)
  fermion:  i (pslash + m) / (p^2 - m^2)
  vector :  -i g^{mu nu} / (p^2 - m^2)          (Feynman / 't Hooft-Feynman)
            (massive: same numerator in Feynman gauge; Goldstone handles the
             p^mu p^nu piece, which is exactly what the GBET study needs)
"""

from __future__ import annotations

import numpy as np

from .dirac import METRIC, gamma_slash, ID4, dot


def _denom(p: np.ndarray, m: float, width: float = 0.0) -> complex:
    s = dot(p, p)
    return s - m**2 + 1j * m * width


def scalar_prop(p: np.ndarray, m: float, width: float = 0.0) -> complex:
    return 1j / _denom(p, m, width)


def fermion_prop(p: np.ndarray, m: float, width: float = 0.0) -> np.ndarray:
    return 1j * (gamma_slash(p) + m * ID4) / _denom(p, m, width)


def vector_prop(p: np.ndarray, m: float, width: float = 0.0) -> np.ndarray:
    """Feynman-gauge vector propagator numerator -i g^{mu nu} over denominator."""
    return -1j * METRIC / _denom(p, m, width)
