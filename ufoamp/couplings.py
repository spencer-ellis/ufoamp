"""
couplings.py
============

Evaluate a UFO model's parameters and couplings numerically, given a choice of
input values.  This is the runtime knob: to study C3=5 vs C3=10 you call
``evaluate(model, overrides={'C3': 10})`` -- no code generation, no rebuild.

The coupling *expressions* are UFO Python (``cmath.sqrt``, ``complex(0,1)``,
``**`` ...), so we evaluate them in a controlled namespace containing the
math functions and the already-computed parameter values.  External parameters
take their UFO defaults unless overridden; internal parameters are evaluated in
declared order (UFO emits them topologically).
"""

from __future__ import annotations

import cmath
import math
from typing import Dict, Tuple

from .ufo_model import Model


def _namespace() -> dict:
    ns = {"cmath": cmath, "math": math, "complex": complex, "complexconjugate": (lambda z: (z).conjugate()),
          "abs": abs, "pi": math.pi}
    # expose common cmath/math names bare too (some UFOs use them unqualified)
    for name in ("sqrt", "sin", "cos", "tan", "exp", "log", "asin", "acos", "atan"):
        ns[name] = getattr(cmath, name)
    ns["re"] = lambda z: complex(z).real
    ns["im"] = lambda z: complex(z).imag
    ns["__builtins__"] = {}
    return ns


class _JaxMath:
    """cmath-compatible namespace backed by jax.numpy (traceable)."""
    def __init__(self):
        import jax.numpy as jnp
        self.pi = jnp.pi; self.e = jnp.e
        for n in ("sqrt", "exp", "log", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh"):
            setattr(self, n, getattr(jnp, n))
        self.asin, self.acos, self.atan = jnp.arcsin, jnp.arccos, jnp.arctan
        self.log10 = jnp.log10
        self.phase = lambda z: jnp.angle(z)


def _namespace_jax():
    import jax.numpy as jnp
    m = _JaxMath()
    ns = {"cmath": m, "math": m, "complex": complex, "complexconjugate": (lambda z: jnp.conj(z)),
          "__builtins__": {}}
    for n in ("sqrt", "exp", "log", "sin", "cos", "tan", "pi", "asin", "acos", "atan"):
        ns[n] = getattr(m, n)
    ns["re"] = lambda z: jnp.real(z); ns["im"] = lambda z: jnp.imag(z)
    return ns


def evaluate(model: Model, overrides: Dict[str, float] = None,
             inject: Dict[str, complex] = None, traceable: bool = False) -> Tuple[Dict[str, complex], Dict[str, complex]]:
    """Return (parameters, couplings) as name->value dicts.

    ``overrides`` replaces external-parameter values (e.g. {'C3': 10}).
    ``inject`` force-sets parameter values (used to pin two models to a common
    physical point); it wins over both defaults and overrides.
    """
    overrides = overrides or {}
    inject = inject or {}
    ns = _namespace_jax() if traceable else _namespace()
    params: Dict[str, complex] = {}

    # externals first (overrides replace defaults)
    for p in model.parameters:
        if p.nature == "external":
            val = overrides.get(p.name, p.value)
            if isinstance(val, (int, float, complex, str)):
                try:
                    val = complex(val) if p.ptype == "complex" else float(val)
                except Exception:
                    val = 0.0
            # else: array/tracer value (differentiable parameter) kept as is
            params[p.name] = val
            ns[p.name] = val
    # injected values are FIXED: set them now so that every internal parameter
    # derived downstream (e.g. lam from MH and vev) is computed consistently.
    for k, v in inject.items():
        params[k] = v
        ns[k] = v
    # internals in declared (topological) order; injected names are not
    # recomputed (that would overwrite the pinned value with the model's own
    # scheme-dependent formula)
    for p in model.parameters:
        if p.nature == "internal":
            if p.name in inject:
                continue
            try:
                v = eval(str(p.value), ns)  # noqa: S307 - controlled namespace
            except Exception:
                v = 0.0
            params[p.name] = v
            ns[p.name] = v

    couplings: Dict[str, complex] = {}
    for c in model.couplings:
        try:
            val = eval(c.value, ns)  # noqa: S307
            couplings[c.name] = val if traceable else complex(val)
        except Exception:
            couplings[c.name] = 0j
    return params, couplings
