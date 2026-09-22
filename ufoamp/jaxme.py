"""
jaxme.py — JAX backend: compiled, differentiable matrix elements.

    from ufoamp.jaxme import JaxMatrixElement
    jme = JaxMatrixElement(me, params=["C2V", "C3"])      # me: a ufoamp MatrixElement
    jme.m2(events)                                        # (N,) jit-compiled
    jme.grad_params(events)                               # (N, n_params)  d|M|^2/dc_i
    jme.grad_momenta(events)                              # (N, n, 4)      d|M|^2/dp
    jme.hessian_params(events)                            # (N, n_params, n_params)

The whole chain UFO parameters -> couplings -> amplitudes -> |M|^2 is traced
once per (process, batch shape) and compiled; the recursion's Python
bookkeeping runs only during tracing.  Results agree with the numpy backend to
machine precision.  Requires `pip install jax jaxlib`.
"""
from __future__ import annotations
from itertools import product
from typing import List, Optional
import numpy as np

from . import backend
from .couplings import evaluate


class JaxMatrixElement:
    def __init__(self, me, params: Optional[List[str]] = None, pol: Optional[dict] = None,
                 average_initial: bool = True, chunk: int = 8):
        import jax
        jax.config.update("jax_enable_x64", True)     # before any jit is built
        self.jax = jax
        self.me = me
        self.params = list(params or [])
        self.avg = average_initial
        skip = getattr(me, "_skip", set())      # exact chirality zeros found by me.prune_helicities
        self.configs = [h for h in product(*me.helicity_sets(pol)) if h not in skip]
        self.chunk = chunk           # events per compiled call (memory vs speed)
        self._fns = {}
        self.base_values = np.array([float(np.real(me.couplings.get(n, me.param(n)))) for n in self.params])
        # The traced graph contains a vertex coupling only if it is "active".
        # A coupling that vanishes at the build point but depends on a
        # differentiable parameter (e.g. an EFT vertex at c=0) must still be
        # traced, otherwise gradients at that point come out as exactly zero.
        if self.params:
            probe = {**me.couplings, **{n: (v if v != 0 else 1.0) * 1.2345 + 0.6789 for n, v in zip(self.params, self.base_values)}}
            _, cprobe = evaluate(me.model, overrides=probe, inject=me._inject or {})
            amp = me.proc.amp
            for k, v in cprobe.items():
                if complex(v) != 0:
                    amp.active[k] = True

    # ---- the traced function --------------------------------------------- #
    def _build(self, N):
        jax = self.jax; me = self.me; proc = me.proc; amp = proc.amp
        configs = self.configs; C = len(configs)
        names = self.params
        n_avg = 1
        if self.avg:
            for i in range(me.n_in):
                n_avg *= len(me._hel_sets[i])

        def fn(momenta, pvals):
            backend.use("jax")
            try:
                ov = {**me.couplings, **{n: pvals[i] for i, n in enumerate(names)}}
                if names:
                    inj = dict(me._inject or {})
                    params, coup = evaluate(me.model, overrides=ov, inject=inj, traceable=True)
                    if me._restrict_couplings:
                        coup = {**coup, **me._restrict_couplings}
                    saved = amp.coup; amp.coup = coup
                else:
                    saved = None
                mom = me._to_frame(momenta) if me.pol_frame not in (None, "lab") else momenta
                v = proc.m2_batch(mom, configs)               # (N, C)
                res = v.sum(axis=1) / n_avg
                if saved is not None:
                    amp.coup = saved
                return res
            finally:
                backend.use("numpy")
        return jax.jit(fn)

    def _fn(self, N):
        if N not in self._fns:
            self._fns[N] = self._build(N)
        return self._fns[N]

    # ---- public --------------------------------------------------------- #
    def _prep(self, events):
        ev = np.asarray(events, dtype=float)
        return ev[None] if ev.ndim == 2 else ev

    def _chunked(self, ev, call):
        """Run `call(chunk_array)` on fixed-size chunks (last one zero-padded)."""
        out = []
        c = self.chunk
        for a in range(0, len(ev), c):
            part = ev[a:a + c]
            pad = c - len(part)
            if pad:
                part = np.concatenate([part, np.repeat(part[-1:], pad, axis=0)])
            r = np.asarray(call(part))
            out.append(r[:len(ev[a:a + c])])
        return np.concatenate(out)

    def m2(self, events, param_values=None):
        ev = self._prep(events); pv = self.base_values if param_values is None else np.asarray(param_values, float)
        return self._chunked(ev, lambda part: self._fn(len(part))(part, pv))

    def grad_params(self, events, param_values=None):
        """d|M|^2/dc_i per event: (N, n_params)."""
        ev = self._prep(events); pv = self.base_values if param_values is None else np.asarray(param_values, float)
        return self._chunked(ev, lambda part: self.jax.jacfwd(lambda p: self._fn(len(part))(part, p))(pv))

    def hessian_params(self, events, param_values=None):
        ev = self._prep(events); pv = self.base_values if param_values is None else np.asarray(param_values, float)
        return self._chunked(ev, lambda part: self.jax.jacfwd(self.jax.jacfwd(lambda p: self._fn(len(part))(part, p)))(pv))

    def grad_momenta(self, events, param_values=None):
        """d|M|^2/dp^mu per event and leg: (N, n, 4) (not momentum-conserving-projected)."""
        ev = self._prep(events); pv = self.base_values if param_values is None else np.asarray(param_values, float)
        def one(part):
            g = np.asarray(self.jax.jacrev(lambda m: self._fn(len(part))(m, pv))(part))   # (c, c, n, 4)
            return g[np.arange(len(part)), np.arange(len(part))]
        return self._chunked(ev, one)
