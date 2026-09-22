"""
process.py
==========

User-facing layer.  A ``Process`` is a model + a list of external particles
(PDG codes, in/out).  Given momenta it returns helicity amplitudes or the
spin-summed/averaged |M|^2.  Coupling values are chosen per call via
``set_couplings(overrides=...)`` -- the runtime knob.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy as xp   # switchable backend

from .ufo_model import Model, load
from .couplings import evaluate
from .recursion import Amplitude, Leg


class Process:
    def __init__(self, model: Model, initial: List[int], final: List[int],
                 overrides: Optional[Dict[str, float]] = None,
                 restrict_couplings: Optional[Dict[str, complex]] = None,
                 inject: Optional[Dict[str, complex]] = None, gauge: str = "auto",
                 drop_vertex=None, max_orders: Optional[Dict[str, int]] = None):
        """drop_vertex(vertex, pdgs)->bool removes whole vertices (safe: UFO
        couplings are shared between vertices, so zeroing by name can have
        collateral effects; e.g. SM WWHH shares its coupling with WWGG)."""
        self.model = model
        self.gauge = gauge
        self.drop_vertex = drop_vertex
        self.max_orders = max_orders
        self.initial = list(initial)
        self.final = list(final)
        self.set_couplings(overrides, restrict_couplings, inject)

    # ---- runtime knob ----------------------------------------------------- #
    def set_couplings(self, overrides=None, restrict_couplings=None, inject=None):
        """overrides: external-parameter values (the C3/CV/C2V knob);
        inject: force parameter values after evaluation (pin two models to a
        common physical point)."""
        self.params, self.coup = evaluate(self.model, overrides=overrides or {},
                                          inject=inject or {})
        if restrict_couplings:
            self.coup = dict(self.coup)
            self.coup.update(restrict_couplings)
        old = getattr(self, "amp", None)
        self.amp = Amplitude(self.model, self.coup, self.params, gauge=self.gauge,
                             drop_vertex=self.drop_vertex, max_orders=self.max_orders)
        from .color import rep_dim
        byp = self.model.p_by_pdg()
        self.amp._external_colored = any(rep_dim(byp[p].color) > 0 for p in self.initial + self.final)
        self.amp._select_color_mode()
        # settings that must survive a coupling change
        if old is not None:
            self.amp.set_amp_orders(getattr(self, "_amp_orders", None))
            self.amp.internal_pol = old.internal_pol
            self.amp.zerowidth_tchannel = old.zerowidth_tchannel

    set_param = set_couplings          # alias (analysis.MatrixElement uses set_param)

    # ---- helicity bookkeeping --------------------------------------------- #
    def _hel_choices(self, pdg: int) -> List[int]:
        sp = self.model.p_by_pdg()[pdg].spin
        if sp == 1:
            return [0]
        if sp == 2:
            return [+1, -1]
        if sp == 3:
            m = self.amp.mass(pdg)
            return [+1, -1, 0] if m > 0 else [+1, -1]
        raise ValueError

    def legs(self, momenta: List[np.ndarray], hels: Tuple[int, ...]) -> List[Leg]:
        pdgs = self.initial + self.final
        n_in = len(self.initial)
        return [Leg(pdg=pdgs[i], incoming=(i < n_in), p=np.asarray(momenta[i], dtype=complex),
                    hel=hels[i]) for i in range(len(pdgs))]

    # ---- amplitudes ------------------------------------------------------- #
    def set_amp_orders(self, amp_orders):
        """Amplitude-level order limits, e.g. {"NP": (1, 1)} keeps only the
        single-insertion part of the amplitude (its square is the pure
        SM x EFT interference structure); {"NP": (0, 1)} = MadGraph's NP<=1.
        Persists across set_couplings()."""
        self._amp_orders = amp_orders
        self.amp.set_amp_orders(amp_orders)

    def m2_batch_diff(self, momenta, configs, setup_a, setup_b):
        """|M_a - M_b|^2 with M_a, M_b the amplitude tensors under two
        amplitude settings (callables that configure self.amp)."""
        self._set_frame(momenta, len(configs))
        legs = self.legs_batch(momenta, configs)
        setup_a(); A = self.amp.amplitude_tensor(legs)
        setup_b(); B = self.amp.amplitude_tensor(legs)
        D = A - B
        m2 = xp.sum(xp.abs(D.reshape(D.shape[0], -1)) ** 2, axis=1)
        from .color import rep_dim
        for lg in legs:
            if lg.incoming:
                d = rep_dim(self.amp.color(lg.pdg))
                if d:
                    m2 = m2 / d
        return m2 if xp.ndim(xp.asarray(momenta)) == 2 else m2.reshape(-1, len(configs))

    def set_internal_pol(self, internal_pol, frame_legs=None):
        """internal_pol: {abs(pdg): (components, where)}; frame_legs: leg indices
        whose summed momentum defines the polarisation frame (None = lab)."""
        self.amp.internal_pol = internal_pol
        self._frame_legs = frame_legs

    def _set_frame(self, momenta, n_configs=1):
        fl = getattr(self, "_frame_legs", None)
        if fl is None:
            self.amp.frame_P = None; return
        M = xp.asarray(momenta)
        if M.ndim == 2:
            self.amp.frame_P = M[fl].sum(axis=0)
        else:
            self.amp.frame_P = xp.repeat(M[:, fl].sum(axis=1), n_configs, axis=0)

    def legs_batch(self, momenta, configs):
        """momenta: (n,4) for one event or (N,n,4) for N events; configs: list
        of helicity tuples.  Flat batch of size N*len(configs), event-major."""
        pdgs = self.initial + self.final
        n_in = len(self.initial)
        H = np.asarray(configs, dtype=int)                       # (C, n)
        M = xp.asarray(momenta)
        if M.ndim == 2:
            return [Leg(pdg=pdgs[i], incoming=(i < n_in), p=M[i] + 0j, hel=H[:, i])
                    for i in range(len(pdgs))]
        N, C = M.shape[0], H.shape[0]
        Prep = xp.repeat(M, C, axis=0)                           # (N*C, n, 4)
        Hrep = np.tile(H, (N, 1))                                # (N*C, n)
        return [Leg(pdg=pdgs[i], incoming=(i < n_in), p=Prep[:, i] + 0j, hel=Hrep[:, i])
                for i in range(len(pdgs))]

    def helicity_amplitude(self, momenta, hels):
        """Amplitude for one helicity configuration.  A complex number for
        colour-singlet processes; for coloured processes the amplitude is a
        tensor over the external colour indices (legs in order) and that array
        is returned (|M|^2 = sum of |.|^2 over it)."""
        self._set_frame(momenta)
        a = np.asarray(self.amp.amplitude(self.legs_batch(momenta, [hels])))[0]
        return complex(a) if a.ndim == 0 else a

    def amplitudes_batch(self, momenta, configs) -> np.ndarray:
        """Amplitudes for helicity configurations (and events) in one pass:
        momenta (n,4) -> (C,);  momenta (N,n,4) -> (N,C)."""
        self._set_frame(momenta, len(configs))
        a = self.amp.amplitude(self.legs_batch(momenta, configs))
        return a if xp.ndim(xp.asarray(momenta)) == 2 else a.reshape((-1, len(configs)) + tuple(a.shape[1:]))

    def m2_batch(self, momenta, configs) -> np.ndarray:
        """Colour-summed |M|^2 per helicity configuration (and event), one pass."""
        self._set_frame(momenta, len(configs))
        v = self.amp.m2_color(self.legs_batch(momenta, configs))
        return v if xp.ndim(xp.asarray(momenta)) == 2 else v.reshape(-1, len(configs))

    def all_helicity_amplitudes(self, momenta) -> Dict[Tuple[int, ...], complex]:
        pdgs = self.initial + self.final
        configs = list(product(*[self._hel_choices(p) for p in pdgs]))
        amps = self.amplitudes_batch(momenta, configs)
        return {h: complex(a) for h, a in zip(configs, amps)}

    def m2(self, momenta, average_initial: bool = True) -> float:
        """Spin- and colour-summed |M|^2 (colour + initial-helicity averaged)."""
        pdgs = self.initial + self.final
        choices = [self._hel_choices(p) for p in pdgs]
        configs = list(product(*choices))
        tot = float(np.sum(self.m2_batch(momenta, configs)))
        if average_initial:
            n_avg = 1
            for p in self.initial:
                n_avg *= len(self._hel_choices(p))
            tot /= n_avg
        return float(tot)
