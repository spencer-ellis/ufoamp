"""
recursion.py
============

Berends-Giele off-shell recursion driven directly by the UFO vertex list.
This replaces explicit Feynman-diagram enumeration: currents for every subset
of external legs are built by combining sub-currents through every model vertex
that fits, and the amplitude is the current of all-but-one legs contracted with
the last leg.  Model, kinematics and coupling values enter separately, so
changing a coupling never touches this code.

Conventions used here (validated in tests/)
-------------------------------------------
* UFO vertex leg labels are OUTGOING particle types (established from the
  charged-current vertex [e+, nu_e, W-] = psibar_nu gamma P_L psi_e).
  A physical external leg therefore "plugs into" a vertex leg labelled by its
  outgoing type: X if outgoing, anti(X) if incoming.
* Momenta are booked all-incoming for conservation (p_in = +p for incoming,
  -p for outgoing legs).  The UFO momentum atom P(mu,k) is taken as
  ``P_SIGN * p_k(incoming)``; P_SIGN is a validated switch (see below).
* Fermion currents carry a type 'col' (psi) or 'bar' (psibar).  A 'col' current
  represents a fermion flowing downstream (propagator +pslash), a 'bar' current
  a fermion flowing upstream (propagator -pslash).
* Vector propagator in Feynman gauge is a pure scalar factor on J^mu.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import numpy as xp   # switchable backend
from . import wf_batched

from .dirac import METRIC, gamma_slash, ID4, dot, GAMMA
from .wavefunctions import u_spinor, v_spinor, polarization_vector
from .dirac import bar as _bar
from .vertex_eval import eval_vertex, eval_vertex_cur, fermion_chains, GAMMA_T, Cur
from .color import rep_dim

# charge conjugation C = i gamma^2 gamma^0 :  C gamma^mu^T C^-1 = -gamma^mu
_C = 1j * GAMMA[2] @ GAMMA[0]
_CINV = np.linalg.inv(_C)
from .ufo_model import Model

# Sign convention for the UFO momentum atom P(mu,k) relative to the all-incoming
# momentum of leg k. Labels are outgoing types; validated by the e+e- -> W+W-
# gauge-cancellation test.
P_SIGN = -1.0   # VALIDATED: e+e- -> W+W- longitudinal gauge cancellation (test_ufo_eeWW)


# --------------------------------------------------------------------------- #
#  external legs                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Leg:
    pdg: int            # physical particle
    incoming: bool
    p: np.ndarray       # physical four-momentum (E>0); shape (4,) or batched (B,4)
    hel: Any            # helicity (int) or array (B,) over the batch (events x configurations)

    def plug(self, model_anti):
        """UFO leg label this external state plugs into (outgoing type)."""
        return self.pdg if not self.incoming else model_anti(self.pdg)

    def p_in(self):
        return self.p if self.incoming else -self.p


# --------------------------------------------------------------------------- #
#  the amplitude engine                                                       #
# --------------------------------------------------------------------------- #
class Amplitude:
    """Currents are dicts  plug_label -> tag -> value, where ``tag`` encodes the
    fermion-line connectivity built so far:  tag = (closed, open)
      closed : frozenset of (bar_leg, unbar_leg) pairs for completed lines
      open   : external fermion leg index this (fermion) current started from,
               or None.
    From the final tag we obtain both the fermion-exchange sign (parity of the
    line permutation) and, for coloured lines, the colour-flow factor."""

    def __init__(self, model: Model, couplings: Dict[str, complex],
                 params: Dict[str, complex], NC: int = 3, gauge: str = "auto",
                 drop_vertex=None, max_orders: Optional[Dict[str, int]] = None):
        """gauge: 'feynman' (needs Goldstone bosons in the model), 'unitary'
        (massive-vector propagator with the k k/M^2 term), or 'auto' = Feynman
        if the UFO ships Goldstones (SM_UFO, HHVBF), else unitary (SMEFTsim and
        most MadGraph models)."""
        self.model = model
        if gauge == "auto":
            gauge = "feynman" if model.has_goldstones() else "unitary"
        self.gauge = gauge
        # drop_vertex(vertex, pdgs) -> True to exclude a vertex from the model
        self.drop_vertex = drop_vertex
        # coupling-order restriction, e.g. {"QCD": 0} reproduces MadGraph's
        # "QCD=0": every vertex coupling whose order exceeds the limit is dropped
        # (uses the UFO's own coupling orders, so it is model-agnostic)
        self.max_orders = dict(max_orders or {})
        # amplitude-level order limits, e.g. {"NP": (0, 1)}: keep only the part
        # of the amplitude whose total coupling order in each tracked name lies
        # in [min, max].  (max_orders acts per vertex; this acts per diagram,
        # like MadGraph's "NP<=1".)  Tracked order counts ride along in the
        # connectivity tag.
        self.amp_orders = None
        self._track = ()
        # static set of non-zero couplings (decided from concrete values, so that
        # traced/differentiable coupling values never enter Python control flow)
        self.active = {k: (complex(v) != 0) for k, v in couplings.items()}
        # Width in spacelike (t-channel) propagators.  False = keep it (MG5
        # standalone, RECOLA complex-mass); True = MG5 event generation with the
        # run-card option zerowidth_tchannel=True.
        self.zerowidth_tchannel = False
        # complex-mass scheme flag (masses in couplings are made complex by
        # Process; here it only affects the unitary-gauge numerator)
        self.cms = False
        self.real_masses = None        # {abs(pdg): (m, width)} kept real in the CMS
        # Majorana fermions: spin-1/2 and self-conjugate.  Their external
        # wavefunctions are u (in) / ubar (out); when a Majorana spinor enters a
        # vertex slot of the other type it is charge-conjugated in place:
        #   psi (col) -> psibar' = -psi^T C^-1 ,   psibar (bar) -> psi' = C psibar^T
        self.majorana = {p.pdg for p in model.particles if p.spin == 2 and p.self_conjugate}
        self._coupling_order = {c.name: c.order for c in model.couplings}
        # internal-line polarisation decomposition (option "propagator
        # decomposition"): {abs(pdg): (components, where)} with components a
        # subset of {"+","-","L","S"} ("T" = {"+","-"}) and where in
        # {"quark_line", "all"}.  self.frame_P (4-vector) defines the frame in
        # which the polarisation vectors are built (rest frame of frame_P).
        self.internal_pol = None
        self.frame_P = None
        self.coup = couplings
        self.params = params
        self.NC = NC
        self.p_by_pdg = model.p_by_pdg()
        self.p_by_py = model.p_by_pyname()
        self.lor = model.lorentz_by_name()
        self._index_vertices()
        self._select_color_mode()

    def _select_color_mode(self):
        """'tensor' if any active vertex has octets or a non-delta colour
        structure, else 'flow' (currents without colour axes, colour sum by
        cycle counting) -- exact and much faster for quark-line processes."""
        self.color_mode = "flow"
        if not getattr(self, "_external_colored", True):
            return
        for v in self.model.vertices:
            pdgs = [self.p_by_py[n].pdg for n in v.particles]
            if any(self.p_by_pdg[q].spin == -1 for q in pdgs):
                continue
            if self.drop_vertex is not None and self.drop_vertex(v, pdgs):
                continue
            if any(rep_dim(self.p_by_pdg[q].color) == 8 for q in pdgs) or any(
                    ("Identity" not in c and c.strip() != "1") for c in v.color):
                if any(self.active.get(cn, False) and not self._order_excluded(cn) for cn in v.couplings.values()):
                    self.color_mode = "tensor"; break

    def _order_excluded(self, cname) -> bool:
        if not self.max_orders:
            return False
        od = self._coupling_order.get(cname, {})
        return any(od.get(k, 0) > v for k, v in self.max_orders.items())

    def color(self, pdg: int) -> int:
        return self.p_by_pdg[pdg].color

    # ---- model helpers ---------------------------------------------------- #
    def anti(self, pdg: int) -> int:
        p = self.p_by_pdg[pdg]
        return pdg if p.self_conjugate else -pdg

    def mass(self, pdg: int) -> float:
        if self.real_masses is not None and abs(pdg) in self.real_masses:
            return self.real_masses[abs(pdg)][0]
        p = self.p_by_pdg[abs(pdg)] if abs(pdg) in self.p_by_pdg else self.p_by_pdg[pdg]
        v = self.params.get(p.mass, 0.0) if p.mass != "ZERO" else 0.0
        return float(np.real(v))

    def width(self, pdg: int) -> float:
        if self.real_masses is not None and abs(pdg) in self.real_masses:
            return self.real_masses[abs(pdg)][1]
        p = self.p_by_pdg[abs(pdg)] if abs(pdg) in self.p_by_pdg else self.p_by_pdg[pdg]
        v = self.params.get(p.width, 0.0) if p.width != "ZERO" else 0.0
        return float(np.real(v))

    def spin(self, pdg: int) -> int:
        return self.p_by_pdg[pdg].spin

    def _index_vertices(self):
        """key: (sorted input plug-labels) -> list of (vertex, output leg idx)."""
        self.vindex: Dict[tuple, List[tuple]] = {}
        self.max_inputs = 2
        for v in self.model.vertices:
            pdgs = [self.p_by_py[n].pdg for n in v.particles]
            # skip ghosts at tree level
            if any(self.p_by_pdg[q].spin == -1 for q in pdgs):
                continue
            if self.drop_vertex is not None and self.drop_vertex(v, pdgs):
                continue
            if self.gauge == "unitary" and any(self.p_by_pdg[q].goldstone for q in pdgs):
                continue                       # Goldstones are absent in unitary gauge
            seen = set()
            for out_i, out_pdg in enumerate(pdgs):
                inputs = tuple(sorted(pdgs[:out_i] + pdgs[out_i + 1:]))
                key = (inputs, out_pdg)
                if key in seen:
                    continue           # identical output particles: count once
                seen.add(key)
                self.vindex.setdefault(inputs, []).append((v, out_i, pdgs))
                self.max_inputs = max(self.max_inputs, len(inputs))

    # ---- external wavefunctions ------------------------------------------ #
    def external_wf(self, leg: Leg, leg_index: int = None) -> Cur:
        """Batched external wavefunction (vectorised; traceable under JAX)."""
        sp = self.spin(leg.pdg)
        m = self.mass(leg.pdg)
        hels = xp.asarray(leg.hel).reshape(-1)
        B = hels.shape[0]
        colored = rep_dim(self.color(leg.pdg)) > 0 and self.color_mode == "tensor"
        ext_self = leg_index if colored else None
        if sp == 1:
            return Cur(xp.ones(B, dtype=complex), "s", 0, (), ext_self)
        P = leg.p
        if P.ndim == 1:
            P = xp.broadcast_to(P, (B, 4))
        is_anti = leg.pdg < 0 and leg.pdg not in self.majorana
        if sp == 3:
            e = wf_batched.polarization_vector(P, hels, m)
            return Cur(e if leg.incoming else xp.conj(e), "v", 0, (), ext_self)
        if leg.incoming and not is_anti:
            arr, kind = wf_batched.u_spinor(P, hels, m), "col"
        elif leg.incoming and is_anti:
            arr, kind = _bar_b(wf_batched.v_spinor(P, hels, m)), "bar"
        elif (not leg.incoming) and not is_anti:
            arr, kind = _bar_b(wf_batched.u_spinor(P, hels, m)), "bar"
        else:
            arr, kind = wf_batched.v_spinor(P, hels, m), "col"
        return Cur(arr, kind, 0, (), ext_self)

    def propagate(self, pdg: int, cur: Cur, P: xp.ndarray, fs=None) -> Cur:
        """P: line momentum (4,) or (B,4).  cur.arr axes (B,[4],[own],*ext)."""
        sp = self.spin(pdg)
        m = self.mass(pdg)
        w = self.width(pdg)
        P = xp.asarray(P)
        Pl = P @ METRIC
        P2 = xp.sum(P * Pl, axis=-1)
        if self.zerowidth_tchannel and w:
            w = xp.where(xp.real(P2) < 0, 0.0, w)
        # pole at p^2 = m^2 - i|m|Gamma: the width term uses |m| (SLHA mass
        # eigenvalues, e.g. neutralinos, can carry a negative sign)
        den = P2 - m**2 + 1j * abs(m) * w                  # scalar or (B,)
        arr = cur.arr
        extra = arr.ndim - (2 if cur.has_lorentz else 1)  # number of colour axes
        def bshape(x):                                     # broadcast (B,) over trailing axes
            return xp.reshape(x, (-1,) + (1,) * (arr.ndim - 1)) if xp.ndim(x) else x
        if sp == 1:
            return Cur(1j * arr * bshape(1 / den), "s", cur.own, cur.ext)
        if sp == 3:
            if self.internal_pol and m > 0 and fs is not None:
                comps = self._line_components(pdg, fs)
                if comps is not None:
                    flat = xp.moveaxis(arr, 1, -1).reshape(-1, 4)          # (B*K, 4)
                    K = flat.shape[0] // arr.shape[0]
                    Pk = xp.repeat(P if P.ndim == 2 else xp.broadcast_to(P, (arr.shape[0], 4)), K, axis=0)
                    fp = self.frame_P
                    if fp is not None and xp.ndim(fp) == 2:
                        fp = xp.repeat(fp, K, axis=0)
                    num = polarised_numerator(Pk, flat, m, comps, fp)
                    num = xp.moveaxis(num.reshape(arr.shape[:1] + arr.shape[2:] + (4,)), -1, 1)
                    return Cur(num * bshape(-1j / den), "v", cur.own, cur.ext)
            if self.gauge == "unitary" and m > 0:
                m2c = (m**2 - 1j * abs(m) * self.width(pdg)) if self.cms else m**2
                kJ = xp.einsum("bm...,bm->b...", arr, xp.broadcast_to(Pl, (arr.shape[0], 4)))
                Pb = xp.broadcast_to(P, (arr.shape[0], 4)).reshape((arr.shape[0], 4) + (1,) * extra)
                return Cur((arr - Pb * kJ[:, None] / m2c) * bshape(-1j / den), "v", cur.own, cur.ext)
            return Cur(arr * bshape(-1j / den), "v", cur.own, cur.ext)
        if sp == 2:
            Plb = xp.broadcast_to(Pl, (arr.shape[0], 4))
            G = xp.einsum("bm,mij->bij", Plb, GAMMA_T)
            if cur.kind == "col":
                out = xp.einsum("bij,bj...->bi...", G, arr) + m * arr
            else:
                out = -xp.einsum("bi...,bij->bj...", arr, G) + m * arr
            return Cur(out * bshape(1j / den), cur.kind, cur.own, cur.ext)
        raise ValueError

    def _conjugate_mismatched(self, ins, chains, pdgs, out_leg):
        """Return inputs with Majorana spinors charge-conjugated where their
        type does not match the slot (row = psibar, col = psi)."""
        need = {}
        for r, c in chains:
            need[r] = "bar"; need[c] = "col"
        out = ins
        for lbl, cur in ins.items():
            want = need.get(lbl)
            if want is None or cur.kind == want or cur.kind in ("s", "v"):
                continue
            # Majorana fermions, and Dirac fermions in fermion-flow-violating
            # ("clashing arrow") vertices as written by FeynRules for e.g.
            # chargino-lepton-sneutrino: reverse the flow by charge conjugation.
            if out is ins:
                out = dict(ins)
            out[lbl] = conjugate_current(cur)
        return out

    def _is_quark_line(self, fs) -> bool:
        """subset = exactly one fermion line (two external fermions), no bosons"""
        return len(fs) == 2 and all(self._is_ferm[i] for i in fs)

    def _line_components(self, pdg, fs):
        """internal_pol formats:
           {abs(pdg): (comps, where)}   where in {"all","quark_line","t_channel"}
           {"t_channel": {initial_leg_index: comps}}  per-line (VBF), any massive V
        Returns the component list or None (ordinary propagator)."""
        ip = self.internal_pol
        if "t_channel" in ip:
            if not self._is_quark_line(fs):
                return None
            ini = [i for i in fs if self._legs[i].incoming]
            if len(ini) != 1:
                return None
            return ip["t_channel"].get(ini[0])
        if abs(pdg) not in ip:
            return None
        comps, where = ip[abs(pdg)]
        if where == "all":
            return comps
        if not self._is_quark_line(fs):
            return None
        if where == "t_channel" and sum(self._legs[i].incoming for i in fs) != 1:
            return None
        return comps

    # ---- vertex evaluation ------------------------------------------------ #
    def vertex_structures(self, vert, pdgs, out_i, inputs, sub_pin):
        """Yield (Cur, chains) for each (colour, Lorentz) structure pair of the
        vertex, leaving leg out_i (0-based) free.  inputs: {0-based leg: Cur}."""
        n = len(pdgs)
        spins = {i + 1: self.spin(pdgs[i]) for i in range(n)}
        cdims = ({i + 1: rep_dim(self.color(pdgs[i])) for i in range(n)} if self.color_mode == "tensor"
                 else {i + 1: 0 for i in range(n)})
        momenta = {i + 1: P_SIGN * sub_pin[i] for i in range(n) if i in sub_pin}
        ins = {i + 1: inputs[i] for i in inputs}
        for (ci, li), cname in vert.couplings.items():
            if not self.active.get(cname, False):
                continue
            g = self.coup[cname]
            if self.max_orders:
                od = self._coupling_order.get(cname, {})
                if any(od.get(k, 0) > v for k, v in self.max_orders.items()):
                    continue
            lstruct = self.lor[vert.lorentz[li]].structure
            chains = fermion_chains(lstruct)
            ins_l = self._conjugate_mismatched(ins, chains, pdgs, out_i + 1)
            cstruct = vert.color[ci] if self.color_mode == "tensor" else "1"
            cur = eval_vertex_cur(lstruct, cstruct, spins, cdims, momenta, ins_l, out_i + 1)
            yield cur.scale(g), chains, self._coupling_order.get(cname, {})

    def amplitude(self, legs: List[Leg]) -> complex:
        """Colour- and fermion-sign-correct amplitude is not a single number in
        general; use amplitude_flows() + m2_color(). This returns the coherent
        sum assuming a single colour flow (valid for colour-trivial processes)."""
        flows = self.amplitude_flows(legs)
        return sum(flows.values())

    def amplitude_flows(self, legs: List[Leg]) -> Dict[tuple, complex]:
        """Return {connectivity_tag: amplitude} with fermion signs applied."""
        n = len(legs)
        S = tuple(range(n - 1))
        plug = [lg.plug(self.anti) for lg in legs]
        pin = [lg.p_in() for lg in legs]
        wf = [self.external_wf(lg, i) for i, lg in enumerate(legs)]
        self._legs = legs
        self._is_ferm = [self.spin(lg.pdg) == 2 for lg in legs]
        # bar end = wavefunction is barred
        self._is_bar = [(self._is_ferm[i] and wf[i].kind == "bar") for i in range(n)]

        J: Dict[frozenset, Dict[int, Dict[tuple, Any]]] = {}
        Pin = {}
        for i in S:
            tag = (frozenset(), i if self._is_ferm[i] else None, (0,) * len(self._track))
            J[frozenset([i])] = {plug[i]: {tag: wf[i]}}
            Pin[frozenset([i])] = pin[i]

        def total_p(sub):
            return sum(pin[i] for i in sub)

        for size in range(2, n - 1):
            for sub in combinations(S, size):
                fs = frozenset(sub)
                Pin[fs] = total_p(sub)
                J[fs] = self._build(fs, J, Pin, propagate=True)

        fsS = frozenset(S)
        Pin[fsS] = total_p(S)
        root_i = n - 1
        flows = self._build(fsS, J, Pin, propagate=False,
                            root_plug=plug[root_i], root_wf=wf[root_i],
                            root_i=root_i)
        # apply fermion-exchange sign per connectivity; apply amplitude-level
        # order limits
        out = {}
        for tag, val in flows.items():
            if self.amp_orders:
                ok = True
                for i, n in enumerate(self._track):
                    lo, hi = self.amp_orders[n]
                    if not (lo <= tag[2][i] <= hi):
                        ok = False
                if not ok:
                    continue
            out[tag] = val * self._fermion_sign(tag)
        return out

    def _m2_flow(self, legs, flows):
        """Colour sum for processes whose only colour structures are deltas on
        quark lines: sum_{a,b} M_a M_b^* N_c^{cycles(a o b^-1)} over the
        fermion-line connectivities (colour-flow matrix)."""
        col_legs = [i for i, lg in enumerate(legs) if rep_dim(self.color(lg.pdg)) == 3]
        vals = list(flows.values())
        if not col_legs:
            a = sum(vals) if vals else 0.0
            return xp.abs(a) ** 2
        def is_color_end(i):
            lg = legs[i]
            return (lg.incoming and lg.pdg > 0) or ((not lg.incoming) and lg.pdg < 0)
        def perm_of(tag):
            m = {}
            for b, u in tag[0]:
                if b in col_legs and u in col_legs:
                    c, ac = (b, u) if is_color_end(b) else (u, b)
                    m[c] = ac
            return m
        items = list(flows.items())
        tot = 0.0
        for ta, A in items:
            pa = perm_of(ta)
            for tb, B in items:
                pb = perm_of(tb); inv_b = {v: k for k, v in pb.items()}
                visited = set(); cycles = 0
                for c in pa:
                    if c in visited:
                        continue
                    cycles += 1; x = c
                    while x not in visited:
                        visited.add(x); x = inv_b[pa[x]]
                tot = tot + xp.real(A * xp.conj(B)) * (self.NC ** cycles)
        n_init = sum(1 for i in col_legs if legs[i].incoming)
        return tot / (self.NC ** n_init)

    def amplitude_tensor(self, legs):
        if self.color_mode == "flow" and any(rep_dim(self.color(lg.pdg)) for lg in legs):
            raise RuntimeError("amplitude_tensor needs color_mode='tensor' for coloured processes")
        """Coherent sum of all flows: colour tensor (B, *external colours)."""
        flows = self.amplitude_flows(legs)
        M = None
        for v in flows.values():
            M = v if M is None else M + v
        return M

    def _fermion_sign(self, tag) -> float:
        closed = tag[0]
        if not closed:
            return 1.0
        seq = []
        for b, u in sorted(closed):
            seq += [b, u]
        ref = sorted(seq)
        # parity of permutation taking ref -> seq
        perm = [ref.index(x) for x in seq]
        sign = 1.0
        visited = [False] * len(perm)
        for i in range(len(perm)):
            if visited[i]:
                continue
            j = i; L = 0
            while not visited[j]:
                visited[j] = True; j = perm[j]; L += 1
            if L % 2 == 0:
                sign = -sign
        return sign

    def m2_color(self, legs: List[Leg]):
        """|M|^2 summed over external colours (the amplitude is a tensor over
        them), averaged over initial colours; per batch entry -> (B,)."""
        flows = self.amplitude_flows(legs)
        if self.color_mode == "flow":
            return self._m2_flow(legs, flows)
        M = None
        for tag, val in flows.items():
            M = val if M is None else M + val
        if M is None:
            return 0.0
        M = xp.asarray(M)
        m2 = xp.sum(xp.abs(M.reshape(M.shape[0], -1)) ** 2, axis=1)
        for lg in legs:
            if lg.incoming:
                d = rep_dim(self.color(lg.pdg))
                if d:
                    m2 = m2 / d
        return m2

    @staticmethod
    def _partitions(fs, k):
        """Unordered partitions of frozenset fs into k non-empty blocks."""
        items = sorted(fs)
        def rec(remaining, k):
            if k == 1:
                yield (frozenset(remaining),)
                return
            first = remaining[0]; rest = remaining[1:]
            # first block always contains the smallest element (canonical order)
            for n in range(0, len(rest) - (k - 1) + 1):
                for comb in combinations(rest, n):
                    A = frozenset((first,) + comb)
                    left = [x for x in rest if x not in comb]
                    for tail in rec(left, k - 1):
                        yield (A,) + tail
        yield from rec(items, k)

    def _partitions2(self, fs):
        items = sorted(fs)
        first = items[0]
        rest = items[1:]
        for k in range(0, len(rest)):
            for comb in combinations(rest, k):
                A = frozenset((first,) + comb)
                B = fs - A
                if B:
                    yield A, B

    def _partitions3(self, fs):
        items = sorted(fs)
        first = items[0]
        rest = items[1:]
        for kA in range(0, len(rest) - 1):
            for combA in combinations(rest, kA):
                A = frozenset((first,) + combA)
                remaining = sorted(fs - A)
                r0 = remaining[0]
                rr = remaining[1:]
                for kB in range(0, len(rr)):
                    for combB in combinations(rr, kB):
                        B = frozenset((r0,) + combB)
                        C = fs - A - B
                        if C:
                            yield A, B, C

    def set_amp_orders(self, amp_orders):
        self.amp_orders = dict(amp_orders) if amp_orders else None
        self._track = tuple(sorted(self.amp_orders)) if self.amp_orders else ()

    def _merge_tags(self, tags_by_leg, out_i, chains, vorder=None):
        """Combine sub-current tags at a vertex using the structure's spinor
        chains. tags_by_leg: {vertex leg idx (0-based): tag}. Returns
        (closed, open, orders)."""
        closed = frozenset().union(*[t[0] for t in tags_by_leg.values()])
        orders = ()
        if self._track:
            tot = [0] * len(self._track)
            for t in tags_by_leg.values():
                for i, v in enumerate(t[2] if len(t) > 2 and t[2] else ()):
                    tot[i] += v
            for i, n in enumerate(self._track):
                tot[i] += (vorder or {}).get(n, 0)
            orders = tuple(tot)
        open_out = None
        for row_leg, col_leg in chains:      # 1-based vertex leg labels
            r, c = row_leg - 1, col_leg - 1
            if out_i in (r, c):
                other = c if out_i == r else r
                t = tags_by_leg.get(other)
                if t is not None and t[1] is not None:
                    open_out = t[1]
            else:
                tr, tc = tags_by_leg.get(r), tags_by_leg.get(c)
                if tr is None or tc is None or tr[1] is None or tc[1] is None:
                    continue
                closed = closed | {(tr[1], tc[1])}          # (bar end, unbar end) by slot
        return (closed, open_out, orders)

    def _build(self, fs, J, Pin, propagate, root_plug=None, root_wf=None, root_i=None):
        out: Dict[int, Dict[tuple, Any]] = {}
        Ptot = Pin[fs]

        def accumulate(out_pdg, tag, current):
            if propagate:
                cur = self.propagate(out_pdg, current, Ptot, fs)
                key = self.anti(out_pdg)
                d = out.setdefault(key, {})
                d[tag] = d[tag].add(cur) if tag in d else cur
            else:
                if out_pdg != root_plug:
                    return
                rw = root_wf
                closed, op = tag[0], tag[1]
                orders = tag[2] if len(tag) > 2 else ()
                if current.kind in ("col", "bar"):
                    want = "bar" if current.kind == "col" else "col"
                    if rw.kind != want:
                        rw = conjugate_current(rw)
                    pair = (root_i, op) if rw.kind == "bar" else (op, root_i)
                    tag = (closed | {pair}, None, orders)
                else:
                    tag = (closed, None, orders)
                val = _contract(current, rw)
                d = out.setdefault("amp", {})
                d[tag] = d.get(tag, 0j) + val

        for k in range(2, min(self.max_inputs, len(fs)) + 1):
            for subs in self._partitions(fs, k):
                self._combine(subs, J, Pin, accumulate)
        return out if propagate else out.get("amp", {})

    def _combine(self, subs, J, Pin, accumulate):
        choices = [[(lbl, tag, cur) for lbl, d in J[s].items() for tag, cur in d.items()]
                   for s in subs]
        def rec(i, chosen):
            if i == len(subs):
                self._apply_vertex(subs, chosen, Pin, accumulate)
                return
            for item in choices[i]:
                rec(i + 1, chosen + [item])
        rec(0, [])

    def _apply_vertex(self, subs, chosen, Pin, accumulate):
        labels = [c[0] for c in chosen]
        key = tuple(sorted(labels))
        for vert, out_i, pdgs in self.vindex.get(key, []):
            legs_avail = [i for i in range(len(pdgs)) if i != out_i]
            assign = _match(labels, [pdgs[i] for i in legs_avail])
            if assign is None:
                continue
            inputs = {}; sub_pin = {}; tags_by_leg = {}
            for k, (lbl, tag, cur) in enumerate(chosen):
                leg_i = legs_avail[assign[k]]
                inputs[leg_i] = cur
                sub_pin[leg_i] = Pin[subs[k]]
                tags_by_leg[leg_i] = tag
            out_pdg = pdgs[out_i]
            for val, chains, vorder in self.vertex_structures(vert, pdgs, out_i, inputs, sub_pin):
                tag = self._merge_tags(tags_by_leg, out_i, chains, vorder)
                accumulate(out_pdg, tag, val)


# --------------------------------------------------------------------------- #
#  small helpers                                                              #
# --------------------------------------------------------------------------- #
def _match(labels, legpdgs):
    """Return assignment list: labels[k] -> index into legpdgs (distinct)."""
    used = [False] * len(legpdgs)
    out = []
    for lbl in labels:
        found = None
        for i, q in enumerate(legpdgs):
            if not used[i] and q == lbl:
                found = i; break
        if found is None:
            return None
        used[found] = True
        out.append(found)
    return out


def _bar_b(psi):
    """Dirac adjoint of batched spinors (B,4): psi^dagger gamma^0."""
    return xp.conj(psi) @ GAMMA[0]


def conjugate_current(cur: Cur) -> Cur:
    """Charge-conjugate a spinor current in place of a fermion-flow reversal:
    col -> bar : -psi^T C^-1 ;  bar -> col : C psibar^T."""
    a = cur.arr
    if cur.kind == "col":
        return Cur(-xp.einsum("bi...,ij->bj...", a, _CINV), "bar", cur.own, cur.ext, cur.ext_self)
    if cur.kind == "bar":
        return Cur(xp.einsum("ij,bj...->bi...", _C, a), "col", cur.own, cur.ext, cur.ext_self)
    return cur


def _scale(val, g):
    return val.scale(g) if isinstance(val, Cur) else g * val

def _add(a, b):
    return a.add(b) if isinstance(a, Cur) else a + b

def _contract(current: Cur, wf: Cur):
    """Contract the current for the root leg with the root's external
    wavefunction.  Result axes: (B, *ext colour of the other legs, [root colour])
    -- the root is the last leg, so its open colour axis goes last."""
    a = current.arr; w = wf.arr
    if current.kind in ("col", "bar"):
        res = xp.einsum("bi...,bi->b...", a, w)
    elif current.kind == "v":
        res = xp.einsum("bm...,mn,bn->b...", a, METRIC, w)
    else:
        res = xp.einsum("b...,b->b...", a, w)
    return res

# --------------------------------------------------------------------------- #
#  internal-line polarisation vectors (propagator decomposition)              #
# --------------------------------------------------------------------------- #
def boost_matrix(P):
    """Lorentz boost(s) taking lab vectors to the rest frame of P: (4,)->(4,4), (B,4)->(B,4,4)."""
    P = xp.real(P)
    single = P.ndim == 1
    P = xp.atleast_2d(P)
    M = xp.sqrt(xp.maximum(P[:, 0]**2 - xp.sum(P[:, 1:]**2, axis=1), 1e-300))
    b = P[:, 1:] / P[:, :1]; b2 = xp.sum(b * b, axis=1)
    g = P[:, 0] / M
    ok = b2 > 1e-30
    b2s = xp.where(ok, b2, 1.0)
    L00 = xp.where(ok, g, 1.0)
    L0i = xp.where(ok[:, None], -g[:, None] * b, 0.0)
    Lij = xp.eye(3)[None] + xp.where(ok, (g - 1) / b2s, 0.0)[:, None, None] * xp.einsum("bi,bj->bij", b, b)
    top = xp.concatenate([L00[:, None, None], L0i[:, None, :]], axis=2)
    bot = xp.concatenate([L0i[:, :, None], Lij], axis=2)
    L = xp.concatenate([top, bot], axis=1)
    return L[0] if single else L


def polarisation_vectors_offshell(k):
    """eps_+, eps_-, eps_L (contravariant) for possibly off-shell k (B,4);
    -g + kk/k^2 = sum_pm eps eps* + sign(k^2) eps_L eps_L*."""
    k = xp.real(xp.atleast_2d(k))
    kv = k[:, 1:]; kk = xp.sqrt(xp.sum(kv**2, axis=1))
    n = xp.where(kk[:, None] < 1e-12, xp.array([0., 0., 1.]), kv / xp.maximum(kk, 1e-300)[:, None])
    a = xp.where(xp.abs(n[:, :1]) < 0.9, xp.array([1., 0., 0.]), xp.array([0., 1., 0.]))
    e1 = a - xp.sum(a * n, axis=1)[:, None] * n; e1 = e1 / xp.sqrt(xp.sum(e1**2, axis=1))[:, None]
    e2 = xp.cross(n, e1)
    z = xp.zeros((k.shape[0], 1))
    ep = xp.concatenate([z, -(e1 + 1j * e2) / xp.sqrt(2.0)], axis=1)
    em = xp.concatenate([z, (e1 - 1j * e2) / xp.sqrt(2.0)], axis=1)
    k2 = k[:, 0]**2 - kk**2
    eL = xp.concatenate([kk[:, None], k[:, :1] * n], axis=1) / xp.sqrt(xp.maximum(xp.abs(k2), 1e-300))[:, None]
    return {"+": ep + 0j, "-": em + 0j, "L": eL + 0j, "k2": k2}


def polarised_numerator(k, J, m, comps, frame_P=None):
    """Return N^{mu nu} J_nu for the selected components of
       N = -g + k k / m^2 = T + L + S,  T = sum_pm eps eps*, L = eps_L eps_L*,
       S = k k (1/m^2 - 1/k^2),
    built in the rest frame of frame_P (lab if None).  Batched: J (B,4), k (4,)
    or (B,4), frame_P (4,) or (B,4)."""
    J = xp.atleast_2d(xp.asarray(J, complex)); B = len(J)
    k = xp.real(xp.asarray(k)); k = xp.broadcast_to(k, (B, 4)) if k.ndim == 1 else k
    if frame_P is None:
        Lam = xp.tile(xp.eye(4), (B, 1, 1))
    else:
        fp = xp.real(xp.asarray(frame_P)); fp = xp.broadcast_to(fp, (B, 4)) if fp.ndim == 1 else fp
        Lam = boost_matrix(fp)
    Linv = xp.linalg.inv(Lam)
    kf = xp.einsum("bij,bj->bi", Lam, k); Jf = xp.einsum("bij,bj->bi", Lam, J)
    eps = polarisation_vectors_offshell(kf)
    k2f = eps["k2"]
    kfl = kf @ METRIC
    out = xp.zeros_like(Jf)
    for c in comps:
        if c in ("+", "-", "L"):
            e = eps[c]
            coef = xp.sum(Jf * (xp.conj(e) @ METRIC), axis=1)           # eps*.J
            sgn = xp.where(k2f > 0, 1.0, -1.0) if c == "L" else 1.0
            out += (sgn * coef)[:, None] * e
        elif c == "S":
            out += (xp.sum(Jf * kfl, axis=1) * (1.0 / m**2 - 1.0 / k2f))[:, None] * kf
        else:
            raise ValueError(f"unknown polarisation component {c!r}")
    return -xp.einsum("bij,bj->bi", Linv, out)
