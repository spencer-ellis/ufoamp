"""
analysis.py — the analysis-facing API.

    from ufoamp.analysis import MatrixElement, read_lhe

    me = MatrixElement("/path/HHVBF_UFO", "u d > u d h h",
                       couplings={"CV": 1, "C2V": 2, "C3": 1})
    m2 = me.m2(momenta)                        # one event: spin+colour summed/averaged
    m2 = me.m2(momenta, pol={"w+": "L"})       # external-polarisation selection
    arr = me.m2_events(events, n_jobs=4)       # (N, n_particles, 4) -> (N,)

Conventions
  * momenta: sequence of 4-vectors [E, px, py, pz] in GeV, metric (+,-,-,-),
    order = initial particles then final particles, as in the process string.
  * couplings: any *external* UFO parameter (Wilson coefficients, C3/CV/C2V,
    masses, widths, CKM parameters, ...).  Change them any time with
    me.set_couplings({...}); no rebuild.
  * pol: {particle: spec} selects external helicities.  particle is a UFO
    name ("w+", "z", "h", "e-"), a PDG code, or a 0-based leg index;
    spec is "L" (longitudinal, h=0), "T" (transverse, h=+-1), "+", "-", or an
    explicit list of helicities.  Unlisted particles are summed.  The result
    is |M|^2 summed over the selected helicity set (no interference between
    different external helicities exists in |M|^2, so sums are additive:
    m2(all) == m2(L) + m2(T) for a given particle).
  * average_initial: divide by the number of initial helicity/colour states.
    If you select initial polarisations you probably want average_initial=False
    (or handle the flux factor yourself) — the flag is applied as given.
"""

from __future__ import annotations

import os, re
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .ufo_model import load, Model
from .process import Process

Spec = Union[str, Sequence[int]]


class MatrixElement:
    def __init__(self, model: Union[str, Model], process: Union[str, Tuple[List[int], List[int]]],
                 couplings: Optional[Dict[str, float]] = None, gauge: str = "auto",
                 restrict_couplings=None, inject=None, drop_vertex=None,
                 massless_light_quarks: bool = True, param_card: Optional[str] = None,
                 max_orders: Optional[Dict[str, int]] = None, pol_frame="lab",
                 restrict="default", zerowidth_tchannel: bool = False, scheme: str = "fixed_width"):
        """param_card: SLHA param_card.dat or MadGraph restrict_*.dat whose
        values are applied before `couplings` (precedence: light-quark defaults
        < card < couplings).  massless_light_quarks: set the u,d,s,c masses AND
        their Yukawa parameters to zero, as MadGraph does by default.
        max_orders: coupling-order limits, e.g. {"QCD": 0} for a QCD=0 sample
        (vertex couplings whose UFO order exceeds the limit are dropped).
        scheme: "fixed_width" (real masses in couplings, i M Gamma in
        propagators -- MG5 default; gauge dependent at O(Gamma/M)) or "cms"
        (complex-mass scheme: M^2 -> M^2 - i M Gamma everywhere, incl. couplings
        and the unitary-gauge numerator; gauge independent -- RECOLA's scheme).
        restrict: model restriction applied first, as MadGraph does on
        `import model`: "default" = <ufo>/restrict_default.dat if it exists,
        a name ("no_b_mass" -> restrict_no_b_mass.dat), a path, or None.
        pol_frame: frame in which EXTERNAL helicities are defined: "lab",
        "partonic_cm" (rest frame of the initial state, MadGraph's me_frame
        default), or a list of leg indices (rest frame of their sum).  |M|^2
        summed over helicities is frame independent; polarised pieces are not."""
        self.model = load(model) if isinstance(model, str) else model
        self._names = self._name_table()
        self._light = {}
        if massless_light_quarks:
            names = {p.name for p in self.model.parameters if p.nature == "external"}
            for n in ("MU", "MD", "MS", "MC", "ymup", "ymdo", "ymstr", "ymcha", "ymc", "yms", "ymu", "ymd",
                      "MUP", "MDO", "MST", "MCH"):
                if n in names:
                    self._light[n] = 0.0
        if isinstance(process, str):
            self.initial, self.final = self.parse_process(process)
        else:
            self.initial, self.final = list(process[0]), list(process[1])
        self.pdgs = self.initial + self.final
        self.n_in = len(self.initial)
        self._restrict = {}
        rpath = None
        model_dir = model if isinstance(model, str) else getattr(self.model, "path", "")
        if restrict is not None and model_dir:
            model = model_dir
            if restrict == "default":
                cand = os.path.join(model, "restrict_default.dat")
                rpath = cand if os.path.isfile(cand) else None
            elif os.path.isfile(restrict):
                rpath = restrict
            else:
                cand = os.path.join(model, f"restrict_{restrict}.dat")
                if not os.path.isfile(cand):
                    raise FileNotFoundError(f"restriction card {restrict!r} not found in {model}")
                rpath = cand
        if rpath:
            self._restrict = dict(read_param_card(rpath, self.model))
        self.restrict_card = rpath
        self._card = dict(read_param_card(param_card, self.model)) if param_card else {}
        self.couplings = {**self._light, **self._restrict, **self._card, **(couplings or {})}
        self._gauge, self._restrict_couplings, self._inject, self._drop = gauge, restrict_couplings, inject, drop_vertex
        self.pol_frame = pol_frame
        self._zwt = zerowidth_tchannel
        self.scheme = scheme
        self.proc = Process(self.model, self.initial, self.final, overrides=self.couplings,
                            restrict_couplings=restrict_couplings, inject=inject, gauge=gauge,
                            drop_vertex=drop_vertex, max_orders=max_orders)
        self._apply_scheme()
        self._hel_sets = [self.proc._hel_choices(p) for p in self.pdgs]

    # ------------------------------------------------------------------ setup
    def _name_table(self) -> Dict[str, int]:
        t = {}
        for p in self.model.particles:
            t[p.name.lower()] = p.pdg
            t[p.py_name.lower()] = p.pdg
        # common aliases
        alias = {"h": 25, "higgs": 25, "z": 23, "a": 22, "gamma": 22, "photon": 22, "g": 21,
                 "w+": 24, "w-": -24, "e+": -11, "e-": 11, "mu+": -13, "mu-": 13,
                 "ve": 12, "ve~": -12, "vm": 14, "vm~": -14}
        for k, v in alias.items():
            t.setdefault(k, v)
        for q, pdg in (("d", 1), ("u", 2), ("s", 3), ("c", 4), ("b", 5), ("t", 6)):
            t.setdefault(q, pdg); t.setdefault(q + "~", -pdg)
        return t

    def parse_process(self, s: str) -> Tuple[List[int], List[int]]:
        """'u d > u d h h'  (also accepts '->' and PDG codes)."""
        s = s.replace("->", ">")
        if ">" not in s:
            raise ValueError("process string must contain '>'")
        lhs, rhs = s.split(">")
        def conv(tok):
            tok = tok.strip()
            if re.fullmatch(r"-?\d+", tok):
                return int(tok)
            key = tok.lower()
            if key not in self._names:
                raise KeyError(f"unknown particle {tok!r}; known: {sorted(set(self._names))[:40]} ...")
            return self._names[key]
        return [conv(t) for t in lhs.split()], [conv(t) for t in rhs.split()]

    def set_param(self, couplings: Dict[str, float], replace: bool = False):
        """Update coupling values (merged into the current set unless replace)."""
        self.couplings = {**self._light, **self._restrict, **self._card, **couplings} if replace else {**self.couplings, **couplings}
        self.proc.set_couplings(overrides=self.couplings, restrict_couplings=self._restrict_couplings,
                                inject=self._inject)
        self._apply_scheme()

    set_couplings = set_param          # backwards-compatible alias

    def _apply_scheme(self):
        self.proc.amp.zerowidth_tchannel = self._zwt
        if self.scheme == "cms":
            # complex masses for every particle with a width, re-derive couplings
            inj = dict(self._inject or {})
            real = {}
            for p in self.model.particles:
                if p.pdg < 0:
                    continue
                m = float(np.real(self.proc.params.get(p.mass, 0))) if p.mass != "ZERO" else 0.0
                w = float(np.real(self.proc.params.get(p.width, 0))) if p.width != "ZERO" else 0.0
                real[p.pdg] = (m, w)
                if m != 0 and w > 0:
                    inj[p.mass] = np.sign(m) * np.sqrt(m**2 - 1j * abs(m) * w)
            self.proc.set_couplings(overrides=self.couplings, restrict_couplings=self._restrict_couplings, inject=inj)
            self.proc.amp.cms = True
            self.proc.amp.real_masses = real
            self.proc.amp.zerowidth_tchannel = self._zwt
        elif self.scheme != "fixed_width":
            raise ValueError("scheme must be 'fixed_width' or 'cms'")

    set_couplings = set_param          # backward-compatible alias

    @property
    def gauge(self) -> str:
        return self.proc.amp.gauge

    def param(self, name: str):
        return self.proc.params[name]

    # ------------------------------------------------------------- helicities
    def _legs_for(self, key) -> List[int]:
        if isinstance(key, int) and 0 <= key < len(self.pdgs) and not (key in self.pdgs and key < 0):
            # ambiguous: small ints are leg indices unless they match a pdg
            if key < len(self.pdgs):
                return [key]
        if isinstance(key, str):
            key = self._names[key.lower()]
        legs = [i for i, p in enumerate(self.pdgs) if p == key]
        if not legs:
            raise KeyError(f"particle {key} not in process {self.pdgs}")
        return legs

    def helicity_sets(self, pol: Optional[Dict] = None) -> List[List[int]]:
        sets = [list(h) for h in self._hel_sets]
        for key, spec in (pol or {}).items():
            for leg in self._legs_for(key):
                full = self._hel_sets[leg]
                if isinstance(spec, str):
                    spec = {"L": [0], "T": [1, -1], "+": [1], "-": [-1], "sum": full, "all": full}[spec.upper() if spec.upper() in ("L", "T") else spec]
                chosen = [h for h in spec if h in full]
                if not chosen:
                    raise ValueError(f"helicities {spec} not available for leg {leg} (pdg {self.pdgs[leg]}: {full})")
                sets[leg] = chosen
        return sets

    # ------------------------------------------------------------- frames
    def _to_frame(self, events):
        """Boost (N,n,4) or (n,4) momenta to self.pol_frame."""
        from .recursion import boost_matrix
        M = np.asarray(events, dtype=float)
        fr = self.pol_frame
        if fr is None or fr == "lab":
            return M
        single = M.ndim == 2
        M = M[None] if single else M
        if fr == "partonic_cm":
            legs = list(range(self.n_in))
        else:
            legs = list(fr)
        P = M[:, legs].sum(axis=1)                       # (N,4)
        Lam = boost_matrix(P)                             # (N,4,4)
        out = np.einsum("bij,bkj->bki", Lam, M)
        return out[0] if single else out

    # ------------------------------------------------------------- evaluation
    def helicity_amplitudes(self, momenta, pol: Optional[Dict] = None) -> Dict[Tuple[int, ...], complex]:
        """{(h1,...,hn): M} for the selected helicity set (colour-summed coherently
        only for colour-trivial processes; use m2 for colour-correct squares)."""
        mom = self._to_frame([np.asarray(p, dtype=float) for p in momenta])
        configs = list(product(*self.helicity_sets(pol)))
        return {h: complex(a) for h, a in zip(configs, self.proc.amplitudes_batch(mom, configs))}

    def prune_helicities(self, momenta, pol: Optional[Dict] = None, tol: float = 1e-28):
        """Find helicity configurations that vanish identically (e.g. chirality
        violating configurations on massless fermion lines) on this event and
        skip them from now on.  Exact for massless-fermion zeros; call
        reset_pruning() if you change masses."""
        mom = self._to_frame([np.asarray(p, dtype=float) for p in momenta])
        configs = list(product(*self.helicity_sets(pol)))
        v = np.real(self.proc.m2_batch(mom, configs))
        mx = float(v.max()) if len(v) else 0.0
        self._skip = {h for h, x in zip(configs, v) if x <= tol * mx}
        return len(self._skip), len(configs)

    def reset_pruning(self):
        self._skip = set()

    # ------------------------------------------- internal-boson polarisation
    _COMPS = {"L": ["L"], "T": ["+", "-"], "+": ["+"], "-": ["-"], "S": ["S"],
              "A": ["+", "-", "L", "S"], "P": ["+", "-", "L"]}   # P = physical (T+L), no S

    def set_amp_orders(self, amp_orders: Optional[Dict[str, tuple]]):
        """Amplitude-level coupling-order window, e.g. {"NP": (0, 1)} (MadGraph
        NP<=1) or {"NP": (1, 1)} (single-insertion part only). None = all."""
        self.proc.set_amp_orders(amp_orders)

    def set_vpol(self, vpol: Optional[str], frame="final_bosons", vbs_only: bool = False):
        """Propagator decomposition for the vector bosons radiated off the quark
        lines (t-channel VBF bosons):  -g + kk/M^2 = T + L + S in the chosen frame.
        vpol: one letter for every line, or one letter per line ordered by the
        initial-state leg (e.g. "LL", "LT", "TT"); letters L, T, +, -, S,
        A (=full propagator), P (=T+L, physical part only).  None switches off.
        frame: "final_bosons" (rest frame of all non-fermion final particles,
        e.g. the HH system), "lab", or a list of leg indices.
        vbs_only: subtract the amplitude with the t-channel propagators removed
        ("S" selection, which for massless quarks contains exactly the diagrams
        that are NOT decomposed: s-channel V*, V emission off the quark lines),
        i.e. |M_XY - M_S|^2.  Use it for processes where the vector bosons also
        couple directly to the quark lines (e.g. z z j j); for VBF-HH the two
        agree since every diagram carries both t-channel bosons."""
        self._vpol = vpol
        self._vbs_only = vbs_only
        if vpol is None:
            self.proc.set_internal_pol(None)
            return
        self._vpol_frame = frame
        vpol = vpol.upper()
        ini = [i for i in range(self.n_in) if self.model.p_by_pdg()[self.pdgs[i]].spin == 2]
        if len(vpol) == 1:
            vpol = vpol * len(ini)
        if len(vpol) != len(ini):
            raise ValueError(f"vpol needs {len(ini)} letters (one per initial fermion line), got {vpol!r}")
        table = {leg: self._COMPS[c] for leg, c in zip(ini, vpol)}
        if frame == "final_bosons":
            fl = [i for i in range(self.n_in, len(self.pdgs)) if self.model.p_by_pdg()[self.pdgs[i]].spin != 2]
        elif frame == "lab":
            fl = None
        else:
            fl = list(frame)
        self.proc.set_internal_pol({"t_channel": table}, frame_legs=fl)

    def m2_vpol(self, momenta, codes=("LL", "LT", "TL", "TT", "A"), frame="final_bosons",
                average_initial: bool = True) -> Dict[str, float]:
        """|M|^2 for several internal-polarisation selections at once."""
        out = {}
        for c in codes:
            self.set_vpol(c, frame); out[c] = self.m2(momenta, average_initial=average_initial)
        self.set_vpol(None)
        return out

    def m2(self, momenta, pol: Optional[Dict] = None, average_initial: bool = True) -> float:
        """|M|^2 for one event, colour-summed, summed over the selected helicities."""
        mom = self._to_frame([np.asarray(p, dtype=float) for p in momenta])
        skip = getattr(self, "_skip", set())
        configs = [h for h in product(*self.helicity_sets(pol)) if h not in skip]
        if not configs:
            tot = 0.0
        elif getattr(self, "_vbs_only", False) and getattr(self, "_vpol", None):
            code, frame = self._vpol, self._vpol_frame
            tot = float(np.sum(np.real(self.proc.m2_batch_diff(
                mom, configs, lambda: self.set_vpol(code, frame, True), lambda: self.set_vpol("S", frame, True)))))
            self.set_vpol(code, frame, True)
        else:
            tot = float(np.sum(np.real(self.proc.m2_batch(mom, configs))))
        if average_initial:
            for i in range(self.n_in):
                tot /= len(self._hel_sets[i])
        return float(tot)

    def m2_by_helicity(self, momenta, pol: Optional[Dict] = None, average_initial: bool = True,
                       prune: bool = True) -> Dict[Tuple[int, ...], float]:
        """{helicity configuration: colour-summed |M|^2} for one event, all
        configurations in one recursion pass, in the pol_frame.  Sum any subset
        of the keys to get polarised pieces (no interference between different
        external helicities), e.g. LL/TT/LT/TL of two Z's and the total from
        one call."""
        configs, vals = self._by_helicity([momenta], pol, average_initial, prune)
        return {h: float(v) for h, v in zip(configs, vals[0])}

    def m2_by_helicity_events(self, events, pol: Optional[Dict] = None, average_initial: bool = True,
                              prune: bool = True, chunk: int = 32):
        """Batched version: returns (configs, array (N, n_configs))."""
        return self._by_helicity(events, pol, average_initial, prune, chunk)

    def _by_helicity(self, events, pol, average_initial, prune, chunk=32):
        events = self._to_frame(np.asarray([[np.asarray(p, float) for p in ev] for ev in events]))
        if prune and not getattr(self, "_skip", None):
            self.prune_helicities(events[0], pol)     # (events already in pol_frame; frame is idempotent)
        configs = list(product(*self.helicity_sets(pol)))
        skip = getattr(self, "_skip", set())
        live = [i for i, h in enumerate(configs) if h not in skip]
        out = np.zeros((len(events), len(configs)))
        if live:
            live_cfg = [configs[i] for i in live]
            for a in range(0, len(events), chunk):
                out[a:a + chunk, live] = np.real(self.proc.m2_batch(events[a:a + chunk], live_cfg))
        if average_initial:
            for i in range(self.n_in):
                out /= len(self._hel_sets[i])
        return configs, out

    def m2_events(self, events, pol: Optional[Dict] = None, average_initial: bool = True,
                  n_jobs: int = 1, prune: bool = True, chunk: int = 32) -> np.ndarray:
        """events: array (N, n_particles, 4) or list of momentum lists -> (N,).
        Events are evaluated in batches of `chunk` (all helicities and events of
        a chunk in one recursion pass).  prune: drop helicity configurations
        that vanish identically (chirality zeros)."""
        events = self._to_frame(np.asarray([[np.asarray(p, float) for p in ev] for ev in events]))   # (N,n,4)
        if prune and len(events) and not getattr(self, "_skip", None):
            self.prune_helicities(events[0], pol)
        if n_jobs == 1:
            skip = getattr(self, "_skip", set())
            configs = [h for h in product(*self.helicity_sets(pol)) if h not in skip]
            if not configs:
                return np.zeros(len(events))
            out = np.empty(len(events))
            for a in range(0, len(events), chunk):
                out[a:a + chunk] = np.real(self.proc.m2_batch(events[a:a + chunk], configs)).sum(axis=1)
            if average_initial:
                for i in range(self.n_in):
                    out /= len(self._hel_sets[i])
            return out
        from multiprocessing import Pool
        chunks = [events[i::n_jobs] for i in range(n_jobs)]
        with Pool(n_jobs) as pool:
            parts = pool.map(_Worker(self, pol, average_initial), chunks)
        out = np.empty(len(events))
        for i, part in enumerate(parts):
            out[i::n_jobs] = part
        return out

    def m2_polarized(self, momenta, particles, average_initial: bool = True) -> Dict[str, float]:
        """Convenience: split |M|^2 by L/T of the given external massive vector(s).
        Returns e.g. {'LL':..., 'LT':..., 'TL':..., 'TT':...} for two particles."""
        keys = ["L", "T"]
        legs = [leg for part in particles for leg in self._legs_for(part)]   # a name -> all its legs
        out = {}
        for combo in product(keys, repeat=len(legs)):
            pol = {leg: k for leg, k in zip(legs, combo)}
            out["".join(combo)] = self.m2(momenta, pol, average_initial)
        return out


class _Worker:
    def __init__(self, me, pol, avg):
        self.me, self.pol, self.avg = me, pol, avg
    def __call__(self, chunk):
        return list(self.me.m2_events(chunk, self.pol, self.avg, n_jobs=1, prune=False))


# ---------------------------------------------------------- LHE precision
def rebalance_momenta(momenta, masses, n_in=2):
    """Restore exact kinematics to LHE momenta (which carry ~11 digits):
    final-state particles are put exactly on shell (energy recomputed from the
    3-momentum and the model mass), the last final particle absorbs the
    transverse-momentum imbalance, and the initial partons are rebuilt from the
    total E and p_z of the final state (massless beams along z).  Relative changes
    are ~1e-8; without this, helicity sums are frame dependent at ~1e-9 and
    comparisons with MadGraph are limited to ~1e-7."""
    M = np.array([np.asarray(p, float) for p in momenta])
    fin = M[n_in:].copy(); m = np.asarray(masses[n_in:], float)
    fin[-1, 1:3] -= fin[:, 1:3].sum(axis=0)                 # zero total pT
    fin[:, 0] = np.sqrt(np.sum(fin[:, 1:]**2, axis=1) + m**2)
    E, pz = fin[:, 0].sum(), fin[:, 3].sum()
    ini = np.zeros((n_in, 4))
    ini[0] = [(E + pz) / 2, 0, 0, (E + pz) / 2]
    ini[1] = [(E - pz) / 2, 0, 0, -(E - pz) / 2]
    return [ini[i] for i in range(n_in)] + [fin[i] for i in range(len(fin))]


# ---------------------------------------------------------------- SLHA cards
def read_param_card(path: str, model: Union[str, Model]) -> Dict[str, float]:
    """Read an SLHA param_card / MadGraph restrict card and return
    {UFO parameter name: value} using the model's lhablock/lhacode.  A restrict
    card is a param_card whose zeros switch vertices off: applying its values
    as overrides reproduces MadGraph's restricted model (couplings that
    evaluate to zero are skipped automatically).  Unknown entries are ignored
    (returned in the ``.unknown`` attribute of the result)."""
    model = load(model) if isinstance(model, str) else model
    table = {}
    for pr in model.parameters:
        if pr.nature == "external" and pr.lhablock:
            table[(pr.lhablock.upper(), tuple(int(c) for c in (pr.lhacode or [])))] = pr.name
    out: Dict[str, float] = {}; unknown = []
    block = None
    with open(path) as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            t = line.split()
            if t[0].upper() == "BLOCK":
                block = t[1].upper(); continue
            if t[0].upper() == "DECAY":
                key = ("DECAY", (int(t[1]),))
                if key in table: out[table[key]] = float(t[2])
                else: unknown.append(key)
                block = None; continue
            if block is None:
                continue
            *codes, val = t
            try:
                key = (block, tuple(int(c) for c in codes)); v = float(val)
            except ValueError:
                continue
            if key in table: out[table[key]] = v
            else: unknown.append(key)
    out = _CardDict(out); out.unknown = unknown
    return out


class _CardDict(dict):
    unknown: list = []


# ---------------------------------------------------------------------- LHE
def read_lhe(path: str, max_events: Optional[int] = None):
    """Minimal Les Houches reader. Yields (pdgs, momenta, status, helicities)
    per event, with momenta as arrays [E, px, py, pz] in the LHE order.
    Handles plain and gzipped files."""
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    n = 0
    with op(path, "rt") as f:
        in_ev = False; lines = []
        for line in f:
            if line.startswith("<event"):
                in_ev = True; lines = []; continue
            if line.startswith("</event"):
                in_ev = False
                body = [l for l in lines if not l.startswith("#") and not l.startswith("<")]
                npart = int(body[0].split()[0])
                pdgs, mom, status, hel = [], [], [], []
                for l in body[1:1 + npart]:
                    t = l.split()
                    pdgs.append(int(t[0])); status.append(int(t[1]))
                    px, py, pz, E = map(float, t[6:10])
                    mom.append(np.array([E, px, py, pz]))
                    hel.append(float(t[12]))
                yield pdgs, mom, status, hel
                n += 1
                if max_events and n >= max_events:
                    return
                continue
            if in_ev:
                lines.append(line.strip())


def lhe_to_process_momenta(pdgs, momenta, status, initial_pdgs, final_pdgs):
    """Reorder an LHE event into the (initial + final) order expected by a
    MatrixElement, matching by PDG code (status -1 = incoming, 1 = outgoing).
    Identical final-state particles are assigned in file order."""
    inc = [(p, m) for p, m, s in zip(pdgs, momenta, status) if s == -1]
    out = [(p, m) for p, m, s in zip(pdgs, momenta, status) if s == 1]
    def pick(pool, wanted):
        res = []
        for w in wanted:
            for i, (p, m) in enumerate(pool):
                if p == w:
                    res.append(m); pool.pop(i); break
            else:
                raise ValueError(f"pdg {w} not found among {[p for p,_ in pool]}")
        return res
    return pick(inc, initial_pdgs) + pick(out, final_pdgs)


# ---------------------------------------------------------------- samples
class SampleEvaluator:
    """Evaluate |M|^2 on a mixed-flavour sample (e.g. MadGraph  p p > ...):
    one MatrixElement is built and cached per partonic channel.

        ev = SampleEvaluator("HHVBF_UFO", couplings={"CV":1,"C2V":2,"C3":1})
        for pdgs, mom, status, hel in read_lhe("events.lhe.gz"):
            w = ev.m2(pdgs, mom, status)             # full
            w_LL = ev.m2(pdgs, mom, status, vpol="LL")  # fusing bosons longitudinal
            w_ZL = ev.m2(pdgs, mom, status, pol={"z": "L"})  # both external Z longitudinal
    Channel key = (initial pdgs in LHE order, final pdgs in LHE order).
    """
    def __init__(self, model, couplings=None, rebalance: bool = True, **kw):
        """rebalance: restore exact kinematics to LHE momenta (see rebalance_momenta)."""
        self.model = load(model) if isinstance(model, str) else model
        self.couplings = dict(couplings or {}); self.kw = kw
        self.rebalance = rebalance
        self.cache: Dict[tuple, MatrixElement] = {}

    def _mom(self, me, pdgs, momenta, status):
        mom = lhe_to_process_momenta(pdgs, momenta, status, me.initial, me.final)
        if self.rebalance:
            masses = [abs(me.proc.amp.mass(p)) for p in me.pdgs]
            mom = rebalance_momenta(mom, masses, me.n_in)
        return mom

    def channel(self, pdgs, status) -> MatrixElement:
        ini = tuple(p for p, s in zip(pdgs, status) if s == -1)
        # canonical final-state order (fermions first, then by pdg): the same
        # physical channel maps to one cached MatrixElement whatever the LHE order
        fin = tuple(sorted((p for p, s in zip(pdgs, status) if s == 1),
                           key=lambda q: (self.model.p_by_pdg()[q].spin != 2, abs(q), q)))
        key = (ini, fin)
        if key not in self.cache:
            self.cache[key] = MatrixElement(self.model, (list(ini), list(fin)), self.couplings, **self.kw)
        return self.cache[key]

    def set_param(self, couplings, replace=False):
        self.couplings = dict(couplings) if replace else {**self.couplings, **couplings}
        for me in self.cache.values():
            me.set_param(self.couplings, replace=True); me.reset_pruning()

    set_couplings = set_param

    def m2_sample(self, events, pol=None, vpol=None, frame="final_bosons", average_initial=True,
                  prune=True, chunk=32, n_jobs=1):
        """Batched evaluation of a whole sample.  events: iterable of
        (pdgs, momenta, status[, ...]) as produced by read_lhe.  Events are
        grouped by partonic channel and evaluated in chunks; the result keeps
        the input order.  Returns an array (N,)."""
        events = list(events)
        groups: Dict[tuple, list] = {}
        for idx, ev in enumerate(events):
            pdgs, mom, status = ev[0], ev[1], ev[2]
            me = self.channel(pdgs, status)
            key = (tuple(me.initial), tuple(me.final))
            groups.setdefault(key, []).append((idx, self._mom(me, pdgs, mom, status)))
        out = np.empty(len(events))
        for key, items in groups.items():
            me = self.cache[key]
            me.set_vpol(vpol, frame)
            try:
                vals = me.m2_events([m for _, m in items], pol=pol, average_initial=average_initial,
                                    n_jobs=n_jobs, prune=prune, chunk=chunk)
            finally:
                me.set_vpol(None)
            for (idx, _), v in zip(items, vals):
                out[idx] = v
        return out

    def m2(self, pdgs, momenta, status, pol=None, vpol=None, frame="final_bosons",
           average_initial=True, prune=True):
        me = self.channel(pdgs, status)
        mom = lhe_to_process_momenta(pdgs, momenta, status, me.initial, me.final)
        if not getattr(me, "_checked", False):
            me._checked = True
            for i, (pdg, q) in enumerate(zip(me.pdgs, mom)):
                m = me.proc.amp.mass(pdg); m2 = q[0]**2 - q[1:] @ q[1:]
                if abs(m2 - m**2) > 1e-3 * max(m**2, 1.0):
                    import warnings
                    warnings.warn(f"leg {i} (pdg {pdg}): event mass^2 {m2:.4g} vs model {m**2:.4g} -- "
                                  f"set the model mass to the value used in the event generator")
        if prune and not getattr(me, "_skip", None):
            me.prune_helicities(mom, pol)
        me.set_vpol(vpol, frame)
        try:
            return me.m2(mom, pol, average_initial)
        finally:
            me.set_vpol(None)
