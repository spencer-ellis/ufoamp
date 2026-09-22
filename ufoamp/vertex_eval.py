"""
vertex_eval.py
==============

General numerical evaluator for UFO Lorentz structures via ``numpy.einsum``.

Given a vertex's Lorentz-structure string, the spin of each leg, the (all
incoming) momenta, and wavefunctions for every leg but one, it returns the
off-shell *current* for the remaining leg -- or, if every leg is supplied, the
fully contracted amplitude contribution.  This single routine is what makes the
generator model-agnostic: SM, HHVBF, SMEFT and arbitrary UFOs all reduce to
strings over the same handful of atoms.

Supported atoms (UFO spec):  Metric(i,j)  P(mu,k)  Gamma(mu,a,b)  ProjM(a,b)
ProjP(a,b)  Identity(a,b)  Gamma5(a,b)  Epsilon(a,b,c,d)  and the constant 1.

Index rules
-----------
* Integer arguments are index *labels*: positive = the leg with that number,
  negative = an internal index summed over.
* A spinor label is one index; whether it sits in the row (psibar) or column
  (psi) position of an atom determines the type of the output current.
* Every Lorentz index is contravariant (gamma^mu, g^{mu nu}, p^mu, eps^mu), so
  each Lorentz contraction inserts exactly one metric g_{mu nu}.

Wavefunction encoding (``wfs`` dict, leg label -> value)
  scalar : complex
  vector : ndarray (4,)  contravariant eps^mu
  fermion: ('bar', ndarray(4,)) for a barred spinor (ubar / vbar), or
           ('col', ndarray(4,)) for an un-barred spinor (u / v)

Output
  vector leg  -> ndarray (4,) contravariant current J^mu
  fermion leg -> ('bar'|'col', ndarray(4,))  (type set by the free slot)
  scalar / all-legs-known -> complex
"""

from __future__ import annotations

import ast
from itertools import count
from typing import Dict, List, Tuple, Any

import numpy as np
import numpy as xp   # switchable backend (see backend.py)

from .dirac import GAMMA, GAMMA5, PROJM, PROJP, ID4, METRIC

# stacked gamma tensor G[mu, a, b] = (gamma^mu)_{ab}
GAMMA_T = np.stack(GAMMA, axis=0)
# Sigma[mu, nu, a, b] = (i/2) [gamma^mu, gamma^nu]_{ab}
SIGMA_T = np.zeros((4, 4, 4, 4), dtype=np.complex128)
for _m in range(4):
    for _n in range(4):
        SIGMA_T[_m, _n] = 0.5j * (GAMMA[_m] @ GAMMA[_n] - GAMMA[_n] @ GAMMA[_m])

# Levi-Civita eps^{mu nu rho sigma}, eps^{0123} = +1
_EPS = np.zeros((4, 4, 4, 4), dtype=np.complex128)
from itertools import permutations as _perm
for perm in _perm(range(4)):
    # parity of permutation
    p = list(perm); sgn = 1
    for i in range(4):
        for j in range(i + 1, 4):
            if p[i] > p[j]:
                sgn = -sgn
    _EPS[perm] = sgn
EPSILON_T = _EPS

_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXY"   # Z = batch axis


# --------------------------------------------------------------------------- #
#  parsing                                                                    #
# --------------------------------------------------------------------------- #
class _Atom:
    __slots__ = ("name", "args")

    def __init__(self, name, args):
        self.name, self.args = name, args


def _int(node):
    if isinstance(node, ast.Constant):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_int(node.operand)
    raise ValueError("non-integer index")


def _expand(node):
    """Fully expand any UFO structure subtree into a list of (coeff, [atoms])."""
    if isinstance(node, ast.Call):
        return [(1 + 0j, [_Atom(node.func.id, [_int(a) for a in node.args])])]
    if isinstance(node, ast.Constant):
        return [(complex(node.value), [])]
    if isinstance(node, ast.UnaryOp):
        t = _expand(node.operand)
        return [(-c, a) for c, a in t] if isinstance(node.op, ast.USub) else t
    if isinstance(node, ast.BinOp):
        L, R = _expand(node.left), _expand(node.right) if not isinstance(node.op, ast.Pow) else None
        if isinstance(node.op, ast.Add):
            return L + R
        if isinstance(node.op, ast.Sub):
            return L + [(-c, a) for c, a in R]
        if isinstance(node.op, ast.Mult):
            return [(cl * cr, al + ar) for cl, al in L for cr, ar in R]
        if isinstance(node.op, ast.Div):
            if any(ar for _, ar in R) or len(R) != 1:
                raise ValueError("division by a non-constant")
            d = R[0][0]
            return [(c / d, a) for c, a in L]
        if isinstance(node.op, ast.Pow):
            n = _int(node.right)
            out = [(1 + 0j, [])]
            for _ in range(n):
                out = [(cl * cr, al + ar) for cl, al in out for cr, ar in L]
            return out
    raise ValueError(f"unsupported {type(node).__name__}")


_cache: Dict[str, list] = {}


def parse(structure: str):
    if structure in _cache:
        return _cache[structure]
    tree = ast.parse(structure.strip(), mode="eval")
    out = [(c, a) for c, a in _expand(tree.body) if c != 0]
    _cache[structure] = out
    return out


_chain_cache: Dict[str, list] = {}


def fermion_chains(structure: str):
    """Return [(row_leg, col_leg), ...]: which external fermion legs are joined
    by one spinor chain (psibar_row ... psi_col) in this Lorentz structure.
    Needed for 4-fermion vertices, where a vertex holds several lines."""
    if structure in _chain_cache:
        return _chain_cache[structure]
    terms = parse(structure)
    chains = []
    if terms:
        atoms = terms[0][1]
        rows, cols = {}, {}          # spinor label -> atom index
        for i, a in enumerate(atoms):
            if a.name in ("Gamma",):
                rows[a.args[1]] = i; cols[a.args[2]] = i
            elif a.name in ("ProjM", "ProjP", "Identity", "Gamma5"):
                rows[a.args[0]] = i; cols[a.args[1]] = i
            elif a.name == "Sigma":
                rows[a.args[2]] = i; cols[a.args[3]] = i
        for lbl, i in rows.items():
            if lbl < 0:
                continue                    # start only at external row labels
            row_ext = lbl
            cur = i
            while True:
                a = atoms[cur]
                if a.name == "Gamma":
                    col = a.args[2]
                elif a.name == "Sigma":
                    col = a.args[3]
                else:
                    col = a.args[1]
                if col > 0:
                    chains.append((row_ext, col)); break
                cur = rows[col]             # internal label: continue at atom where it is a row
    _chain_cache[structure] = chains
    return chains


_plan_cache: Dict[tuple, list] = {}


def _compile(expr, operands):
    """Turn an einsum expression into a fixed list of contraction steps
    [(positions, sub_expr), ...] from numpy's optimal path, so that at run time
    only small direct einsums are executed (no path search)."""
    ins, out = expr.split("->")
    subs = ins.split(",")
    path = np.einsum_path(expr, *operands, optimize="optimal" if len(operands) <= 7 else "greedy")[0][1:]
    steps = []
    for contr in path:
        pos = sorted(contr, reverse=True)
        picked = [subs.pop(i) for i in pos]            # pop from the back
        rest = "".join(subs) + out
        keep = "".join(dict.fromkeys(c for c in "".join(picked) if c in rest))
        steps.append((pos, f"{','.join(picked)}->{keep}"))
        subs.append(keep)
    steps.append((None, f"{subs[0]}->{out}"))
    return steps


def _einsum_cached(expr, operands):
    key = (expr, tuple(tuple(op.shape) for op in operands))
    plan = _plan_cache.get(key)
    if plan is None:
        plan = _compile(expr, [np.zeros(op.shape) for op in operands]); _plan_cache[key] = plan
    ops = list(operands)
    for pos, e in plan[:-1]:
        picked = [ops.pop(i) for i in pos]
        ops.append(xp.einsum(e, *picked))
    e = plan[-1][1]
    a, b = e.split("->")
    return ops[0] if a == b else xp.einsum(e, ops[0])


# --------------------------------------------------------------------------- #
#  evaluation                                                                 #
# --------------------------------------------------------------------------- #
def eval_vertex(structure: str,
                spins: Dict[int, int],
                momenta: Dict[int, np.ndarray],
                wfs: Dict[int, Any],
                out_leg: int | None) -> Any:
    """Evaluate one Lorentz structure; see module docstring."""
    labels = sorted(spins)
    if out_leg is not None:
        # momentum of the off-shell leg by conservation (all incoming)
        momenta = dict(momenta)
        momenta[out_leg] = -sum(momenta[l] for l in labels if l != out_leg)

    total = None
    for coeff, atoms in parse(structure):
        val = _eval_term(coeff, atoms, spins, momenta, wfs, out_leg)
        if total is None:
            total = val
        elif isinstance(val, tuple):          # fermion output: (kind, array)
            assert val[0] == total[0], "inconsistent spinor slot between terms"
            total = (total[0], total[1] + val[1])
        else:
            total = total + val
    return total


def _eval_term(coeff, atoms, spins, momenta, wfs, out_leg):
    operands: List[np.ndarray] = []
    subs: List[str] = []
    letters = iter(_LETTERS)
    spin_letter: Dict[int, str] = {}          # spinor label -> letter
    lor_occ: Dict[int, List[Tuple[int, int]]] = {}  # lorentz label -> [(operand#, pos)]
    # which spinor slot (row/col) each spinor label occupies on atoms
    slot_of: Dict[int, str] = {}

    def spin_l(lbl):
        if lbl not in spin_letter:
            spin_letter[lbl] = next(letters)
        return spin_letter[lbl]

    def add(op, idx, batch=False):  # idx: list of ('L',label) or ('S',label)
        operands.append(op)
        s = "Z" if batch else ""
        for kind, lbl in idx:
            if kind == "S":
                s += spin_l(lbl)
            else:  # Lorentz: placeholder, resolved after all occurrences known
                s += "?"
                lor_occ.setdefault(lbl, []).append((len(operands) - 1, len(s) - 1))
        subs.append(s)

    scalar = coeff
    batched = False
    # --- atoms (momenta may carry a batch axis) ---------------------------- #
    for a in atoms:
        n, ar = a.name, a.args
        if n == "Metric":
            add(METRIC, [("L", ar[0]), ("L", ar[1])])
        elif n == "P":
            pk = np.asarray(momenta[ar[1]])
            add(pk, [("L", ar[0])], batch=(pk.ndim == 2)); batched |= (pk.ndim == 2)
        elif n == "Gamma":
            add(GAMMA_T, [("L", ar[0]), ("S", ar[1]), ("S", ar[2])])
            slot_of.setdefault(ar[1], "row"); slot_of.setdefault(ar[2], "col")
        elif n in ("ProjM", "ProjP", "Identity", "Gamma5"):
            M = {"ProjM": PROJM, "ProjP": PROJP, "Identity": ID4, "Gamma5": GAMMA5}[n]
            add(M, [("S", ar[0]), ("S", ar[1])])
            slot_of.setdefault(ar[0], "row"); slot_of.setdefault(ar[1], "col")
        elif n == "Sigma":
            add(SIGMA_T, [("L", ar[0]), ("L", ar[1]), ("S", ar[2]), ("S", ar[3])])
            slot_of.setdefault(ar[2], "row"); slot_of.setdefault(ar[3], "col")
        elif n == "Epsilon":
            add(EPSILON_T, [("L", x) for x in ar])
        else:
            raise ValueError(f"unsupported atom {n}")

    # --- wavefunctions of known legs -------------------------------------- #
    # wavefunctions may carry a leading batch axis (helicity configurations);
    # einsum letter 'Z' is reserved for it.
    for lbl, sp in spins.items():
        if lbl == out_leg:
            continue
        w = wfs[lbl]
        if sp == 1:
            w = np.asarray(w)
            if w.ndim == 1:
                operands.append(w); subs.append("Z"); batched = True
            else:
                scalar *= complex(w)
        elif sp == 3:
            w = np.asarray(w)
            add(w, [("L", lbl)], batch=(w.ndim == 2)); batched |= (w.ndim == 2)
        elif sp == 2:
            kind, arr = w
            arr = np.asarray(arr)
            add(arr, [("S", lbl)], batch=(arr.ndim == 2)); batched |= (arr.ndim == 2)
        else:
            raise ValueError(f"spin {sp} not supported")

    # --- resolve Lorentz indices: insert a metric per contraction ---------- #
    out_sub = ""
    for lbl, occ in lor_occ.items():
        if len(occ) == 2:
            x, y = next(letters), next(letters)
            (o1, p1), (o2, p2) = occ
            subs[o1] = subs[o1][:p1] + x + subs[o1][p1 + 1:]
            subs[o2] = subs[o2][:p2] + y + subs[o2][p2 + 1:]
            operands.append(METRIC); subs.append(x + y)
        elif len(occ) == 1:
            x = next(letters)
            (o1, p1) = occ[0]
            subs[o1] = subs[o1][:p1] + x + subs[o1][p1 + 1:]
            out_sub += x                    # free Lorentz index -> output
        else:
            raise ValueError(f"Lorentz label {lbl} appears {len(occ)} times")

    # free spinor index (the output fermion leg)
    out_kind = None
    if out_leg is not None and spins[out_leg] == 2:
        out_sub += spin_letter[out_leg]
        # free row slot -> unbarred/column output; free col slot -> barred output
        out_kind = "col" if slot_of.get(out_leg) == "row" else "bar"

    if batched:
        out_sub = "Z" + out_sub
    expr = ",".join(subs) + "->" + out_sub
    res = _einsum_cached(expr, operands) if operands else np.array(1.0 + 0j)
    res = scalar * res
    if out_kind is not None:
        return (out_kind, res)
    if out_sub == "" :
        return complex(res)
    return res


# --------------------------------------------------------------------------- #
#  self-test against the hand-built FFV of lorentz_eval                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .wavefunctions import u_spinor, v_spinor, polarization_vector
    from .dirac import bar
    rng = np.random.default_rng(1)
    p1 = np.array([5.0, 1.0, 2.0, np.sqrt(25 - 5)], dtype=complex)
    p2 = np.array([5.0, -1.0, 0.5, -np.sqrt(25 - 1.25)], dtype=complex)
    u1 = u_spinor(p1, +1, 0.0)
    v2bar = bar(v_spinor(p2, -1, 0.0))
    # current for the vector leg 3 from Gamma(3,2,1): J^mu = vbar_2 gamma^mu u_1
    J = eval_vertex("Gamma(3,2,1)", {1: 2, 2: 2, 3: 3}, {1: p1, 2: p2},
                    {1: ("col", u1), 2: ("bar", v2bar)}, out_leg=3)
    ref = np.array([v2bar @ GAMMA[mu] @ u1 for mu in range(4)])
    assert np.allclose(J, ref), "vector current mismatch"
    # scalar amplitude: contract J with a polarization -> eps_mu J^mu (via metric)
    eps = polarization_vector(-(p1 + p2), +1, 0.0) if False else np.array([0, 1, 1j, 0]) / np.sqrt(2)
    A = eval_vertex("Gamma(3,2,1)", {1: 2, 2: 2, 3: 3}, {1: p1, 2: p2, 3: -(p1 + p2)},
                    {1: ("col", u1), 2: ("bar", v2bar), 3: eps}, out_leg=None)
    ref_A = sum(METRIC[m, m] * eps[m] * ref[m] for m in range(4))
    assert np.isclose(A, ref_A), "full contraction mismatch"
    # Metric(1,2) VVS: current for scalar leg 3 = eps1 . eps2
    e1 = np.array([0, 1, 0, 0], complex); e2 = np.array([0, 0, 1, 0], complex)
    S = eval_vertex("Metric(1,2)", {1: 3, 2: 3, 3: 1}, {1: p1, 2: p2}, {1: e1, 2: e2}, out_leg=3)
    assert np.isclose(S, -0.0 if False else sum(METRIC[m, m] * e1[m] * e2[m] for m in range(4)))
    print("vertex_eval OK: FFV current, full contraction, Metric VVS")


# =========================================================================== #
#  Colour-aware currents                                                      #
# =========================================================================== #
from . import color as _col


class Cur:
    """An off-shell current or external wavefunction.
    arr axes: (Z batch, [4: Lorentz or spinor], [own colour], *ext colour)
    kind : 's' scalar, 'v' vector, 'col' psi, 'bar' psibar
    own  : dimension of the own colour axis (0 = none / colour singlet)
    ext  : tuple of (external leg index, dim) for open external colour axes,
           sorted by leg index
    ext_self : for an external coloured wavefunction, its own leg index (its
           colour index is left open at the first vertex it enters)"""
    __slots__ = ("arr", "kind", "own", "ext", "ext_self")

    def __init__(self, arr, kind, own=0, ext=(), ext_self=None):
        self.arr, self.kind, self.own, self.ext, self.ext_self = arr, kind, own, tuple(ext), ext_self

    def scale(self, g):
        return Cur(g * self.arr, self.kind, self.own, self.ext, self.ext_self)

    def add(self, other):
        assert self.kind == other.kind and self.own == other.own and self.ext == other.ext
        return Cur(self.arr + other.arr, self.kind, self.own, self.ext, self.ext_self)

    @property
    def has_lorentz(self):
        return self.kind != "s"


def eval_vertex_cur(lstruct: str, cstruct: str, spins: Dict[int, int], cdims: Dict[int, int],
                    momenta: Dict[int, Any], inputs: Dict[int, Cur], out_leg: int) -> Cur:
    """Evaluate one (Lorentz structure, colour structure) pair of a vertex,
    leaving vertex leg `out_leg` free.  spins/cdims: per vertex leg label.
    Returns the current for the free leg (own colour axis = its colour dim)."""
    labels = sorted(spins)
    momenta = dict(momenta)
    momenta[out_leg] = -sum(momenta[l] for l in labels if l != out_leg)
    total = None
    for ccoeff, catoms in _col.parse(cstruct):
        for lcoeff, latoms in parse(lstruct):
            val = _eval_term_cur(lcoeff * ccoeff, latoms, catoms, spins, cdims, momenta, inputs, out_leg)
            total = val if total is None else total.add(val)
    return total


def _eval_term_cur(coeff, latoms, catoms, spins, cdims, momenta, inputs, out_leg):
    operands: List[np.ndarray] = []
    subs: List[str] = []
    letters = iter(_LETTERS)
    spin_letter: Dict[int, str] = {}
    col_letter: Dict[int, str] = {}          # colour label (leg or internal) -> letter
    ext_letter: Dict[int, str] = {}          # external leg index -> letter
    lor_occ: Dict[int, List[Tuple[int, int]]] = {}
    slot_of: Dict[int, str] = {}
    ext_dims: Dict[int, int] = {}

    def spin_l(lbl):
        if lbl not in spin_letter: spin_letter[lbl] = next(letters)
        return spin_letter[lbl]

    def col_l(lbl):
        if lbl not in col_letter: col_letter[lbl] = next(letters)
        return col_letter[lbl]

    def ext_l(leg, dim):
        if leg not in ext_letter:
            ext_letter[leg] = next(letters); ext_dims[leg] = dim
        return ext_letter[leg]

    def add(op, idx, prefix=""):
        operands.append(op)
        s = prefix
        for kind, lbl in idx:
            if kind == "S":
                s += spin_l(lbl)
            elif kind == "C":
                s += col_l(lbl)
            else:
                s += "?"
                lor_occ.setdefault(lbl, []).append((len(operands) - 1, len(s) - 1))
        subs.append(s)

    scalar = coeff
    # Lorentz atoms
    for a in latoms:
        n, ar = a.name, a.args
        if n == "Metric":
            add(METRIC, [("L", ar[0]), ("L", ar[1])])
        elif n == "P":
            pk = momenta[ar[1]]
            add(pk, [("L", ar[0])], prefix="Z" if pk.ndim == 2 else "")
        elif n == "Gamma":
            add(GAMMA_T, [("L", ar[0]), ("S", ar[1]), ("S", ar[2])])
            slot_of.setdefault(ar[1], "row"); slot_of.setdefault(ar[2], "col")
        elif n in ("ProjM", "ProjP", "Identity", "Gamma5"):
            M = {"ProjM": PROJM, "ProjP": PROJP, "Identity": ID4, "Gamma5": GAMMA5}[n]
            add(M, [("S", ar[0]), ("S", ar[1])])
            slot_of.setdefault(ar[0], "row"); slot_of.setdefault(ar[1], "col")
        elif n == "Sigma":
            add(SIGMA_T, [("L", ar[0]), ("L", ar[1]), ("S", ar[2]), ("S", ar[3])])
            slot_of.setdefault(ar[2], "row"); slot_of.setdefault(ar[3], "col")
        elif n == "Epsilon":
            add(EPSILON_T, [("L", x) for x in ar])
        else:
            raise ValueError(f"unsupported Lorentz atom {n}")
    # colour atoms
    for a in catoms:
        tens, ar = _col.atom_tensor(a, cdims)
        add(tens, [("C", x) for x in ar])
    # inputs
    for lbl, sp in spins.items():
        if lbl == out_leg:
            continue
        c = inputs[lbl]
        s = "Z"
        if sp == 3:
            s += "?"; pos = len(s) - 1
        elif sp == 2:
            s += spin_l(lbl)
        if c.own:
            s += col_l(lbl)                       # internal coloured current: contract
        elif cdims.get(lbl, 0) and c.ext_self is not None:
            # external coloured leg: its vertex colour index stays open
            ext_letter[c.ext_self] = col_l(lbl); ext_dims[c.ext_self] = cdims[lbl]
        for leg, dim in c.ext:
            s += ext_l(leg, dim)
        operands.append(c.arr); subs.append(s)
        if sp == 3:
            lor_occ.setdefault(lbl, []).append((len(operands) - 1, pos))
    # resolve Lorentz contractions with one metric each
    out_lor = ""
    for lbl, occ in lor_occ.items():
        if len(occ) == 2:
            x, y = next(letters), next(letters)
            (o1, p1), (o2, p2) = occ
            subs[o1] = subs[o1][:p1] + x + subs[o1][p1 + 1:]
            subs[o2] = subs[o2][:p2] + y + subs[o2][p2 + 1:]
            operands.append(METRIC); subs.append(x + y)
        elif len(occ) == 1:
            x = next(letters); (o1, p1) = occ[0]
            subs[o1] = subs[o1][:p1] + x + subs[o1][p1 + 1:]
            out_lor += x
        else:
            raise ValueError(f"Lorentz label {lbl} appears {len(occ)} times")
    out_sp = spins[out_leg]
    kind = "s"
    out_sub = "Z" + out_lor
    if out_sp == 2:
        out_sub += spin_letter[out_leg]
        kind = "col" if slot_of.get(out_leg) == "row" else "bar"
    elif out_sp == 3:
        kind = "v"
    own = cdims.get(out_leg, 0)
    if own:
        out_sub += col_l(out_leg)
    ext_sorted = tuple(sorted(ext_dims.items()))
    out_sub += "".join(ext_letter[leg] for leg, _ in ext_sorted)
    expr = ",".join(subs) + "->" + out_sub
    res = scalar * _einsum_cached(expr, operands)
    return Cur(res, kind, own, ext_sorted, None)
