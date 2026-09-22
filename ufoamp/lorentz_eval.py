"""
lorentz_eval.py
===============

Numerically evaluate UFO Lorentz-structure strings.

UFO writes vertex Lorentz structures as strings over a small algebra of atoms:
``Gamma(mu,i,j)``, ``ProjM(i,j)``, ``ProjP(i,j)``, ``Identity(i,j)``,
``Gamma5(i,j)``, ``Metric(i,j)``, ``P(mu,i)``, ``Epsilon(...)``.  The integer
arguments are *leg labels* (1..N for external legs of the vertex) or negative
internal indices that are summed.

For Phase 1 (SM, fermion-vector and fermion-scalar vertices) we evaluate the
*Dirac* part: an FFV/FFS structure is reduced to a Dirac matrix (or, for FFV, a
set of four Dirac matrices carrying the open vector index mu) such that the
fermion bilinear is

        current^mu = psibar_out  Gamma^mu  psi_in .

The chaining of atoms follows the spinor-index contraction, so e.g.
``Gamma(3,2,-1)*ProjM(-1,1)`` -> gamma^mu P_L  (rows=leg2, cols=leg1).

This module is intentionally small and explicit; it is the piece most exposed to
convention errors, so every structure it supports is covered by a validation
test.  It is written to extend cleanly to Metric/P/Epsilon atoms (needed for the
scalar-vector and SMEFT sectors) later.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Tuple

import numpy as np

from .dirac import GAMMA, GAMMA5, PROJM, PROJP, ID4


# --------------------------------------------------------------------------- #
#  parse a structure string into sum of (coefficient, [atom, ...])            #
# --------------------------------------------------------------------------- #
class Atom:
    def __init__(self, name: str, args: List[int]):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"{self.name}({','.join(map(str, self.args))})"


def _int(node):
    if isinstance(node, ast.Constant):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_int(node.operand)
    raise ValueError(f"non-integer index {ast.dump(node)}")


def _flatten_mul(node) -> Tuple[complex, List[Atom]]:
    """Return (numeric coeff, [atoms]) for a product term."""
    if isinstance(node, ast.Call):
        name = node.func.id
        return 1.0 + 0j, [Atom(name, [_int(a) for a in node.args])]
    if isinstance(node, ast.Constant):
        return complex(node.value), []
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        c, atoms = _flatten_mul(node.operand)
        return -c, atoms
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        cl, al = _flatten_mul(node.left)
        cr, ar = _flatten_mul(node.right)
        return cl * cr, al + ar
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        cl, al = _flatten_mul(node.left)
        cr, ar = _flatten_mul(node.right)
        if ar:
            raise ValueError("division by atom not supported")
        return cl / cr, al
    raise ValueError(f"unsupported node {ast.dump(node)}")


def _split_terms(node) -> List[Tuple[int, ast.AST]]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _split_terms(node.left)
        right = _split_terms(node.right)
        sgn = 1 if isinstance(node.op, ast.Add) else -1
        return left + [(s * sgn, n) for s, n in right]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return [(-s, n) for s, n in _split_terms(node.operand)]
    return [(1, node)]


def parse_structure(s: str) -> List[Tuple[complex, List[Atom]]]:
    tree = ast.parse(s.strip(), mode="eval")
    out = []
    for sgn, node in _split_terms(tree.body):
        c, atoms = _flatten_mul(node)
        out.append((sgn * c, atoms))
    return out


# --------------------------------------------------------------------------- #
#  Dirac-chain evaluation for fermion-line structures                         #
# --------------------------------------------------------------------------- #
# atoms that are 4x4 Dirac matrices in spinor space (row index, col index)
_DIRAC_ATOMS = {"ProjM", "ProjP", "Identity", "Gamma5"}


def _dirac_atom_matrix(atom: Atom):
    if atom.name == "ProjM":
        return PROJM
    if atom.name == "ProjP":
        return PROJP
    if atom.name == "Identity":
        return ID4
    if atom.name == "Gamma5":
        return GAMMA5
    raise ValueError(f"not a plain Dirac atom: {atom}")


def ffv_vertex_matrices(structure: str,
                        vector_leg: int,
                        out_leg: int,
                        in_leg: int) -> List[np.ndarray]:
    """Return [Gamma^0,..,Gamma^3] (4x4 Dirac matrices) for an FFV structure so
    that current^mu = psibar_out Gamma^mu psi_in.

    ``vector_leg`` is the leg label carrying the open Lorentz index; ``out_leg``
    and ``in_leg`` are the (barred, un-barred) fermion leg labels.
    """
    terms = parse_structure(structure)
    result = [np.zeros((4, 4), dtype=np.complex128) for _ in range(4)]
    for coeff, atoms in terms:
        mats_mu = _chain_ffv(atoms, coeff, vector_leg, out_leg, in_leg)
        for mu in range(4):
            result[mu] += mats_mu[mu]
    return result


def _chain_ffv(atoms, coeff, vlab, out_leg, in_leg):
    """Build coeff * (Dirac chain) as 4 matrices carrying the vector index.

    The chain is ordered from the barred (out) leg to the un-barred (in) leg by
    following spinor-index contractions. Exactly one Gamma carries the open
    vector index ``vlab``.
    """
    # index each atom by its (row, col) spinor labels; Gamma(mu,i,j)->row i,col j
    # collect a map from spinor-label -> list of (atom, end) to order the chain
    seq = []  # list of (matrix_or_gamma_flag, row, col)
    gamma_positions = []
    for a in atoms:
        if a.name == "Gamma":
            mu, i, j = a.args
            seq.append(("G", i, j, mu))
        elif a.name in _DIRAC_ATOMS:
            i, j = a.args
            seq.append((a.name, i, j, None))
        else:
            raise ValueError(f"atom {a} not supported in FFV chain yet")

    # order the sequence into a chain from out_leg to in_leg
    ordered = _order_chain(seq, out_leg, in_leg)

    # build the 4 matrices: multiply atoms; the Gamma with vector index vlab
    # becomes the "slot" carrying mu.
    mats = [None] * 4
    # For each mu, build the product replacing the open Gamma by GAMMA[mu]
    for mu in range(4):
        M = ID4.copy()
        for (kind, i, j, gmu) in ordered:
            if kind == "G":
                if gmu == vlab:
                    A = GAMMA[mu]
                else:
                    raise ValueError("second open gamma index not supported")
            else:
                A = _dirac_atom_matrix_by_name(kind)
            M = M @ A
        mats[mu] = coeff * M
    return mats


def _dirac_atom_matrix_by_name(name):
    return {"ProjM": PROJM, "ProjP": PROJP, "Identity": ID4, "Gamma5": GAMMA5}[name]


def _order_chain(seq, out_leg, in_leg):
    """Order atoms so spinor indices contract: start at row=out_leg, follow the
    column index to the next atom whose row matches, end at col=in_leg."""
    # build lookup: row-label -> entry
    by_row: Dict[int, tuple] = {}
    for entry in seq:
        kind, i, j, gmu = entry
        by_row[i] = entry
    ordered = []
    cur = out_leg
    used = set()
    for _ in range(len(seq)):
        entry = by_row.get(cur)
        if entry is None or id(entry) in used:
            # fall back to original order if chain can't be followed
            return seq
        ordered.append(entry)
        used.add(id(entry))
        cur = entry[2]  # column index -> next row
        if cur == in_leg:
            break
    return ordered if len(ordered) == len(seq) else seq


# --------------------------------------------------------------------------- #
#  self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Gamma(3,2,1) -> gamma^mu
    G = ffv_vertex_matrices("Gamma(3,2,1)", vector_leg=3, out_leg=2, in_leg=1)
    assert all(np.allclose(G[mu], GAMMA[mu]) for mu in range(4))
    # Gamma(3,2,-1)*ProjM(-1,1) -> gamma^mu P_L
    GL = ffv_vertex_matrices("Gamma(3,2,-1)*ProjM(-1,1)", 3, 2, 1)
    assert all(np.allclose(GL[mu], GAMMA[mu] @ PROJM) for mu in range(4))
    # linear combo: Gamma.ProjM + 2 Gamma.ProjP
    GC = ffv_vertex_matrices("Gamma(3,2,-1)*ProjM(-1,1) + 2*Gamma(3,2,-1)*ProjP(-1,1)", 3, 2, 1)
    assert all(np.allclose(GC[mu], GAMMA[mu] @ PROJM + 2 * GAMMA[mu] @ PROJP) for mu in range(4))
    print("lorentz_eval OK: Gamma, Gamma.ProjM/P, linear combinations")
