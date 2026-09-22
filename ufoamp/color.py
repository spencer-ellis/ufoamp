"""
color.py — numeric tensors for UFO colour structures.

Colour is treated exactly like Lorentz: each atom is a numeric tensor with one
index per leg (or a contracted internal index), contracted in the same einsum as
the Lorentz structure.  Conventions (FeynRules/UFO):
  T(a,i,j)   = (T^a)_{ij},  T^a = lambda^a/2,  Tr(T^a T^b) = delta^{ab}/2
  f(a,b,c)   :  [T^a, T^b] = i f^{abc} T^c
  d(a,b,c)   :  {T^a, T^b} = delta^{ab}/3 + d^{abc} T^c
  Identity(i,j): delta (dimension taken from the legs)
  Epsilon(i,j,k), EpsilonBar(i,j,k): fundamental Levi-Civita
Sextet structures (K6, T6) are recognised but not implemented.
"""
from __future__ import annotations
import ast
from typing import Dict, List, Tuple
import numpy as np

NC = 3
# Gell-Mann matrices
_l = np.zeros((8, 3, 3), dtype=complex)
_l[0][0, 1] = _l[0][1, 0] = 1
_l[1][0, 1] = -1j; _l[1][1, 0] = 1j
_l[2][0, 0] = 1; _l[2][1, 1] = -1
_l[3][0, 2] = _l[3][2, 0] = 1
_l[4][0, 2] = -1j; _l[4][2, 0] = 1j
_l[5][1, 2] = _l[5][2, 1] = 1
_l[6][1, 2] = -1j; _l[6][2, 1] = 1j
_l[7][0, 0] = _l[7][1, 1] = 1 / np.sqrt(3); _l[7][2, 2] = -2 / np.sqrt(3)
T = _l / 2                                              # T[a, i, j]
# structure constants: f^{abc} = -2i Tr([T^a,T^b] T^c)
F = np.zeros((8, 8, 8), dtype=complex)
D = np.zeros((8, 8, 8), dtype=complex)
for a in range(8):
    for b in range(8):
        comm = T[a] @ T[b] - T[b] @ T[a]
        acomm = T[a] @ T[b] + T[b] @ T[a]
        for c in range(8):
            F[a, b, c] = -2j * np.trace(comm @ T[c])
            D[a, b, c] = 2 * np.trace(acomm @ T[c])
F = F.real.astype(complex); D = D.real.astype(complex)
EPS3 = np.zeros((3, 3, 3), dtype=complex)
for p, sgn in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1), ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)):
    EPS3[p] = sgn
ID3 = np.eye(3, dtype=complex); ID8 = np.eye(8, dtype=complex)


def rep_dim(color: int) -> int:
    """UFO particle 'color' attribute -> index dimension (1 = singlet)."""
    return {1: 0, 3: 3, -3: 3, 8: 8, 6: 6, -6: 6}[color]


class _Atom:
    __slots__ = ("name", "args")
    def __init__(self, name, args): self.name, self.args = name, args


def _int(node):
    if isinstance(node, ast.Constant): return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return -_int(node.operand)
    raise ValueError("bad colour index")


def _expand(node):
    if isinstance(node, ast.Call):
        return [(1 + 0j, [_Atom(node.func.id, [_int(a) for a in node.args])])]
    if isinstance(node, ast.Constant):
        return [(complex(node.value), [])]
    if isinstance(node, ast.UnaryOp):
        t = _expand(node.operand)
        return [(-c, a) for c, a in t] if isinstance(node.op, ast.USub) else t
    if isinstance(node, ast.BinOp):
        L = _expand(node.left); R = _expand(node.right)
        if isinstance(node.op, ast.Add): return L + R
        if isinstance(node.op, ast.Sub): return L + [(-c, a) for c, a in R]
        if isinstance(node.op, ast.Mult): return [(cl * cr, al + ar) for cl, al in L for cr, ar in R]
        if isinstance(node.op, ast.Div):
            d = R[0][0]; return [(c / d, a) for c, a in L]
    raise ValueError(f"unsupported colour expression node {type(node).__name__}")


_cache: Dict[str, list] = {}


def parse(structure: str):
    if structure not in _cache:
        tree = ast.parse(structure.strip(), mode="eval")
        _cache[structure] = [(c, a) for c, a in _expand(tree.body) if c != 0]
    return _cache[structure]


def atom_tensor(atom: _Atom, leg_dims: Dict[int, int]):
    """(tensor, index labels) for a colour atom.  leg_dims: leg label -> dim."""
    n, ar = atom.name, atom.args
    if n == "T":
        return T, ar
    if n == "f":
        return F, ar
    if n == "d":
        return D, ar
    if n == "Identity":
        dims = [leg_dims.get(x, 0) for x in ar]
        d = max(dims) if max(dims) else 3
        return (ID3 if d == 3 else ID8), ar
    if n in ("Epsilon", "EpsilonBar"):
        return EPS3, ar
    raise NotImplementedError(f"colour atom {n} (sextets) not supported")
