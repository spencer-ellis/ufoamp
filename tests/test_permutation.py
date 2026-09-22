"""
Test 9: |M|^2 must not depend on the order of the external particles (the
recursion's root leg changes with the order).  Also unit-tests multi-term
chiral structures with a fermion output (the ProjM + 2 ProjP form used
for the Z vertex).
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)
from ufoamp.analysis import MatrixElement
from ufoamp.phasespace import rambo
from ufoamp.vertex_eval import eval_vertex
from ufoamp.dirac import GAMMA, PROJP, PROJM, METRIC

# (a) multi-term structure, fermion output
u = np.array([0, 22.36, 0, 22.36], complex); J = np.array([1.0, 0.3, -0.2j, 0.5], complex)
p1 = np.array([500., 0, 0, -500.]); p3 = np.array([100., 10, 5, 20.])
out = eval_vertex("Gamma(3,2,-1)*ProjM(-1,1) - 2*Gamma(3,2,-1)*ProjP(-1,1)", {1: 2, 2: 2, 3: 3}, {1: p1, 3: p3}, {1: ("col", u), 3: J}, 2)
Jl = METRIC @ J; hand = sum(Jl[m] * GAMMA[m] for m in range(4)) @ (PROJM - 2 * PROJP) @ u
ok = np.allclose(out[1], hand)
print("(a) multi-term chiral structure with fermion output:", "PASS" if ok else "FAIL"); assert ok

# (b) permutation invariance
c = {}
ok = True
for procs, masses in ((("u d > u d h h", "u d > h h u d", "u d > u h h d"), [0, 0, 125., 125.]),
                      (("u d > u d z z h", "u d > z z h u d", "u d > z h u z d", "d u > z h u z d"), [0, 0, 91.1876, 91.1876, 125.])):
    rng = np.random.default_rng(1); rs = 1000. if len(masses) == 4 else 2000.
    fin = rambo(rs, masses, rng); E = rs / 2
    vals = []
    for s in procs:
        me = MatrixElement(SM_UFO, s, couplings=c)
        pool = {2: [fin[0]], 1: [fin[1]], 23: [fin[2], fin[3]], 25: [fin[-1]] if len(masses) == 5 else [fin[2], fin[3]]}
        if len(masses) == 5: pool[25] = [fin[4]]
        ini = {2: np.array([E, 0, 0, E]), 1: np.array([E, 0, 0, -E])}
        mom = [ini[p] for p in me.initial] + [pool[p].pop(0) for p in me.final]
        vals.append(me.m2(mom))
    rel = max(abs(v / vals[0] - 1) for v in vals)
    print(f"    {procs[0]:18s}: {len(procs)} orderings agree to {rel:.1e}")
    ok &= rel < 1e-10
print("(b) permutation invariance:", "PASS" if ok else "FAIL"); assert ok
