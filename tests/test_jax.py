"""Test 13: JAX backend equals numpy; autodiff gradient equals finite differences (sm)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)
try:
    import jax
except ImportError:
    print("SKIP: jax not installed"); sys.exit(0)
from ufoamp.analysis import MatrixElement
from ufoamp.phasespace import rambo
from ufoamp.jaxme import JaxMatrixElement
rng = np.random.default_rng(0)
ev = np.array([[np.array([500., 0, 0, 500.]), np.array([500., 0, 0, -500.])] + list(rambo(1000., [0, 0, 125., 125.], rng)) for _ in range(8)])
me = MatrixElement(SM_UFO, "u d > u d h h", max_orders={"QCD": 0})
ok = np.max(np.abs(JaxMatrixElement(me).m2(ev) / me.m2_events(ev, prune=False) - 1)) < 1e-12
print("(a) jax == numpy on u d > u d h h:", "PASS" if ok else "FAIL")
# gradient with respect to alpha_s on u u~ > g g: |M|^2 is exactly proportional to aS^2
p = ev[:1, :4].copy(); p[0, 2:] = rambo(1000., [0, 0], np.random.default_rng(1))
meg = MatrixElement(SM_UFO, "u u~ > g g", max_orders={"QED": 0})
aS = float(np.real(meg.param("aS")))
g = JaxMatrixElement(meg, params=["aS"]).grad_params(p)[0, 0]
exact = 2 * meg.m2_events(p, prune=False)[0] / aS
ok2 = abs(g / exact - 1) < 1e-10
print(f"(b) d|M|^2/d(alpha_s) on u u~ > g g: autodiff {g:.6e} vs exact 2|M|^2/aS {exact:.6e}:", "PASS" if ok2 else "FAIL")
assert ok and ok2
