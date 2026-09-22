"""
Test 8: gauge invariance.  SM_UFO in Feynman gauge (with Goldstones) must equal
SM_UFO in unitary gauge (Goldstone vertices DROPPED, kk/M^2 propagator) for
W_L W_L -> HH and for u d -> u d H H.  This validates the unitary-gauge
propagator used for Goldstone-free UFOs such as SMEFTsim.
"""
import numpy as np
from ufoamp.ufo_model import load
from ufoamp.process import Process
from ufoamp.phasespace import rambo
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)

SM = load(SM_UFO)
is_G = lambda v, pdgs: any(abs(q) in (250, 251) for q in pdgs)
light = lambda v, pdgs: (25 in pdgs and any(abs(q) in (1, 2, 3, 4) for q in pdgs)) or 21 in pdgs
MW = float(np.real(Process(SM, [24,-24],[25,25]).params['MW'])); MH = float(np.real(Process(SM, [24,-24],[25,25]).params['MH']))
def cm(rs, th):
    E = rs/2; pi = np.sqrt(E**2-MW**2); pf = np.sqrt(E**2-MH**2)
    return [np.array(x, float) for x in ([E,0,0,pi],[E,0,0,-pi],[E,pf*np.sin(th),0,pf*np.cos(th)],[E,-pf*np.sin(th),0,-pf*np.cos(th)])]
ok = True
pF = Process(SM, [24,-24], [25,25], gauge="feynman")
pU = Process(SM, [24,-24], [25,25], gauge="unitary", drop_vertex=is_G)
for rs in (500., 2000.):
    for h in [(0,0,0,0),(1,-1,0,0),(1,1,0,0),(0,1,0,0)]:
        a = pF.helicity_amplitude(cm(rs,1.0), h); b = pU.helicity_amplitude(cm(rs,1.0), h)
        ok &= abs(a-b) < 1e-9*abs(a)
        print(f"WW->HH rs={rs:5.0f} hel={h}: Feynman {a:.6e}  unitary {b:.6e}")
qF = Process(SM, [2,1], [2,1,25,25], overrides={"MU":0,"MD":0}, gauge="feynman", drop_vertex=light)
qU = Process(SM, [2,1], [2,1,25,25], overrides={"MU":0,"MD":0}, gauge="unitary", drop_vertex=lambda v,p: light(v,p) or is_G(v,p))
rng = np.random.default_rng(5)
for i in range(3):
    fin = rambo(1000., [0,0,MH,MH], rng); mom = [np.array([500.,0,0,500.]), np.array([500.,0,0,-500.])] + list(fin)
    a, b = qF.m2(mom), qU.m2(mom); ok &= abs(a/b-1) < 1e-6
    print(f"VBF-HH pt{i}: Feynman {a:.6e}  unitary {b:.6e}  ratio {b/a:.10f}")
print("gauge invariance Feynman == unitary:", "PASS" if ok else "FAIL")
assert ok
