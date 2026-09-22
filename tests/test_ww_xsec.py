"""Absolute normalisation: sigma(e+e- -> W+W-) at LEP2. Known ~17 pb at 200 GeV."""
import numpy as np
from ufoamp.ufo_model import load
from ufoamp.process import Process
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)

M = load(SM_UFO)
GEV2_TO_PB = 0.3893793e9

def xsec(rs):
    proc = Process(M, initial=[11,-11], final=[24,-24], overrides={"Me":0.0})
    BY = M.p_by_pyname(); zero={}
    for v in M.vertices:
        if 25 in [BY[n].pdg for n in v.particles]:
            for cn in v.couplings.values(): zero[cn]=0j
    proc.set_param(overrides={"Me":0.0}, restrict_couplings=zero)
    MW = float(np.real(proc.params["MW"])); E=rs/2; pf=np.sqrt(E**2-MW**2); s=rs**2
    beta = 2*pf/rs
    # dsigma/dOmega = |M|^2_avg * beta / (64 pi^2 s); integrate over cos theta, phi
    xs, w = np.polynomial.legendre.leggauss(40)
    tot=0.0
    for c,wt in zip(xs,w):
        th=np.arccos(c)
        p1=[E,0,0,E];p2=[E,0,0,-E];p3=[E,pf*np.sin(th),0,pf*c];p4=[E,-pf*np.sin(th),0,-pf*c]
        m2=proc.m2([np.array(x,float) for x in (p1,p2,p3,p4)])
        tot += wt * m2*beta/(64*np.pi**2*s) * 2*np.pi
    return tot*GEV2_TO_PB

for rs in (161.0, 183.0, 200.0, 209.0):
    print(f"sqrt(s)={rs:6.1f} GeV   sigma(e+e- -> W+W-) = {xsec(rs):7.3f} pb")
print("(LEP2 measured: ~3.7 pb @161, ~15.4 @183, ~16.9 @200, ~17.3 @209; SM prediction at LEP2 within ~1%)")
