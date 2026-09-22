"""
Example: per-helicity amplitudes for W+ W- -> H H (the GBET testbed).
Shows longitudinal (0,0) vs transverse contributions and the Goldstone amplitude.
"""
import sys, os, numpy as np
from ufoamp.ufo_model import load
from ufoamp.process import Process

ufo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("UFOAMP_HHVBF_UFO", "HHVBF_UFO")
M = load(ufo)
def cm(rs, th, mi, mo):
    E=rs/2; pi=np.sqrt(E**2-mi**2); pf=np.sqrt(E**2-mo**2)
    return [np.array(x,float) for x in ([E,0,0,pi],[E,0,0,-pi],[E,pf*np.sin(th),0,pf*np.cos(th)],[E,-pf*np.sin(th),0,-pf*np.cos(th)])]
for c2v in (1.0, 2.0):
    pW = Process(M, [24,-24], [25,25], overrides={"CV":1,"C2V":c2v,"C3":1})
    pG = Process(M, [251,-251], [25,25], overrides={"CV":1,"C2V":c2v,"C3":1})
    MW=float(np.real(pW.params["MW"])); MH=float(np.real(pW.params["MH"]))
    rs=2000.0; mom=cm(rs,np.pi/3,MW,MH)
    print(f"\nC2V={c2v}, sqrt(s)={rs:.0f} GeV, theta=60:")
    for h, a in pW.all_helicity_amplitudes(mom).items():
        print(f"  W+({h[0]:+d}) W-({h[1]:+d}) -> HH : |M| = {abs(a):.5e}")
    g = pG.helicity_amplitude(cm(rs,np.pi/3,pG.amp.mass(251),MH),(0,0,0,0))
    print(f"  G+ G- -> HH (Goldstone)  : |M| = {abs(g):.5e}")
