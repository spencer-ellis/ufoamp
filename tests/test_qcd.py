"""
Test 11: QCD colour algebra vs the classic massless 2->2 results
(Ellis-Stirling-Webber Table 7.1), spin- and colour-averaged, in units g_s^4:
  q q' -> q q'    : 4/9 (s^2+u^2)/t^2
  q q  -> q q     : 4/9 [(s^2+u^2)/t^2 + (s^2+t^2)/u^2] - 8/27 s^2/(t u)
  q qb -> q' qb'  : 4/9 (t^2+u^2)/s^2
  q qb -> g g     : 32/27 (t^2+u^2)/(t u) - 8/3 (t^2+u^2)/s^2
  g g  -> q qb    : 1/6 (t^2+u^2)/(t u) - 3/8 (t^2+u^2)/s^2
  q g  -> q g     : -4/9 (s^2+u^2)/(s u) + (u^2+s^2)/t^2
  g g  -> g g     : 9/2 (3 - t u/s^2 - s u/t^2 - s t/u^2)
These test T^a, f^abc, the four-gluon vertex, colour sums/averages and the
colour-interference of identical quarks.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import SM_UFO, require
require(SM_UFO)
from ufoamp.analysis import MatrixElement

def cm(E, th):
    return [np.array(x, float) for x in ([E,0,0,E],[E,0,0,-E],[E,E*np.sin(th),0,E*np.cos(th)],[E,-E*np.sin(th),0,-E*np.cos(th)])]

light = {"MU": 0, "MD": 0}
gs = None
tests = {
    "u d > u d":   lambda s,t,u: 4/9*(s**2+u**2)/t**2,
    "u u > u u":   lambda s,t,u: 4/9*((s**2+u**2)/t**2 + (s**2+t**2)/u**2) - 8/27*s**2/(t*u),
    "u u~ > d d~": lambda s,t,u: 4/9*(t**2+u**2)/s**2,
    "u u~ > g g":  lambda s,t,u: 32/27*(t**2+u**2)/(t*u) - 8/3*(t**2+u**2)/s**2,
    "g g > u u~":  lambda s,t,u: 1/6*(t**2+u**2)/(t*u) - 3/8*(t**2+u**2)/s**2,
    "u g > u g":   lambda s,t,u: -4/9*(s**2+u**2)/(s*u) + (u**2+s**2)/t**2,
    "g g > g g":   lambda s,t,u: 9/2*(3 - t*u/s**2 - s*u/t**2 - s*t/u**2),
}
ok = True
E = 50.0; s = (2*E)**2
for proc, formula in tests.items():
    me = MatrixElement(SM_UFO, proc, couplings=light, max_orders={"QED": 0})
    if gs is None:
        gs = float(np.real(me.param("gs") if "gs" in me.proc.params else me.param("G")))
    worst = 0.0
    for deg in (40, 90, 130):
        th = np.radians(deg); c = np.cos(th)
        t = -s*(1-c)/2; u = -s*(1+c)/2
        num = me.m2(cm(E, th)); ana = gs**4*formula(s, t, u)
        worst = max(worst, abs(num/ana - 1))
    ok &= worst < 1e-8
    print(f"  {proc:12s}  max |ratio-1| = {worst:.1e}")
print("QCD 2->2 colour algebra vs ESW:", "PASS" if ok else "FAIL")
assert ok
