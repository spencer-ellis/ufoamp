"""
Test 12: independent reference -- MadGraph5_aMC@NLO standalone C++ on identical
phase-space points and parameter cards (needs MG5 + g++; set UFOAMP_MG5).
Compares colour, EW, Higgs and top processes at machine precision (unitary
gauge, as MG5).  Skipped when MG5 is not available.
"""
import os, sys, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MG = os.environ.get("UFOAMP_MG5", "/home/claude/models/mg5amcnlo")
if not os.path.isfile(os.path.join(MG, "bin", "mg5_aMC")) or shutil.which("g++") is None:
    print("SKIP: MadGraph5 (UFOAMP_MG5) or g++ not available"); sys.exit(0)
from ufoamp.mg5ref import MG5Reference
from ufoamp.analysis import MatrixElement
from ufoamp.phasespace import rambo
SMU = os.path.join(MG, "models", "sm")
rng = np.random.default_rng(0); rs = 1000.; E = rs / 2
ok = True
for proc, masses in (("e+ e- > mu+ mu-", [0, 0]), ("e+ e- > w+ w-", [80.419, 80.419]), ("u u~ > g g", [0, 0]),
                     ("g g > g g", [0, 0]), ("g g > t t~ h", [173., 173., 125.]), ("u d > u d h h QCD=0", [0, 0, 125., 125.]),
                     ("u u > u u z z h QCD=0", [0, 0, 91.188, 91.188, 125.]), ("u d > u d z z h QCD=0", [0, 0, 91.188, 91.188, 125.])):
    ref = MG5Reference(MG, "sm", proc)
    me = MatrixElement(SMU, proc.replace(" QCD=0", ""), param_card=ref.param_card,
                       max_orders={"QCD": 0} if "QCD=0" in proc else None, gauge="unitary")
    mi = me.proc.amp.mass(me.initial[0]); pi = np.sqrt(E**2 - mi**2)
    ev = [[np.array([E, 0, 0, pi]), np.array([E, 0, 0, -pi])] + list(rambo(rs, masses, rng)) for _ in range(4)]
    w, _ = ref.compare(me, ev, rtol=1e-6)
    ok &= w < 1e-6
print("ufoamp vs MG5 standalone:", "PASS" if ok else "FAIL"); assert ok
