"""
How to use ufoamp inside an analysis: build one MatrixElement, then evaluate
|M|^2 event by event (e.g. on a MadGraph LHE sample, or on your own 4-vectors).

  UFOAMP_HHVBF_UFO=/path/HHVBF_UFO python3 examples/analysis_example.py [events.lhe(.gz)]
"""
import os, sys, numpy as np
from ufoamp.analysis import MatrixElement, read_lhe, lhe_to_process_momenta

ufo = os.environ.get("UFOAMP_HHVBF_UFO", "HHVBF_UFO")

# 1. process + model + coupling values.  Light quarks massless (as in MadGraph).
me = MatrixElement(ufo, "u d > u d h h",
                   couplings={"CV": 1.0, "C2V": 1.0, "C3": 1.0, "MU": 0.0, "MD": 0.0})
print("gauge:", me.gauge)

# 2. events: from an LHE file if given, else a few random phase-space points
if len(sys.argv) > 1:
    events = []
    for pdgs, mom, status, hel in read_lhe(sys.argv[1], max_events=200):
        try:
            events.append(lhe_to_process_momenta(pdgs, mom, status, me.initial, me.final))
        except ValueError:
            pass            # event has a different flavour channel
    print(f"{len(events)} events of this channel read")
else:
    from ufoamp.phasespace import rambo
    MH = float(np.real(me.param("MH"))); rng = np.random.default_rng(1)
    events = []
    for _ in range(5):
        fin = rambo(1000.0, [0, 0, MH, MH], rng)
        events.append([np.array([500., 0, 0, 500.]), np.array([500., 0, 0, -500.])] + list(fin))

# 3. |M|^2 per event, for several coupling hypotheses -> reweighting factors
w_sm = me.m2_events(events)
for c2v in (0.0, 2.0):
    me.set_param({"C2V": c2v})
    w = me.m2_events(events)
    print(f"C2V={c2v}: <|M|^2/|M_SM|^2> = {np.mean(w / w_sm):.3f}   first events: {np.round(w / w_sm, 3)[:5]}")
me.set_param({"C2V": 1.0})

# 4. polarisations of EXTERNAL bosons: W+ W- -> H H by helicity class
w2 = MatrixElement(ufo, "w+ w- > h h", couplings={"CV": 1, "C2V": 1, "C3": 1})
MW = float(np.real(w2.param("MW"))); MH = float(np.real(w2.param("MH")))
E = 500.0; pi = np.sqrt(E**2 - MW**2); pf = np.sqrt(E**2 - MH**2); th = 1.2
mom = [[E, 0, 0, pi], [E, 0, 0, -pi], [E, pf*np.sin(th), 0, pf*np.cos(th)], [E, -pf*np.sin(th), 0, -pf*np.cos(th)]]
for c2v in (1.0, 2.0):
    w2.set_param({"C2V": c2v})
    pol = w2.m2_polarized(mom, ["w+", "w-"], average_initial=False)
    print(f"W+W- -> HH at C2V={c2v}: " + "  ".join(f"{k}={v:.3e}" for k, v in pol.items()))

# 5. polarisation of the INTERNAL fusing bosons in VBF (propagator decomposition)
me.set_param({"C2V": 1.0}); me.reset_pruning()
for c2v in (1.0, 2.0):
    me.set_param({"C2V": c2v})
    r = me.m2_vpol(events[0], codes=("LL", "LT", "TL", "TT", "A"))
    print(f"VBF-HH event 0, C2V={c2v}: " + "  ".join(f"{k}={v:.3e}" for k, v in r.items()))
