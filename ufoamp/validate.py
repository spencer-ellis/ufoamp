"""
validate.py — capability report and the `ufoamp validate` command.

    python -m ufoamp.validate --model /path/UFO --process "u d > u d h h" \
        [--orders QCD=0] [--energy 1000] [--npoints 5] [--gauge unitary] \
        [--mg5 /path/mg5amcnlo] [--mg5-model sm] [--save-reference ref.json] \
        [--reference ref.json]

* capability report: what the UFO needs (spins, colour reps, Lorentz and
  colour atoms, custom propagators, Majorana fermions, Goldstones) against
  what ufoamp supports -- printed before anything is evaluated.
* with --mg5: MadGraph standalone C++ is generated for the same process and
  compared on random phase-space points (identical param card).
* --save-reference writes the points, MG5 values and card to JSON so the same
  comparison runs later without MadGraph (--reference).
"""
from __future__ import annotations
import argparse, json, os, re, sys
from typing import Dict, List
import numpy as np

from .ufo_model import load, Model

SUPPORTED = {
    "spins": {1, 2, 3},                 # -1 ghosts are skipped at tree level
    "colors": {1, 3, -3, 8},
    "lorentz_atoms": {"Metric", "P", "Gamma", "ProjM", "ProjP", "Identity", "Gamma5", "Sigma", "Epsilon"},
    "color_atoms": {"T", "f", "d", "Identity", "Epsilon", "EpsilonBar"},
}


def capabilities(model) -> Dict:
    m = load(model) if isinstance(model, str) else model
    spins = {p.spin for p in m.particles}
    colors = {p.color for p in m.particles}
    latoms = set()
    for l in m.lorentz:
        latoms |= set(re.findall(r"\b([A-Za-z][A-Za-z0-9]*)\(", l.structure))
    catoms = set()
    for v in m.vertices:
        for c in v.color:
            catoms |= set(re.findall(r"\b([A-Za-z][A-Za-z0-9]*)\(", c))
    rep = {
        "model": m.name,
        "particles": len(m.particles), "vertices": len(m.vertices), "couplings": len(m.couplings),
        "spins": sorted(spins), "unsupported_spins": sorted(spins - SUPPORTED["spins"] - {-1}),
        "colors": sorted(colors), "unsupported_colors": sorted(set(colors) - SUPPORTED["colors"]),
        "lorentz_atoms": sorted(latoms), "unsupported_lorentz_atoms": sorted(latoms - SUPPORTED["lorentz_atoms"]),
        "color_atoms": sorted(catoms), "unsupported_color_atoms": sorted(catoms - SUPPORTED["color_atoms"]),
        "majorana": [p.name for p in m.particles if p.spin == 2 and p.self_conjugate],
        "goldstones": m.has_goldstones(),
        "default_gauge": "feynman" if m.has_goldstones() else "unitary",
        "max_vertex_legs": max(len(v.particles) for v in m.vertices) if m.vertices else 0,
        "custom_propagators": bool(getattr(m, "propagators", None)),
    }
    rep["supported"] = not (rep["unsupported_spins"] or rep["unsupported_colors"]
                            or rep["unsupported_lorentz_atoms"] or rep["unsupported_color_atoms"])
    return rep


def print_capabilities(rep: Dict):
    print(f"model {rep['model']}: {rep['particles']} particles, {rep['vertices']} vertices, "
          f"{rep['couplings']} couplings, vertices up to {rep['max_vertex_legs']} legs")
    print(f"  spins {rep['spins']}  colour reps {rep['colors']}  Majorana {rep['majorana'] or 'none'}")
    print(f"  Lorentz atoms {rep['lorentz_atoms']}")
    print(f"  colour atoms  {rep['color_atoms']}")
    print(f"  Goldstones: {rep['goldstones']} -> default gauge {rep['default_gauge']}")
    for k in ("unsupported_spins", "unsupported_colors", "unsupported_lorentz_atoms", "unsupported_color_atoms"):
        if rep[k]:
            print(f"  UNSUPPORTED {k.replace('unsupported_', '')}: {rep[k]}")
    print("  => " + ("fully supported by ufoamp" if rep["supported"] else "NOT fully supported (see above)"))


def random_points(me, energy, n, seed=0):
    from .phasespace import rambo
    rng = np.random.default_rng(seed)
    E = energy / 2
    mi = [me.proc.amp.mass(p) for p in me.initial]
    pi = [np.sqrt(max(E**2 - m**2, 0)) for m in mi]
    masses = [abs(me.proc.amp.mass(p)) for p in me.final]
    ev = []
    for _ in range(n):
        fin = rambo(energy, masses, rng)
        ev.append([np.array([E, 0, 0, pi[0]]), np.array([E, 0, 0, -pi[1]])] + list(fin))
    return np.array(ev)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate ufoamp on a model/process (capability report + MG5 comparison)")
    ap.add_argument("--model", required=True, help="UFO directory")
    ap.add_argument("--process", help='e.g. "u d > u d h h"')
    ap.add_argument("--orders", nargs="*", default=[], help="coupling-order limits, e.g. QCD=0")
    ap.add_argument("--energy", type=float, default=1000.0)
    ap.add_argument("--npoints", type=int, default=5)
    ap.add_argument("--gauge", default="unitary", help="unitary (MG5) | feynman | auto")
    ap.add_argument("--param-card", default=None)
    ap.add_argument("--mg5", default=os.environ.get("UFOAMP_MG5"), help="MG5_aMC directory (needs g++)")
    ap.add_argument("--mg5-model", default=None, help="model name as MG5 knows it (default: path given to --model)")
    ap.add_argument("--save-reference", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--rtol", type=float, default=1e-6)
    a = ap.parse_args(argv)

    rep = capabilities(a.model); print_capabilities(rep)
    if not a.process:
        return 0
    from .analysis import MatrixElement
    orders = {k: int(v) for k, v in (o.split("=") for o in a.orders)} or None
    proc_ufoamp = re.sub(r"\s+\w+=\d+", "", a.process)
    order_str = " ".join(a.orders)

    if a.reference:
        ref = json.load(open(a.reference))
        me = MatrixElement(a.model, proc_ufoamp, param_card=ref["param_card_path"] if os.path.isfile(ref.get("param_card_path", "")) else a.param_card,
                           max_orders=orders, gauge=a.gauge)
        if ref.get("param_card_text") and not os.path.isfile(ref.get("param_card_path", "")):
            tmp = "/tmp/ufoamp_ref_card.dat"; open(tmp, "w").write(ref["param_card_text"])
            me = MatrixElement(a.model, proc_ufoamp, param_card=tmp, max_orders=orders, gauge=a.gauge)
        ev = np.array(ref["momenta"]); target = np.array(ref["m2"]) * ref.get("symmetry_factor", 1.0)
        got = me.m2_events(ev, prune=False)
        worst = float(np.max(np.abs(got / target - 1)))
        print(f"ufoamp vs stored reference ({ref['source']}) {a.process}: {len(ev)} points, max |ratio-1| = {worst:.2e} -> {'PASS' if worst < a.rtol else 'FAIL'}")
        return 0 if worst < a.rtol else 1

    if a.mg5:
        from .mg5ref import MG5Reference
        mg5_model = a.mg5_model or a.model
        ref = MG5Reference(a.mg5, mg5_model, (a.process + " " + order_str).strip())
        me = MatrixElement(a.model, proc_ufoamp, param_card=ref.param_card, max_orders=orders, gauge=a.gauge)
        ev = random_points(me, a.energy, a.npoints)
        worst, ratio = ref.compare(me, ev, rtol=a.rtol)
        if a.save_reference:
            json.dump({"source": f"MG5 standalone_cpp model={mg5_model}", "process": a.process, "model": a.model,
                       "momenta": ev.tolist(), "m2": ref.m2_events(ev).tolist(),
                       "symmetry_factor": ref.symmetry_factor(me), "param_card_path": ref.param_card,
                       "param_card_text": open(ref.param_card).read(), "gauge": a.gauge, "orders": a.orders},
                      open(a.save_reference, "w"))
            print("reference saved to", a.save_reference)
        return 0 if worst < a.rtol else 1

    me = MatrixElement(a.model, proc_ufoamp, param_card=a.param_card, max_orders=orders, gauge=a.gauge)
    ev = random_points(me, a.energy, a.npoints)
    for i, v in enumerate(me.m2_events(ev, prune=False)):
        print(f"point {i}: |M|^2 = {v:.10e}")
    print("(no --mg5 or --reference given: values printed only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
