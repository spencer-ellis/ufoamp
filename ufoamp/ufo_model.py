"""
ufo_model.py
============

Minimal, self-contained UFO loader: imports a UFO package (modern relative-import
or legacy absolute-import style) and harvests it into plain dataclasses.  The
model is treated as *data*; coupling value-expressions are kept as strings and
only turned into numbers later, at call time, by :mod:`couplings` -- which is the
design decision that makes "change C3" a runtime argument instead of a rebuild.
"""

from __future__ import annotations

import importlib.util
import re
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Particle:
    pdg: int
    name: str
    spin: int          # UFO: 1 scalar, 2 fermion, 3 vector, -1 ghost
    color: int
    mass: str          # parameter name ('ZERO' if massless)
    width: str
    self_conjugate: bool
    py_name: str = ""
    goldstone: bool = False


@dataclass
class Parameter:
    name: str
    nature: str        # external | internal
    ptype: str         # real | complex
    value: Any
    lhablock: Optional[str] = None
    lhacode: Optional[List[int]] = None


@dataclass
class Coupling:
    name: str
    value: str
    order: Dict[str, int] = field(default_factory=dict)


@dataclass
class Lorentz:
    name: str
    spins: List[int]
    structure: str


@dataclass
class Vertex:
    name: str
    particles: List[str]                       # UFO py-names, ordered
    color: List[str]
    lorentz: List[str]
    couplings: Dict[Tuple[int, int], str] = field(default_factory=dict)


@dataclass
class Model:
    def has_goldstones(self) -> bool:
        return any(p.goldstone for p in self.particles)

    name: str
    particles: List[Particle]
    parameters: List[Parameter]
    couplings: List[Coupling]
    lorentz: List[Lorentz]
    vertices: List[Vertex]
    orders: List[str]
    path: str = ""          # UFO directory (set by load); used to find restriction cards

    def p_by_pyname(self):
        return {p.py_name: p for p in self.particles}

    def p_by_pdg(self):
        return {p.pdg: p for p in self.particles}

    def lorentz_by_name(self):
        return {l.name: l for l in self.lorentz}


_SUBMODULES = ["object_library", "function_library", "parameters", "particles",
               "coupling_orders", "couplings", "lorentz", "propagators",
               "vertices", "decays"]


# --------------------------------------------------------------------------- #
#  robust, library-free import: works for Python-2 UFOs (SMEFTsim, legacy)    #
# --------------------------------------------------------------------------- #
class _UFOObj:
    """Generic UFO object: stores every keyword argument as an attribute."""
    def __init__(self, *args, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.__dict__.setdefault("_args", args)

    def __getitem__(self, k):
        return getattr(self, k)


class _Particle(_UFOObj):
    def anti(self):
        d = dict(self.__dict__)
        d.pop("_args", None)
        d["name"], d["antiname"] = self.antiname, self.name
        d["pdg_code"] = -self.pdg_code
        if "charge" in d:
            d["charge"] = -d["charge"]
        if "color" in d and d["color"] not in (1, 8):
            d["color"] = -d["color"]
        if "texname" in d and "antitexname" in d:
            d["texname"], d["antitexname"] = d["antitexname"], d["texname"]
        if "Y" in d:
            d["Y"] = -d["Y"]
        return type(self)(**d)   # registered subclass -> appended to all_particles


def _make_object_library():
    ol = types.ModuleType("object_library")
    lists = {}
    def mk(clsname, listname, base=_UFOObj):
        lst = []
        lists[listname] = lst
        def __init__(self, *a, **kw):
            base.__init__(self, *a, **kw)
            lst.append(self)
        cls = type(clsname, (base,), {"__init__": __init__})
        setattr(ol, clsname, cls)
        setattr(ol, listname, lst)
    mk("Particle", "all_particles", _Particle)
    mk("Parameter", "all_parameters")
    mk("Vertex", "all_vertices")
    mk("Coupling", "all_couplings")
    mk("Lorentz", "all_lorentz")
    mk("CouplingOrder", "all_orders")
    mk("Decay", "all_decays")
    mk("CTVertex", "all_CTvertices")
    mk("CTParameter", "all_CTparameters")
    mk("FormFactor", "all_form_factors")
    mk("Propagator", "all_propagators")
    class UFOError(Exception):
        pass
    ol.UFOError = UFOError
    ol.all_functions = []
    return ol


def _make_function_library():
    import cmath
    fl = types.ModuleType("function_library")
    class Function:
        def __init__(self, name=None, arguments=None, expression=None, **kw):
            self.name, self.arguments, self.expression = name, arguments, expression
        def __call__(self, *a):
            return None
    fl.Function = Function
    fl.complexconjugate = lambda z: complex(z).conjugate()
    fl.re = lambda z: complex(z).real
    fl.im = lambda z: complex(z).imag
    fl.csc = lambda z: 1 / cmath.sin(z)
    fl.sec = lambda z: 1 / cmath.cos(z)
    fl.acsc = lambda z: cmath.asin(1 / z)
    fl.asec = lambda z: cmath.acos(1 / z)
    fl.cot = lambda z: 1 / cmath.tan(z)
    fl.theta_function = lambda x, y, z: y if x else z
    fl.cond = lambda c, a, b: a if c == 0 else b
    fl.reglog = lambda z: 0 if z == 0 else cmath.log(z)
    fl.reglogp = fl.reglog
    fl.reglogm = fl.reglog
    fl.arg = lambda z: cmath.phase(z)
    return fl


_PY2_FIXES = [
    (re.compile(r"raise\s+(\w+)\s*,\s*(.+)$", re.M), r"raise \1(\2)"),
    (re.compile(r"except\s+(\w+)\s*,\s*(\w+)\s*:"), r"except \1 as \2:"),
    (re.compile(r"^\s*print\s+([^(].*)$", re.M), r"pass"),
]


def _exec_datafile(path, modname, base):
    src = open(path, encoding="utf-8", errors="replace").read()
    for rx, rep in _PY2_FIXES:
        src = rx.sub(rep, src)
    m = types.ModuleType(modname)
    m.__file__ = path
    m.__package__ = base
    sys.modules[modname] = m
    sys.modules[base + "." + modname] = m
    setattr(sys.modules[base], modname, m)
    exec(compile(src, path, "exec"), m.__dict__)  # noqa: S102 - trusted model data
    return m


_DATAFILES = ["coupling_orders", "parameters", "lorentz", "propagators", "particles",
              "couplings", "vertices"]


def _import(ufo_dir: str):
    ufo_dir = os.path.abspath(ufo_dir)
    base = "_ufoamp_ufo_" + os.path.basename(ufo_dir).replace("-", "_").replace(".", "_")
    saved = {n: sys.modules.get(n) for n in ["object_library", "function_library"] + _DATAFILES}
    pkg = types.ModuleType(base)
    pkg.__path__ = [ufo_dir]
    pkg.__package__ = base
    ol = _make_object_library()
    fl = _make_function_library()
    try:
        sys.modules[base] = pkg
        for n, m in (("object_library", ol), ("function_library", fl)):
            sys.modules[n] = m
            sys.modules[base + "." + n] = m
            setattr(pkg, n, m)
        imported = {}
        for n in _DATAFILES:
            p = os.path.join(ufo_dir, n + ".py")
            if os.path.isfile(p):
                imported[n] = _exec_datafile(p, n, base)
        synth = types.ModuleType("_ufo")
        for attr in ("all_particles", "all_parameters", "all_couplings",
                     "all_lorentz", "all_vertices", "all_orders"):
            setattr(synth, attr, getattr(ol, attr))
        synth.particles = imported.get("particles")
        return synth
    finally:
        for n in ["object_library", "function_library"] + _DATAFILES:
            sys.modules.pop(n, None)
            sys.modules.pop(base + "." + n, None)
        sys.modules.pop(base, None)
        for n, mm in saved.items():
            if mm is not None:
                sys.modules[n] = mm


def load(ufo_dir: str) -> Model:
    if not os.path.isfile(os.path.join(ufo_dir, "vertices.py")):
        raise FileNotFoundError(f"{ufo_dir!r} is not a UFO model directory (no vertices.py)")
    mod = _import(ufo_dir)
    pmod = getattr(mod, "particles", None)
    pyname = {id(v): k for k, v in vars(pmod).items()
              if not k.startswith("_")} if pmod else {}

    particles = []
    for p in mod.all_particles:
        mass = getattr(p, "mass", None); width = getattr(p, "width", None)
        particles.append(Particle(
            pdg=int(getattr(p, "pdg_code", 0)),
            name=str(getattr(p, "name", "")),
            spin=int(getattr(p, "spin", 1)),
            color=int(getattr(p, "color", 1)),
            mass=str(getattr(mass, "name", "ZERO")) if mass is not None else "ZERO",
            width=str(getattr(width, "name", "ZERO")) if width is not None else "ZERO",
            self_conjugate=getattr(p, "name", 1) == getattr(p, "antiname", 2),
            py_name=pyname.get(id(p), str(getattr(p, "name", ""))),
            goldstone=bool(getattr(p, "goldstone", False) or getattr(p, "GoldstoneBoson", False) or getattr(p, "goldstoneboson", False)),
        ))
    parameters = [Parameter(str(pr.name), str(getattr(pr, "nature", "internal")),
                            str(getattr(pr, "type", "real")),
                            getattr(pr, "value", 0),
                            getattr(pr, "lhablock", None),
                            list(getattr(pr, "lhacode", []) or []) or None)
                  for pr in mod.all_parameters]
    couplings = [Coupling(str(c.name), str(getattr(c, "value", "0")),
                          {str(k): int(v) for k, v in (getattr(c, "order", {}) or {}).items()})
                 for c in mod.all_couplings]
    lorentz = [Lorentz(str(l.name), list(getattr(l, "spins", []) or []),
                       str(getattr(l, "structure", "1"))) for l in mod.all_lorentz]

    part_py = {id(p): pyname.get(id(p), str(getattr(p, "name", "")))
               for p in mod.all_particles}
    vertices = []
    for v in mod.all_vertices:
        legs = [part_py.get(id(pp), "") for pp in getattr(v, "particles", [])]
        cpl = {(int(k[0]), int(k[1])): str(getattr(cc, "name", cc))
               for k, cc in (getattr(v, "couplings", {}) or {}).items()}
        vertices.append(Vertex(str(v.name), legs,
                               [str(x) for x in getattr(v, "color", [])],
                               [str(getattr(x, "name", x)) for x in getattr(v, "lorentz", [])],
                               cpl))
    orders = [str(getattr(o, "name", o)) for o in getattr(mod, "all_orders", [])]
    name = os.path.basename(os.path.abspath(ufo_dir))
    for suf in ("_UFO", "_ufo"):
        if name.endswith(suf):
            name = name[:-len(suf)]
    m = Model(name, particles, parameters, couplings, lorentz, vertices, orders)
    m.path = os.path.abspath(ufo_dir)
    return m
