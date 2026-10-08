# ufoamp v1


A standalone Python matrix-element generator that reads a UFO model directly
and evaluates tree-level helicity amplitudes and |M|² numerically, with
**coupling values chosen at run time** (no code generation, no recompilation).
Built for the VBF di-Higgs / Goldstone-equivalence study: switching
CV, C2V, C3 — or switching from the SM UFO to HHVBF — is a function call.

Validated for the Standard Model, the HHVBF UFO and SMEFTsim 3 (all nine
flavour/scheme variants load; U35 MW-scheme validated in detail; see *Validation*).

## Quick start

```bash
pip install ufoamp-1.0.0-py3-none-any.whl     # or: pip install -e /path/to/ufoamp (source tree)
python3 examples/vbf_hh_scan.py /path/to/HHVBF_UFO
python3 examples/helicity_amplitudes.py /path/to/HHVBF_UFO

# validation ladder: tell it where the UFOs are, then every line must say PASS
export UFOAMP_MODELS=/path/to/models   # containing SM_UFO/, HHVBF_UFO/, SMEFTsim/UFO_models/
#   or individually: UFOAMP_SM_UFO, UFOAMP_HHVBF_UFO, UFOAMP_SMEFTSIM_UFO
./run_tests.sh
```
SMEFTsim: `git clone https://github.com/SMEFTsim/SMEFTsim` and point
`UFOAMP_SMEFTSIM_UFO` at `UFO_models/SMEFTsim_U35_MwScheme_UFO`.

```python
from ufoamp.ufo_model import load
from ufoamp.process import Process

M = load("HHVBF_UFO")
proc = Process(M, initial=[2, 1], final=[2, 1, 25, 25],       # u d -> u d H H
               overrides={"CV": 1, "C2V": 2, "C3": 1, "MU": 0, "MD": 0})
m2  = proc.m2(momenta)                          # spin+colour summed, averaged
amp = proc.helicity_amplitude(momenta, (1,-1,1,-1,0,0))   # one helicity config
me.set_param({"C2V": 0, "C3": 5})                     # runtime knob (MatrixElement API)
```

`momenta` is a list of NumPy 4-vectors `[E, px, py, pz]` (metric +−−−), in the
order `initial + final`. `ufoamp.phasespace.rambo` generates flat phase space.

### Coupling control
* `overrides={name: value}` — set any **external** UFO parameter (C3, CV, C2V,
  masses, `cabi`, …). Internal parameters and all couplings are re-derived
  consistently.
* `restrict_couplings={GC_name: value}` — force individual UFO couplings
  (e.g. switch off a vertex class by setting its couplings to 0).
* `inject={name: value}` — pin derived parameters (ee, sw, vev, …) to compare
  two models at one physical point.

## Using it in an analysis (`ufoamp.analysis`)

```python
from ufoamp.analysis import MatrixElement, read_lhe, lhe_to_process_momenta

me = MatrixElement("HHVBF_UFO", "u d > u d h h",            # process string (UFO names or PDG codes)
                   couplings={"CV": 1, "C2V": 2, "C3": 1})   # any external UFO parameter
me.m2(momenta)                                   # one event -> |M|^2 (spin/colour summed, initial averaged)
me.m2_events(events, n_jobs=8)                   # (N, n_particles, 4) -> (N,)
me.set_param({"C2V": 0})                     # switch hypothesis, no rebuild -> reweighting
me.helicity_amplitudes(momenta)                  # {(h1,...,hn): M}

# polarisations of EXTERNAL particles (massive vectors: L = h0, T = h+-1)
w = MatrixElement("HHVBF_UFO", "w+ w- > h h", couplings={...})
w.m2(momenta, pol={"w+": "L"})                   # W+ longitudinal, W- summed
w.m2_polarized(momenta, ["w+", "w-"])            # {'LL','LT','TL','TT'}  (sums to the total exactly)

# MadGraph events
for pdgs, mom, status, hel in read_lhe("events.lhe.gz"):
    ev = lhe_to_process_momenta(pdgs, mom, status, me.initial, me.final)
    print(me.m2(ev))
```
* **Restrict cards / param cards**: `MatrixElement(..., param_card="restrict_4FSZeroYukawa.dat")`
  or `SampleEvaluator(..., param_card="param_card.dat")` reads any SLHA card
  (the UFO's `restrict_*.dat` or the `param_card.dat` of a MadGraph run) and
  applies its values — zeros switch vertices off exactly as MadGraph's
  restriction does. Precedence: light-quark defaults < card < `couplings`.
  `read_param_card(path, model)` returns the {parameter: value} dict.
* **`max_orders={"QCD": 0}`** reproduces MadGraph's `QCD=0`: vertex couplings whose
  UFO coupling order exceeds the limit are dropped (model-agnostic; prefer it
  to dropping "vertices with a gluon"). Without it, gluon exchange is included
  and dominates identical-quark channels.
* **`pol_frame`** sets the frame in which *external* helicities are defined:
  `"lab"` (default), `"partonic_cm"` (initial-state rest frame — MadGraph's
  `me_frame` default for polarised samples), or a list of legs. The
  helicity-summed |M|² is frame independent; polarised pieces are not, so match
  the frame of whatever you compare to.
* `SampleEvaluator` handles mixed-flavour samples (`p p > ...`): one
  `MatrixElement` per partonic channel, built and cached on first use.
* `massless_light_quarks=True` (default) zeroes the u,d,s,c masses **and** their
  Yukawa parameters (`ymup`, `ymdo`, …), as MadGraph does; otherwise Higgs and
  Goldstone emission off the quark lines survives at O(y_q).
* Events and helicity configurations are batched through one pass of the
  recursion (a flat batch axis on every wavefunction, current, momentum and
  propagator), every vertex contraction runs a pre-compiled pairwise einsum
  plan, and exactly-vanishing (chirality) helicity configurations are pruned.
  Cost per core: ~1.6 ms per VBF-HH event, ~20 ms per `u d > u d z z h`
  event; 2→2 processes are ~0.1 ms. `m2_events(..., chunk=32, n_jobs=N)`;
  for LHE samples use `SampleEvaluator.m2_sample(read_lhe(path), vpol="LL")`.
* **Internal (t-channel) boson polarisation — propagator decomposition.**
  For the W/Z radiated off the quark lines the unitary-gauge numerator is split
  exactly, −g^{μν} + k^μk^ν/M² = T + L + S, with ε_± and ε_L built from the
  off-shell k in a chosen frame (default: rest frame of the non-fermion final
  state, i.e. the HH system); for spacelike k the completeness relation is
  −g + kk/k² = T − ε_Lε_L*. This is the "propagator decomposition" definition:
  frame-dependent, and the components interfere in |M|².
  ```python
  me.set_vpol("LL")                 # both fusing bosons longitudinal (one letter per initial quark)
  me.m2(momenta)                    # coherent |M_LL|^2
  me.m2_vpol(momenta, codes=("LL","LT","TL","TT","A"))   # A = full propagator
  me.set_vpol("LT", frame="lab"); me.set_vpol(None)      # frame choice / switch off
  ```
  Letters: L, T, +, −, S (k k remainder), P (=T+L), A (=everything). For
  massless quark lines k·J = 0, so S ≡ 0 and the split is gauge-independent.
  Only t-channel lines (initial↔final fermion) are decomposed; s-channel
  u d → W* diagrams keep the full propagator.

## What is inside

| module | role |
|---|---|
| `ufo_model.py` | self-contained UFO loader (modern and legacy UFOs) |
| `couplings.py` | evaluates parameters → couplings with runtime overrides |
| `dirac.py`, `wavefunctions.py`, `propagators.py` | frozen conventions: metric (+−−−), Dirac basis, helicity spinors, polarization vectors incl. longitudinal, Feynman-gauge propagators |
| `vertex_eval.py` | **general `einsum` evaluator**: any UFO Lorentz string (Metric, P, Gamma, ProjM/P, Identity, Gamma5, Epsilon) → off-shell current |
| `recursion.py` | **Berends–Giele recursion** driven by the UFO vertex table: no diagram enumeration; currents tagged by fermion-line connectivity → exchange signs and colour flow |
| `process.py` | user interface: helicity amplitudes, |M|² |
| `phasespace.py` | RAMBO (massive) |
| `analysis.py` | `MatrixElement`: process strings, runtime couplings, polarisation selection, event arrays, LHE reader |

### Conventions established (and validated) from the UFO files
* UFO vertex leg labels are **outgoing** particle types; a physical leg plugs
  into label X if outgoing, anti(X) if incoming.
* `P(mu,k)` is the outgoing momentum of leg k (`P_SIGN = −1` relative to the
  all-incoming bookkeeping). Fixed by the e⁺e⁻→W⁺W⁻ gauge cancellation: the
  other sign makes |M_LL|² grow like s².
* `Gamma(mu,a,b)` = (γ^μ)_{ab}; a is the ψ̄ (row) slot, b the ψ (column) slot.
* UFO couplings already contain the `i`; Feynman rule = coupling × colour × Lorentz.
* **Gauge is auto-detected** (`gauge="auto"`): Feynman gauge when the UFO
  ships Goldstone bosons (SM_UFO, HHVBF), unitary gauge (propagator with the
  k^μk^ν/M² term) when it does not (SMEFTsim and most MadGraph models). Can be
  forced with `gauge="feynman"|"unitary"`. Feynman ≡ unitary is verified
  numerically on W_LW_L→HH and VBF-HH.
* `drop_vertex=lambda vertex, pdgs: ...` removes whole vertices. Prefer it to
  zeroing couplings by name: UFO couplings are shared across vertices (the SM
  WWHH contact shares its coupling with WWGG).

## Validation (all exact unless stated)
| test | checks | result |
|---|---|---|
| e⁺e⁻→μ⁺μ⁻, photon | engine + UFO plumbing vs e⁴(1+cos²θ) | 1e-9 |
| e⁺e⁻→μ⁺μ⁻, γ+Z | chiral slots, Z width, interference; A_FB sign change across the pole | 1e-7 |
| σ(e⁺e⁻→μ⁺μ⁻) | absolute normalisation vs 4πα²/3s | 1.00000000 |
| e⁺e⁻→W⁺W⁻ | longitudinal gauge cancellation (VVV, t-channel ν, γ/Z) | |M_LL|² → const |
| Bhabha | fermion-exchange sign via s/t interference | 1e-8 |
| e⁺e⁻→uū | colour sum N_c Q_u² | exact |
| **u d → u d H H** | **HHVBF(1,1,1) ≡ SM UFO, 6-point, colour flows, CKM** | **1.000000000** |
| u d → u d H H | amplitude is C2V·A + CV²·B + CV·C3·C + E (E = Goldstone t-channel) | 1e-9 |
| W_L W_L → HH vs G⁺G⁻ → HH | Goldstone equivalence theorem | ratio → 1.0004 at 8 TeV |
| SM Feynman vs SM unitary | gauge invariance: W_LW_L→HH (all helicities), VBF-HH | exact / 1e-8 |
| SMEFTsim e⁺e⁻→μ⁺μ⁻ | SM limit; 4-fermion contact vs photon normalisation (LL only); **Fierz identity cll↔cll1** (crossed chains + fermion sign); linearity | exact |
| **SMEFTsim(c=0) vs SM_UFO, VBF-HH** | two UFOs, two gauges, 6-point | **1e-9** |
| HH→HHHH via \|H\|⁶ | 6-leg vertex plumbing | exact |
| SMEFTsim cH, cHbox, cHDD, cHW on VBF-HH | Higgs-sector coefficients act; cH exactly linear | — |

σ(e⁺e⁻→W⁺W⁻) = 20.8 pb at 200 GeV is the tree-level α(M_Z)-scheme Born value;
the LEP2 number (~17 pb) includes ISR (~−11 %) and a scheme shift (~−7 %).

## Physics finding you should know (HHVBF UFO)
HHVBF scales only the physical VVH (CV), VVHH (C2V), HHH (C3) vertices; every
Goldstone–Higgs vertex is left at its SM value. Consequences, all reproduced
numerically here:
* In Feynman gauge the VBF-HH amplitude is `C2V·A + CV²·B + CV·C3·C + E`
  where `E` is the unscaled Goldstone t-channel exchange.
* The O(s) growth of W_L W_L → HH is ∝ **(C2V − 1)**, independent of CV, in
  Feynman gauge (ufoamp, RECOLA). In unitary gauge (MadGraph default) it is
  ∝ (C2V − CV²). The two agree only at CV = 1. Away from CV = 1 the model is
  gauge-dependent, so MadGraph and RECOLA will disagree there; C2V and C3 scans
  at CV = 1 are safe.

## QCD colour
Every UFO colour structure (`T`, `f`, `d`, `Identity`, `Epsilon`, products with
contracted indices) is a numeric tensor contracted in the same compiled einsum
as the Lorentz structure; currents carry their open external colour indices as
tensor axes and the amplitude is a tensor over external colours
(`helicity_amplitude` returns it; |M|² is its norm). Validated to ~1e-15
against the analytic massless 2→2 results (qq'→qq', qq→qq, qq̄→q'q̄', qq̄→gg,
gg→qq̄, qg→qg, gg→gg) and against MadGraph. Cost scales as 8^(#gluons):
≤4 gluons is fast (VBF+jet ~60 ms/event), gg→ggg is ~17 s/event. Colour
sextets are not supported.

## Majorana fermions
Majorana particles (spin-1/2, self-conjugate) and Dirac fermions in
fermion-flow-violating ("clashing arrow") vertices, as FeynRules writes e.g.
chargino–lepton–sneutrino couplings, are handled by charge-conjugating the
spinor in place whenever it enters a slot of the other type
(ψ → −ψᵀC⁻¹, ψ̄ → Cψ̄ᵀ, C = iγ²γ⁰); fermion signs come from the chain slots.
Validated against MG5 on MSSM_SLHA2 at 1e-14–1e-16: e⁺e⁻→χ̃⁰₁χ̃⁰₁ (both flow
orientations of the selectron exchange), χ̃⁰₁χ̃⁰₂, χ̃⁰₃χ̃⁰₄ (negative SLHA
masses), χ̃⁺₁χ̃⁻₁, u d̄→χ̃⁰₁χ̃⁺₁, gluino pairs from qq̄ and gg, and 6-point
e⁺e⁻→χ̃⁰₁χ̃⁰₁ℓ⁺ℓ⁻. Propagator poles use |m| in the width term so negative mass
eigenvalues are handled correctly.

Not yet supported: spin-2, colour sextets, custom propagators/form factors.

## Independent reference: MadGraph5 standalone (`ufoamp.mg5ref`)
```python
from ufoamp.mg5ref import MG5Reference
ref = MG5Reference("/path/to/mg5amcnlo", "sm", "u d > u d z z h QCD=0")   # generates + compiles C++ once
me  = MatrixElement("/path/to/mg5amcnlo/models/sm", "u d > u d z z h",
                    param_card=ref.param_card, max_orders={"QCD": 0}, gauge="unitary")
ref.compare(me, events)         # max |ratio-1| on identical points; symmetry factors handled
```
Needs MG5 (Python) and g++ only. `tests/test_mg5ref.py` runs eight SM
processes (EW, QCD, top, Higgs, VBF, ZZHjj) at machine precision.

### Conventions needed to match MadGraph exactly (all now defaults or options)
* `restrict="default"`: `<ufo>/restrict_default.dat` is applied implicitly, as
  MG5 does on `import model` (e.g. MG's `sm` has a diagonal CKM; restricted
  parameters are absent from the run's param_card).
* `gauge="unitary"` (drops Goldstone vertices automatically). MG5 standalone
  keeps widths in t-channel propagators (`zerowidth_tchannel=False`, default);
  MG5 *event generation* defaults to `zerowidth_tchannel=True` in the run card.
* `scheme="fixed_width"` (default, MG5) or `"cms"` (complex-mass scheme:
  complex masses in couplings and the unitary numerator; RECOLA-like).
* **Processes with external unstable bosons (e.g. `z z h j j`) are gauge
  dependent at O(Γ/M) (~0.3–0.7 %) with any width scheme**, because on-shell
  external Z's sit on the real mass shell: this is why Feynman-gauge ufoamp and
  unitary-gauge MG5 agree to "4 digits" only; use `gauge="unitary"` to
  reproduce MG5, or compare with widths set to zero.

## SMEFT notes (SMEFTsim 3)
* Loader is library-free (never imports the UFO's Python-2 `object_library`), so
  legacy and SMEFTsim UFOs load directly. Wilson coefficients are ordinary
  `overrides` (`cH`, `cHbox`, `cll`, …, `LambdaSMEFT`).
* n-point vertices (5- and 6-leg), 4-fermion operators (spinor chains read off
  each Lorentz structure; Fierz-crossed structures tagged separately), the
  `Sigma` atom, and atom powers (`P(-1,1)**2`) are supported; every Lorentz
  structure in all nine SMEFTsim UFOs parses.
* Input-scheme subtleties reproduced: `cll1` shifts G_F (vev, dgw, …) and adds
  the photon-vertex coupling `GC_344`; `lam` is `Gf·MH²/√2`; couplings use
  `vevhat`. SMEFTsim ships the SM loop-induced Hγγ/HZγ/Hgg effective vertices
  as tree vertices and a Wolfenstein CKM (`CKMlambda`).

## Scope and next steps
* Done: SM, HHVBF, SMEFTsim; any tree process from n-point vertices incl.
  4-fermion; colour for quark lines (colour-flow matrix N_c^cycles); Feynman
  and unitary gauges.
* Not yet: external gluons / T^a, f^abc colour algebra (QCD-induced processes),
  Majorana fermions, spin-2; vectorisation over phase-space points for fast MC
  integration.
