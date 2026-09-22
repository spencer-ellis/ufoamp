"""
mg5ref.py — independent reference: MadGraph5_aMC@NLO standalone C++ output.

    from ufoamp.mg5ref import MG5Reference
    ref = MG5Reference(mg5_dir, model="sm", process="u u~ > g g")
    ref.m2(momenta)                    # MG5 |M|^2 (initial spin+colour averaged)
    ref.compare(ufoamp_matrix_element, n_points=20)

MG5 is run once per (model, process) to generate and compile the C++ code
(cached under `workdir`); afterwards evaluations are a subprocess call with
the momenta on stdin.  The param_card MG5 writes is what ufoamp should be
pointed at (`param_card=ref.param_card`) so both use identical parameters.
Requires MG5 (python) and g++; no Fortran needed.
"""
from __future__ import annotations
import os, re, subprocess, hashlib, shutil
from typing import Optional
import numpy as np

_DRIVER = r'''
#include <iostream>
#include <iomanip>
#include <vector>
#include "CPPProcess.h"
int main(int argc, char** argv){
  CPPProcess process;
  process.initProc(argv[1]);
  int n = process.nexternal;
  std::vector<double*> p(n);
  for (int i = 0; i < n; i++) p[i] = new double[4];
  double x;
  std::cout << std::setprecision(17);
  while (std::cin >> x) {
    p[0][0] = x;
    for (int i = 0; i < n; i++) for (int k = (i == 0 ? 1 : 0); k < 4; k++) std::cin >> p[i][k];
    process.setMomenta(p);
    process.sigmaKin();
    const double* me = process.getMatrixElements();
    std::cout << me[0] << std::endl;
  }
  return 0;
}
'''


class MG5Reference:
    def __init__(self, mg5_dir: str, model: str, process: str, workdir: str = "/tmp/ufoamp_mg5ref",
                 model_options: str = "", restrict: Optional[str] = None):
        self.mg5_dir = mg5_dir
        self.model = model
        self.process = process
        key = hashlib.md5(f"{model}|{model_options}|{restrict}|{process}".encode()).hexdigest()[:10]
        self.dir = os.path.join(workdir, key)
        if not os.path.isfile(os.path.join(self.dir, "driver")):
            self._build(model_options, restrict)
        self.param_card = os.path.join(self.dir, "Cards", "param_card.dat")
        self.nexternal = self._nexternal()

    # ---- build ------------------------------------------------------------ #
    def _build(self, model_options, restrict):
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(os.path.dirname(self.dir), exist_ok=True)
        mdl = self.model + (f"-{restrict}" if restrict else "")
        card = f"import model {mdl} {model_options}\ngenerate {self.process}\noutput standalone_cpp {self.dir}\n"
        cf = self.dir + ".mg5"
        open(cf, "w").write(card)
        r = subprocess.run(["python3", os.path.join(self.mg5_dir, "bin", "mg5_aMC"), cf],
                           capture_output=True, text=True, timeout=1800)
        if not os.path.isdir(os.path.join(self.dir, "SubProcesses")):
            raise RuntimeError("MG5 generation failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
        sub = [d for d in os.listdir(os.path.join(self.dir, "SubProcesses")) if d.startswith("P")]
        if len(sub) != 1:
            raise RuntimeError(f"expected exactly one subprocess directory, got {sub}")
        self.subdir = os.path.join(self.dir, "SubProcesses", sub[0])
        open(os.path.join(self.subdir, "driver.cpp"), "w").write(_DRIVER)
        r = subprocess.run(["make"], cwd=self.subdir, capture_output=True, text=True, timeout=1800)
        src = os.path.join(self.dir, "src")
        libs = " ".join(sorted(f for f in os.listdir(self.dir + "/lib") if f.endswith(".a"))) if os.path.isdir(self.dir + "/lib") else ""
        cmd = f"g++ -O2 -std=c++11 -I{src} -I. driver.cpp CPPProcess.o -L{self.dir}/lib -lmodel_{self._model_tag()} -o driver"
        r2 = subprocess.run(cmd, shell=True, cwd=self.subdir, capture_output=True, text=True)
        if r2.returncode != 0:
            raise RuntimeError("driver compilation failed:\n" + cmd + "\n" + r2.stderr[-3000:] + r.stderr[-2000:])
        shutil.copy(os.path.join(self.subdir, "driver"), os.path.join(self.dir, "driver"))

    def _model_tag(self):
        libs = os.listdir(os.path.join(self.dir, "lib"))
        for f in libs:
            m = re.match(r"libmodel_(.+)\.a", f)
            if m:
                return m.group(1)
        raise RuntimeError(f"no model library in {self.dir}/lib: {libs}")

    def _nexternal(self):
        for d in os.listdir(os.path.join(self.dir, "SubProcesses")):
            if d.startswith("P"):
                h = open(os.path.join(self.dir, "SubProcesses", d, "CPPProcess.h")).read()
                m = re.search(r"nexternal\s*=\s*(\d+)", h)
                return int(m.group(1))
        raise RuntimeError

    # ---- evaluate --------------------------------------------------------- #
    def m2_events(self, events) -> np.ndarray:
        events = np.asarray(events, dtype=float)
        if events.ndim == 2:
            events = events[None]
        lines = []
        for ev in events:
            lines.append(" ".join(f"{x:.17g}" for p in ev for x in p))
        r = subprocess.run([os.path.join(self.dir, "driver"), self.param_card], input="\n".join(lines) + "\n",
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError("driver failed: " + r.stderr[-2000:])
        vals = []
        for line in r.stdout.splitlines():
            t = line.strip()
            try:
                vals.append(float(t))
            except ValueError:
                pass                               # MG5 chatter (param card echo)
        return np.array(vals)

    def m2(self, momenta) -> float:
        return float(self.m2_events([momenta])[0])

    @staticmethod
    def symmetry_factor(me) -> float:
        from collections import Counter
        from math import factorial
        f = 1.0
        for n in Counter(me.final).values():
            f *= factorial(n)
        return f

    def compare(self, me, events, rtol=1e-8, verbose=True):
        """Compare a ufoamp MatrixElement on the given events.  Returns
        (max relative difference, array of ratios)."""
        a = np.asarray(me.m2_events(events, prune=False))
        b = self.m2_events(events) * self.symmetry_factor(me)   # MG5 includes 1/n! for identical final particles
        ratio = a / b
        worst = float(np.max(np.abs(ratio - 1)))
        if verbose:
            print(f"ufoamp vs MG5 [{self.model}] {self.process}: {len(events)} points, "
                  f"max |ratio-1| = {worst:.2e}  ->  {'PASS' if worst < rtol else 'FAIL'}")
        return worst, ratio
