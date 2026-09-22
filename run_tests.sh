#!/bin/bash
# Test-suite: every test prints PASS (or SKIP if a model/tool is absent).
# Needs MadGraph's Standard Model: set UFOAMP_MG5 (MadGraph installation) or UFOAMP_SM_UFO.
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
for t in test_ee_mumu test_ufo_eemumu test_ufo_eemumu_Z test_sign_color test_ww_xsec test_ufo_eeWW test_gauge_invariance test_permutation test_orders_frames test_qcd test_mg5ref test_jax; do
  echo "--- $t ---"
  out=$(python3 tests/$t.py 2>&1 | grep -vE "SyntaxWarning|texname|escape|ComplexWarning|return \\[")
  echo "$out" | grep -E "PASS|FAIL|SKIP|=> LL|ratio" | head -8
  echo "$out" | grep -qE "PASS|FAIL|SKIP|=> LL" || echo "$out" | tail -3
done
