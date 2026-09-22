"""Locate the models used by the test-suite: MadGraph's Standard Model (sm).
Set UFOAMP_MG5 (the MadGraph installation) or UFOAMP_SM_UFO (the sm UFO directory)."""
import os, sys
MG5 = os.environ.get("UFOAMP_MG5", "/home/claude/models/mg5amcnlo")
SM_UFO = os.environ.get("UFOAMP_SM_UFO", os.path.join(MG5, "models", "sm"))

def require(*paths):
    missing = [p for p in paths if not os.path.isfile(os.path.join(p, "vertices.py"))]
    if missing:
        print("SKIP: UFO model directory not found: " + ", ".join(missing))
        print("      set UFOAMP_MG5 (MadGraph installation) or UFOAMP_SM_UFO")
        sys.exit(0)
