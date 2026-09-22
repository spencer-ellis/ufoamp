"""wf_batched.py — fully vectorised external wavefunctions (same conventions as
wavefunctions.py, validated against it), for arrays of momenta (B,4) and
helicities (B,).  Uses the switchable backend `xp` (numpy or jax.numpy)."""
import numpy as xp   # replaced by backend.use()

def _angles(p3):
    pm = xp.sqrt(xp.sum(p3**2, axis=1))
    ok = pm > 1e-14
    pz = xp.where(ok, p3[:, 2], 1.0); px = xp.where(ok, p3[:, 0], 0.0); py = xp.where(ok, p3[:, 1], 0.0)
    pmm = xp.where(ok, pm, 1.0)
    theta = xp.arccos(xp.clip(pz / pmm, -1, 1)); phi = xp.arctan2(py, px)
    return pm, theta, phi

def _chi(theta, phi, h):
    """helicity 2-spinor chi_h, h = +-1 arrays -> (B,2)"""
    c = xp.cos(theta / 2); s = xp.sin(theta / 2)
    up = xp.stack([c + 0j, s * xp.exp(1j * phi)], axis=1)
    dn = xp.stack([-s * xp.exp(-1j * phi), c + 0j], axis=1)
    return xp.where((h == 1)[:, None], up, dn)

def u_spinor(p, h, m):
    E = xp.real(p[:, 0]); p3 = xp.real(p[:, 1:4])
    pm, th, ph = _angles(p3); chi = _chi(th, ph, h)
    a = xp.sqrt(E + m); kappa = xp.where(E + m > 0, h * pm / xp.where(E + m > 0, E + m, 1.0), 0.0)
    return xp.concatenate([a[:, None] * chi, (a * kappa)[:, None] * chi], axis=1)

def v_spinor(p, h, m):
    E = xp.real(p[:, 0]); p3 = xp.real(p[:, 1:4])
    pm, th, ph = _angles(p3); chi = _chi(th, ph, -h)
    a = xp.sqrt(E + m); kappa = xp.where(E + m > 0, -h * pm / xp.where(E + m > 0, E + m, 1.0), 0.0)
    return xp.concatenate([(a * kappa)[:, None] * chi, a[:, None] * chi], axis=1)

def polarization_vector(p, lam, m):
    """lam array in {+1,-1,0}; longitudinal only meaningful for m>0."""
    E = xp.real(p[:, 0]); p3 = xp.real(p[:, 1:4])
    pm = xp.sqrt(xp.sum(p3**2, axis=1)); ok = pm > 1e-14
    phat = xp.where(ok[:, None], p3 / xp.where(ok, pm, 1.0)[:, None], xp.array([0.0, 0.0, 1.0]))
    ref = xp.where((xp.abs(phat[:, :1]) < 0.9), xp.array([1.0, 0.0, 0.0]), xp.array([0.0, 1.0, 0.0]))
    e1 = xp.cross(ref, phat); e1 = e1 / xp.sqrt(xp.sum(e1**2, axis=1))[:, None]
    e2 = xp.cross(phat, e1)
    mm = m if m > 0 else 1.0
    long = xp.concatenate([(pm / mm)[:, None], (E / mm)[:, None] * phat], axis=1) + 0j
    sign = xp.where(lam == 1, -1.0, 1.0)
    space = sign[:, None] * (e1 + 1j * lam[:, None] * e2) / xp.sqrt(2.0)
    trans = xp.concatenate([xp.zeros((p.shape[0], 1)) + 0j, space], axis=1)
    return xp.where((lam == 0)[:, None], long, trans)
