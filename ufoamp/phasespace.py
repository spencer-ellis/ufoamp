"""
phasespace.py — RAMBO flat phase-space generator (Kleiss, Stirling, Ellis),
with the standard massive extension.  Used for validation points and for Monte
Carlo integration.  Returns momenta in the CM frame with total (sqrt_s,0,0,0).
"""
import numpy as np

def rambo(sqrt_s, masses, rng):
    n = len(masses)
    # massless isotropic momenta
    q = np.zeros((n, 4))
    for i in range(n):
        c = 2*rng.random()-1; phi = 2*np.pi*rng.random()
        E = -np.log(rng.random()*rng.random())
        s = np.sqrt(1-c*c)
        q[i] = [E, E*s*np.cos(phi), E*s*np.sin(phi), E*c]
    Q = q.sum(axis=0)
    Mq = np.sqrt(Q[0]**2 - Q[1:]@Q[1:])
    b = -Q[1:]/Mq; x = sqrt_s/Mq; g = Q[0]/Mq; a = 1/(1+g)
    p = np.zeros((n, 4))
    for i in range(n):
        bq = b @ q[i, 1:]
        p[i, 0] = x*(g*q[i, 0] + bq)
        p[i, 1:] = x*(q[i, 1:] + b*q[i, 0] + a*bq*b)
    masses = np.asarray(masses, float)
    if np.all(masses == 0):
        return p
    # massive: find xi with sum sqrt(m^2 + xi^2 p^2) = sqrt_s
    pm = np.linalg.norm(p[:, 1:], axis=1)
    xi = 1.0
    for _ in range(50):
        E = np.sqrt(masses**2 + (xi*pm)**2)
        f = E.sum() - sqrt_s
        df = (xi*pm**2/E).sum()
        xi -= f/df
        if abs(f) < 1e-12: break
    out = np.zeros((n, 4))
    out[:, 1:] = xi*p[:, 1:]
    out[:, 0] = np.sqrt(masses**2 + (xi*pm)**2)
    return out
