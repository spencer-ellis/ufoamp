"""backend.py — switch the numerical array library (numpy | jax)."""
import numpy as _np
_current = "numpy"
xp = _np

def use(name: str):
    """Select 'numpy' (default) or 'jax'. Affects vertex_eval, recursion, process."""
    global _current, xp
    if name == "jax":
        import jax, jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        xp = jnp
    elif name == "numpy":
        xp = _np
    else:
        raise ValueError(name)
    _current = name
    from . import vertex_eval, recursion, process, wf_batched
    for mod in (vertex_eval, recursion, process, wf_batched):
        mod.xp = xp

def current() -> str:
    return _current
