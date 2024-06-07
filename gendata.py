from typing import Callable
from pysindy.differentiation import SpectralDerivative
import numpy as np
import scipy


INTEGRATOR_KEYWORDS = {"rtol": 1e-12, "method": "LSODA", "atol": 1e-12}


def diffusion(t, u, dx, nx):
    u = np.reshape(u, nx)
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    return np.reshape(uxx, nx)


def burgers(t, u, dx, nx):
    u = np.reshape(u, nx)
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    ux = SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    return np.reshape((0.1 * uxx - u * ux), nx)


def ks(t, u, dx, nx):
    u = np.reshape(u, nx)
    ux = SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    uxxxx = SpectralDerivative(d=4, axis=0)._differentiate(u, dx)
    return np.reshape(-uxx - uxxxx - u * ux, nx)


def gen_data(
    rhs_func: Callable,
    init_cond: np.ndarray,
    args: tuple,
    dt: float = 0.01,
    t_end: int = 100,
):
    t = np.arange(0, t_end, dt)
    t_span = (t[0], t[-1])
    u = []
    u.append(
        scipy.integrate.solve_ivp(
            rhs_func,
            t_span,
            init_cond,
            t_eval=t,
            args=args,
            **INTEGRATOR_KEYWORDS,
        ).y.T
    )
    u = np.stack(u)
    u = np.squeeze(u)
    return u.T
