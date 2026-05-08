"""
PTA window functions and GW response sampler.

Window functions map (f_source, f_mode, T_obs) -> complex weight.
Available: sinc, tm (timing-model), tophat, whitened.
"""

import os
import time

import numpy as np
from scipy.interpolate import interp1d

R_MEAN_SQ = 4.0 / 15.0  # <|R|^2> in the fL >> 1 limit


def w_sinc(f, fk, T):
    """Standard sinc window."""
    return np.sinc(T * (f - fk))


def w_tm(f, fk, T):
    """Timing-model-subtracted sinc window."""
    f   = np.asarray(f, dtype=float)
    pT  = np.pi * T
    arg = pT * (f - fk)
    sin_bracket = (  3/(pT**3*f**2*fk) - 15/(pT**3*f*fk**2) + 45/(pT**5*f**3*fk**2))
    cos_bracket = (  3/(pT**2*f*fk)    + 45/(pT**4*f**2*fk**2))
    return np.sinc(T*(f-fk)) + np.sin(arg)*sin_bracket - np.cos(arg)*cos_bracket


def w_tophat(f, fk, T):
    mask = (np.abs(f - fk) < 0.5 / T)
    # return mask.astype(float) * (f / fk)
    return mask.astype(float)




def w_whitened(f, fk, T):
    """Spectrally whitened sinc window."""
    f = np.asarray(f, dtype=float)
    return (np.abs(f)/fk)**(13/6) * np.sinc(T*(f-fk))


WINDOWS = {
    'sinc':     w_sinc,
    'tm':       w_tm,
    'tophat':   w_tophat,
    'whitened': w_whitened,
}


def _build_R_sampler(n_r=2000, n_z=300, n_psi=300):
    """
    Tabulate the inverse CDF of |R| from the paper's double-integral formula
    and return a linear-interpolation sampler.

    p(|R|) = (4/π²) ∫₀¹ dz ∫₀^{π/4} dψ Θ(2|T|−|R|) arccosh(2|T|/|R|) / |T|
    |T|² = (1/8)[1 + 6z² + z⁴ + (1−z²)² cos(4ψ)],  z = cos ι

    Valid in the fL ≫ 1 limit; satisfies ⟨|R|²⟩ = 4/15.
    The result is cached alongside this module as ``_R_inv_cdf.npz``.
    """
    _cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_R_inv_cdf.npz')
    if os.path.exists(_cache):
        d = np.load(_cache)
        cdf_a = d['cdf'].astype(np.float64)
        r_a   = d['r'].astype(np.float64)
        return (interp1d(cdf_a, r_a, kind='linear',
                         bounds_error=False, fill_value=(0.0, 2.0)),
                cdf_a, r_a)

    print('  gwadpy: building |R| CDF table ...', end='', flush=True)
    t0 = time.perf_counter()

    z   = np.linspace(0, 1, n_z)
    psi = np.linspace(0, np.pi / 4, n_psi)
    Z, PSI = np.meshgrid(z, psi, indexing='ij')          # (n_z, n_psi)
    T2 = (1/8) * (1 + 6*Z**2 + Z**4 + (1 - Z**2)**2 * np.cos(4*PSI))
    T  = np.sqrt(np.maximum(T2, 0.0))
    norm = (4 / np.pi**2) * (z[1] - z[0]) * (psi[1] - psi[0])

    r_grid = np.linspace(1e-4, 2.0, n_r)
    pdf    = np.zeros(n_r)
    for i, r in enumerate(r_grid):
        mask  = (2*T > r) & (T > 0)
        ratio = np.where(mask, 2*T / r, 1.0)
        intgd = np.where(mask, np.arccosh(ratio) / T, 0.0)
        pdf[i] = norm * intgd.sum()

    dr  = r_grid[1] - r_grid[0]
    cdf = np.concatenate([[0.0], np.cumsum(pdf) * dr])
    r_f = np.concatenate([[0.0], r_grid])
    cdf /= cdf[-1]

    np.savez(_cache, r=r_f, cdf=cdf)
    print(f' done ({time.perf_counter() - t0:.1f} s).')
    return (interp1d(cdf, r_f, kind='linear', bounds_error=False, fill_value=(0.0, 2.0)),
            cdf.astype(np.float64), r_f.astype(np.float64))


_R_inv_cdf, _R_CDF_AXIS, _R_VALS_AXIS = _build_R_sampler()


def sample_absR(n_samples, rng=None):
    """Sample |R| from the paper's exact distribution (fL ≫ 1 limit, ⟨|R|²⟩=4/15)."""
    _rng = rng if rng is not None else np.random
    return _R_inv_cdf(_rng.random(n_samples)).astype(float)


def sample_R(n_samples, rng=None):
    """Sample complex GW antenna response R over random sky/polarisation/inclination."""
    _rng = rng if rng is not None else np.random
    r         = _rng.random((n_samples, 4))
    cos_theta = r[:, 0] * 2.0 - 1.0
    psi       = r[:, 1] * np.pi
    cos_iota  = r[:, 2] * 2.0 - 1.0
    phi       = r[:, 3] * (2.0 * np.pi)
    sin2_half = (1 - cos_theta) / 2
    Fp  = sin2_half * np.cos(2*psi)
    Fx  = sin2_half * np.sin(2*psi)
    pol = (1 + cos_iota**2)/2 * Fp - 1j*cos_iota*Fx
    return (1.0 - np.exp(-1j*phi)) * pol
