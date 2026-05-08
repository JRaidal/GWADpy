"""
GW amplitude distribution (GWAD): dN/(dA d ln f).

BrokenPowerLawGWAD  — frequency-independent broken power law (direct parameters).
_gwad_density       — cosmological integral for ModelI / ModelII.
calculate_gwad      — public interface; returns density and number per bin.
"""

import time
import threading
import numpy as np
from scipy.integrate import simpson
from numpy.polynomial.legendre import leggauss

from .constants import KPC_TO_SECONDS, TSun, YEAR_IN_SEC
from .cosmology import DVc, DLz, m1m2, residence_time
from .merger_rates import ModelI, ModelII

# ── Optional fine-grained profiling ──────────────────────────────────────────
# Call enable_gwad_profiling() before precompute, then print_gwad_profile() after.
_prof_lock    = threading.Lock()
_prof_enabled = False
_prof_counts  = {}
_prof_totals  = {}

def enable_gwad_profiling():
    global _prof_enabled
    with _prof_lock:
        _prof_enabled = True
        _prof_counts.clear()
        _prof_totals.clear()

def print_gwad_profile():
    with _prof_lock:
        if not _prof_totals:
            print("  (no gwad profile data)")
            return
        n = max(_prof_counts.values())
        total = sum(_prof_totals.values())
        print(f"\n  _gwad_density profile  ({n} calls, Σ={total*1e3:.1f} ms/call × n_calls):")
        keys = ('f_b+Mc_2d', 'tau_2d', 'R_eff_eval', 'integrate_2d',
                'broadcast+m1m2', 'rate_model', 'integrand', 'simpson')
        for k in keys:
            v = _prof_totals.get(k, 0.0)
            if v > 0 or k in ('f_b+Mc_2d', 'tau_2d'):
                print(f"    {k:<20s}: {v/n*1e3:7.2f} ms/call  ({100*v/total:.1f}%)")

_GL_XI, _GL_WI = leggauss(2)

# ── Cosmological grids (module-level, built once) ─────────────────────────────
_LOGZ_GWAD  = np.linspace(np.log10(1e-9), np.log10(8.0), 150)
_Z_GWAD     = 10.0 ** _LOGZ_GWAD
_DV_GWAD    = DVc(_Z_GWAD) * 1e-9 * _Z_GWAD * np.log(10) / (1.0 + _Z_GWAD)
_DL_SEC_GWAD = DLz(_Z_GWAD) * KPC_TO_SECONDS
_ETA_GWAD   = np.linspace(1e-3, 0.249, 40)
_DISC_GWAD  = 1.0 - 4.0 * _ETA_GWAD
_KERN_GWAD  = np.where(_DISC_GWAD > 1e-10,
                       1.0 / (_ETA_GWAD * np.sqrt(_DISC_GWAD)),
                       0.0)

# ── m1/m2 coefficients: m1 = Mc·_M1C_GWAD[k], m2 = Mc·_M2C_GWAD[k] ──────────
_s_eta    = np.sqrt(np.maximum(1.0 - 4.0 * _ETA_GWAD, 0.0))
_A_term   = 1.0 + _s_eta + _ETA_GWAD * (-5.0 - 3.0*_s_eta + (5.0 + _s_eta) * _ETA_GWAD)
_A5_eta   = np.power(_A_term, 0.2)
_21_5     = np.power(2.0, 0.2)
_M1C_GWAD = _A5_eta / (_21_5 * np.power(_ETA_GWAD, 0.6))
_M2C_GWAD = -((-1.0 + _s_eta + 2.0 * _ETA_GWAD) * _A5_eta) / (
              2.0 * _21_5 * np.power(_ETA_GWAD, 1.6))
del _s_eta, _A_term, _A5_eta, _21_5

# ── Trapz weights for the (z, η) double integration ──────────────────────────
_dlogz     = _LOGZ_GWAD[1] - _LOGZ_GWAD[0]
_W_Z_GWAD  = np.full(len(_LOGZ_GWAD), _dlogz); _W_Z_GWAD[0] *= 0.5; _W_Z_GWAD[-1] *= 0.5
_deta      = _ETA_GWAD[1] - _ETA_GWAD[0]
_W_ETA_GWAD = np.full(len(_ETA_GWAD), _deta); _W_ETA_GWAD[0] *= 0.5; _W_ETA_GWAD[-1] *= 0.5
_W2D_GWAD  = _W_Z_GWAD[:, None] * _W_ETA_GWAD[None, :]   # (n_z, n_eta)


class BrokenPowerLawGWAD:
    r"""
    Frequency-independent broken power-law:
    dN/dA = N_b (p+q)^s / [ q (A/A_b)^{p/s} + p (A/A_b)^{q/s} ]^s
    """
    def __init__(self, N_b, A_b, p, q, s=2.0):
        self.N_b=N_b; self.A_b=A_b; self.p=p; self.q=q; self.s=s

    def __call__(self, A_array, f=None):
        A = np.asarray(A_array, dtype=float)
        x = A / self.A_b
        denom = (self.q * x**(self.p/self.s) + self.p * x**(self.q/self.s))**self.s
        return self.N_b * (self.p + self.q)**self.s / denom


def _gwad_density(A_array, f, rate_model, env_params, z_min=0.0):
    """Compute dN/(dA d ln f) at a single frequency f."""
    if isinstance(rate_model, BrokenPowerLawGWAD):
        return rate_model(A_array)

    f_ref     = (env_params or {}).get('f_ref', 1e-20)
    alpha_env = (env_params or {}).get('alpha', 8/3)
    beta_env  = (env_params or {}).get('beta',  5/8)

    logz     = _LOGZ_GWAD
    Z        = _Z_GWAD
    dV_dlogz = _DV_GWAD
    DL_sec   = _DL_SEC_GWAD

    def _t():
        return time.perf_counter() if _prof_enabled else 0.0

    def _acc(key, t0):
        if _prof_enabled:
            with _prof_lock:
                _prof_totals[key]  = _prof_totals.get(key, 0.0)  + time.perf_counter() - t0
                _prof_counts[key]  = _prof_counts.get(key, 0)    + 1

    t0 = _t()
    f_b    = f * (1.0 + Z) / 2.0
    A_arr  = np.asarray(A_array)
    Mc_2d  = ((A_arr[:, None] * DL_sec[None, :]) /
              (4.0 * (1.0 + Z[None, :]) * (2.0*np.pi*f_b[None, :])**(2.0/3.0)))**(3.0/5.0) / TSun
    _acc('f_b+Mc_2d', t0)

    t0 = _t()
    tau_2d = residence_time(f_b[None, :], Mc_2d, Z[None, :],
                            f_ref=f_ref, alpha=alpha_env,
                            beta=beta_env) / YEAR_IN_SEC
    _acc('tau_2d', t0)

    if isinstance(rate_model, ModelII):
        rate      = rate_model(Mc_2d, np.broadcast_to(Z[None, :], Mc_2d.shape))
        integrand = rate * (0.6 * Mc_2d / A_arr[:, None]) * tau_2d * dV_dlogz[None, :]
        return simpson(integrand, x=logz, axis=1)

    elif hasattr(rate_model, 'R_eff_eval'):
        # ModelI fast path: η-integrated rate precomputed as 2-D table.
        t0 = _t()
        R_eff_2d = rate_model.R_eff_eval(Mc_2d, Z)
        _acc('R_eff_eval', t0)

        t0 = _t()
        integrand_2d = R_eff_2d * (0.6 / A_arr[:, None]) * tau_2d * dV_dlogz[None, :]
        result = integrand_2d @ _W_Z_GWAD
        _acc('integrate_2d', t0)
        return result

    else:  # ModelI without R_eff precomputed (fallback)
        t0 = _t()
        Mc_3d  = Mc_2d[:, :, None]
        Z_3d   = np.broadcast_to(Z[None, :, None], (*Mc_2d.shape, len(_ETA_GWAD)))
        m1_3d  = Mc_3d * _M1C_GWAD[None, None, :]
        m2_3d  = Mc_3d * _M2C_GWAD[None, None, :]
        _acc('broadcast+m1m2', t0)

        t0 = _t()
        rate = rate_model(m1_3d, m2_3d, Z_3d)
        _acc('rate_model', t0)

        t0 = _t()
        integrand = (_KERN_GWAD[None, None, :] * rate
                     * (0.6 / A_arr[:, None, None])
                     * tau_2d[:, :, None]
                     * dV_dlogz[None, :, None])
        _acc('integrand', t0)

        t0 = _t()
        result = np.einsum('ijk,jk->i', integrand, _W2D_GWAD)
        _acc('simpson', t0)
        return result


def calculate_gwad(A_array, f_obs, rate_model, env_params=None, z_min=0.0, f_width=2e-9):
    """Compute dN/(dA d ln f) at f_obs; returns dict with 'density' and 'number'."""
    f_min = max(f_obs - f_width/2.0, f_obs*1e-3)
    f_max = f_obs + f_width/2.0

    if isinstance(rate_model, BrokenPowerLawGWAD):
        density   = rate_model(A_array)
        delta_lnf = np.log(f_max) - np.log(f_min)
        return {'density': density, 'number': density * delta_lnf}

    density  = _gwad_density(A_array, f_obs, rate_model, env_params, z_min)
    ln_lo, ln_hi = np.log(f_min), np.log(f_max)
    f_quad = np.exp(0.5*(ln_lo+ln_hi) + 0.5*(ln_hi-ln_lo)*_GL_XI)
    w_quad = 0.5 * (ln_hi - ln_lo) * _GL_WI
    number = np.zeros(len(A_array))
    for fq, wq in zip(f_quad, w_quad):
        number += wq * _gwad_density(A_array, fq, rate_model, env_params, z_min)
    return {'density': density, 'number': number}
