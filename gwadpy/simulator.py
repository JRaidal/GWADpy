"""
GlobalResidualsSimulator: Monte Carlo simulator for PTA timing residuals.
Pre-computes per-bin GWAD statistics; at run time draws strong sources explicitly
and adds the weak background as Gaussian noise.
"""

import os
import hashlib
import pickle
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.random import default_rng as _default_rng
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import interp1d

from .gwad import calculate_gwad, _gwad_density
from .windows import (sample_absR, R_MEAN_SQ, WINDOWS,
                      _R_CDF_AXIS, _R_VALS_AXIS)
from ._nb_kernels import (NUMBA_AVAILABLE, nb_accumulate_strong,
                          nb_accumulate_tail, warmup as _nb_warmup)

_GL_XI_SIM, _GL_WI_SIM = leggauss(8)


@contextmanager
def _timed(label, timings):
    """Simple context-manager timer; appends elapsed seconds to timings[label]."""
    t0 = time.perf_counter()
    yield
    timings.setdefault(label, 0.0)
    timings[label] += time.perf_counter() - t0


class GlobalResidualsSimulator:
    def __init__(self, gwad_model, env_params, T_obs, n_modes,
                 source_f_edges, A_common, window_fn=None,
                 cache_dir='.bin_cache'):
        self.gwad_model     = gwad_model
        self.env_params     = env_params or {}
        self.T_obs          = T_obs
        self.n_modes        = n_modes
        self.f_obs          = np.arange(1, n_modes + 1) / T_obs
        self.source_f_edges = np.asarray(source_f_edges)
        self.bin_centers    = 0.5 * (source_f_edges[:-1] + source_f_edges[1:])
        self.n_bins         = len(self.bin_centers)
        self.A_common       = np.asarray(A_common)
        self.window_fn      = window_fn
        self.cache_dir      = cache_dir
        self._bin_cache       = None
        self._cached_n_strong = None
        # Infer window name for numba dispatch (None → numpy fallback)
        self.window_name = next(
            (name for name, fn in WINDOWS.items() if fn is window_fn), None)

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, n_strong):
        h = hashlib.md5()
        h.update(b"v18_gwad_dens")
        h.update(self.source_f_edges.tobytes())
        h.update(self.A_common.tobytes())
        h.update(self.f_obs.tobytes())
        h.update(f"{n_strong}|{self.T_obs}".encode())
        h.update(repr(sorted(self.env_params.items())).encode())
        h.update((self.window_name or 'custom').encode())
        try:
            tm = np.array([1e6, 1e8, 1e9, 1e11, 1e12])
            tz = np.array([0.1, 0.5, 1.0, 2.0,  4.0])
            h.update(np.asarray(self.gwad_model(tm, tm, tz)).tobytes())
        except Exception:
            pass
        return h.hexdigest()

    def _cache_path(self, n_strong):
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"bin_stats_{self._cache_key(n_strong)}.pkl")

    # ── Per-bin statistics ────────────────────────────────────────────────────

    def _compute_single_bin(self, i, n_strong):
        flo = self.source_f_edges[i]; fhi = self.source_f_edges[i + 1]
        fc  = 0.5 * (flo + fhi)
        A   = self.A_common

        # ── Tophat early exit ─────────────────────────────────────────────────
        if self.window_name == 'tophat':
            half_bw = 0.5 / self.T_obs
            if not np.any((self.f_obs - half_bw < fhi) & (self.f_obs + half_bw > flo)):
                return dict(idx=i, fc=fc, flo=flo, fhi=fhi,
                            A_th=A[-1], N_tot=0.0, N_strong=0.0,
                            strong_cdf=None, strong_A_arr=None,
                            C_fs=0.0,
                            sigma2_weak_per_mode=np.zeros(self.n_modes),
                            delta_lnf=float(np.log(fhi / flo)),
                            gwad_log_A=np.array([]), gwad_log_D=np.array([]))

        gwad  = calculate_gwad(A, fc, self.gwad_model,
                               env_params=self.env_params,
                               f_width=fhi - flo)
        dN_dA     = gwad['number']
        delta_lnf = float((gwad['number'] / gwad['density'])[0])

        # ── Fine log-log grid for N(>A), CDF, and C_fs ───────────────────────
        _pos = dN_dA > 0
        if _pos.sum() >= 2:
            _fi    = interp1d(np.log(A[_pos]), np.log(dN_dA[_pos]),
                              kind='linear', bounds_error=False)
            A_fine  = np.logspace(np.log10(A[0]), np.log10(A[-1]), 2000)
            dN_fine = np.exp(_fi(np.log(A_fine)))   # NaN outside support → 0 for nanmax
        else:
            A_fine  = A.copy()
            dN_fine = dN_dA.copy()

        # N(<A) = ∫ A dN/dA d(lnA)  — cumulative trapezoid in log space
        N_lt = cumulative_trapezoid(dN_fine * A_fine, x=np.log(A_fine), initial=0)
        N_gt = N_lt[-1] - N_lt
        N_tot = float(N_lt[-1])

        # ── Threshold amplitude A_th ──────────────────────────────────────────
        if N_tot <= n_strong:
            A_th = float(A_fine[0])
        elif N_gt[-1] >= n_strong:
            A_th = float(A_fine[-1])
        else:
            A_th = float(np.exp(interp1d(
                np.log(np.maximum(N_gt[::-1], 1e-300)),
                np.log(A_fine[::-1]), kind='linear',
                bounds_error=False,
                fill_value=(np.log(A_fine[-1]), np.log(A_fine[0]))
            )(np.log(n_strong))))

        # ── Strong-source CDF on fine grid ────────────────────────────────────
        strong_cdf = None; strong_A_arr = None
        mask_s = A_fine >= A_th
        if np.any(mask_s) and int(n_strong) > 0:
            cdf = cumulative_trapezoid(dN_fine[mask_s] * A_fine[mask_s],
                                       x=np.log(A_fine[mask_s]), initial=0)
            if cdf[-1] > 0:
                strong_cdf   = cdf / cdf[-1]
                strong_A_arr = A_fine[mask_s]

        # ── C_fs = max_A [A^4 · dN/dA] on fine grid (nanmax ignores NaN fill) ─
        C_fs = float(np.nanmax(A_fine**4 * dN_fine))

        # ── sigma2_weak: GL quadrature over log(f), simpson on coarse A grid ──
        lnlo      = np.log(flo); lnhi = np.log(fhi)
        half_dlnf = 0.5 * (lnhi - lnlo)
        f_quad    = np.exp(0.5*(lnlo+lnhi) + 0.5*(lnhi-lnlo)*_GL_XI_SIM)
        sigma2_weak_per_mode = np.zeros(self.n_modes)

        mask_weak = A < A_th
        for qi in range(len(_GL_XI_SIM)):
            fq = f_quad[qi]
            wq = _GL_WI_SIM[qi]
            if not np.any(mask_weak):
                continue
            w2 = np.abs(self.window_fn(fq, self.f_obs, self.T_obs))**2
            if not np.any(w2 > 0):
                continue
            dens_q   = _gwad_density(A, fq, self.gwad_model, self.env_params, z_min=0.0)
            sum_A2_q = simpson(dens_q[mask_weak] * A[mask_weak]**3,
                               x=np.log(A[mask_weak]))
            sigma2_weak_per_mode += wq * half_dlnf * sum_A2_q / (4*np.pi*fq)**2 * w2

        idx_th   = min(np.searchsorted(A_fine, A_th), len(N_gt) - 1)
        N_strong = float(N_gt[idx_th])

        _dens_c = gwad['density']
        _pos_c  = _dens_c > 0
        gwad_log_A = np.log(A[_pos_c])       if _pos_c.any() else np.array([])
        gwad_log_D = np.log(_dens_c[_pos_c]) if _pos_c.any() else np.array([])

        return dict(idx=i, fc=fc, flo=flo, fhi=fhi,
                    A_th=A_th, N_tot=N_tot, N_strong=N_strong,
                    strong_cdf=strong_cdf, strong_A_arr=strong_A_arr,
                    C_fs=C_fs,
                    sigma2_weak_per_mode=sigma2_weak_per_mode,
                    delta_lnf=delta_lnf,
                    gwad_log_A=gwad_log_A, gwad_log_D=gwad_log_D)

    def precompute_bin_stats(self, n_strong, n_workers=None, force=False):
        """Pre-compute and cache per-bin amplitude statistics."""
        cache_path = self._cache_path(n_strong)
        if not force and os.path.exists(cache_path):
            with open(cache_path, 'rb') as fh:
                self._bin_cache = pickle.load(fh)
            self._cached_n_strong = n_strong
            print(f"  Bin stats: {len(self._bin_cache)} bins loaded from cache.")
            if NUMBA_AVAILABLE and self.window_name is not None:
                _nb_warmup(self.window_name, self.n_modes)
            return

        n_w = n_workers or min(self.n_bins, os.cpu_count() or 4)
        print(f"  Precomputing {self.n_bins} bins (n_workers={n_w}) ...", end='', flush=True)
        t0 = time.perf_counter()

        self._bin_cache = [None] * self.n_bins
        with ThreadPoolExecutor(max_workers=n_w) as ex:
            futs = {ex.submit(self._compute_single_bin, i, n_strong): i
                    for i in range(self.n_bins)}
            for fut in as_completed(futs):
                r = fut.result()
                self._bin_cache[r['idx']] = r

        print(f" done ({time.perf_counter()-t0:.1f} s)")
        self._cached_n_strong = n_strong
        with open(cache_path, 'wb') as fh:
            pickle.dump(self._bin_cache, fh)
        if NUMBA_AVAILABLE and self.window_name is not None:
            _nb_warmup(self.window_name, self.n_modes)

    # ── Monte Carlo draw ──────────────────────────────────────────────────────

    @staticmethod
    def _process_strong_and_tail(stat, n_real, f_obs, T_obs,
                                  window_fn, window_name, chunk_size,
                                  n_tail_samples, rng):
        """Compute strong-source residuals for one bin (Poisson draws per realisation)."""
        n_modes = len(f_obs)
        res_s_local = np.zeros((n_real, n_modes), dtype=complex)
        tn_local    = np.zeros(n_modes)  # always zero; tail norm computed separately

        flo, fhi = stat['flo'], stat['fhi']

        # ── Tophat early exit ─────────────────────────────────────────────────
        if window_name == 'tophat':
            half_bw = 0.5 / T_obs
            if not np.any((f_obs - half_bw < fhi) & (f_obs + half_bw > flo)):
                return res_s_local, tn_local

        # ── Strong sources: Poisson(N_strong) per realisation ─────────────────
        N_mean = stat['N_strong']
        if N_mean > 0 and stat['strong_cdf'] is not None:
            log_flo = np.log(flo); log_fhi = np.log(fhi)
            n_src_arr = rng.poisson(N_mean, size=n_real).astype(np.int64)

            if NUMBA_AVAILABLE and window_name is not None:
                nb_accumulate_strong(
                    res_s_local, n_src_arr, f_obs, T_obs,
                    log_flo, log_fhi,
                    stat['strong_cdf'], stat['strong_A_arr'],
                    _R_CDF_AXIS, _R_VALS_AXIS,
                    window_name)
            else:
                # Numpy fallback: pad each chunk to max Poisson draw
                cdf_xp = stat['strong_cdf']
                cdf_fp = stat['strong_A_arr']
                for start in range(0, n_real, chunk_size):
                    end   = min(start + chunk_size, n_real)
                    nc    = end - start
                    n_src = n_src_arr[start:end]
                    n_max = int(n_src.max()) if n_src.max() > 0 else 0
                    if n_max == 0:
                        continue
                    u    = rng.random((nc, n_max))
                    A_s  = np.interp(u, cdf_xp, cdf_fp)
                    f_s  = np.exp(log_flo + rng.random((nc, n_max)) * (log_fhi - log_flo))
                    absR = sample_absR(nc * n_max, rng=rng).reshape(nc, n_max)
                    dbar = rng.random((nc, n_max)) * (2 * np.pi)
                    slot_mask = np.arange(n_max)[None, :] < n_src[:, None]
                    A_s  = A_s * slot_mask
                    f_s3 = f_s[:, :, None]; fo3 = f_obs[None, None, :]
                    w_p  = window_fn( f_s3, fo3, T_obs)
                    w_m  = window_fn(-f_s3, fo3, T_obs)
                    pref = A_s * absR / (4*np.pi*1j*f_s)
                    ep   = np.exp(1j*dbar); em = np.conj(ep)
                    res_s_local[start:end] += (
                        pref[:, :, None] * (ep[:, :, None]*w_p - em[:, :, None]*w_m)
                    ).sum(axis=1)

        return res_s_local, tn_local

    def compute_gwad_pdf(self, x_grid, ki, N_R=300, n_f_pts=10):
        """
        Evaluate dP/d ln|δt_k| via the GWAD integral, using dN/dA from _bin_cache.

        dP/d ln x ≈ ∫ d|R| p(|R|) ∫ d ln f  A* · dN/(dA d ln f)|_{A*=4πfx/(|R||w_k⁺(f)|}
        """
        fk     = self.f_obs[ki]
        x_grid = np.asarray(x_grid, dtype=float)
        N_x    = len(x_grid)

        # Build per-bin data from cached log-density arrays
        bin_data = []
        for s in self._bin_cache:
            if s['N_tot'] <= 0 or s['C_fs'] <= 0:
                continue
            logA = s['gwad_log_A'];  logD = s['gwad_log_D']
            if len(logA) < 2:
                continue
            bin_data.append({
                'flo': s['flo'], 'fhi': s['fhi'],
                'logA': logA, 'logD': logD,
                'A_min': np.exp(logA[0]),
                'A_max': np.exp(logA[-1]),
            })

        R_samp = sample_absR(N_R)
        pdf    = np.zeros(N_x)

        for bd in bin_data:
            flo, fhi = bd['flo'], bd['fhi']
            logA = bd['logA'];  logD = bd['logD']

            f_pts  = np.linspace(flo, fhi, n_f_pts)
            w_abs  = np.abs(self.window_fn(f_pts, fk, self.T_obs))  # (N_f,)

            # Loop over f-points; each iteration works on (N_x, N_R) arrays.
            # np.interp on 250k elements is ~10× faster than interp1d.
            integrand_lnf = np.zeros((N_x, n_f_pts))
            for fi, (fq, wq) in enumerate(zip(f_pts, w_abs)):
                if wq <= 0:
                    continue
                A_star = (4.0 * np.pi * fq / wq) * x_grid[:, None] / R_samp[None, :]
                valid  = (A_star >= bd['A_min']) & (A_star <= bd['A_max'])
                logA_s = np.log(np.where(A_star > 0, A_star, 1.0))
                logD_v = np.interp(logA_s.ravel(), logA, logD,
                                   left=logD[0], right=-np.inf)
                dens   = np.where(valid, np.exp(logD_v.reshape(N_x, N_R)), 0.0)
                integrand_lnf[:, fi] = np.mean(A_star * dens, axis=1)

            pdf += np.trapz(integrand_lnf, x=np.log(f_pts), axis=1)

        return pdf

    def compute_tail_norm(self, n_pts=40):
        """
        Compute tail normalisation Λ_k deterministically:
            Λ_k = (1/256π³) Σ_bins ∫_flo^fhi df/f^4 · C(f) · |w_k^+(f)|^3
        Returns (n_modes,) array of Λ_k [s^3].
        """
        tn = np.zeros(self.n_modes)
        for s in self._bin_cache:
            if s['C_fs'] <= 0:
                continue
            flo, fhi = s['flo'], s['fhi']
            C_f   = s['C_fs'] / np.log(fhi / flo)   # plateau per unit ln-f
            f_pts = np.linspace(flo, fhi, n_pts)
            for ki, fk in enumerate(self.f_obs):
                w3 = np.abs(self.window_fn(f_pts, fk, self.T_obs))**3
                tn[ki] += np.trapezoid(C_f / f_pts**4 * w3, f_pts)
        return tn / (256.0 * np.pi**3)

    def get_residuals(self, n_real, n_strong=None, n_workers=None,
                      chunk_size=1000, n_tail_samples=None, verbose=True):
        """
        Simulate n_real realisations of the GW timing residual vector.
        Returns (res, res_strong, res_weak, tail_norm).
        """
        timings = {}

        if n_strong is None:
            n_strong = self._cached_n_strong
        if self._bin_cache is None or self._cached_n_strong != n_strong:
            self.precompute_bin_stats(n_strong)

        # ── Weak sources ──────────────────────────────────────────────────────
        with _timed('weak', timings):
            total_sigma2 = sum(s['sigma2_weak_per_mode'] for s in self._bin_cache)
            sigma_k = np.sqrt(np.maximum(total_sigma2 * R_MEAN_SQ / 2.0, 0.0))
            res_w = (np.random.standard_normal((n_real, self.n_modes)) +
                     1j * np.random.standard_normal((n_real, self.n_modes))) * sigma_k[None, :]

        # ── Strong sources + tail: parallelised via numba (numpy fallback) ──────
        if NUMBA_AVAILABLE and self.window_name is not None:
            n_w = 1
        else:
            n_w = n_workers if n_workers is not None else (os.cpu_count() or 4)
            n_w = max(1, min(n_w, self.n_bins))

        res_s = np.zeros((n_real, self.n_modes), dtype=complex)

        bin_rngs = [_default_rng() for _ in self._bin_cache]

        with _timed('strong', timings):
            with ThreadPoolExecutor(max_workers=n_w) as ex:
                futs = [ex.submit(self._process_strong_and_tail,
                                  stat, n_real,
                                  self.f_obs, self.T_obs, self.window_fn,
                                  self.window_name, chunk_size, n_tail_samples, rng)
                        for stat, rng in zip(self._bin_cache, bin_rngs)]
                for fut in as_completed(futs):
                    s_bin, _ = fut.result()
                    res_s += s_bin

        # ── Tail normalisation: deterministic window integral ─────────────────
        with _timed('tail_norm', timings):
            tn = self.compute_tail_norm()

        if verbose:
            total   = sum(timings.values())
            backend = 'numba' if NUMBA_AVAILABLE and self.window_name else 'numpy'
            print(f"  Simulation: {n_real:,} realisations in {total:.2f} s  "
                  f"(backend={backend})")

        return res_s + res_w, res_s, res_w, tn
