"""Numba-JIT kernels for strong-source and σ₀ Monte Carlo accumulation."""

import numpy as np

try:
    from numba import njit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ─── Everything inside the guard is only defined when numba is installed ──────

if NUMBA_AVAILABLE:

    # ── Helpers ───────────────────────────────────────────────────────────────

    @njit(cache=True)
    def _interp1(x, xp, fp):
        """Clamped binary-search linear interpolation (scalar)."""
        if x <= xp[0]:
            return fp[0]
        if x >= xp[-1]:
            return fp[-1]
        lo, hi = 0, len(xp) - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if xp[mid] <= x:
                lo = mid
            else:
                hi = mid
        t = (x - xp[lo]) / (xp[hi] - xp[lo])
        return fp[lo] + t * (fp[hi] - fp[lo])

    @njit(cache=True, fastmath=True)
    def _sample_absR(r_cdf, r_vals):
        """Sample |R| from the precomputed inverse CDF (paper's exact distribution)."""
        return _interp1(np.random.random(), r_cdf, r_vals)

    # ── Scalar window functions ───────────────────────────────────────────────

    @njit(cache=True, fastmath=True)
    def _w_tophat(f, fk, T):
        return 1.0 if abs(f - fk) < 0.5 / T else 0.0

    @njit(cache=True, fastmath=True)
    def _w_sinc(f, fk, T):
        px = np.pi * T * (f - fk)
        return 1.0 if abs(px) < 1e-10 else np.sin(px) / px

    @njit(cache=True, fastmath=True)
    def _w_whitened(f, fk, T):
        px = np.pi * T * (f - fk)
        s  = 1.0 if abs(px) < 1e-10 else np.sin(px) / px
        return (abs(f) / fk) ** (13.0 / 6.0) * s

    @njit(cache=True, fastmath=True)
    def _w_tm(f, fk, T):
        pT    = np.pi * T
        arg   = pT * (f - fk)
        sinc_v = 1.0 if abs(arg) < 1e-10 else np.sin(arg) / arg
        sin_b = (  3.0 / (pT**3 * f**2 * fk)
                 - 15.0 / (pT**3 * f  * fk**2)
                 + 45.0 / (pT**5 * f**3 * fk**2))
        cos_b = (  3.0 / (pT**2 * f  * fk)
                 + 45.0 / (pT**4 * f**2 * fk**2))
        return sinc_v + np.sin(arg) * sin_b - np.cos(arg) * cos_b

    # ── Core kernel (window function passed as first-class function) ──────────

    @njit(cache=False, parallel=True, fastmath=True)
    def _strong_core(res_s, n_src_arr, f_obs, T_obs,
                     log_flo, log_fhi, cdf_x, cdf_fp, r_cdf, r_vals, w_fn):
        """Accumulate strong-source residuals into res_s (n_real, n_modes), in-place."""
        n_real  = res_s.shape[0]
        n_modes = f_obs.shape[0]
        INV4PI  = 0.25 / np.pi
        TWO_PI  = 2.0 * np.pi

        for i in prange(n_real):
            for j in range(n_src_arr[i]):
                A    = _interp1(np.random.random(), cdf_x, cdf_fp)
                f_s  = np.exp(log_flo + np.random.random() * (log_fhi - log_flo))
                absR = _sample_absR(r_cdf, r_vals)
                dbar = np.random.random() * TWO_PI
                # pref = A|R|/(4πf): pref·e^{iδ} = P(sinδ − i·cosδ)
                P     = A * absR * INV4PI / f_s
                sin_d = np.sin(dbar)
                cos_d = np.cos(dbar)
                pe_r  =  P * sin_d    # real part of pref·e^{iδ}
                pe_i  = -P * cos_d    # imag part

                for k in range(n_modes):
                    fk = f_obs[k]
                    wp = w_fn( f_s, fk, T_obs)
                    wm = w_fn(-f_s, fk, T_obs)
                    res_s[i, k] += complex(pe_r * (wp + wm),
                                           pe_i * (wp - wm))

    # ── Per-window compiled instances (one specialisation per window) ─────────

    @njit(cache=False, fastmath=True)
    def _strong_tophat(res_s, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, r_cdf, r_vals):
        _strong_core(res_s, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, r_cdf, r_vals, _w_tophat)

    @njit(cache=False, fastmath=True)
    def _strong_sinc(res_s, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, r_cdf, r_vals):
        _strong_core(res_s, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, r_cdf, r_vals, _w_sinc)

    @njit(cache=False, fastmath=True)
    def _strong_whitened(res_s, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, r_cdf, r_vals):
        _strong_core(res_s, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, r_cdf, r_vals, _w_whitened)

    @njit(cache=False, fastmath=True)
    def _strong_tm(res_s, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, r_cdf, r_vals):
        _strong_core(res_s, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, r_cdf, r_vals, _w_tm)

    _NB_STRONG_DISPATCH = {
        'tophat':   _strong_tophat,
        'sinc':     _strong_sinc,
        'whitened': _strong_whitened,
        'tm':       _strong_tm,
    }

    # ── σ₀² kernel ────────────────────────────────────────────────────────────

    @njit(cache=False, parallel=True, fastmath=True)
    def _sigma2_core(s2_out, n_src_arr, f_obs, T_obs,
                     log_flo, log_fhi, cdf_x, cdf_fp, prefactor, w_fn):
        """Accumulate σ₀² into s2_out (n_real, n_modes), in-place."""
        n_real  = s2_out.shape[0]
        n_modes = f_obs.shape[0]
        log_span = log_fhi - log_flo
        for i in prange(n_real):
            for j in range(n_src_arr[i]):
                A   = _interp1(np.random.random(), cdf_x, cdf_fp)
                f_s = np.exp(log_flo + np.random.random() * log_span)
                pref = prefactor * A * A / (f_s * f_s)
                for k in range(n_modes):
                    wp = w_fn( f_s, f_obs[k], T_obs)
                    wm = w_fn(-f_s, f_obs[k], T_obs)
                    s2_out[i, k] += pref * (wp * wp + wm * wm)

    @njit(cache=False, fastmath=True)
    def _sigma2_tophat(s2_out, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, pf):
        _sigma2_core(s2_out, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, pf, _w_tophat)

    @njit(cache=False, fastmath=True)
    def _sigma2_sinc(s2_out, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, pf):
        _sigma2_core(s2_out, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, pf, _w_sinc)

    @njit(cache=False, fastmath=True)
    def _sigma2_whitened(s2_out, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, pf):
        _sigma2_core(s2_out, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, pf, _w_whitened)

    @njit(cache=False, fastmath=True)
    def _sigma2_tm(s2_out, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, pf):
        _sigma2_core(s2_out, n_src_arr, f_obs, T_obs, lf_lo, lf_hi, cx, cf, pf, _w_tm)

    _NB_SIGMA2_DISPATCH = {
        'tophat':   _sigma2_tophat,
        'sinc':     _sigma2_sinc,
        'whitened': _sigma2_whitened,
        'tm':       _sigma2_tm,
    }

    # ── Tail kernel: accumulate <|g_k|^3> over MC samples ────────────────────

    @njit(cache=False, fastmath=True)
    def _tail_core(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals, w_fn):
        """
        Accumulate <|g_k|^3> for each mode into tn_out (written in-place).
        tn_out is divided by n_samples at the end — pass a zero-initialised array.
        Source frequency is drawn log-uniformly in [exp(log_flo), exp(log_fhi)].
        """
        n_modes  = f_obs.shape[0]
        INV4PI   = 0.25 / np.pi
        TWO_PI   = 2.0 * np.pi
        log_span = log_fhi - log_flo

        for j in range(n_samples):
            f_s  = np.exp(log_flo + np.random.random() * log_span)
            dbar = np.random.random() * TWO_PI
            absR = _sample_absR(r_cdf, r_vals)
            P     = absR * INV4PI / f_s
            cos_d = np.cos(dbar)
            sin_d = np.sin(dbar)
            for k in range(n_modes):
                fk = f_obs[k]
                wp = w_fn( f_s, fk, T_obs)
                wm = w_fn(-f_s, fk, T_obs)
                gr = P * cos_d * (wp - wm)
                gi = P * sin_d * (wp + wm)
                tn_out[k] += (gr*gr + gi*gi) ** 1.5

        for k in range(n_modes):
            tn_out[k] /= n_samples

    @njit(cache=False, fastmath=True)
    def _tail_tophat(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals):
        _tail_core(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals, _w_tophat)

    @njit(cache=False, fastmath=True)
    def _tail_sinc(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals):
        _tail_core(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals, _w_sinc)

    @njit(cache=False, fastmath=True)
    def _tail_whitened(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals):
        _tail_core(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals, _w_whitened)

    @njit(cache=False, fastmath=True)
    def _tail_tm(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals):
        _tail_core(tn_out, n_samples, f_obs, T_obs, log_flo, log_fhi, r_cdf, r_vals, _w_tm)

    _NB_TAIL_DISPATCH = {
        'tophat':   _tail_tophat,
        'sinc':     _tail_sinc,
        'whitened': _tail_whitened,
        'tm':       _tail_tm,
    }

    # ── Trilinear interpolator for ModelI rate grid ───────────────────────────

    @njit(cache=True, fastmath=True)
    def _nb_trilinear(m1_flat, m2_flat, z_flat, grid,
                      lm1_min, lm1_step, lm2_min, lm2_step, lz_min, lz_step):
        """Trilinear interpolation on a (log m1, log m2, log z) regular grid."""
        n   = len(m1_flat)
        out = np.empty(n)
        n0  = grid.shape[0]
        n1  = grid.shape[1]
        n2  = grid.shape[2]
        lim0 = float(n0) - 1.0001
        lim1 = float(n1) - 1.0001
        lim2 = float(n2) - 1.0001
        for ii in range(n):
            m1c = m1_flat[ii]
            if m1c < 1e5:  m1c = 1e5
            if m1c > 1e13: m1c = 1e13
            m2c = m2_flat[ii]
            if m2c < 1e5:  m2c = 1e5
            if m2c > 1e13: m2c = 1e13
            zc  = z_flat[ii]
            if zc  < 1e-5: zc  = 1e-5
            if zc  > 6.0:  zc  = 6.0
            x0 = (np.log10(m1c) - lm1_min) / lm1_step
            x1 = (np.log10(m2c) - lm2_min) / lm2_step
            x2 = (np.log10(zc)  - lz_min)  / lz_step
            if x0 < 0.0: x0 = 0.0
            if x0 > lim0: x0 = lim0
            if x1 < 0.0: x1 = 0.0
            if x1 > lim1: x1 = lim1
            if x2 < 0.0: x2 = 0.0
            if x2 > lim2: x2 = lim2
            i0 = int(x0); i1 = int(x1); i2 = int(x2)
            t0 = x0 - i0;  t1 = x1 - i1;  t2 = x2 - i2
            c00 = grid[i0,   i1,   i2  ] * (1.0-t2) + grid[i0,   i1,   i2+1] * t2
            c01 = grid[i0,   i1+1, i2  ] * (1.0-t2) + grid[i0,   i1+1, i2+1] * t2
            c10 = grid[i0+1, i1,   i2  ] * (1.0-t2) + grid[i0+1, i1,   i2+1] * t2
            c11 = grid[i0+1, i1+1, i2  ] * (1.0-t2) + grid[i0+1, i1+1, i2+1] * t2
            c0  = c00 * (1.0-t1) + c01 * t1
            c1  = c10 * (1.0-t1) + c11 * t1
            out[ii] = c0 * (1.0-t0) + c1 * t0
        return out

    def nb_model_i_eval(m1, m2, z, grid,
                        lm1_min, lm1_step, lm2_min, lm2_step, lz_min, lz_step):
        """Evaluate ModelI rate grid; returns array with same shape as m1."""
        shape  = np.asarray(m1).shape
        m1f = np.ascontiguousarray(np.ravel(m1), dtype=np.float64)
        m2f = np.ascontiguousarray(np.ravel(m2), dtype=np.float64)
        zf  = np.ascontiguousarray(np.ravel(z),  dtype=np.float64)
        return _nb_trilinear(m1f, m2f, zf,
                             np.ascontiguousarray(grid, dtype=np.float64),
                             float(lm1_min), float(lm1_step),
                             float(lm2_min), float(lm2_step),
                             float(lz_min),  float(lz_step)).reshape(shape)

    # ── Public entry points ───────────────────────────────────────────────────

    def nb_accumulate_strong(res_s, n_src_arr, f_obs, T_obs,
                             log_flo, log_fhi, cdf_x, cdf_fp,
                             r_cdf, r_vals, window_name):
        """Accumulate strong-source residuals into res_s (n_real, n_modes), in-place."""
        kernel = _NB_STRONG_DISPATCH.get(window_name)
        if kernel is None:
            raise ValueError(
                f"No numba kernel for window '{window_name}'. "
                f"Available: {list(_NB_STRONG_DISPATCH)}")
        kernel(res_s,
               np.ascontiguousarray(n_src_arr, dtype=np.int64),
               np.ascontiguousarray(f_obs, dtype=np.float64),
               float(T_obs),
               float(log_flo),
               float(log_fhi),
               np.ascontiguousarray(cdf_x,  dtype=np.float64),
               np.ascontiguousarray(cdf_fp, dtype=np.float64),
               np.ascontiguousarray(r_cdf,  dtype=np.float64),
               np.ascontiguousarray(r_vals, dtype=np.float64))

    def nb_accumulate_tail(tn_out, n_samples, f_obs, T_obs,
                           log_flo, log_fhi, r_cdf, r_vals, window_name):
        """Accumulate <|g_k|^3> into tn_out (n_modes,), in-place."""
        kernel = _NB_TAIL_DISPATCH.get(window_name)
        if kernel is None:
            raise ValueError(
                f"No numba tail kernel for window '{window_name}'. "
                f"Available: {list(_NB_TAIL_DISPATCH)}")
        kernel(np.ascontiguousarray(tn_out, dtype=np.float64),
               int(n_samples),
               np.ascontiguousarray(f_obs, dtype=np.float64),
               float(T_obs),
               float(log_flo),
               float(log_fhi),
               np.ascontiguousarray(r_cdf,  dtype=np.float64),
               np.ascontiguousarray(r_vals, dtype=np.float64))

    def nb_accumulate_sigma2(s2_out, n_src_arr, f_obs, T_obs,
                              log_flo, log_fhi, cdf_x, cdf_fp,
                              prefactor, window_name):
        """Accumulate σ₀² into s2_out (n_real, n_modes), in-place."""
        kernel = _NB_SIGMA2_DISPATCH.get(window_name)
        if kernel is None:
            raise ValueError(
                f"No numba sigma2 kernel for window '{window_name}'. "
                f"Available: {list(_NB_SIGMA2_DISPATCH)}")
        kernel(np.ascontiguousarray(s2_out,   dtype=np.float64),
               np.ascontiguousarray(n_src_arr, dtype=np.int64),
               np.ascontiguousarray(f_obs,     dtype=np.float64),
               float(T_obs),
               float(log_flo),
               float(log_fhi),
               np.ascontiguousarray(cdf_x,  dtype=np.float64),
               np.ascontiguousarray(cdf_fp, dtype=np.float64),
               float(prefactor))

    def warmup(window_name='sinc', n_modes=14, r_cdf=None, r_vals=None):
        """Trigger JIT compilation; call once before the first real run."""
        from .windows import _R_CDF_AXIS, _R_VALS_AXIS
        if r_cdf is None:
            r_cdf = _R_CDF_AXIS
        if r_vals is None:
            r_vals = _R_VALS_AXIS
        dummy_s   = np.zeros((2, n_modes), dtype=complex)
        cdf_x     = np.array([0.0, 1.0])
        cdf_fp    = np.array([1e-16, 1e-12])
        n_src_dummy = np.array([1, 1], dtype=np.int64)
        nb_accumulate_strong(dummy_s, n_src_dummy, np.ones(n_modes) * 1e-8, 5e8,
                             np.log(1e-9), np.log(1e-8),
                             cdf_x, cdf_fp, r_cdf, r_vals, window_name)
        dummy_t = np.zeros(n_modes)
        nb_accumulate_tail(dummy_t, 2, np.ones(n_modes) * 1e-8, 5e8,
                           np.log(1e-9), np.log(1e-8), r_cdf, r_vals, window_name)
        dummy_s2 = np.zeros((2, n_modes), dtype=np.float64)
        nb_accumulate_sigma2(dummy_s2, n_src_dummy, np.ones(n_modes) * 1e-8, 5e8,
                             np.log(1e-9), np.log(1e-8),
                             cdf_x, cdf_fp, 1e-5, window_name)
        dummy_grid = np.zeros((2, 2, 2), dtype=np.float64)
        nb_model_i_eval(np.array([1e8]), np.array([1e8]), np.array([0.1]),
                        dummy_grid, 5.0, 8.0, 5.0, 8.0, -8.0, 1.0)

else:
    # Stubs so imports never fail
    def nb_accumulate_strong(*args, **kwargs):
        raise RuntimeError("numba is not installed.")

    def nb_accumulate_tail(*args, **kwargs):
        raise RuntimeError("numba is not installed.")

    def nb_accumulate_sigma2(*args, **kwargs):
        raise RuntimeError("numba is not installed.")

    def nb_model_i_eval(*args, **kwargs):
        raise RuntimeError("numba is not installed.")

    def warmup(*args, **kwargs):
        pass
