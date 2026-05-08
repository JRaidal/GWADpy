"""
σ₀ mixing distribution: dP/d ln σ₀.

σ₀²_k = σ²_weak,k + Σ_i PREFACTOR · A_i² · |w_k⁺(f_i)|² / f_i²

Requires a GlobalResidualsSimulator with _bin_cache populated.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from time import time as _time

from .windows import R_MEAN_SQ
from ._nb_kernels import NUMBA_AVAILABLE, nb_accumulate_sigma2

# PREFACTOR = <|R|²> / (2(4π)²) = R_MEAN_SQ / (2(4π)²)
_PREFACTOR = 2*R_MEAN_SQ / ((4.0 * np.pi) ** 2)


# ── Weak floor ────────────────────────────────────────────────────────────────

def _sigma2_weak(sim):
    """Deterministic weak-background per-component variance.  Returns (n_modes,)."""
    base = sum(s['sigma2_weak_per_mode'] for s in sim._bin_cache)
    return base * _PREFACTOR * (4.0 * np.pi) ** 2


# ── Analytic mean (weak floor + strong-source tail) ──────────────────────────

def _sigma2_mean(sim, n_f=20):
    """Analytic mean of σ₀²_k including the strong-source tail; returns (n_modes,)."""
    mean  = _sigma2_weak(sim).copy()
    f_obs = sim.f_obs
    for s in sim._bin_cache:
        if s['N_strong'] <= 0 or s['strong_cdf'] is None:
            continue
        A_arr   = s['strong_A_arr']
        cdf_arr = s['strong_cdf']
        A_mid   = np.sqrt(A_arr[:-1] * A_arr[1:])
        A2_mean = float(np.sum(A_mid**2 * np.diff(cdf_arr)))
        f_q     = np.exp(np.linspace(np.log(s['flo']), np.log(s['fhi']), n_f))
        Gk      = (np.abs(sim.window_fn(f_q[:, None], f_obs[None, :], sim.T_obs))**2
                   / f_q[:, None]**2)
        mean   += s['N_strong'] * _PREFACTOR * A2_mean * Gk.mean(axis=0)
    return mean


# ── Monte Carlo σ₀² sampler ──────────────────────────────────────────────────

def sample_sigma2(sim, n_real, use_wm=True, rng=None, chunk=5_000):
    """Draw n_real realisations of σ₀²_k for all modes; returns (n_real, n_modes)."""
    if rng is None:
        rng = np.random.default_rng()
    sw      = _sigma2_weak(sim)
    n_modes = sim.n_modes
    f_obs   = sim.f_obs
    s2      = np.zeros((n_real, n_modes), dtype=np.float64)

    use_nb = NUMBA_AVAILABLE and sim.window_name is not None

    for s in sim._bin_cache:
        if s['N_strong'] <= 0 or s['strong_cdf'] is None:
            continue

        if use_nb:
            # ── Numba path: parallel over realisations, no chunking needed ──
            n_src_arr = rng.poisson(s['N_strong'], size=n_real).astype(np.int64)
            nb_accumulate_sigma2(
                s2, n_src_arr, f_obs, sim.T_obs,
                np.log(s['flo']), np.log(s['fhi']),
                s['strong_cdf'], s['strong_A_arr'],
                _PREFACTOR, sim.window_name)
        else:
            # ── Numpy fallback: chunked to limit memory ───────────────────────
            for c0 in range(0, n_real, chunk):
                nc    = min(chunk, n_real - c0)
                n_src = rng.poisson(s['N_strong'], size=nc)
                n_max = int(n_src.max())
                if n_max == 0:
                    continue
                u    = rng.random((nc, n_max))
                A    = np.interp(u, s['strong_cdf'], s['strong_A_arr'])
                logf = (np.log(s['flo'])
                        + rng.random((nc, n_max)) * np.log(s['fhi'] / s['flo']))
                f    = np.exp(logf)
                slot = np.arange(n_max)[None, :] < n_src[:, None]
                sf   = np.where(slot, f, 1.0)
                base = np.where(slot, _PREFACTOR * A ** 2 / sf ** 2, 0.0)
                f3   = sf[:, :, None]
                fk   = f_obs[None, None, :]
                wp2  = np.abs(sim.window_fn(f3, fk, sim.T_obs)) ** 2
                if use_wm:
                    wp2 = wp2 + np.abs(sim.window_fn(-f3, fk, sim.T_obs)) ** 2
                s2[c0:c0 + nc] += (base[:, :, None] * wp2).sum(axis=1)

    s2 += sw[None, :]
    return s2


# ── Analytic Campbell tail ────────────────────────────────────────────────────

def compute_sigma0_tail(sim, sigma0_grid, sw=None, n_f_quad=30):
    """
    Analytic Campbell n=1 tail of dP/d ln σ₀; returns (n_s0, n_modes).
    Integrates Dirac-δ at each (A, f) → σ₀² contribution; gives σ₀^{-3} for large σ₀.
    """
    if sw is None:
        sw = _sigma2_weak(sim)
    s0_sq    = sigma0_grid ** 2
    f_obs    = sim.f_obs
    n_modes  = sim.n_modes
    pdf_tail = np.zeros((len(sigma0_grid), n_modes))

    for s in sim._bin_cache:
        if s['N_strong'] <= 0 or s['strong_cdf'] is None:
            continue
        Nb      = s['N_strong']
        A_arr   = s['strong_A_arr']
        cdf_arr = s['strong_cdf']
        A_lo, A_hi = A_arr[0], A_arr[-1]

        # Â·p_b(Â) = dCDF/d(ln A) — precompute on log-A midpoints
        dlnA     = np.diff(np.log(np.maximum(A_arr, 1e-300)))
        dlnA     = np.where(dlnA > 0, dlnA, 1e-300)
        ApA_vals = np.diff(cdf_arr) / dlnA          # (len(A_arr)−1,)
        A_mid_c  = np.sqrt(A_arr[:-1] * A_arr[1:])  # log midpoints

        # Log-uniform f quadrature within bin
        f_q   = np.exp(np.linspace(np.log(s['flo']), np.log(s['fhi']), n_f_quad))
        fq_2d = f_q[:, None]
        fk_2d = f_obs[None, :]
        # G_k(f) = |w_k⁺(f)|²/f²:  (n_f_quad, n_modes)
        Gk = np.abs(sim.window_fn(fq_2d, fk_2d, sim.T_obs)) ** 2 / fq_2d ** 2

        # δ = σ₀² − σ²_weak,k ≥ 0:  (n_s0, n_modes)
        delta = np.maximum(s0_sq[:, None] - sw[None, :], 0.0)

        # Â[f, s0, k] = √(δ / (_PREFACTOR·G_k)):  (n_f, n_s0, n_modes)
        safe_G = np.maximum(_PREFACTOR * Gk[:, None, :], 1e-300)
        Ahat   = np.sqrt(delta[None, :, :] / safe_G)

        # Interpolate Â·p_b(Â); zero outside amplitude support
        in_rng   = (Ahat >= A_lo) & (Ahat <= A_hi) & (delta[None, :, :] > 0)
        ApA_flat = np.interp(Ahat.ravel(), A_mid_c, ApA_vals, left=0.0, right=0.0)
        ApA_3d   = ApA_flat.reshape(Ahat.shape)

        # Sum over f → (n_s0, n_modes)
        sum_f  = (Nb / n_f_quad) * np.sum(np.where(in_rng, ApA_3d, 0.0), axis=0)
        safe_d = np.where(delta > 0, delta, 1.0)
        pref   = np.where(delta > 0, s0_sq[:, None] / safe_d, 0.0)
        pdf_tail += pref * sum_f

    return pdf_tail   # (n_s0, n_modes)


# ── Composite PDF (MC bulk + analytic tail) ───────────────────────────────────

def composite_sigma0_pdf(sim, n_real=20_000, n_bins=80, min_counts=15,
                          smooth_sigma=0.8, n_tail_dex=1.5, n_f_quad=30,
                          rng=None, verbose=True):
    """
    Composite dP/d ln σ₀ for all modes: MC histogram bulk + analytic Campbell tail.

    Returns a dict with keys: 's2_draws', 'sw', 'univ_grid', 'tail_all',
    'bins_e', 'centers', 'pdf_mc', 'x_comp', 'y_comp', 'tail_x', 'tail_y', 'x_hi'.
    """
    if rng is None:
        rng = np.random.default_rng()

    sw      = _sigma2_weak(sim)            # (n_modes,)
    n_modes = sim.n_modes

    # ── Phase 1: MC sampling ─────────────────────────────────────────────────
    if verbose:
        print(f"  [σ₀] Phase 1: MC sampling ({n_real:,} realisations) ...", flush=True)
    _t0 = _time()
    s2  = sample_sigma2(sim, n_real, rng=rng)
    _dt_mc = _time() - _t0
    if verbose:
        print(f"  [σ₀] Phase 1 done: {_dt_mc:.2f}s")

    # ── Phase 2: histograms ───────────────────────────────────────────────────
    if verbose:
        print(f"  [σ₀] Phase 2: building histograms ...", end='', flush=True)
    _t0 = _time()
    mode_data = []
    sigma0_max_global = 0.0
    for ki in range(n_modes):
        sigma0 = np.sqrt(np.maximum(s2[:, ki], 0.0))
        sigma0 = sigma0[sigma0 > 0]
        bins_e = np.geomspace(sigma0.min() / 2, sigma0.max() * 2, n_bins + 1)
        dlnb   = np.log(bins_e[1] / bins_e[0])
        counts, _ = np.histogram(sigma0, bins=bins_e)
        centers   = np.sqrt(bins_e[:-1] * bins_e[1:])
        pdf_mc    = counts / (len(sigma0) * dlnb)
        good  = np.where(counts >= min_counts)[0]
        ci_hi = int(good[-1]) if len(good) > 0 else len(centers) - 1
        x_hi  = float(centers[ci_hi])
        sigma0_max_global = max(sigma0_max_global, float(sigma0.max()))
        mode_data.append(dict(bins_e=bins_e, centers=centers,
                               counts=counts, pdf_mc=pdf_mc, x_hi=x_hi))
    if verbose:
        print(f" done ({_time() - _t0:.3f}s)")

    # ── Phase 3: analytic tail (ONE call) ────────────────────────────────────
    if verbose:
        print(f"  [σ₀] Phase 3: analytic Campbell tail (n_f_quad={n_f_quad}, "
              f"grid=600) ...", end='', flush=True)
    _t0 = _time()
    univ_lo   = float(np.sqrt(sw.min())) * 0.05
    univ_hi   = sigma0_max_global * 10 ** n_tail_dex
    univ_grid = np.geomspace(univ_lo, univ_hi, 600)
    tail_all  = compute_sigma0_tail(sim, univ_grid, sw=sw, n_f_quad=n_f_quad)
    if verbose:
        print(f" done ({_time() - _t0:.2f}s)")
    # tail_all: (600, n_modes)

    # ── Phase 4: composite assembly ───────────────────────────────────────────
    if verbose:
        print(f"  [σ₀] Phase 4: composite assembly ...", end='', flush=True)
    _t0 = _time()
    bins_e_list = []; centers_list = []; pdf_mc_list = []
    x_comp_list = []; y_comp_list  = []
    tail_x_list = []; tail_y_list  = []; x_hi_list   = []

    for ki in range(n_modes):
        d       = mode_data[ki]
        x_hi    = d['x_hi']
        bins_e  = d['bins_e']
        centers = d['centers']
        pdf_mc  = d['pdf_mc']

        # Smooth only within the positive range (preserves sharp edges)
        pos_idx    = np.where(pdf_mc > 0)[0]
        pdf_smooth = np.zeros_like(pdf_mc)
        if len(pos_idx) > 0:
            i0, i1 = pos_idx[0], pos_idx[-1] + 1
            _slice = pdf_mc[i0:i1]
            _tiny  = _slice[_slice > 0].min() * 1e-6
            pdf_smooth[i0:i1] = np.exp(
                gaussian_filter1d(np.log(np.maximum(_slice, _tiny)), sigma=smooth_sigma))

        # Extract tail slice from universal grid
        m_univ = univ_grid > x_hi * 0.3
        tail_x = univ_grid[m_univ]
        tail_y = tail_all[m_univ, ki]
        m_tail = tail_x > x_hi

        # Composite x/y (bulk centres + tail portion of universal grid)
        bulk_mask = (centers <= x_hi) & (pdf_smooth > 0)
        x_comp    = np.concatenate([centers[bulk_mask], tail_x[m_tail]])
        y_comp    = np.concatenate([pdf_smooth[bulk_mask], tail_y[m_tail]])

        bins_e_list.append(bins_e)
        centers_list.append(centers)
        pdf_mc_list.append(pdf_mc)
        x_comp_list.append(x_comp)
        y_comp_list.append(y_comp)
        tail_x_list.append(tail_x)
        tail_y_list.append(tail_y)
        x_hi_list.append(x_hi)

    if verbose:
        print(f" done ({_time() - _t0:.3f}s)")

    return {
        's2_draws':  s2,
        'sw':        sw,
        'univ_grid': univ_grid,
        'tail_all':  tail_all,
        'bins_e':    bins_e_list,
        'centers':   centers_list,
        'pdf_mc':    pdf_mc_list,
        'x_comp':    x_comp_list,
        'y_comp':    y_comp_list,
        'tail_x':    tail_x_list,
        'tail_y':    tail_y_list,
        'x_hi':      x_hi_list,
    }


# ── Production plot ───────────────────────────────────────────────────────────

def make_sigma0_plot(sim, data, out_path, k_modes=None, n_tail_dex=1.5, model_label=None):
    """
    Save a production dP/d ln σ₀ figure matching the plot_pdfs.py style.

    Parameters
    ----------
    sim      : GlobalResidualsSimulator
    data     : dict returned by composite_sigma0_pdf
    out_path : str  — output file path (.pdf or .png)
    k_modes  : list[int] or None  — 1-indexed modes to show (default: [1,5,10,14])
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if k_modes is None:
        candidates = [1, 5, 10, 14]
        k_modes = [k for k in candidates if k <= sim.n_modes]
        if not k_modes:
            k_modes = list(range(1, min(5, sim.n_modes + 1)))

    plt.rcParams.update({
        'text.usetex':        True,
        'font.family':        'serif',
        'font.serif':         ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size':          15,
        'axes.labelsize':     15,
        'axes.titlesize':     15,
        'xtick.labelsize':    13,
        'ytick.labelsize':    13,
        'legend.fontsize':    11,
        'figure.dpi':         150,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.05,
        'lines.linewidth':    1.5,
        'axes.linewidth':     1.0,
        'axes.grid':          True,
        'grid.alpha':         0.3,
        'grid.linewidth':     0.5,
        'xtick.major.width':  1.0,
        'ytick.major.width':  1.0,
        'xtick.minor.width':  0.5,
        'ytick.minor.width':  0.5,
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.top':          True,
        'ytick.right':        True,
    })

    sw         = data['sw']
    univ_grid  = data['univ_grid']
    tail_all   = data['tail_all']
    n_cols     = len(k_modes)

    fig = plt.figure(figsize=(4.7 * n_cols, 4.7))
    gs  = gridspec.GridSpec(1, n_cols, wspace=0.2)
    axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]

    def _line_angle(ax, x0, y0, x1, y1):
        d0 = ax.transData.transform((x0, y0))
        d1 = ax.transData.transform((x1, y1))
        return np.degrees(np.arctan2(d1[1] - d0[1], d1[0] - d0[0]))

    _label_kw = dict(fontsize=9, ha='left', va='bottom',
                     rotation_mode='anchor',
                     bbox=dict(boxstyle='round,pad=0.0',
                               fc='white', ec='none', alpha=0.9))

    for col_idx, K_DEMO in enumerate(k_modes):
        ki  = K_DEMO - 1
        ax  = axes[col_idx]
        f_k = sim.f_obs[ki]

        bins_e  = data['bins_e'][ki]
        pdf_mc  = data['pdf_mc'][ki]
        x_comp  = data['x_comp'][ki]
        y_comp  = data['y_comp'][ki]
        tail_x  = data['tail_x'][ki]
        tail_y  = data['tail_y'][ki]
        x_hi    = data['x_hi'][ki]
        sw_k    = float(np.sqrt(sw[ki]))

        # Axis limits (computed before plotting so baseline can use _ylim[0])
        _xmax  = x_hi * 10 ** n_tail_dex
        m_tail = (tail_x > x_hi) & (tail_x <= _xmax)
        _all_y = np.concatenate([pdf_mc[pdf_mc > 0], tail_y[m_tail][tail_y[m_tail] > 0]])
        _ymin  = _all_y.min() / 10 if len(_all_y) > 0 else 1e-6
        _xlim  = (sw_k * 0.3, _xmax)
        _ylim  = (_ymin, _all_y.max() * 5 if len(_all_y) > 0 else 1)

        # Filled histogram (staircase → fill down to plot floor)
        _x_s = np.repeat(bins_e, 2)[1:-1]
        _y_s = np.repeat(pdf_mc, 2)
        ax.fill_between(_x_s, _y_s, y2=_ylim[0],
                        color='C4', alpha=0.35, label='MC samples')
        ax.plot(_x_s, np.where(_y_s > 0, _y_s, np.nan),
                '-', color='C4', lw=1.2, alpha=0.85)

        # Smooth composite total (black) — clipped to displayed range
        pos_c = (y_comp > 0) & (x_comp <= _xmax)
        ax.plot(x_comp[pos_c], y_comp[pos_c], '-', color='k', lw=2.5, label='Total')

        # σ₀^{−3} reference line (dashed gray, over displayed x range)
        ref_y   = tail_all[:, ki]
        ref_pos = (univ_grid >= _xlim[0]) & (univ_grid <= _xmax) & (ref_y > 0)
        if ref_pos.any():
            ax.plot(univ_grid[ref_pos], ref_y[ref_pos], '--',
                    color='0.45', lw=1.6, alpha=0.85, zorder=0)

        # σ_weak vertical line
        ax.axvline(sw_k, color='0.4', lw=1.2, ls='--', label=r'$\sigma_{\rm weak}$')

        # σ₀^{−3} slope annotation
        if ref_pos.any():
            _rx  = univ_grid[ref_pos]
            _ry  = ref_y[ref_pos]
            _lx0 = np.log10(_xlim[0]); _lx1 = np.log10(_xlim[1])
            _xb  = 10 ** (_lx0 + 0.82 * (_lx1 - _lx0))
            _xb2 = _xb * 2
            _yb  = float(np.interp(_xb,  _rx, _ry))
            _yb2 = float(np.interp(_xb2, _rx, _ry))
            if _yb > 0 and _yb2 > 0 and _ylim[0] < _yb * 10 ** 0.8 < _ylim[1]:
                _ang = _line_angle(ax, _xb, _yb, _xb2, _yb2)
                ax.text(_xb, _yb * 10 ** 0.8, r'$\propto\sigma_0^{-3}$',
                        rotation=_ang, color='black', **_label_kw)

        ax.set(xscale='log', yscale='log', xlim=_xlim, ylim=_ylim,
               title=rf'$k={K_DEMO}$,  $f={f_k * 1e9:.1f}$ nHz',
               xlabel=r'$\sigma_0\;[\mathrm{s}]$',
               ylabel=(r'$\mathrm{d}P/\mathrm{d}\ln\sigma_0$'
                       if col_idx == 0 else ''))
        if col_idx > 0:
            ax.tick_params(labelleft=False)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':')

        if col_idx == 0:
            ax.legend(loc='upper right', frameon=True,
                      fancybox=False, edgecolor='black')

    _title = (r'No-interference distribution $\mathrm{d}P/\mathrm{d}\ln\sigma_0$'
              + (f' --- {model_label}' if model_label else ''))
    fig.suptitle(_title, fontsize=13, y=1.01)

    plt.savefig(out_path)
    plt.close()
    print(f"  σ₀ plot → {out_path}")


# ── Combined variance plot (3 × dP/d ln σ₀² + violin) ───────────────────────

def make_variance_plot(sim, data, ng_data, out_path, model_label=None,
                       k_modes=None, ylim=(-17.0, -11.0), n_tail_dex=1.5):
    """Four-panel figure: dP/d ln σ₀² for k=1,7,14 + σ₀² violin vs PTA data."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if k_modes is None:
        k_modes = [k for k in [1, 7, 14] if k <= sim.n_modes]

    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 15, 'axes.labelsize': 15, 'axes.titlesize': 15,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
        'lines.linewidth': 1.5, 'axes.linewidth': 1.0,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })

    sw        = data['sw']
    univ_grid = data['univ_grid']
    tail_all  = data['tail_all']
    s2_draws  = data['s2_draws']
    x_pos_v   = np.log10(sim.f_obs)

    fig = plt.figure(figsize=(18, 4.5))
    gs  = gridspec.GridSpec(1, 4, wspace=0.35, width_ratios=[1, 1, 1, 1.6])
    axes_pdf = [fig.add_subplot(gs[i]) for i in range(3)]
    ax_vln   = fig.add_subplot(gs[3])

    def _line_angle(ax, x0, y0, x1, y1):
        d0 = ax.transData.transform((x0, y0))
        d1 = ax.transData.transform((x1, y1))
        return np.degrees(np.arctan2(d1[1] - d0[1], d1[0] - d0[0]))

    _label_kw = dict(fontsize=9, ha='left', va='bottom', rotation_mode='anchor',
                     bbox=dict(boxstyle='round,pad=0.0', fc='white', ec='none', alpha=0.9))

    # ── σ₀² PDF panels ────────────────────────────────────────────────────────
    for col_idx, K_DEMO in enumerate(k_modes):
        ki  = K_DEMO - 1
        ax  = axes_pdf[col_idx]
        f_k = sim.f_obs[ki]

        bins_e = data['bins_e'][ki]
        pdf_mc = data['pdf_mc'][ki]
        x_comp = data['x_comp'][ki]
        y_comp = data['y_comp'][ki]
        tail_x = data['tail_x'][ki]
        tail_y = data['tail_y'][ki]
        x_hi   = data['x_hi'][ki]
        sw2_k  = float(sw[ki])          # σ²_weak (s²)

        # σ₀² axis limits
        _xmax2   = (x_hi * 10**n_tail_dex)**2
        m_tail   = (tail_x > x_hi) & (tail_x <= x_hi * 10**n_tail_dex)
        _py      = np.concatenate([(pdf_mc/2)[pdf_mc > 0],
                                   (tail_y[m_tail]/2)[tail_y[m_tail] > 0]])
        _ymin    = _py.min() / 10 if len(_py) > 0 else 1e-6
        _xlim    = (sw2_k * 0.3, _xmax2)
        _ylim_ax = (_ymin, _py.max() * 5 if len(_py) > 0 else 1)

        # Histogram staircase (σ₀² x-axis)
        _x_s = np.repeat(bins_e**2, 2)[1:-1]
        _y_s = np.repeat(pdf_mc / 2, 2)
        ax.fill_between(_x_s, _y_s, y2=_ylim_ax[0],
                        color='C4', alpha=0.35, label='MC samples')
        ax.plot(_x_s, np.where(_y_s > 0, _y_s, np.nan),
                '-', color='C4', lw=1.2, alpha=0.85)

        # Composite total line
        pos_c = (y_comp > 0) & (x_comp <= x_hi * 10**n_tail_dex)
        ax.plot(x_comp[pos_c]**2, y_comp[pos_c] / 2,
                '-', color='k', lw=2.5, label='Total')

        # Reference tail
        ref_y   = tail_all[:, ki]
        ref_pos = (univ_grid**2 >= _xlim[0]) & (univ_grid**2 <= _xlim[1]) & (ref_y > 0)
        if ref_pos.any():
            ax.plot(univ_grid[ref_pos]**2, ref_y[ref_pos] / 2,
                    '--', color='0.45', lw=1.6, alpha=0.85, zorder=0)

        # σ²_weak marker
        ax.axvline(sw2_k, color='0.4', lw=1.2, ls='--',
                   label=r'$\sigma^2_{\rm weak}$')

        # Slope annotation
        if ref_pos.any():
            _rx2 = univ_grid[ref_pos]**2
            _ry2 = ref_y[ref_pos] / 2
            _lx0 = np.log10(_xlim[0]); _lx1 = np.log10(_xlim[1])
            _xb2 = 10**(_lx0 + 0.75 * (_lx1 - _lx0))
            _xb4 = _xb2 * 4
            _yb2 = float(np.interp(_xb2, _rx2, _ry2))
            _yb4 = float(np.interp(_xb4, _rx2, _ry2))
            if _yb2 > 0 and _yb4 > 0 and _ylim_ax[0] < _yb2 * 10**0.8 < _ylim_ax[1]:
                _ang = _line_angle(ax, _xb2, _yb2, _xb4, _yb4)
                ax.text(_xb2, _yb2 * 10**0.8, r'$\propto(\sigma_0^2)^{-3/2}$',
                        rotation=_ang, color='black', **_label_kw)

        ax.set(xscale='log', yscale='log', xlim=_xlim, ylim=_ylim_ax,
               title=rf'$k={K_DEMO}$,  $f={f_k*1e9:.1f}$ nHz',
               xlabel=r'$\sigma_0^2\;[\mathrm{s}^2]$',
               ylabel=(r'$\mathrm{d}P/\mathrm{d}\ln\sigma_0^2$'
                       if col_idx == 0 else ''))
        if col_idx > 0:
            ax.tick_params(labelleft=False)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':')
        if col_idx == 0:
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    # ── Violin panel ──────────────────────────────────────────────────────────
    data_log = np.log10(s2_draws[:20_000] + 1e-60)
    parts = ax_vln.violinplot(
        [data_log[:, i] for i in range(sim.n_modes)],
        positions=x_pos_v, widths=0.054,
        showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('C0'); pc.set_edgecolor('black')
        pc.set_alpha(0.50); pc.set_linewidth(1.0)

    if ng_data is not None:
        y_grid = np.linspace(-20, -8, 4000)
        for (fng, pdf_fn) in ng_data[:sim.n_modes]:
            xc = np.log10(fng)
            if xc < x_pos_v.min() - 0.2 or xc > x_pos_v.max() + 0.2:
                continue
            xd = pdf_fn(y_grid / 2.0)
            ax_vln.fill_betweenx(y_grid, xc - xd * 0.027, xc + xd * 0.027,
                                 facecolor='#FF9900', edgecolor='black',
                                 linewidth=0.8, alpha=0.40, zorder=3)
            ax_vln.scatter(xc, y_grid[np.argmax(xd)], s=12, color='white', zorder=5)

    ref_x  = np.log10(ng_data[0][0]) if ng_data else x_pos_v[0]
    x_line = np.linspace(x_pos_v.min() - 0.05, x_pos_v.max() + 0.05, 100)
    ax_vln.plot(x_line, -13/3 * (x_line - ref_x) - 12.4, 'k--', lw=2.0, alpha=0.7)

    mean_s2  = _sigma2_mean(sim)
    mean_log = np.log10(mean_s2 + 1e-60)
    ax_vln.plot(x_pos_v, mean_log, '-.', color='C0', lw=1.8)

    ax_vln.set(xlabel=r'$\log_{10}(f\,[\mathrm{Hz}])$',
               ylabel=r'$\log_{10}(\sigma_0^2\,[\mathrm{s}^2])$',
               xlim=(x_pos_v.min() - 0.10, x_pos_v.max() + 0.10),
               ylim=ylim)
    ax_vln.tick_params(which='both', direction='in', top=True, right=True)

    legend_handles = [
        Patch(facecolor='C0', edgecolor='black', alpha=0.6, label='Simulation'),
        Line2D([0], [0], color='black', lw=2, ls='--', label=r'Power law $-13/3$'),
        Line2D([0], [0], color='C0', lw=1.8, ls='-.', label=r'Mean $\sigma_0^2$'),
    ]
    if ng_data:
        legend_handles.insert(1, Patch(facecolor='#FF9900', edgecolor='black',
                                       alpha=0.5, label='NANOGrav 15yr'))
    ax_vln.legend(handles=legend_handles, loc='upper right', frameon=True,
                  fancybox=False, edgecolor='black', framealpha=0.95)

    _title = (r'No-interference variance $\mathrm{d}P/\mathrm{d}\ln\sigma_0^2$'
              + (f' --- {model_label}' if model_label else ''))
    fig.suptitle(_title, fontsize=13, y=1.01)

    plt.savefig(out_path)
    plt.close()
    print(f"  Variance plot → {out_path}")
