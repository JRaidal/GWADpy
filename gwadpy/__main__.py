"""
Entry point for `python -m GWADpy`.

Run `python -m GWADpy --help` for full usage.
"""

import argparse
import os
from pathlib import Path
from time import time as _time

import numpy as np
import matplotlib
matplotlib.use('Agg')

from .constants import YEAR_IN_SEC
from .merger_rates import ModelI, ModelII
from .gwad import BrokenPowerLawGWAD
from .windows import WINDOWS
from .simulator import GlobalResidualsSimulator
from .analysis import compute_pdfs
from .plotting import make_validation_plot


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m gwadpy',
        description='GW timing residual analysis — compute PDFs, validation plot, and likelihood.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    # Output
    p.add_argument('--output-dir', default='.',             help='Directory for output files')
    p.add_argument('--prefix',     default='gw_residuals',  help='Output filename prefix')

    # Simulation
    p.add_argument('--t-obs',          type=float, default=16.0,    help='Observation time [years]')
    p.add_argument('--n-modes',        type=int,   default=14,      help='Number of PTA Fourier modes')
    p.add_argument('--n-real',         type=int,   default=10_000,  help='Monte Carlo realisations')
    p.add_argument('--n-strong',       type=int,   default=10,      help='Strong-source threshold per bin')

    # Source frequency grid
    p.add_argument('--f-start', type=float, default=1e-10,  help='Source frequency bin start [Hz]')
    p.add_argument('--f-end',   type=float, default=100e-9, help='Source frequency bin end [Hz]')
    p.add_argument('--n-bins',  type=int,   default=103,    help='Number of source frequency bins')

    # Window function
    p.add_argument('--window', choices=list(WINDOWS), default='tophat',
                               help='PTA window function')

    # Environment (optional)
    p.add_argument('--env-f-ref', type=float, default=None,
                                  help='Environmental hardening f_ref [Hz] (omit for GW-only)')
    p.add_argument('--env-alpha', type=float, default=8/3,
                                  help='Environmental hardening frequency power-law index (default: 8/3)')
    p.add_argument('--env-beta',  type=float, default=5/8,
                                  help='Environmental hardening mass-ratio power-law index (default: 5/8)')

    # Parallelisation
    p.add_argument('--n-workers', type=int, default=None,
                                  help='Worker threads for bin precomputation and simulation '
                                       '(default: auto; set to 1 to disable parallelisation)')

    # KDE
    p.add_argument('--kde-bw',      type=float, default=0.1,    help='KDE bandwidth')
    p.add_argument('--kde-max-pts', type=int,   default=10_000,
                   help='Max realisations used to fit each KDE (subsampled if n_real exceeds this)')

    # Gaussian approximation (no strong sources, no power-law tail)
    p.add_argument('--gaussian', action='store_true',
                                 help='Treat all sources as Gaussian (forces n_strong=0, '
                                      'disables the power-law high-residual tail)')

    # Variance distribution (σ₀²)
    p.add_argument('--variance', action='store_true',
                                 help='Also produce a combined dP/d ln σ₀² + violin plot')
    p.add_argument('--variance-n-real', type=int, default=100_000,
                                        help='MC realisations for the σ₀² histogram bulk')

    # PTA data (optional — needed for likelihood)
    p.add_argument('--pta-data-dir', default=None,
                                     help='PTA data directory containing density.npy, '
                                          'log10rhogrid.npy, freqs.npy')

    # ── Model subcommands ────────────────────────────────────────────────────
    sub = p.add_subparsers(dest='model', required=True, title='merger rate model')

    mI = sub.add_parser('modelI',
                         help='Semi-numerical rate (halo merger tree + MBH–halo relation)',
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mI.add_argument('--a',     type=float, default=8.95, help='MBH–halo amplitude')
    mI.add_argument('--b',     type=float, default=1.4,  help='MBH–halo slope')
    mI.add_argument('--sigma', type=float, default=0.47, help='MBH–halo scatter [dex]')
    mI.add_argument('--pbh',   type=float, default=0.06, help='Binary fraction')

    mII = sub.add_parser('modelII',
                          help='Analytic phenomenological merger rate',
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mII.add_argument('--R0',    type=float, default=4e-14, help='Rate normalisation')
    mII.add_argument('--M-star',type=float, default=2.5e9, help='Chirp mass scale [Msun]')
    mII.add_argument('--c',     type=float, default=-0.2,  help='Low-mass power-law index')
    mII.add_argument('--d',     type=float, default=6.0,   help='Redshift evolution index')
    mII.add_argument('--z0',    type=float, default=0.3,   help='Redshift decay scale')

    mb = sub.add_parser('bpl',
                         help='Broken power-law GWAD (specify amplitude distribution directly)',
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mb.add_argument('--N-b', type=float, required=True, help='Normalisation at A_b [strain^-1]')
    mb.add_argument('--A-b', type=float, required=True, help='Break amplitude [strain]')
    mb.add_argument('--p',   type=float, required=True, help='High-A slope')
    mb.add_argument('--q',   type=float, required=True, help='Low-A slope')
    mb.add_argument('--s',   type=float, default=2.0,   help='Transition sharpness')

    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.prefix)

    # ── Build model ──────────────────────────────────────────────────────────
    if args.env_f_ref is not None:
        env_params = {'f_ref': args.env_f_ref, 'alpha': args.env_alpha, 'beta': args.env_beta}
    else:
        env_params = {}

    if args.model == 'modelI':
        model = ModelI(a=args.a, b=args.b, sigma=args.sigma, pbh=args.pbh)
        model_label = (rf'Model I ($a={args.a},\,b={args.b},'
                       rf'\,\sigma={args.sigma},\,p_{{BH}}={args.pbh}$)')
    elif args.model == 'modelII':
        model = ModelII(R0=args.R0, M_star=args.M_star, c=args.c, d=args.d, z0=args.z0)
        model_label = rf'Model II ($R_0={args.R0:.1e},\,M_*={args.M_star:.2e}$)'
    else:  # bpl
        model = BrokenPowerLawGWAD(N_b=args.N_b, A_b=args.A_b,
                                    p=args.p, q=args.q, s=args.s)
        model_label = (rf'BPL ($N_b={args.N_b:.2e},\,A_b={args.A_b:.2e},'
                       rf'\,p={args.p},\,q={args.q},\,s={args.s}$)')

    if args.gaussian:
        args.n_strong = 0

    T_OBS          = args.t_obs * YEAR_IN_SEC
    window         = WINDOWS[args.window]
    source_f_edges = np.linspace(args.f_start, args.f_end, args.n_bins + 1)
    A_COMMON       = np.logspace(-27, -9, 306)

    # ── Simulator ────────────────────────────────────────────────────────────
    sim = GlobalResidualsSimulator(
        gwad_model     = model,
        env_params     = env_params,
        T_obs          = T_OBS,
        n_modes        = args.n_modes,
        source_f_edges = source_f_edges,
        A_common       = A_COMMON,
        window_fn      = window,
    )
    sim.precompute_bin_stats(n_strong=args.n_strong, n_workers=args.n_workers)

    # ── Simulate ─────────────────────────────────────────────────────────────
    res, res_strong, res_weak, tail_norm = sim.get_residuals(
        args.n_real, n_strong=args.n_strong, n_workers=args.n_workers)

    # ── Compute and save PDFs ─────────────────────────────────────────────────
    print("  Computing KDE+tail PDFs ...", end='', flush=True)
    _t0 = _time()
    dt_grids, pdf_out, dt_cross = compute_pdfs(
        res, tail_norm, args.n_modes, sim.f_obs,
        kde_bw=args.kde_bw, n_kde_max=args.kde_max_pts, sim=sim,
        gaussian=args.gaussian, verbose=False)
    npz_path = f"{prefix}_pdfs.npz"
    np.savez(npz_path,
             f_modes   = sim.f_obs,
             dt_grids  = dt_grids,
             pdf       = pdf_out,
             tail_norm = tail_norm,
             dt_cross  = dt_cross)
    print(f" done  ({_time() - _t0:.1f}s)  →  {npz_path}")

    # ── Load NANOGrav data ────────────────────────────────────────────────────
    ng_data = None
    if args.pta_data_dir is not None:
        try:
            from scipy.interpolate import interp1d
            data_dir = Path(args.pta_data_dir)
            prob     = np.load(data_dir / 'density.npy')[0]
            L10rho   = np.load(data_dir / 'log10rhogrid.npy')
            fNG      = np.load(data_dir / 'freqs.npy')
            ng_data  = []
            for j, fv in enumerate(fNG):
                p = np.exp(prob[j]); p /= p.max()
                ng_data.append((fv, interp1d(L10rho, p, kind='linear',
                                             bounds_error=False, fill_value=0.0)))
            print(f"  NANOGrav data: {len(ng_data)} modes loaded.")
        except FileNotFoundError as e:
            print(f"  Warning: PTA data not found ({e}). Plotting without data.")

    # ── Validation plot ───────────────────────────────────────────────────────
    plot_path = f"{prefix}_validation.pdf"
    make_validation_plot(res, res_strong, res_weak, tail_norm,
                         sim, plot_path,
                         args.n_real, args.n_strong, model_label,
                         gaussian=args.gaussian)
    print(f"  Validation plot  →  {plot_path}")

    # ── Variance distribution ─────────────────────────────────────────────────
    if args.variance:
        from .sigma0 import composite_sigma0_pdf, make_variance_plot
        print(f"  σ₀² MC ({args.variance_n_real:,} realisations) ...", end='', flush=True)
        _t0 = _time()
        s0_data = composite_sigma0_pdf(
            sim,
            n_real = args.variance_n_real,
            rng    = np.random.default_rng(42),
        )
        print(f" done  ({_time() - _t0:.1f}s)")

        print("  Saving σ₀² data ...", end='', flush=True)
        npz_s0 = f"{prefix}_variance.npz"
        np.savez(
            npz_s0,
            f_modes   = sim.f_obs,
            sw        = s0_data['sw'],
            s2_draws  = s0_data['s2_draws'],
            univ_grid = s0_data['univ_grid'],
            tail_all  = s0_data['tail_all'],
        )
        print(f" done  →  {npz_s0}")

        print("  Plotting variance distribution ...", end='', flush=True)
        _t0 = _time()
        plot_var = f"{prefix}_variance.pdf"
        make_variance_plot(sim, s0_data, ng_data, plot_var, model_label=model_label)
        print(f" done  ({_time() - _t0:.1f}s)  →  {plot_var}")



    # ── σ₀ likelihood (runs whenever --pta-data-dir is given) ────────────────
    if args.pta_data_dir is not None:
        try:
            from .analysis import compute_sigma0_likelihood
            # Reuse draws already computed by --variance; otherwise sample fresh.
            _s2 = s0_data['s2_draws'] if args.variance else None
            log_L_modes, log_L_total = compute_sigma0_likelihood(
                sim, args.variance_n_real, args.pta_data_dir,
                kde_bw=args.kde_bw, s2_draws=_s2)
            n_valid = int(np.sum(np.isfinite(log_L_modes)))
            print(f"\n  σ₀ log-likelihood over {n_valid} modes:")
            for k in range(args.n_modes):
                tag = '  (no overlap)' if not np.isfinite(log_L_modes[k]) else ''
                print(f"    k={k+1:2d}  {sim.f_obs[k]*1e9:5.1f} nHz  "
                      f"ln L = {log_L_modes[k]:9.4f}{tag}")
            print(f"  Total: ln L = {log_L_total:.4f}")
        except FileNotFoundError as e:
            print(f"  Warning: could not compute likelihood ({e})")


if __name__ == '__main__':
    main()
