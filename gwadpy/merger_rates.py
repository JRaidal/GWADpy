"""
Merger rate models.

ModelI  — semi-numerical rate from halo merger trees + MBH–halo relation.
           Rate grid is pre-computed and cached to disk.
ModelII — analytic phenomenological rate (power-law in Mc and redshift).
"""

import os
import numpy as np
from scipy.interpolate import make_interp_spline, RegularGridInterpolator
from scipy.integrate import quad as scipy_quad
from scipy import stats, optimize
from scipy.stats import qmc

from .constants import h_cosmo, Omega_M, Omega_L
from ._nb_kernels import NUMBA_AVAILABLE, nb_model_i_eval

# ── Sheth–Tormen halo mass function & EPS progenitor rate ────────────────────

class _PhysicsEngine:
    """Singleton that initialises the CDM matter power spectrum on first use."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("  Initialising matter power spectrum ...", end='', flush=True)
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
            print(" done")
        return cls._instance

    def _initialize(self):
        self.kpc_int   = 1.0
        self.meter_int = 3.24e-20 * self.kpc_int
        self.s_int     = 1.0
        self.kg_int    = 1.0
        self.h_param   = h_cosmo
        self.H0_int    = 100 * self.h_param * self.meter_int
        self.Omega_M   = Omega_M;  self.Omega_b = 0.0493
        self.Omega_L   = Omega_L
        self.Omega_c   = self.Omega_M - self.Omega_b
        self.sigma8    = 0.811;    self.zeq  = 3402
        self.T0        = 2.7255;   self.ns   = 0.965
        self.Omega_R   = self.Omega_M * (1+self.zeq)**3 / (1+self.zeq)**4
        G_int  = 6.674e-11 * self.meter_int**3 / self.kg_int / self.s_int**2
        rho_c  = 3 * self.H0_int**2 / (8 * np.pi * G_int)
        self.rho_M0 = rho_c * self.Omega_M / 1.989e30 * self.kpc_int**3
        c_int  = 2.998e8 * self.meter_int / self.s_int
        self.keq = (self.H0_int * np.sqrt(
            self.Omega_M*(1+self.zeq)**3 + self.Omega_R*(1+self.zeq)**4 + self.Omega_L
        )) / (1+self.zeq) * self.kpc_int / c_int
        log10m = np.arange(-16, 32.1, 0.2)
        m_vals = 10**log10m
        s2_vals = np.array([self._sigma_f(m) for m in m_vals])
        tmp  = make_interp_spline(m_vals, s2_vals, k=1)
        nf   = self.sigma8 / tmp(2.7803939422903778e14)
        self.sigma_CDM_spline = make_interp_spline(m_vals, s2_vals * nf, k=1)
        self.dsigma_dM_spline = self.sigma_CDM_spline.derivative()

    def _AH_int(self, z):
        return np.sqrt(self.Omega_M*(1+z)**3 + self.Omega_R*(1+z)**4 + self.Omega_L)

    def _H_int(self, z):
        return self.H0_int * self._AH_int(z)

    def _T_transfer(self, kk):
        h  = self.h_param; Om, Ob, Oc = self.Omega_M, self.Omega_b, self.Omega_c
        ksilk = 1.6*(Ob*h**2)**0.52*(Om*h**2)**0.73*(1+(10.4*Om*h**2)**-0.95)/1e3
        q     = lambda k: k/(13.41*self.keq)
        a1    = (46.9*Om*h**2)**0.67*(1+(32.1*Om*h**2)**-0.532)
        a2    = (12.0*Om*h**2)**0.424*(1+(45.0*Om*h**2)**-0.582)
        alpha_c  = a1**(-Ob/Om) * a2**(-(Ob/Om)**3)
        b1 = 0.944*(1+(458*Om*h**2)**-0.708)**-1; b2 = (0.395*Om*h**2)**(-0.026)
        beta_c   = 1 / (1 + b1*((Oc/Om)**b2 - 1))
        C1  = lambda k: 14.2/alpha_c + 386/(1+69.9*q(k)**1.08)
        To1 = lambda k, ac, bc: (np.log(np.e+1.8*bc*q(k)) /
                                 (np.log(np.e+1.8*bc*q(k)) + C1(k)*q(k)**2))
        s2  = 44.5*np.log(9.83/(Om*h**2))/np.sqrt(1+10*(Ob*h**2)**(3/4))*1e3
        Tc  = lambda k: (1/(1+(k*s2/5.4)**4)*To1(k,1,beta_c)
                         + (1-1/(1+(k*s2/5.4)**4))*To1(k,alpha_c,beta_c))
        zeq_v = 2.50e4*Om*h**2*(self.T0/2.7)**-4
        b3 = 0.313*(Om*h**2)**-0.419*(1+0.607*(Om*h**2)**0.674)
        b4 = 0.238*(Om*h**2)**0.223
        zd   = 1291*(Om*h**2)**0.251/(1+0.659*(Om*h**2)**0.828)*(1+b3*(Ob*h**2)**b4)
        Rd   = 31.5*Ob*h**2*(self.T0/2.7)**-4*(zd/1e3)**-1
        alpha_b = 2.07*self.keq*s2*(1+Rd)**(-3/4)*(5/2*((1+zeq_v)/(1+zd))*
                  (-(6*np.sqrt(1+(1+zeq_v)/(1+zd)))+(2+3*(1+zeq_v)/(1+zd))*
                   np.log((np.sqrt(1+(1+zeq_v)/(1+zd))+1)/
                          (np.sqrt(1+(1+zeq_v)/(1+zd))-1))))
        beta_b    = 0.5+Ob/Om+(3-2*Ob/Om)*np.sqrt(1+(17.2*Om*h**2)**2)
        beta_node = 8.41*(Om*h**2)**0.435
        s3 = lambda k: s2/(1+(beta_node/(k*s2))**3)**(1/3)
        Tb = lambda k: (To1(k,1,1)/(1+(k*s2/5.2)**2)+alpha_b/(1+(beta_b/(k*s2))**3)
                        *np.exp(-(k/ksilk)**1.4))*np.sin(k*s3(k))/(k*s3(k))
        return Ob/Om*Tb(kk) + Oc/Om*Tc(kk)

    def _sigma_f(self, M):
        r    = lambda Mc: (3*Mc/(4*np.pi*self.rho_M0))**(1/3)
        WFT  = lambda x: 3*(np.sin(x)-x*np.cos(x))/(x**3)
        def P_spec(k):
            dh = 0.00005; ci = 2.998e8*self.meter_int/self.s_int
            CD = np.sqrt(dh**2*(ci*k/self.H0_int)**(3+self.ns)*np.abs(self._T_transfer(k))**2)
            return CD**2*(2*np.pi**2)/k**3
        val, _ = scipy_quad(lambda lk: (10**lk)**3*WFT(r(M)*10**lk)**2*P_spec(10**lk),
                            -20, 20, epsrel=1e-2)
        return 1/(np.sqrt(2)*np.pi)*np.sqrt(np.log(10)*val)

    def sigma_CDM(self, M): return self.sigma_CDM_spline(M)
    def dsigma_dM(self, M): return self.dsigma_dM_spline(M)

    def delta_c(self, z):
        Om, Ol = self.Omega_M, self.Omega_L
        def OmMz(z): return Om*(1+z)**3/self._AH_int(z)**2
        def OmLz(z): return Ol/self._AH_int(z)**2
        def Dg(z):
            om, ol = OmMz(z), OmLz(z)
            return 5/2*om/(om**(4/7)-ol+(1+om/2)*(1+ol/70))/(1+z)/0.78694
        term = 0.123*np.log10(Ol*(1+z)**3/(Ol*(1+z)**3+1-Ol))
        return 3/5*(3*np.pi/2)**(2/3)/Dg(z)*(1+term)


def _dndlogm(M, z, eng):
    return (eng.rho_M0 * 2 * eng.sigma_CDM(M) * np.abs(eng.dsigma_dM(M))
            * _pfcST(eng.delta_c(z), eng.sigma_CDM(M)**2))

def _pfcST(delta1, S1):
    return (0.114963*S1**(-3/2)*np.exp(-0.4*delta1**2/S1)
            *delta1*(1+1.06923*(delta1**2/S1)**(-3/10)))

def _dPdM0dt(M, M0, z, eng):
    S, S0 = eng.sigma_CDM(M)**2, eng.sigma_CDM(M0)**2
    if np.isscalar(S) and S0 >= S: return 0.0
    dz = 1e-5
    def de(d, s): return np.sqrt(0.707)*d*(1+0.485*(0.707*d**2/s)**(-0.615))
    ddelta = np.abs((de(eng.delta_c(z+dz),S0)-de(eng.delta_c(z-dz),S0))/(2*dz))
    nu, nu0 = eng.delta_c(z)**2/S, eng.delta_c(z)**2/S0
    a, p, q = 0.707, 0.3, 0.8
    A_fac = (1+(a*nu0)**(-p))**(-1)
    t1 = A_fac/np.sqrt(2*np.pi)*(1+(a*nu)**(-p))*(1+(q*nu0)**(-p))/(1+(q*nu)**(-p))
    t2 = np.nan_to_num((S/(S0*(S-S0)))**(3/2)*np.exp(-q*(nu0-nu)/2))
    t3 = ddelta*(1+z)*eng._H_int(z)*np.abs(2*eng.sigma_CDM(M0)*eng.dsigma_dM(M0))
    return t1*t2*t3

def _Rloglog(M1, M2, z, eng):
    Mp = np.minimum(M1, M2); M = M1+M2
    # return np.log(10)**2*_dndlogm(Mp,z,eng)*M*_dPdM0dt(Mp,M,z,eng)
    return _dndlogm(Mp,z,eng) * np.maximum(M1, M2) * _dPdM0dt(Mp,M,z,eng)

def _logMBH(Mh, z, a, b):
    Az = 0.046*(1+z)**(-0.38); logMaz = 11.79+0.2*z
    Gz = 0.709*(1+z)**(-0.18); Bz = 0.043*z+0.92
    denom = (Mh/10**logMaz)**(-Bz) + (Mh/10**logMaz)**Gz
    return a + b*np.log10(2*Az*Mh/denom/1e11)

def _pMBH(mbh, Mh, z, a, b, sigma):
    return stats.norm.pdf(np.log10(mbh), loc=_logMBH(Mh,z,a,b), scale=sigma)

def _logmvMin(mbh, z, a, b, sigma):
    """Minimum log10(Mhalo) where pMBH is significant (log10 PDF >= -3)."""
    def scaling_diff(Mh):
        return _logMBH(Mh, z, a, b) - np.log10(mbh)
    try:
        if scaling_diff(1e7) * scaling_diff(1e18) > 0:
            Mh_peak = 1e12
        else:
            Mh_peak = optimize.root_scalar(scaling_diff, bracket=[1e7, 1e18],
                                           method='brentq').root
    except Exception:
        Mh_peak = 1e12
    def f(Mh):
        if Mh <= 100: return -100.0
        lp = stats.norm.logpdf(np.log10(mbh), loc=_logMBH(Mh, z, a, b), scale=sigma)
        return lp / np.log(10) - (-3.0)
    try:
        return np.log10(optimize.root_scalar(f, bracket=[1.0, Mh_peak],
                                             method='brentq').root)
    except ValueError:
        return np.log10(Mh_peak)


def _Rastro_qmc(m1, m2, z, a, b, sigma, n=2**12, merger_rate_fn=None):
    if merger_rate_fn is None:
        merger_rate_fn = _Rloglog
    eng  = _PhysicsEngine()
    UNIT_CONV = 1e9 * 3.15576e7
    m1, m2, z = np.broadcast_arrays(np.atleast_1d(m1), np.atleast_1d(m2), np.atleast_1d(z))
    res  = np.zeros_like(m1, dtype=float)
    u_base = qmc.Sobol(d=2, scramble=False).random(n)
    hi = 18.0
    for m1v, m2v, zv, rv in np.nditer([m1,m2,z,res], op_flags=[['readonly'],['readonly'],['readonly'],['writeonly']]):
        m1f, m2f, zf = float(m1v), float(m2v), float(zv)
        lo1 = _logmvMin(m1f, zf, a, b, sigma)
        lo2 = _logmvMin(m2f, zf, a, b, sigma)
        lM1 = lo1 + (hi - lo1) * u_base[:, 0]
        lM2 = lo2 + (hi - lo2) * u_base[:, 1]
        M1h, M2h = 10**lM1, 10**lM2
        area = (hi - lo1) * (hi - lo2)
        val = np.nanmean(merger_rate_fn(M1h, M2h, zf, eng)
                         * _pMBH(m1f, M1h, zf, a, b, sigma)
                         * _pMBH(m2f, M2h, zf, a, b, sigma)) * area
        rv[...] = val * UNIT_CONV
    return res


# ── Spherical collapse (Press-Schechter / Lacey-Cole 1993) merger rate ────────

def _nps_SC(M, z, eng):
    """Press-Schechter mass function dn/d ln M for spherical collapse."""
    sig  = eng.sigma_CDM(M)
    dc_z = eng.delta_c(z)
    return (np.sqrt(2.0 / np.pi) * eng.rho_M0
            * dc_z / sig**2 * np.abs(eng.dsigma_dM(M))
            * np.exp(-dc_z**2 / (2.0 * sig**2)))


def _Q_SC(M1, M2, z, eng):
    """Lacey-Cole (1993) conditional merger kernel [kpc^3 s^-1]."""
    Mf    = M1 + M2
    sig1  = eng.sigma_CDM(M1)
    sig2  = eng.sigma_CDM(M2)
    sigf  = eng.sigma_CDM(Mf)
    dsigf = eng.dsigma_dM(Mf)
    dsig2 = eng.dsigma_dM(M2)
    dc_z  = eng.delta_c(z)

    # dt/dz = -1/((1+z)*H(z))  [s, negative]
    dtz_dz = -1.0 / ((1.0 + z) * eng._H_int(z))

    dz = 1e-5
    ddelta_dz = (eng.delta_c(z + dz) - eng.delta_c(z - dz)) / (2.0 * dz)

    exp_arg = -dc_z**2 / 2.0 * (1.0/sigf**2 - 1.0/sig1**2 - 1.0/sig2**2)

    return ((-M2 / dtz_dz)                      # > 0
            * (1.0 / eng.rho_M0)
            * (ddelta_dz / dc_z)
            * (dsigf / dsig2)                    # both < 0 → ratio > 0
            * (sig2 / sigf)**2
            * (1.0 - (sigf / sig1)**2)**(-1.5)
            * np.exp(exp_arg))


def _Rloglog_SC(M1, M2, z, eng):
    """Halo merger rate — spherical collapse (Press-Schechter/Lacey-Cole).

    Same interface and units as _Rloglog: d²R/(d ln M1 d ln M2 dt) [kpc^-3 s^-1].
    """
    m_s = np.minimum(M1, M2)
    m_l = np.maximum(M1, M2)
    return _nps_SC(m_s, z, eng) * _nps_SC(m_l, z, eng) * _Q_SC(m_s, m_l, z, eng)


# ── Public model classes ──────────────────────────────────────────────────────

class MergerRateModel:
    """Abstract base for merger rate models."""
    pass


class ModelI(MergerRateModel):
    """
    Semi-numerical SMBHB merger rate derived from the EPS halo merger tree
    convolved with a log-normal MBH–halo mass relation.

    Parameters
    ----------
    a, b    : MBH–halo mass relation (amplitude and slope)
    sigma   : scatter in MBH–halo relation [dex]
    pbh     : binary fraction (scales the overall rate)
    """
    _grid_cache = {}

    def __init__(self, a=9.0, b=1.5, sigma=0.5, pbh=1.0):
        self.a=a; self.b=b; self.sigma=sigma; self.pbh=pbh
        self.params_key = (a, b, sigma, pbh)
        cached = ModelI._grid_cache.get(self.params_key)
        if isinstance(cached, dict):
            self.interpolator    = cached['interp']
            self._grid_data      = cached['grid']
            self._lm1_min        = cached['lm1_min']
            self._lm1_step       = cached['lm1_step']
            self._lm2_min        = cached['lm2_min']
            self._lm2_step       = cached['lm2_step']
            self._lz_min         = cached['lz_min']
            self._lz_step        = cached['lz_step']
            self._R_eff_grid     = cached.get('R_eff_grid')
            self._R_eff_lMc_min  = cached.get('R_eff_lMc_min', 5.0)
            self._R_eff_lMc_step = cached.get('R_eff_lMc_step', 8.0/299.0)
            self._R_eff_n_Mc     = cached.get('R_eff_n_Mc', 300)
        else:
            self._load_or_build_grid()

    def _load_or_build_grid(self):
        cache_dir = os.path.join(os.getcwd(), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"rate_grid_a{self.a}_b{self.b}_sig{self.sigma}_pbh{self.pbh}.npy"
        fpath = os.path.join(cache_dir, fname)
        m_min,m_max,m_pts = 5,13,50
        z_min,z_max,z_pts = -8,np.log10(8),30
        log_m = np.linspace(m_min,m_max,m_pts)
        log_z = np.linspace(z_min,z_max,z_pts)
        vals  = None
        if os.path.exists(fpath):
            try:
                print(f"  ModelI: loading rate grid from cache ...", end='', flush=True)
                vals = np.load(fpath)
                if vals.shape != (m_pts,m_pts,z_pts): vals = None
                if vals is not None: print(" done")
            except Exception: vals = None
        if vals is None:
            print("  ModelI: building rate grid (this may take a few minutes) ...")
            vals = np.zeros((m_pts, m_pts, z_pts))
            _PhysicsEngine()
            m_v = 10**log_m; z_v = 10**log_z
            for k, zk in enumerate(z_v):
                M1, M2 = np.meshgrid(m_v, m_v, indexing='ij')
                vals[:,:,k] = _Rastro_qmc(M1,M2,max(zk,1e-8),
                                          self.a,self.b,self.sigma,n=2**12)*self.pbh
            np.save(fpath, vals)
            print(f"  ModelI: rate grid saved → {fname}")
        interp = RegularGridInterpolator((log_m,log_m,log_z), vals,
                                         bounds_error=False, fill_value=0.0)
        lm_step = (m_max - m_min) / (m_pts - 1)
        lz_step = (z_max - z_min) / (z_pts - 1)
        self._grid_data = np.ascontiguousarray(vals, dtype=np.float64)
        self._lm1_min   = float(m_min);  self._lm1_step = float(lm_step)
        self._lm2_min   = float(m_min);  self._lm2_step = float(lm_step)
        self._lz_min    = float(z_min);  self._lz_step  = float(lz_step)
        self.interpolator = interp
        self._R_eff_grid = None  # built lazily on first _gwad_density call
        ModelI._grid_cache[self.params_key] = {
            'interp':   interp,
            'grid':     self._grid_data,
            'lm1_min':  self._lm1_min,  'lm1_step': self._lm1_step,
            'lm2_min':  self._lm2_min,  'lm2_step': self._lm2_step,
            'lz_min':   self._lz_min,   'lz_step':  self._lz_step,
        }

    def _ensure_R_eff(self):
        """Build and cache the η-integrated rate table R_eff(Mc, z) if not done yet."""
        if self._R_eff_grid is not None:
            return
        # Lazy import to avoid circular dependency (gwad imports merger_rates)
        from .gwad import (_ETA_GWAD, _KERN_GWAD, _M1C_GWAD, _M2C_GWAD,
                           _W_ETA_GWAD, _LOGZ_GWAD, _Z_GWAD)

        n_Mc_pts = 300
        lMc_min, lMc_max = 5.0, 13.0
        lMc_step = (lMc_max - lMc_min) / (n_Mc_pts - 1)
        Mc_v = 10.0 ** np.linspace(lMc_min, lMc_max, n_Mc_pts)

        n_eta = len(_ETA_GWAD)
        n_z   = len(_Z_GWAD)

        # Build all (Mc, eta, z) evaluation points in one batch — no Python loop.
        # m1[i,k] = Mc[i] * M1C[k],  shape (n_Mc, n_eta)
        m1_2d = Mc_v[:, None] * _M1C_GWAD[None, :]   # (100, 40)
        m2_2d = Mc_v[:, None] * _M2C_GWAD[None, :]

        # Broadcast to (n_Mc, n_eta, n_z) then flatten
        m1_flat = np.ascontiguousarray(
            np.broadcast_to(m1_2d[:, :, None], (n_Mc_pts, n_eta, n_z)).reshape(-1))
        m2_flat = np.ascontiguousarray(
            np.broadcast_to(m2_2d[:, :, None], (n_Mc_pts, n_eta, n_z)).reshape(-1))
        z_flat  = np.ascontiguousarray(
            np.broadcast_to(_Z_GWAD[None, None, :], (n_Mc_pts, n_eta, n_z)).reshape(-1))

        rate = nb_model_i_eval(m1_flat, m2_flat, z_flat,
                               self._grid_data,
                               self._lm1_min, self._lm1_step,
                               self._lm2_min, self._lm2_step,
                               self._lz_min,  self._lz_step
                               ).reshape(n_Mc_pts, n_eta, n_z)

        # Integrate over eta: R_eff[i, j] = sum_k rate[i,k,j] * kernel[k] * w_eta[k]
        W = (_KERN_GWAD * _W_ETA_GWAD).astype(np.float64)
        R_eff = np.einsum('iej,e->ij', rate, W)   # (n_Mc_pts, n_z)

        self._R_eff_grid     = R_eff
        self._R_eff_lMc_min  = lMc_min
        self._R_eff_lMc_step = lMc_step
        self._R_eff_n_Mc     = n_Mc_pts

        # Update the in-process cache so subsequent ModelI instances skip the build
        cached = ModelI._grid_cache.get(self.params_key, {})
        cached.update(R_eff_grid=R_eff, R_eff_lMc_min=lMc_min,
                      R_eff_lMc_step=lMc_step, R_eff_n_Mc=n_Mc_pts)
        ModelI._grid_cache[self.params_key] = cached

    def R_eff_eval(self, Mc_2d, Z_1d):
        """
        Evaluate the η-integrated rate R_eff(Mc, z) at an (n_A, n_z) grid.

        R_eff(Mc, z) = ∫ R(m1(Mc,η), m2(Mc,η), z) · kernel(η) dη

        Parameters
        ----------
        Mc_2d : (n_A, n_z) array — chirp masses [Msun]
        Z_1d  : (n_z,) array  — redshifts (must match the module-level z grid)

        Returns
        -------
        (n_A, n_z) array
        """
        self._ensure_R_eff()
        # 1D linear interpolation in Mc for each z column.
        # The z axis of _R_eff_grid matches _LOGZ_GWAD exactly, so j is the
        # column index directly — no interpolation in the z direction.
        lMc  = np.log10(np.clip(Mc_2d, 10.0**self._R_eff_lMc_min, 1e13))
        fidx = (lMc - self._R_eff_lMc_min) / self._R_eff_lMc_step
        np.clip(fidx, 0.0, float(self._R_eff_n_Mc) - 1.0001, out=fidx)
        idx  = fidx.astype(np.intp)
        t    = fidx - idx
        j    = np.arange(Mc_2d.shape[1])[None, :]   # (1, n_z)
        return (self._R_eff_grid[idx, j] * (1.0 - t) +
                self._R_eff_grid[idx + 1, j] * t)

    def __call__(self, m1, m2, z):
        if NUMBA_AVAILABLE:
            return nb_model_i_eval(m1, m2, z,
                                   self._grid_data,
                                   self._lm1_min, self._lm1_step,
                                   self._lm2_min, self._lm2_step,
                                   self._lz_min,  self._lz_step)
        lm1 = np.log10(np.clip(m1, 1e5, 1e13))
        lm2 = np.log10(np.clip(m2, 1e5, 1e13))
        lz  = np.log10(np.clip(z,  1e-5, 6))
        shape = np.asarray(m1).shape
        pts = np.column_stack((np.ravel(lm1), np.ravel(lm2), np.ravel(lz)))
        return self.interpolator(pts).reshape(shape)


class ModelII(MergerRateModel):
    """
    Analytic phenomenological merger rate:
        d²N/(dM_c dz) ∝ (Mc/M*)^c * exp(-Mc/M*) * (1+z)^d * exp(-z/z0)

    Parameters
    ----------
    R0     : overall normalisation
    M_star : chirp-mass scale [Msun]
    c      : low-mass power-law index
    d      : redshift evolution index
    z0     : redshift decay scale
    """
    def __init__(self, R0=4e-14, M_star=2.5e9, c=-0.2, d=6.0, z0=0.3):
        self.R0=R0; self.M_star=M_star; self.c=c; self.d=d; self.z0=z0

    def __call__(self, Mc, z):
        return (self.R0/Mc * (Mc/self.M_star)**self.c * np.exp(-Mc/self.M_star)
                * (1+z)**self.d * np.exp(-z/self.z0))


class ModelISC(ModelI):
    """
    ModelI with spherical collapse (Press-Schechter / Lacey-Cole 1993) merger rate.

    Drop-in replacement for ModelI; all post-grid machinery is identical.
    The only difference is that the halo merger rate uses _Rloglog_SC instead
    of the EPS-based _Rloglog.
    """
    _grid_cache = {}

    def __init__(self, a=9.0, b=1.5, sigma=0.5, pbh=1.0):
        # Bypass ModelI.__init__ grid lookup; use our own cache dict.
        self.a=a; self.b=b; self.sigma=sigma; self.pbh=pbh
        self.params_key = (a, b, sigma, pbh)
        cached = ModelISC._grid_cache.get(self.params_key)
        if isinstance(cached, dict):
            self.interpolator    = cached['interp']
            self._grid_data      = cached['grid']
            self._lm1_min        = cached['lm1_min']
            self._lm1_step       = cached['lm1_step']
            self._lm2_min        = cached['lm2_min']
            self._lm2_step       = cached['lm2_step']
            self._lz_min         = cached['lz_min']
            self._lz_step        = cached['lz_step']
            self._R_eff_grid     = cached.get('R_eff_grid')
            self._R_eff_lMc_min  = cached.get('R_eff_lMc_min', 5.0)
            self._R_eff_lMc_step = cached.get('R_eff_lMc_step', 8.0/299.0)
            self._R_eff_n_Mc     = cached.get('R_eff_n_Mc', 300)
        else:
            self._load_or_build_grid()

    def _load_or_build_grid(self):
        cache_dir = os.path.join(os.getcwd(), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"rate_grid_SC_a{self.a}_b{self.b}_sig{self.sigma}_pbh{self.pbh}.npy"
        fpath = os.path.join(cache_dir, fname)
        m_min,m_max,m_pts = 5,13,50
        z_min,z_max,z_pts = -8,np.log10(8),30
        log_m = np.linspace(m_min,m_max,m_pts)
        log_z = np.linspace(z_min,z_max,z_pts)
        vals  = None
        if os.path.exists(fpath):
            try:
                print(f"  ModelISC: loading rate grid from cache ...", end='', flush=True)
                vals = np.load(fpath)
                if vals.shape != (m_pts,m_pts,z_pts): vals = None
                if vals is not None: print(" done")
            except Exception: vals = None
        if vals is None:
            print("  ModelISC: building rate grid (this may take a few minutes) ...")
            vals = np.zeros((m_pts, m_pts, z_pts))
            _PhysicsEngine()
            m_v = 10**log_m; z_v = 10**log_z
            for k, zk in enumerate(z_v):
                M1, M2 = np.meshgrid(m_v, m_v, indexing='ij')
                vals[:,:,k] = _Rastro_qmc(M1, M2, max(zk, 1e-8),
                                          self.a, self.b, self.sigma, n=2**12,
                                          merger_rate_fn=_Rloglog_SC) * self.pbh
            np.save(fpath, vals)
            print(f"  ModelISC: rate grid saved → {fname}")
        interp = RegularGridInterpolator((log_m,log_m,log_z), vals,
                                         bounds_error=False, fill_value=0.0)
        lm_step = (m_max - m_min) / (m_pts - 1)
        lz_step = (z_max - z_min) / (z_pts - 1)
        self._grid_data = np.ascontiguousarray(vals, dtype=np.float64)
        self._lm1_min   = float(m_min);  self._lm1_step = float(lm_step)
        self._lm2_min   = float(m_min);  self._lm2_step = float(lm_step)
        self._lz_min    = float(z_min);  self._lz_step  = float(lz_step)
        self.interpolator = interp
        self._R_eff_grid = None
        ModelISC._grid_cache[self.params_key] = {
            'interp':   interp,
            'grid':     self._grid_data,
            'lm1_min':  self._lm1_min,  'lm1_step': self._lm1_step,
            'lm2_min':  self._lm2_min,  'lm2_step': self._lm2_step,
            'lz_min':   self._lz_min,   'lz_step':  self._lz_step,
        }

    def _ensure_R_eff(self):
        """Same as ModelI._ensure_R_eff but updates ModelISC._grid_cache."""
        if self._R_eff_grid is not None:
            return
        from .gwad import (_ETA_GWAD, _KERN_GWAD, _M1C_GWAD, _M2C_GWAD,
                           _W_ETA_GWAD, _LOGZ_GWAD, _Z_GWAD)
        n_Mc_pts = 300
        lMc_min, lMc_max = 5.0, 13.0
        lMc_step = (lMc_max - lMc_min) / (n_Mc_pts - 1)
        Mc_v = 10.0 ** np.linspace(lMc_min, lMc_max, n_Mc_pts)
        n_eta = len(_ETA_GWAD)
        n_z   = len(_Z_GWAD)
        m1_2d = Mc_v[:, None] * _M1C_GWAD[None, :]
        m2_2d = Mc_v[:, None] * _M2C_GWAD[None, :]
        m1_flat = np.ascontiguousarray(
            np.broadcast_to(m1_2d[:, :, None], (n_Mc_pts, n_eta, n_z)).reshape(-1))
        m2_flat = np.ascontiguousarray(
            np.broadcast_to(m2_2d[:, :, None], (n_Mc_pts, n_eta, n_z)).reshape(-1))
        z_flat  = np.ascontiguousarray(
            np.broadcast_to(_Z_GWAD[None, None, :], (n_Mc_pts, n_eta, n_z)).reshape(-1))
        rate = nb_model_i_eval(m1_flat, m2_flat, z_flat,
                               self._grid_data,
                               self._lm1_min, self._lm1_step,
                               self._lm2_min, self._lm2_step,
                               self._lz_min,  self._lz_step
                               ).reshape(n_Mc_pts, n_eta, n_z)
        W = (_KERN_GWAD * _W_ETA_GWAD).astype(np.float64)
        R_eff = np.einsum('iej,e->ij', rate, W)
        self._R_eff_grid     = R_eff
        self._R_eff_lMc_min  = lMc_min
        self._R_eff_lMc_step = lMc_step
        self._R_eff_n_Mc     = n_Mc_pts
        cached = ModelISC._grid_cache.get(self.params_key, {})
        cached.update(R_eff_grid=R_eff, R_eff_lMc_min=lMc_min,
                      R_eff_lMc_step=lMc_step, R_eff_n_Mc=n_Mc_pts)
        ModelISC._grid_cache[self.params_key] = cached
