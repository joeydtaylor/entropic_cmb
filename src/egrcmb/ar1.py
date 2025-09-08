import numpy as np
from scipy.optimize import fminbound

def ar1_whiten(z, rho):
    z = np.asarray(z, float)
    if abs(rho) < 1e-12:  return z, 1.0
    out = np.empty_like(z)
    out[0] = z[0]*np.sqrt(1.0 - rho**2)
    out[1:] = (z[1:] - rho*z[:-1])
    return out, 1.0

def fit_ar1_rho(z, bounds=(-0.9, 0.9)):
    z = np.asarray(z, float)
    a, b = bounds
    grid = np.linspace(a, b, 121)
    ss = []
    for r in grid:
        zw, _ = ar1_whiten(z, r)
        ss.append(float(np.dot(zw, zw)))
    j = int(np.argmin(ss)); step = grid[1]-grid[0]
    lo, hi = max(a, grid[j]-2*step), min(b, grid[j]+2*step)
    def obj(r): zw,_ = ar1_whiten(z, r); return float(np.dot(zw, zw))
    rho_star = float(fminbound(obj, lo, hi)); chi2_star = obj(rho_star)
    return rho_star, chi2_star

def redchi_ar1_for_epsilon(Dl, yhat, sig_planck, eps_frac):
    err = np.sqrt(np.asarray(sig_planck, float)**2 + (float(eps_frac)*np.asarray(Dl, float))**2)
    z   = (np.asarray(Dl, float) - np.asarray(yhat, float)) / err
    rho_star, chi2_star = fit_ar1_rho(z)
    dof = max(len(Dl) - 7, 1)
    return rho_star, chi2_star / dof

def calibrate_epsilon_ar1(Dl, yhat, sig_planck, s_init=0.075):
    def objective(s): _, rc = redchi_ar1_for_epsilon(Dl, yhat, sig_planck, s); return (rc - 1.0)**2
    s_star = float(fminbound(objective, 0.0, max(0.5, 3.0*s_init)))
    rho_star, rc_star = redchi_ar1_for_epsilon(Dl, yhat, sig_planck, s_star)
    return s_star, rho_star, rc_star

def whitened_residual_stats(Dl, preds, err):
    r = (np.asarray(Dl, float) - np.asarray(preds, float)) / np.asarray(err, float)
    return float(np.mean(r)), float(np.std(r)), r

def residual_acf(r, maxlag=50):
    r = np.asarray(r, float) - np.mean(r)
    denom = float(np.dot(r, r)) + 1e-30
    acf = [1.0]
    for L in range(1, maxlag+1):
        acf.append(float(np.dot(r[:-L], r[L:]))/denom)
    return np.array(acf)

def ledger_band_overhead(ell, sig_meas, y_obs, eps_frac,
                         edges=(2, 30, 200, 800, 2000, 3000)):
    sig2 = np.asarray(sig_meas, float)**2
    add2 = (float(eps_frac) * np.asarray(y_obs, float))**2
    ratio = add2 / (sig2 + 1e-30)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ell >= a) & (ell < b); n = int(np.sum(m))
        if n == 0: out.append((a, b, 0.0, 0.0)); continue
        H = 0.5 * float(np.sum(np.log1p(ratio[m])))
        out.append((a, b, H, H / n))
    return out
