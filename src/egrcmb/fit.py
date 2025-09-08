import numpy as np
from scipy.optimize import curve_fit
from .constants import MAXFEV_FIT

def build_total_error(sig_planck, data_vals, epsilon_model: float):
    eps = float(np.clip(epsilon_model, 0.0, 1.0))
    return np.sqrt(np.asarray(sig_planck, float)**2 + (eps*np.asarray(data_vals, float))**2)

def ledger_info_overhead(sig_meas, y_obs, eps_frac):
    sig2  = np.asarray(sig_meas, float)**2
    add2  = (float(eps_frac) * np.asarray(y_obs, float))**2
    ratio = add2 / (sig2 + 1e-30)
    total_nats = 0.5 * np.sum(np.log1p(ratio))
    n = max(len(y_obs), 1)
    pm = total_nats / n
    ln2 = np.log(2.0)
    return total_nats, pm, total_nats/ln2, pm/ln2

def multiple_random_starts(func, ell, Dl_data, Dl_err, p0, bounds, n_starts=12, seed=42):
    rng = np.random.default_rng(seed)
    best_sol, best_pcov, lowest_chi2 = None, None, float('inf')
    lb, ub = bounds
    for _ in range(n_starts):
        factor = 1.0 + 0.30*(2.0*rng.random(len(p0))-1.0)
        p0_rand = np.clip(p0 * factor, lb, ub)
        try:
            popt, pcov = curve_fit(
                func, ell, Dl_data,
                p0=p0_rand, sigma=Dl_err,
                bounds=(lb, ub),
                maxfev=MAXFEV_FIT, method='trf',
                loss='soft_l1', f_scale=0.5
            )
            preds = func(ell, *popt)
            chi2 = float(np.sum(((Dl_data - preds)/Dl_err)**2))
            if chi2 < lowest_chi2 and np.all(np.isfinite(popt)):
                best_sol, best_pcov, lowest_chi2 = popt, pcov, chi2
        except Exception:
            continue
    return best_sol, best_pcov

def run_fit(epsilon_model, ell, Dl_data, Dl_err_planck, model_func, p0, lb, ub, nstarts=12, seed=42):
    err = build_total_error(Dl_err_planck, Dl_data, epsilon_model)
    popt, _ = multiple_random_starts(model_func, ell, Dl_data, err, p0, (lb, ub), n_starts=nstarts, seed=seed)
    if popt is None:
        return None
    preds = model_func(ell, *popt)
    chi2  = float(np.sum(((Dl_data - preds)/err)**2))
    dof   = max(len(ell) - len(popt), 1)
    Qexp  = float(popt[0] * popt[-1])
    H_tot_nats, H_pp_nats, H_tot_bits, H_pp_bits = ledger_info_overhead(Dl_err_planck, Dl_data, epsilon_model)
    hit = np.isclose(popt, lb, atol=1e-10) | np.isclose(popt, ub, atol=1e-10)
    return {
        "epsilon_model": float(epsilon_model),
        "popt": popt,
        "hit_bounds": hit.tolist(),
        "Qexp": Qexp,
        "chi2_red": chi2/dof,
        "err": err,
        "preds": preds,
        "deltaH_tot_nats": float(H_tot_nats),
        "deltaH_pp_nats": float(H_pp_nats),
        "deltaH_tot_bits": float(H_tot_bits),
        "deltaH_pp_bits": float(H_pp_bits)
    }
