import numpy as np
import matplotlib.pyplot as plt

def gof_and_normality(residual_z, dof):
    import scipy.stats as sps
    z = np.asarray(residual_z, float)
    N = z.size
    chi2 = float(np.dot(z, z))
    p_chi = float(sps.chi2.sf(chi2, dof))
    ks_stat, ks_p = sps.kstest((z - z.mean())/z.std(ddof=0), 'norm')
    jb_stat, jb_p = sps.jarque_bera(z)
    mean = float(np.mean(z)); std = float(np.std(z, ddof=0))
    skew = float(sps.skew(z, bias=False)); kurt = float(sps.kurtosis(z, fisher=True, bias=False))
    return {"N": int(N), "dof": int(dof), "chi2": chi2, "p_chi2": p_chi,
            "mean": mean, "std": std, "skew": skew, "kurtosis_excess": kurt,
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "jb_stat": float(jb_stat), "jb_p": float(jb_p)}

def aic_bic(chi2, k, N):
    AIC = float(chi2 + 2*k)
    BIC = float(chi2 + k*np.log(N))
    return AIC, BIC

def save_fit_table(path, ell, Dl, pred, err, z):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell","D_ell","model","sigma_eff","z_whitened"])
        for i in range(len(ell)):
            w.writerow([int(ell[i]), float(Dl[i]), float(pred[i]), float(err[i]), float(z[i])])

def qqplot_normal(path, z):
    import scipy.stats as sps
    z = np.asarray(z, float)
    N = z.size
    probs = (np.arange(1, N+1) - 0.5)/N
    q_theory = sps.norm.ppf(probs)
    q_sample = np.sort((z - z.mean())/z.std(ddof=0))
    plt.figure(figsize=(6,6))
    plt.plot(q_theory, q_sample, '.', ms=2)
    lim = float(np.max(np.abs(np.concatenate([q_theory, q_sample]))))
    lim = max(lim, 3.5)
    plt.plot([-lim, lim], [-lim, lim], '-', lw=1, alpha=0.7)
    plt.xlabel("Normal quantiles"); plt.ylabel("Sample quantiles")
    plt.title("Q–Q plot (whitened residuals)"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()

def corr_heatmap_from_chain(path, chain, names):
    if chain is None or chain.size == 0: return None
    C = np.corrcoef(chain, rowvar=False)
    plt.figure(figsize=(6,5))
    im = plt.imshow(C, vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.title("Posterior correlation"); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()
    return C
